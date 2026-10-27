"""The search loop: select an action, write code, run it, grade it, record the node."""

import time
from collections.abc import Callable
from pathlib import Path

from kagent import harness
from kagent.config import CompetitionConfig, Settings
from kagent.grader import metrics
from kagent.llm.client import ChatClient
from kagent.search.operators import Context, Operator, OperatorError, summarize
from kagent.search.policies import SearchPolicy
from kagent.search.tree import Node, SolutionTree, Status
from kagent.workspace import Workspace

MIN_STEP_S = 60  # do not start a step with less than this left
LOG_TAIL = 3000

RunScript = Callable[[Workspace, Path, Settings], harness.RunRecord]


class IntegrityAbort(RuntimeError):
    """A run changed protected files. The search stops: the harness can no longer be trusted."""


class OperatorFailures(RuntimeError):
    """The coder keeps returning unusable code; stop instead of burning the budget."""


def open_tree(ws: Workspace, config: CompetitionConfig) -> SolutionTree:
    higher = metrics.METRICS[config.metric].higher_is_better
    return SolutionTree(ws.search_dir / "nodes.jsonl", higher_is_better=higher)


def _tail(path: Path, limit: int = LOG_TAIL) -> str:
    return path.read_text(errors="replace")[-limit:] if path.exists() else ""


def _outcome(record: harness.RunRecord, step_timeout_s: int) -> tuple[bool, str]:
    """(usable, error). usable means a graded cv_score exists."""
    grade = record.grade
    if record.status == "ok" and grade is not None and grade.cv_score is not None:
        return True, ""
    if record.status == "timeout":
        return False, f"timed out after {step_timeout_s}s"
    if record.status == "failed":
        return False, record.stderr_tail[-LOG_TAIL:] or f"exit code {record.exit_code}"
    if record.status == "invalid_submission":
        errors = grade.errors if grade else []
        return False, "invalid submission: " + ("; ".join(errors) or "no submission.csv written")
    detail = "; ".join(grade.cv_errors) if grade else ""
    return False, "cross-validation failed: " + (detail or "no oof.csv written")


def run_search(
    ws: Workspace,
    config: CompetitionConfig,
    settings: Settings,
    tree: SolutionTree,
    policy: SearchPolicy,
    operators: dict[str, Operator],
    planner: ChatClient,
    *,
    description: str,
    eda: str,
    budget_s: float,
    max_nodes: int | None = None,
    max_debug: int = 3,
    max_operator_failures: int = 3,
    run: RunScript = harness.run_script,
    log: Callable[[str], None] = lambda _: None,
    clock: Callable[[], float] = time.monotonic,
) -> SolutionTree:
    target = ws.target_column(config)
    deadline = clock() + budget_s
    start_size = len(tree)
    failures = 0
    (ws.search_dir / "nodes").mkdir(parents=True, exist_ok=True)

    while True:
        remaining = deadline - clock()
        if remaining < MIN_STEP_S:
            log("time budget used up")
            break
        if max_nodes is not None and len(tree) - start_size >= max_nodes:
            log(f"reached max nodes ({max_nodes})")
            break

        action = policy.select(tree)
        parent = action.parent
        step_timeout = int(min(settings.step_timeout_s, remaining))
        ctx = Context(
            description=description,
            eda=eda,
            metric=config.metric,
            higher_is_better=tree.higher_is_better,
            id_column=config.id_column,
            target=target,
            tree=tree,
            step_timeout_s=step_timeout,
            remaining_s=remaining,
            sandbox_cpus=settings.sandbox_cpus,
            sandbox_memory=settings.sandbox_memory,
            error_log=parent.error if action.operator == "debug" and parent else "",
        )

        try:
            code = operators[action.operator].apply(parent, ctx)
        except OperatorError as exc:
            failures += 1
            log(
                f"{action.operator}: unusable code from model ({exc}) [{failures}/{max_operator_failures}]"
            )
            if failures >= max_operator_failures:
                raise OperatorFailures(str(exc)) from exc
            continue
        failures = 0

        node_id = tree.next_id()
        code_path = ws.search_dir / "nodes" / f"{node_id}.py"
        code_path.write_text(code)
        record = run(ws, code_path, settings.model_copy(update={"step_timeout_s": step_timeout}))

        debug_depth = parent.debug_depth + 1 if action.operator == "debug" and parent else 0
        base = Node(
            id=node_id,
            parent=parent.id if parent else None,
            operator=action.operator,
            code_path=code_path,
            status="buggy",
            run_id=record.run_id,
            debug_depth=debug_depth,
        )

        if record.status == "integrity_violation":
            base.status, base.error = (
                "dead",
                "integrity violation: " + "; ".join(record.integrity_changes),
            )
            tree.add(base)
            raise IntegrityAbort(base.error)

        usable, error = _outcome(record, step_timeout)
        if usable:
            grade = record.grade
            assert grade is not None
            base.status = "ok"
            base.cv_score, base.cv_std = grade.cv_score, grade.cv_std
            base.summary = summarize(
                planner,
                code=code,
                log=_tail(ws.run_dir(record.run_id) / "stdout.log"),
                metric=config.metric,
                cv_score=grade.cv_score,
                cv_std=grade.cv_std,
            )
        else:
            status: Status = "dead" if debug_depth >= max_debug else "buggy"
            base.status, base.error = status, error
        tree.add(base)

        if base.cv_score is not None:
            detail = f"cv={base.cv_score:.5f}"
        else:
            detail = base.error.strip().splitlines()[-1][:100] if base.error.strip() else ""
        origin = f" <- {base.parent}" if base.parent else ""
        log(f"{base.id} {action.operator}{origin}: {base.status} {detail}")

    return tree
