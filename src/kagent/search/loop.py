"""The search loop: select an action, write code, run it, grade it, record the node."""

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from kagent import harness
from kagent.config import CompetitionConfig, Settings
from kagent.deps.gate import DepGate, denial_message
from kagent.deps.lockfile import sandbox_versions
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


def _evaluate(
    base: Node,
    record: harness.RunRecord,
    *,
    code: str,
    ws: Workspace,
    config: CompetitionConfig,
    tree: SolutionTree,
    planner: ChatClient,
    deps: DepGate | None,
    step_timeout: int,
    max_debug: int,
) -> None:
    """Fill in the node's status, score and error from what the run produced."""
    if record.status == "integrity_violation":
        base.status = "dead"
        base.error = "integrity violation: " + "; ".join(record.integrity_changes)
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
        return

    failed: Status = "dead" if base.debug_depth >= max_debug else "buggy"
    request = deps.request_for_failure(base.id, record.stderr_tail) if deps else None
    if request is None:
        base.status, base.error = failed, error
    elif request.status == "denied":
        base.status, base.error = failed, denial_message(request, error)
    else:  # pending: park the node, other branches keep running
        base.status, base.error, base.dep_request = "blocked", error, request.id


def _describe(node: Node, operator: str) -> str:
    if node.cv_score is not None:
        detail = f"cv={node.cv_score:.5f}"
    elif node.status == "blocked":
        detail = f"waiting on {node.dep_request}"
    else:
        detail = node.error.strip().splitlines()[-1][:100] if node.error.strip() else ""
    origin = f" <- {node.parent}" if node.parent else ""
    return f"{node.id} {operator}{origin}: {node.status} {detail}"


def _settle_blocked(
    ws: Workspace,
    config: CompetitionConfig,
    settings: Settings,
    tree: SolutionTree,
    planner: ChatClient,
    deps: DepGate,
    run: RunScript,
    log: Callable[[str], None],
    *,
    step_timeout: int,
    max_debug: int,
) -> None:
    """Act on human decisions for parked nodes: rerun, hand the denial to debug, or give up."""
    for node in tree.blocked_nodes():
        request = deps.current(node.dep_request or "")
        if request.status == "pending" and deps.is_stale(request):
            request = deps.expire(request)

        if request.status == "approved":
            log(f"{node.id}: {request.package} {request.resolved_version} installed, rerunning")
            record = run(
                ws, node.code_path, settings.model_copy(update={"step_timeout_s": step_timeout})
            )
            fresh = replace(node, run_id=record.run_id, status="buggy", error="", dep_request=None)
            _evaluate(
                fresh, record, code=node.code_path.read_text(), ws=ws, config=config, tree=tree,
                planner=planner, deps=deps, step_timeout=step_timeout, max_debug=max_debug,
            )  # fmt: skip
            tree.add(fresh)
            log(_describe(fresh, "rerun"))
        elif request.status == "denied":
            status: Status = "dead" if node.debug_depth >= max_debug else "buggy"
            tree.add(
                replace(
                    node, status=status, error=denial_message(request, node.error), dep_request=None
                )
            )
            log(f"{node.id}: {request.package} denied, node returned to debugging")
        elif request.status == "expired":
            tree.add(
                replace(
                    node,
                    status="dead",
                    error=node.error + "\n(dependency request expired)",
                    dep_request=None,
                )
            )
            log(f"{node.id}: dependency request for {request.package} expired, node dead")


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
    deps: DepGate | None = None,
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

        step_timeout = int(min(settings.step_timeout_s, remaining))
        if deps is not None:
            _settle_blocked(
                ws, config, settings, tree, planner, deps, run, log,
                step_timeout=step_timeout, max_debug=max_debug,
            )  # fmt: skip

        action = policy.select(tree)
        parent = action.parent
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
            libraries=sandbox_versions(settings.docker_dir),
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

        base = Node(
            id=node_id,
            parent=parent.id if parent else None,
            operator=action.operator,
            code_path=code_path,
            status="buggy",
            run_id=record.run_id,
            debug_depth=parent.debug_depth + 1 if action.operator == "debug" and parent else 0,
        )
        _evaluate(
            base, record, code=code, ws=ws, config=config, tree=tree, planner=planner,
            deps=deps, step_timeout=step_timeout, max_debug=max_debug,
        )  # fmt: skip
        tree.add(base)
        log(_describe(base, action.operator))

    return tree
