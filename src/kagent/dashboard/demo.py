"""A pretend search that writes real workspace files, to see the dashboard without models or Docker.

It uses the real tree, progress and dependency-store classes, so what the dashboard reads here is
what it reads from a live search. The numbers are invented.
"""

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from kagent.config import CompetitionConfig
from kagent.deps.models import DependencyRequest, PreCheck
from kagent.deps.store import DependencyStore
from kagent.search.progress import Progress
from kagent.search.tree import Node, SolutionTree
from kagent.workspace import Workspace

PLANNER = "mlx-community/Qwen3.8-Flash-Next-4bit"
CODER = "mlx-community/Qwen3-Coder-Next-4bit"

PAGE_INFO = {
    "kind": "Kaggle · Getting Started Prediction Competition · Ongoing",
    "tags": [
        "Beginner",
        "Tabular",
        "Binary Classification",
        "Categorization Accuracy",
    ],
    "prizes": ["Knowledge", "Does not award Points or Medals"],
    "prize_warning": None,
}


def _trace(line: int, code: str, error: str) -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "/work/script.py", line {line}, in <module>\n'
        f"    {code}\n{error}"
    )


@dataclass(frozen=True)
class Step:
    operator: str
    parent: str | None
    status: str  # ok | buggy | dead | blocked
    plan: str
    cv: float | None = None
    std: float | None = None
    stage: str = "validate"
    error: str = ""
    summary: str = ""


MEMORY = _trace(22, "X = pd.get_dummies(df, sparse=False)", "MemoryError")

SCRIPT_TEMPLATE = '''# PLAN: __PLAN__
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

DATA = "/data"
SEED = 42


def load():
    train = pd.read_csv(f"{DATA}/train.csv")
    folds = pd.read_csv(f"{DATA}/folds.csv")
    print("STAGE: load")
    return train.merge(folds, on="PassengerId")


def features(df):
    """Split Cabin into deck, number and side, and add the spend total."""
    parts = df["Cabin"].str.split("/", expand=True)
    df["deck"], df["num"], df["side"] = parts[0], parts[1], parts[2]
    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["spend"] = df[spend].fillna(0).sum(axis=1)  # a missing amount means no spending
    print("STAGE: features")
    return df


def main():
    df = features(load())
    cols = [c for c in df.columns if c not in ("Transported", "fold")]
    scores = []
    for fold in sorted(df["fold"].unique()):
        valid = df["fold"] == fold
        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED)
        model.fit(df.loc[~valid, cols], df.loc[~valid, "Transported"])
        scores.append(model.score(df.loc[valid, cols], df.loc[valid, "Transported"]))
    print(f"cv accuracy: {np.mean(scores):.4f} +- {np.std(scores):.4f}")
    print("STAGE: cv")


if __name__ == "__main__":
    main()
'''


def script_text(step: Step) -> str:
    return SCRIPT_TEMPLATE.replace("__PLAN__", step.plan)


def plan_text(step: Step) -> str:
    where = f"on top of {step.parent}" if step.parent else "as a fresh starting point"
    return (
        f"Idea: {step.plan}.\n\n"
        f"Why: this builds {where} and changes one thing, so a score change can be read.\n"
        "Steps:\n"
        "1. Keep the data loading and the fold split from folds.csv unchanged.\n"
        f"2. Apply the change: {step.plan}.\n"
        "3. Report accuracy per fold and write oof.csv and submission.csv.\n"
        "Risk: more features can overfit, so watch the spread between folds."
    )


def reasoning_text(step: Step) -> str:
    base = step.parent or "nothing yet"
    return (
        f"The search has {base} to build on. The plan is: {step.plan}. "
        "One change at a time keeps the score difference readable against the fold noise. "
        "The script must still read folds.csv and write both oof.csv and submission.csv."
    )


def _reply(step: Step) -> str:
    return f"Here is the script:\n\n```python\n{script_text(step)}```\n"

# fmt: off
SCRIPT = [
    Step("draft", None, "ok", "logistic regression on one-hot features", 0.7861, 0.0071,
         summary="Logistic regression with one-hot encoding of the categorical columns, median fill for numbers."),
    Step("draft", None, "buggy", "split Cabin into deck, number and side, then LightGBM", stage="features",
         error=_trace(31, "df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)", "ValueError: Columns must be same length as key")),
    Step("debug", "n002", "ok", "split Cabin into deck, number and side, then LightGBM", 0.8012, 0.0068,
         summary="LightGBM, 500 trees, on deck/side and group features. Warnings about unused columns."),
    Step("draft", None, "ok", "random forest with ordinal-encoded categoricals", 0.7934, 0.0075,
         summary="Random forest, 400 trees, ordinal-encoded categoricals. Slowest fold took 41 s."),
    Step("improve", "n003", "ok", "add travel-group size and total spend features", 0.8068, 0.0071,
         summary="LightGBM plus group size, total spend and a cryosleep flag. Spend features dominate importance."),
    Step("improve", "n005", "ok", "lower the learning rate and add 300 trees", 0.8071, 0.0069,
         summary="Same features, learning rate 0.03 with 800 trees. Gain is within fold noise."),
    Step("improve", "n005", "blocked", "CatBoost with native categorical handling", stage="cv",
         error=_trace(4, "from catboost import CatBoostClassifier", "ModuleNotFoundError: No module named 'catboost'")),
    Step("improve", "n005", "ok", "target-encode HomePlanet and Destination", 0.7990, 0.0074,
         summary="Out-of-fold target encoding for two columns. Clearly lower than its parent."),
    Step("improve", "n005", "ok", "blend LightGBM with logistic regression", 0.8124, 0.0066,
         summary="Rank-average of LightGBM and logistic regression predictions, weights 0.7 and 0.3."),
    Step("improve", "n009", "buggy", "stack with a ridge meta-learner", stage="cv",
         error=_trace(58, "meta.fit(oof_matrix, y)", "ValueError: Found input variables with inconsistent numbers of samples: [8693, 6954]")),
    Step("debug", "n010", "ok", "stack with a ridge meta-learner", 0.8119, 0.0069,
         summary="Ridge stacker over out-of-fold predictions of three models. No gain over the blend."),
    Step("improve", "n009", "ok", "add interaction between spend and cryosleep", 0.8131, 0.0067,
         summary="Blend plus a spend-by-cryosleep interaction. Gain is within fold noise."),
    Step("improve", "n009", "buggy", "one-hot every categorical, 2000 trees", stage="features", error=MEMORY),
    Step("debug", "n013", "buggy", "one-hot every categorical, 2000 trees", stage="features", error=MEMORY),
    Step("debug", "n014", "dead", "one-hot every categorical, 2000 trees", stage="features", error=MEMORY),
]
# fmt: on


class DemoRun:
    def __init__(self, ws: Workspace, *, seed: int = 7, budget_s: float = 3600.0) -> None:
        self.ws = ws
        self.rng = random.Random(seed)
        ws.create()
        config = CompetitionConfig(slug=ws.root.name, metric="accuracy", id_column="PassengerId")
        ws.save_config(config)
        ws.page_dir.mkdir(parents=True, exist_ok=True)
        (ws.page_dir / "info.json").write_text(json.dumps(PAGE_INFO))
        (ws.search_dir / "nodes").mkdir(parents=True, exist_ok=True)
        self.tree = SolutionTree(ws.search_dir / "nodes.jsonl", higher_is_better=True)
        self.progress = Progress(
            ws,
            budget_s=budget_s,
            params={
                "policy": "greedy",
                "noise_k": 0.5,
                "num_drafts": 3,
                "max_debug": 2,
                "planner_model": PLANNER,
                "coder_model": CODER,
                "demo": True,
            },
        )
        self.deps = DependencyStore(ws.root / "deps")
        self.next_step = 0

    @property
    def done(self) -> bool:
        return self.next_step >= len(SCRIPT)

    # -- writing what a real search would have written -------------------------------------

    @staticmethod
    def _messages(role: str, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"You are the {role}."},
            {"role": "user", "content": f"(demo prompt) {text}"},
        ]

    def _call(
        self, role: str, seconds: float, when: datetime, text: str, response: str | None = None
    ) -> None:
        prompt_tokens = self.rng.randint(2500, 6500)
        entry = {
            "ts": when.isoformat(),
            "role": role,
            "model": PLANNER if role == "planner" else CODER,
            "messages": self._messages(role, text),
            "response": response if response is not None else f"(demo reply) {text}",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(seconds * self.rng.uniform(14, 22)),
            },
            "seconds": round(seconds, 2),
        }
        with (self.ws.search_dir / "trajectory.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _write_run(self, run_id: str, step: Step) -> None:
        ok = step.status == "ok"
        run_dir = self.ws.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "slug": self.ws.root.name,
            "status": "ok" if ok else "failed",
            "exit_code": 0 if ok else 1,
            "duration_s": round(self.rng.uniform(45, 320), 2),
            "stage_reached": step.stage,
            "integrity_changes": [],
            "grade": {"valid": ok, "errors": [], "cv_score": step.cv, "cv_errors": []},
        }
        (run_dir / "run.json").write_text(json.dumps(record))
        stdout = "STAGE: load\nSTAGE: features\n"
        if ok:
            stdout += f"cv accuracy: {step.cv}\nSTAGE: cv\n"
        (run_dir / "stdout.log").write_text(stdout)
        (run_dir / "stderr.log").write_text(step.error)

    def _write_node(self, step: Step, when: datetime) -> Node:
        node_id = self.tree.next_id()
        run_id = f"demo-{node_id}"
        code_path = self.ws.search_dir / "nodes" / f"{node_id}.py"
        code_path.write_text(script_text(step))
        self._write_run(run_id, step)
        parent = self.tree.get(step.parent) if step.parent else None
        node = Node(
            id=node_id,
            parent=step.parent,
            operator=step.operator,
            code_path=code_path,
            status=step.status,  # type: ignore[arg-type]
            cv_score=step.cv,
            cv_std=step.std,
            summary=step.summary,
            error=step.error,
            run_id=run_id,
            debug_depth=parent.debug_depth + 1 if step.operator == "debug" and parent else 0,
            stage=step.stage,
            dep_request="dep-001" if step.status == "blocked" else None,
            created_at=when.isoformat(),
        )
        self.tree.add(node)
        if step.status == "blocked":
            self._request_catboost(node_id)
        self.progress.event(self._describe(node), node=node_id)
        return node

    def _request_catboost(self, node_id: str) -> None:
        precheck = PreCheck(
            exists_on_index=True,
            age_days=2400,
            latest_release_age_days=34,
            new_transitive=["graphviz", "plotly"],
            resolved_version="1.2.8",
        )
        self.deps.save(
            DependencyRequest(
                id="dep-001",
                package="catboost",
                reason="ModuleNotFoundError",
                node_id=node_id,
                precheck=precheck,
            )
        )

    @staticmethod
    def _describe(node: Node) -> str:
        origin = f" <- {node.parent}" if node.parent else ""
        if node.status == "ok":
            return f"{node.id} {node.operator}{origin}: ok cv={node.cv_score:.5f}"
        last_line = node.error.splitlines()[-1]
        return (
            f"{node.id} {node.operator}{origin}: {node.status} (reached {node.stage}) {last_line}"
        )

    # -- driving it ------------------------------------------------------------------------

    def populate(self, count: int) -> None:
        """Write the first `count` steps at once, spaced into the past."""
        start = datetime.now(UTC) - timedelta(minutes=4 * count)
        while self.next_step < min(count, len(SCRIPT)):
            step = SCRIPT[self.next_step]
            when = start + timedelta(minutes=4 * self.next_step)
            if step.operator != "debug":
                self._call("planner", self.rng.uniform(20, 70), when, step.plan, plan_text(step))
            self._call("coder", self.rng.uniform(35, 110), when, step.plan, _reply(step))
            if step.status == "ok":
                self._call("planner", self.rng.uniform(6, 14), when, f"summary of {step.plan}")
            self._write_node(step, when)
            self.next_step += 1

    def _stream(
        self,
        role: str,
        step: Step,
        seconds: float,
        sleep: Callable[[float], None],
        *,
        ticks: int = 40,
    ) -> str:
        """Write a reply a little at a time, the way a model does. Returns the finished reply."""
        planner = role == "planner"
        thinking = reasoning_text(step) if planner else ""
        answer = plan_text(step) if planner else _reply(step)
        if planner:
            purpose = "plan a new approach" if step.operator == "draft" else "plan an improvement"
        else:
            purpose = "fix the failing script" if step.operator == "debug" else "write the script"
        self.progress.call_started(
            role,
            model=PLANNER if planner else CODER,
            purpose=purpose,
            messages=self._messages(role, step.plan),
        )
        quiet = ticks // 5  # the wait for the first token
        thought_end = quiet + (ticks // 5 if thinking else 0)
        for tick in range(1, ticks + 1):
            seen_thought = len(thinking) * min(max(tick - quiet, 0), thought_end - quiet)
            seen_thought //= max(thought_end - quiet, 1)
            seen_answer = len(answer) * max(tick - thought_end, 0) // (ticks - thought_end)
            self.progress.call_progress(
                role,
                thinking=thinking[:seen_thought],
                answer=answer[:seen_answer],
                chunks=(seen_thought + seen_answer) // 4,
            )
            sleep(seconds / ticks)
        self.progress.call_finished(role)
        return answer

    def play(self, *, speed: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        """Run the remaining steps in real time, moving through the phases a search goes through."""
        progress = self.progress
        while not self.done:
            step = SCRIPT[self.next_step]
            detail = f"from {step.parent}" if step.parent else "new draft"
            progress.phase(
                "writing",
                node=self.tree.next_id(),
                op=step.operator,
                parent=step.parent,
                detail=detail,
            )
            for role, seconds in (("planner", 6.0), ("coder", 9.0)):
                if role == "planner" and step.operator == "debug":
                    continue
                reply = self._stream(role, step, seconds / speed, sleep)
                self._call(role, seconds * 6, datetime.now(UTC), step.plan, reply)
            progress.phase("running", detail="sandbox and grader")
            sleep(6.0 / speed)
            progress.phase("evaluating", detail="grading and summarizing")
            sleep(2.0 / speed)
            self._write_node(step, datetime.now(UTC))
            self.next_step += 1
        progress.phase("selecting", node=None, op=None)
        sleep(1.0 / speed)
        progress.finish("finished", "demo complete")
