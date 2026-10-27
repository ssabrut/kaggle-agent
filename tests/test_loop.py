"""Search loop with scripted models and a stubbed runner (no Docker, no network)."""

import json
import re

import pytest

from fakes import FakeChat, code_block
from kagent.grader.result import GradeResult
from kagent.harness import RunRecord
from kagent.search.loop import IntegrityAbort, OperatorFailures, open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy


def fake_run(ws, script, settings):
    """Outcome comes from markers in the script: '# CV=0.8', '# CRASH', '# TAMPER', '# NOCV'.

    '# STAGE=features' sets how far the run is reported to have got.
    """
    text = script.read_text()
    stage = re.search(r"# STAGE=(\w+)", text)
    base = {
        "run_id": f"run-{script.stem}",
        "slug": "toy",
        "exit_code": 0,
        "duration_s": 1.0,
        "script_sha256": "x",
        "stage_reached": stage.group(1) if stage else "start",
    }
    if "# TAMPER" in text:
        return RunRecord(
            **base, status="integrity_violation", integrity_changes=["public/train.csv modified"]
        )
    if "# CRASH" in text:
        return RunRecord(
            **{**base, "exit_code": 1},
            status="failed",
            stderr_tail="Traceback ...\nKeyError: 'foo'",
        )
    if "# NOCV" in text:
        return RunRecord(
            **base, status="ok", grade=GradeResult(valid=True, cv_errors=["oof.csv missing rows"])
        )
    cv = float(re.search(r"# CV=([\d.]+)", text).group(1))
    return RunRecord(**base, status="ok", grade=GradeResult(valid=True, cv_score=cv, cv_std=0.01))


def marker(text: str) -> str:
    return code_block(f"{text}\nprint('x')")


@pytest.fixture
def parts(ws, settings):
    config = ws.load_config()
    planner = FakeChat(["- plan"] * 20, role="planner")
    return config, planner


def _search(ws, settings, parts, coder_replies, *, policy=None, **kwargs):
    config, planner = parts
    coder = FakeChat(coder_replies, role="coder")
    tree = open_tree(ws, config)
    operators = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    logs: list[str] = []
    run_search(
        ws, config, settings, tree, policy or GreedyPolicy(num_drafts=2, debug_prob=1.0), operators, planner,
        description="d", eda="e", budget_s=3600, run=fake_run, log=logs.append, **kwargs,
    )  # fmt: skip
    return tree, coder, logs


def test_draft_debug_improve_sequence(ws, settings, parts):
    tree, coder, logs = _search(
        ws, settings, parts,
        [marker("# CV=0.70"), marker("# CRASH"), marker("# CV=0.80"), marker("# CV=0.85")],
        max_nodes=4,
    )  # fmt: skip

    n1, n2, n3, n4 = tree.nodes
    assert (n1.operator, n1.status, n1.cv_score) == ("draft", "ok", 0.70)
    assert (n2.operator, n2.status) == ("draft", "buggy") and "KeyError: 'foo'" in n2.error
    assert (n3.operator, n3.parent, n3.status, n3.debug_depth) == ("debug", "n002", "ok", 1)
    assert (n4.operator, n4.parent, n4.cv_score) == ("improve", "n003", 0.85)
    assert tree.best().id == "n004"
    assert n1.summary == "- plan"  # planner summarizes ok nodes
    assert "KeyError: 'foo'" in coder.user_text(2)  # the debug prompt carries the traceback
    assert "n002 draft: buggy" in logs[1] and logs[-1] == "reached max nodes (4)"


def test_run_ids_link_nodes_to_run_records(ws, settings, parts):
    tree, _, _ = _search(ws, settings, parts, [marker("# CV=0.7")], max_nodes=1)
    assert tree.nodes[0].run_id == "run-n001"
    assert tree.nodes[0].code_path.read_text().startswith("# CV=0.7")


def test_node_dies_after_max_debug_attempts_and_search_restarts(ws, settings, parts):
    crash = marker("# CRASH")
    tree, _, _ = _search(
        ws, settings, parts, [crash] * 4, policy=GreedyPolicy(num_drafts=1, debug_prob=1.0),
        max_nodes=4, max_debug=2,
    )  # fmt: skip

    assert [(n.status, n.debug_depth) for n in tree.nodes] == [
        ("buggy", 0), ("buggy", 1), ("dead", 2), ("buggy", 0),
    ]  # fmt: skip
    assert tree.nodes[3].operator == "draft"


def test_status_and_events_follow_the_search_and_end_cleanly(ws, settings, parts):
    _search(ws, settings, parts, [marker("# CV=0.7")], max_nodes=1)

    status = json.loads((ws.search_dir / "status.json").read_text())
    assert (status["state"], status["message"]) == ("finished", "reached max nodes (1)")
    events = [
        json.loads(line) for line in (ws.search_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert any(e["node"] == "n001" and "n001 draft: ok" in e["text"] for e in events)


def test_status_says_stopped_when_the_search_fails(ws, settings, parts):
    with pytest.raises(OperatorFailures):
        _search(ws, settings, parts, ["no code"] * 10, max_nodes=5)

    status = json.loads((ws.search_dir / "status.json").read_text())
    assert status["state"] == "stopped" and "OperatorFailures" in status["message"]


def test_stage_is_recorded_logged_and_handed_to_debug(ws, settings, parts):
    tree, coder, logs = _search(
        ws, settings, parts, [marker("# CRASH\n# STAGE=features"), marker("# CV=0.8")],
        policy=GreedyPolicy(num_drafts=1, debug_prob=1.0), max_nodes=2,
    )  # fmt: skip

    assert tree.get("n001").stage == "features"
    assert "(reached features)" in logs[0]
    assert "How far it got: The features were built" in coder.user_text(1)


def test_next_draft_is_told_about_the_branch_that_died(ws, settings, parts):
    _search(
        ws, settings, parts, [marker("# CRASH")] * 4,
        policy=GreedyPolicy(num_drafts=1, debug_prob=1.0), max_nodes=4, max_debug=2,
    )  # fmt: skip
    planner = parts[1]

    assert "failed with KeyError: 'foo'" in planner.user_text(1)  # the draft after the dead node
    assert "Failed or did not help" not in planner.user_text(0)  # nothing had failed yet


def test_missing_oof_is_a_failure_not_a_score(ws, settings, parts):
    tree, _, _ = _search(ws, settings, parts, [marker("# NOCV")], max_nodes=1)
    node = tree.nodes[0]
    assert node.status == "buggy" and node.cv_score is None
    assert "oof.csv missing rows" in node.error


def test_integrity_violation_stops_the_search_and_is_recorded(ws, settings, parts):
    config, _ = parts
    with pytest.raises(IntegrityAbort, match="public/train.csv modified"):
        _search(ws, settings, parts, [marker("# TAMPER")], max_nodes=5)
    tree = open_tree(ws, config)
    assert [n.status for n in tree.nodes] == ["dead"]


def test_repeated_unusable_code_aborts(ws, settings, parts):
    with pytest.raises(OperatorFailures):
        _search(ws, settings, parts, ["no code"] * 10, max_nodes=5)


def test_stops_when_the_time_budget_is_spent(ws, settings, parts):
    clock_now = [0.0]

    def run(ws_, script, settings_):
        clock_now[0] += 30
        return fake_run(ws_, script, settings_)

    config, planner = parts
    coder = FakeChat([marker("# CV=0.7")] * 10, role="coder")
    tree = open_tree(ws, config)
    ops = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    logs: list[str] = []
    run_search(
        ws, config, settings, tree, GreedyPolicy(num_drafts=2), ops, planner,
        description="", eda="", budget_s=130, run=run, log=logs.append, clock=lambda: clock_now[0],
    )  # fmt: skip

    assert len(tree) == 3  # 130 -> 100 -> 70 -> 40 s left, below the 60 s minimum
    assert logs[-1] == "time budget used up"


def test_step_timeout_is_capped_by_remaining_budget(ws, settings, parts):
    seen = []

    def run(ws_, script, settings_):
        seen.append(settings_.step_timeout_s)
        return fake_run(ws_, script, settings_)

    config, planner = parts
    coder = FakeChat([marker("# CV=0.7")], role="coder")
    ops = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    run_search(
        ws, config, settings, open_tree(ws, config), GreedyPolicy(num_drafts=1), ops, planner,
        description="", eda="", budget_s=200, max_nodes=1, run=run, clock=lambda: 0.0,
    )  # fmt: skip

    assert seen == [200]  # settings default is 600


def test_search_resumes_an_existing_tree(ws, settings, parts):
    _search(ws, settings, parts, [marker("# CV=0.7")], max_nodes=1)
    tree, _, _ = _search(ws, settings, parts, [marker("# CV=0.8")], max_nodes=1)
    assert [n.id for n in tree.nodes] == ["n001", "n002"]


def test_oof_failure_prompt_reaches_debug(ws, settings, parts):
    _, coder, _ = _search(
        ws, settings, parts, [marker("# NOCV"), marker("# CV=0.9")],
        policy=GreedyPolicy(num_drafts=1, debug_prob=1.0), max_nodes=2,
    )  # fmt: skip
    assert "cross-validation failed: oof.csv missing rows" in coder.user_text(1)
