from pathlib import Path

import pytest

from kagent.search.policies import GreedyPolicy
from kagent.search.select import select_best, selected_run_id
from kagent.search.tree import Node, SolutionTree


def _tree(tmp_path, higher=True):
    return SolutionTree(tmp_path / "nodes.jsonl", higher_is_better=higher)


def _add(tree, status="ok", cv=None, std=None, parent=None, **kw):
    return tree.add(
        Node(
            id=tree.next_id(),
            parent=parent,
            operator=kw.pop("operator", "draft" if parent is None else "improve"),
            code_path=Path(f"{tree.next_id()}.py"),
            status=status,
            cv_score=cv,
            cv_std=std,
            **kw,
        )
    )


def test_nodes_persist_and_reload_with_last_version_winning(tmp_path):
    tree = _tree(tmp_path)
    first = _add(tree, "buggy", error="boom")
    _add(tree, "ok", cv=0.8, parent=first.id, summary="lgbm")
    tree.update(first.id, status="dead")

    reloaded = _tree(tmp_path)
    assert [n.id for n in reloaded.nodes] == ["n001", "n002"]
    assert reloaded.get("n001").status == "dead"
    assert reloaded.get("n002").code_path == Path("n002.py")
    assert reloaded.next_id() == "n003"


def test_best_respects_metric_direction(tmp_path):
    high = _tree(tmp_path / "h", higher=True)
    low = _tree(tmp_path / "l", higher=False)
    for tree in (high, low):
        _add(tree, cv=0.5)
        _add(tree, cv=0.9)
    assert high.best().cv_score == 0.9
    assert low.best().cv_score == 0.5


def test_best_ignores_failed_nodes_and_handles_empty(tmp_path):
    tree = _tree(tmp_path)
    assert tree.best() is None
    _add(tree, "buggy")
    _add(tree, "dead")
    assert tree.best() is None


def test_robust_best_penalizes_spread(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, cv=0.90, std=0.10)  # 0.85 after penalty
    _add(tree, cv=0.88, std=0.01)  # ~0.875 after penalty
    assert tree.best().id == "n001"
    assert tree.best(robust=True).id == "n002"


def test_robust_best_penalizes_spread_for_lower_is_better(tmp_path):
    tree = _tree(tmp_path, higher=False)
    _add(tree, cv=0.10, std=0.10)  # 0.15 after penalty
    _add(tree, cv=0.12, std=0.01)  # ~0.125 after penalty
    assert tree.best().id == "n001"
    assert tree.best(robust=True).id == "n002"


def test_buggy_leaves_exclude_nodes_already_debugged(tmp_path):
    tree = _tree(tmp_path)
    broken = _add(tree, "buggy")
    _add(tree, "buggy")
    _add(tree, "ok", cv=0.7, parent=broken.id, operator="debug")
    assert [n.id for n in tree.buggy_leaves()] == ["n002"]


def test_policy_drafts_until_enough_roots(tmp_path):
    tree = _tree(tmp_path)
    policy = GreedyPolicy(num_drafts=2)
    assert policy.select(tree).operator == "draft"
    _add(tree, cv=0.5)
    action = policy.select(tree)
    assert action.operator == "draft" and action.parent is None


def test_policy_improves_best_when_nothing_is_broken(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, cv=0.5)
    _add(tree, cv=0.8)
    action = GreedyPolicy(num_drafts=2).select(tree)
    assert (action.operator, action.parent.id) == ("improve", "n002")


def test_policy_debugs_when_probability_says_so(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, cv=0.8)
    _add(tree, "buggy")
    always = GreedyPolicy(num_drafts=2, debug_prob=1.0).select(tree)
    never = GreedyPolicy(num_drafts=2, debug_prob=0.0).select(tree)
    assert (always.operator, always.parent.id) == ("debug", "n002")
    assert (never.operator, never.parent.id) == ("improve", "n001")


def test_policy_always_debugs_when_nothing_has_worked(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, "buggy")
    action = GreedyPolicy(num_drafts=1, debug_prob=0.0).select(tree)
    assert action.operator == "debug"


def test_policy_restarts_with_a_draft_when_everything_died(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, "dead")
    action = GreedyPolicy(num_drafts=1).select(tree)
    assert action.operator == "draft" and action.parent is None


def test_select_best_records_choice_for_submit(ws, tmp_path):
    tree = _tree(tmp_path)
    _add(tree, cv=0.9, std=0.1, run_id="run-a")
    _add(tree, cv=0.88, std=0.01, run_id="run-b")

    assert select_best(ws, tree, "cv").id == "n001"
    assert selected_run_id(ws) == "run-a"
    assert select_best(ws, tree, "robust_cv").id == "n002"
    assert selected_run_id(ws) == "run-b"


def test_select_best_errors(ws, tmp_path):
    with pytest.raises(ValueError, match="no node"):
        select_best(ws, _tree(tmp_path))
    with pytest.raises(ValueError, match="unknown selection rule"):
        select_best(ws, _tree(tmp_path), "by_vibes")
    with pytest.raises(FileNotFoundError):
        selected_run_id(ws)
