from pathlib import Path

import pytest

from fakes import FakeChat, code_block
from kagent.search.operators import (
    Context,
    DebugOperator,
    DraftOperator,
    ImproveOperator,
    OperatorError,
    extract_code,
    summarize,
)
from kagent.search.tree import Node, SolutionTree


@pytest.fixture
def tree(tmp_path):
    return SolutionTree(tmp_path / "nodes.jsonl", higher_is_better=True)


@pytest.fixture
def ctx(tree):
    return Context(
        description="Predict churn.",
        eda="train.csv: 200 rows",
        metric="auc",
        higher_is_better=True,
        id_column="id",
        target="target",
        tree=tree,
        step_timeout_s=600,
        remaining_s=3600,
        libraries={
            "numpy": "2.4.6",
            "pandas": "2.3.3",
            "scikit-learn": "1.9.1",
            "lightgbm": "4.7.0",
        },
    )


def _node(tree, tmp_path, code="print('parent')\n", **kw):
    node_id = tree.next_id()
    path = tmp_path / f"{node_id}.py"
    path.write_text(code)
    defaults = {
        "parent": None,
        "operator": "draft",
        "status": "ok",
        "cv_score": 0.8,
        "cv_std": 0.01,
    }
    return tree.add(Node(id=node_id, code_path=path, **{**defaults, **kw}))


def test_extract_code_takes_the_fenced_block():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"


def test_extract_code_prefers_the_longest_block():
    reply = "```python\nx = 1\n```\n```python\nx = 1\ny = 2\n```"
    assert extract_code(reply) == "x = 1\ny = 2\n"


def test_extract_code_accepts_unfenced_code():
    assert extract_code("x = 1") == "x = 1\n"
    assert extract_code("print(1)") == "print(1)\n"


@pytest.mark.parametrize("prose", ["nope", "sorry", "model.fit"])
def test_extract_code_rejects_prose_that_happens_to_parse(prose):
    with pytest.raises(OperatorError, match="not a script"):
        extract_code(prose)


@pytest.mark.parametrize(
    "reply", ["", "```python\n\n```", "just prose, no code", "```python\nx = (\n```"]
)
def test_extract_code_rejects_unusable_replies(reply):
    with pytest.raises(OperatorError):
        extract_code(reply)


def test_first_draft_asks_for_a_baseline_and_embeds_the_contract(ctx):
    planner = FakeChat(["- LightGBM on raw features"])
    coder = FakeChat([code_block("# PLAN: lgbm\nprint(1)")])

    code = DraftOperator(planner, coder).apply(None, ctx)

    assert code == "# PLAN: lgbm\nprint(1)\n"
    plan_prompt = planner.user_text(0)
    assert "Predict churn." in plan_prompt and "train.csv: 200 rows" in plan_prompt
    assert "simple, robust baseline" in plan_prompt
    code_prompt = coder.user_text(0)
    assert "- LightGBM on raw features" in code_prompt
    assert "/data/folds.csv" in code_prompt and "/work/out/oof.csv" in code_prompt
    assert "Commit to ONE final model" in code_prompt
    assert "600 s wall clock" in code_prompt


def test_later_drafts_see_earlier_approaches(ctx, tree, tmp_path):
    _node(tree, tmp_path, summary="Logistic regression, cv 0.71")
    planner = FakeChat(["- different"])
    coder = FakeChat([code_block("print(2)")])

    DraftOperator(planner, coder).apply(None, ctx)

    prompt = planner.user_text(0)
    assert "Logistic regression, cv 0.71" in prompt
    assert "differs meaningfully" in prompt and "simple, robust baseline" not in prompt


def test_improve_sends_parent_code_score_and_sibling_results(ctx, tree, tmp_path):
    parent = _node(
        tree, tmp_path, code="print('parent code')\n", cv_score=0.8123, summary="LGBM base"
    )
    _node(
        tree,
        tmp_path,
        parent=parent.id,
        operator="improve",
        cv_score=0.79,
        summary="Added TF-IDF, worse",
    )
    _node(
        tree,
        tmp_path,
        parent=parent.id,
        operator="improve",
        status="buggy",
        cv_score=None,
        error="KeyError: 'zip'",
    )
    planner = FakeChat(["Tune the learning rate."])
    coder = FakeChat([code_block("print('child')")])

    code = ImproveOperator(planner, coder).apply(parent, ctx)

    assert code == "print('child')\n"
    idea_prompt = planner.user_text(0)
    assert "print('parent code')" in idea_prompt and "0.81230" in idea_prompt
    assert "LGBM base" in idea_prompt
    assert "Added TF-IDF, worse" in idea_prompt and "KeyError: 'zip'" in idea_prompt
    code_prompt = coder.user_text(0)
    assert "Tune the learning rate." in code_prompt and "print('parent code')" in code_prompt


def test_debug_sends_code_and_error_but_skips_the_planner(ctx, tree, tmp_path):
    broken = _node(tree, tmp_path, code="df['zip']\n", status="buggy", cv_score=None)
    ctx.error_log = "KeyError: 'zip'"
    coder = FakeChat([code_block("df.get('zip')")])

    code = DebugOperator(coder).apply(broken, ctx)

    assert code == "df.get('zip')\n"
    prompt = coder.user_text(0)
    assert "df['zip']" in prompt and "KeyError: 'zip'" in prompt


def test_unusable_reply_gets_one_repair_attempt(ctx):
    planner = FakeChat(["- plan"])
    coder = FakeChat(["I cannot do that", code_block("print(1)")])

    assert DraftOperator(planner, coder).apply(None, ctx) == "print(1)\n"
    assert len(coder.calls) == 2
    assert "unusable" in coder.user_text(1)


def test_two_unusable_replies_raise(ctx):
    coder = FakeChat(["nope", "still nope"])
    with pytest.raises(OperatorError):
        DraftOperator(FakeChat(["- plan"]), coder).apply(None, ctx)


def test_summarize_truncates_long_replies():
    planner = FakeChat(["x" * 2000])
    assert (
        len(summarize(planner, code="c", log="l", metric="auc", cv_score=0.8, cv_std=None)) == 600
    )


def test_long_code_is_clipped_in_prompts(ctx, tree, tmp_path):
    parent = _node(tree, tmp_path, code="x = 1\n" * 10_000)
    planner = FakeChat(["idea"])
    ImproveOperator(planner, FakeChat([code_block("x = 2")])).apply(parent, ctx)
    assert "characters cut" in planner.user_text(0)
    assert len(planner.user_text(0)) < 20_000
    assert isinstance(Path(parent.code_path), Path)


def test_contract_states_the_installed_library_versions(ctx):
    contract = ctx.contract()
    assert "numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0" in contract
    assert "FrozenEstimator" in contract


def test_contract_follows_the_libraries_it_is_given(ctx):
    ctx.libraries = {"numpy": "2.4.6", "catboost": "1.2.8"}
    assert "numpy 2.4.6, catboost 1.2.8" in ctx.contract()


def test_planner_prompts_name_the_installed_libraries_and_the_cost_of_others(ctx, tree, tmp_path):
    planner = FakeChat(["- plan"])
    DraftOperator(planner, FakeChat([code_block("print(1)")])).apply(None, ctx)
    parent = _node(tree, tmp_path)
    improver = FakeChat(["idea"])
    ImproveOperator(improver, FakeChat([code_block("print(2)")])).apply(parent, ctx)

    for text in (planner.user_text(0), improver.user_text(0)):
        assert "standard library plus numpy, pandas, scikit-learn, lightgbm" in text
        assert "until a human approves an install request" in text
