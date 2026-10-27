"""CV scoring and the search loop against the real sandbox and grader containers."""

import pytest

from fakes import FakeChat, code_block
from kagent import harness
from kagent.search.loop import open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy

pytestmark = pytest.mark.docker

CV_SCRIPT = """
# PLAN: logistic regression on the two raw features
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("/data/train.csv")
test = pd.read_csv("/data/test.csv")
folds = pd.read_csv("/data/folds.csv")
df = train.merge(folds, on="id")
feats = {features}

oof = np.zeros(len(df))
for k in sorted(df["fold"].unique()):
    fit, held = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
    model = LogisticRegression().fit(df.loc[fit, feats], df.loc[fit, "target"])
    oof[held] = model.predict_proba(df.loc[held, feats])[:, 1]
pd.DataFrame({{"id": df["id"], "target": oof}}).to_csv("/work/out/oof.csv", index=False)

model = LogisticRegression().fit(df[feats], df["target"])
out = pd.DataFrame({{"id": test["id"], "target": model.predict_proba(test[feats])[:, 1]}})
out.to_csv("/work/out/submission.csv", index=False)
print("done")
"""

GOOD = CV_SCRIPT.format(features='["x1", "x2"]')
BROKEN = CV_SCRIPT.format(features='["x1", "x_missing"]')  # KeyError at fit time
WEAK = CV_SCRIPT.format(features='["x1"]')


def test_script_gets_a_cv_score_computed_by_the_grader(ws, settings, tmp_path):
    script = tmp_path / "s.py"
    script.write_text(GOOD)
    record = harness.run_script(ws, script, settings)

    assert record.status == "ok", record.stderr_tail
    assert record.grade.cv_errors == []
    assert 0.9 < record.grade.cv_score <= 1.0
    assert record.grade.cv_std is not None and record.grade.cv_std >= 0
    assert record.grade.score > 0.9  # held-out score is reported but separate


def test_missing_oof_gives_a_valid_submission_but_no_cv(ws, settings, tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        GOOD.replace('.to_csv("/work/out/oof.csv", index=False)', ".head(1).to_csv('/tmp/x.csv')")
    )
    record = harness.run_script(ws, script, settings)
    assert record.status == "ok" and record.grade.cv_score is None


def test_search_loop_end_to_end(ws, settings):
    config = ws.load_config()
    planner = FakeChat(["- plan"] * 10, role="planner")
    coder = FakeChat(
        [code_block(WEAK), code_block(BROKEN), code_block(GOOD), code_block(GOOD)],
        role="coder",
    )
    ops = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    tree = open_tree(ws, config)
    run_search(
        ws, config, settings, tree, GreedyPolicy(num_drafts=2, debug_prob=1.0), ops, planner,
        description="toy", eda="toy", budget_s=3600, max_nodes=4,
    )  # fmt: skip

    weak, broken, fixed, improved = tree.nodes
    assert weak.status == "ok" and weak.cv_score is not None
    assert broken.status == "buggy" and "KeyError" in broken.error
    assert fixed.operator == "debug" and fixed.parent == broken.id and fixed.status == "ok"
    assert "KeyError" in coder.user_text(2)  # the real traceback reached the debug prompt
    assert improved.operator == "improve" and improved.status == "ok"
    assert tree.best().cv_score >= weak.cv_score
    assert all(harness.load_record(ws, n.run_id).integrity_changes == [] for n in tree.nodes)
