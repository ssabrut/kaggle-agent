import numpy as np
import pandas as pd
import pytest

from kagent.folds import make_folds
from kagent.grader.cv import score_oof
from kagent.grader.main import grade


def _oof(ws, **changes):
    df = pd.read_csv(ws.public / "train.csv")[["id", "target"]]
    df["target"] = df["target"].astype(float)
    for col, value in changes.items():
        df[col] = value
    path = ws.root / "oof.csv"
    df.to_csv(path, index=False)
    return path


def _score(ws, path, metric="auc"):
    return score_oof(ws.public, path, metric, "id", "target")


def test_perfect_oof_scores_one_with_zero_spread(ws):
    score, std, errors = _score(ws, _oof(ws))
    assert (score, std, errors) == (1.0, 0.0, [])


def test_noisy_oof_has_fold_spread(ws):
    rng = np.random.default_rng(1)
    target = pd.read_csv(ws.public / "train.csv")["target"].to_numpy()
    path = _oof(ws, target=target + rng.normal(scale=0.8, size=len(target)))
    score, std, errors = _score(ws, path)
    assert errors == [] and 0.5 < score < 1.0 and std > 0


def test_wrong_columns_rejected(ws):
    path = _oof(ws)
    pd.read_csv(path).rename(columns={"target": "pred"}).to_csv(path, index=False)
    assert "columns must be" in _score(ws, path)[2][0]


def test_wrong_row_count_ids_and_nan_rejected(ws):
    df = pd.read_csv(_oof(ws)).iloc[:-5]
    df.loc[df.index[0], "id"] = 10_000
    df.loc[df.index[1], "target"] = None
    path = ws.root / "bad.csv"
    df.to_csv(path, index=False)
    errors = _score(ws, path)[2]
    assert any("rows" in e for e in errors)
    assert any("differ" in e for e in errors)
    assert any("missing" in e for e in errors)


def test_missing_folds_file_is_reported(ws):
    path = _oof(ws)
    (ws.public / "folds.csv").unlink()
    assert "cannot read oof inputs" in _score(ws, path)[2][0]


def test_grade_reports_cv_alongside_held_out_score(ws):
    sub = pd.read_csv(ws.private / "labels.csv").astype({"target": float})
    sub_path = ws.root / "submission.csv"
    sub.to_csv(sub_path, index=False)

    result = grade(ws.public, ws.private, sub_path, "auc", "id", _oof(ws))
    assert result.score == 1.0 and result.cv_score == 1.0 and result.cv_errors == []


def test_grade_without_oof_leaves_cv_empty(ws):
    sub_path = ws.root / "submission.csv"
    pd.read_csv(ws.private / "labels.csv").astype({"target": float}).to_csv(sub_path, index=False)
    result = grade(ws.public, ws.private, sub_path, "auc", "id")
    assert result.valid and result.cv_score is None and result.cv_errors == []


def test_folds_cover_every_row_and_are_balanced(ws):
    folds = pd.read_csv(ws.public / "folds.csv")
    train = pd.read_csv(ws.public / "train.csv")
    assert list(folds["id"]) == list(train["id"])
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    assert folds["fold"].value_counts().nunique() == 1  # 200 rows -> 40 per fold


def test_folds_are_stratified_for_classification(ws):
    merged = pd.read_csv(ws.public / "train.csv").merge(pd.read_csv(ws.public / "folds.csv"))
    rates = merged.groupby("fold")["target"].mean()
    assert rates.max() - rates.min() < 0.05


def test_folds_skipped_without_train(settings):
    from kagent.config import CompetitionConfig
    from kagent.workspace import Workspace

    empty = Workspace.for_slug(settings, "empty")
    empty.create()
    assert make_folds(empty, CompetitionConfig(slug="empty", metric="auc")) is False


@pytest.mark.parametrize("metric", ["rmse", "mae"])
def test_regression_metrics_use_plain_kfold(ws, metric):
    from kagent.config import CompetitionConfig

    config = CompetitionConfig(slug="toy", metric=metric)
    assert make_folds(ws, config)
    assert pd.read_csv(ws.public / "folds.csv")["fold"].nunique() == 5
