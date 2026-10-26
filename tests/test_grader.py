import pandas as pd
import pytest

from kagent.grader.main import grade


def _write(ws, df, name="submission.csv"):
    path = ws.root / name
    df.to_csv(path, index=False)
    return path


def _grade(ws, path, metric="auc"):
    return grade(ws.public, ws.private, path, metric, "id")


@pytest.fixture
def perfect(ws):
    return pd.read_csv(ws.private / "labels.csv").astype({"target": float})


def test_perfect_submission_scores_one(ws, perfect):
    result = _grade(ws, _write(ws, perfect))
    assert result.valid and result.score == 1.0


def test_inverted_submission_scores_zero(ws, perfect):
    perfect["target"] = 1 - perfect["target"]
    assert _grade(ws, _write(ws, perfect)).score == 0.0


def test_wrong_columns_rejected(ws, perfect):
    result = _grade(ws, _write(ws, perfect.rename(columns={"target": "pred"})))
    assert not result.valid and "columns" in result.errors[0]


def test_missing_rows_rejected(ws, perfect):
    result = _grade(ws, _write(ws, perfect.iloc[:-5]))
    assert not result.valid and "rows" in result.errors[0]


def test_duplicate_and_foreign_ids_rejected(ws, perfect):
    perfect.loc[0, "id"] = perfect.loc[1, "id"]
    result = _grade(ws, _write(ws, perfect))
    assert not result.valid
    assert any("duplicate" in e for e in result.errors)
    assert any("differ" in e for e in result.errors)


def test_nan_and_non_numeric_rejected(ws, perfect):
    perfect["target"] = perfect["target"].astype(object)
    perfect.loc[0, "target"] = None
    perfect.loc[1, "target"] = "high"
    result = _grade(ws, _write(ws, perfect))
    assert not result.valid
    assert any("missing" in e for e in result.errors)
    assert any("numeric" in e for e in result.errors)


def test_unreadable_file_rejected(ws):
    path = ws.root / "submission.csv"
    path.write_bytes(b"")
    result = _grade(ws, path)
    assert not result.valid and "cannot read" in result.errors[0]


def test_no_private_labels_means_valid_but_unscored(ws, perfect):
    (ws.private / "labels.csv").unlink()
    result = _grade(ws, _write(ws, perfect))
    assert result.valid and result.score is None


def test_unknown_metric(ws, perfect):
    result = _grade(ws, _write(ws, perfect), metric="bogus")
    assert not result.valid and "unknown metric" in result.errors[0]


def test_rmse_matches_hand_calculation(ws, perfect):
    noisy = perfect.copy()
    noisy["target"] = 0.5
    assert _grade(ws, _write(ws, noisy), metric="rmse").score == pytest.approx(0.5)
