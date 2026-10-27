"""Label-like targets (bool, text) need the same dtype in submissions and out-of-fold files."""

import pandas as pd
import pytest

from kagent.grader.cv import score_oof
from kagent.grader.validate import kind_mismatch, validate_submission


def _sample(values):
    return pd.DataFrame({"id": [f"a{i}" for i in range(len(values))], "y": values})


def test_bool_sample_requires_true_false_not_zero_one():
    sample = _sample([False, False, False])
    ints = _sample([0, 1, 0])
    bools = _sample([True, False, True])
    assert "True/False" in validate_submission(ints, sample, "id")[0]
    assert validate_submission(bools, sample, "id") == []


def test_text_sample_rejects_numbers_and_accepts_text():
    sample = _sample(["cat", "cat", "cat"])
    assert "text labels" in validate_submission(_sample([0, 1, 0]), sample, "id")[0]
    assert validate_submission(_sample(["dog", "cat", "dog"]), sample, "id") == []


def test_numeric_sample_accepts_ints_floats_and_rejects_text():
    sample = _sample([0.5, 0.5, 0.5])
    assert validate_submission(_sample([1, 0, 1]), sample, "id") == []
    assert validate_submission(_sample([0.1, 0.2, 0.3]), sample, "id") == []
    assert "numeric" in validate_submission(_sample(["a", "b", "c"]), sample, "id")[0]


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (pd.Series([True, False]), pd.Series([True, True]), None),
        (pd.Series([True, False]), pd.Series([1, 0]), "True/False"),
        (pd.Series([1, 0]), pd.Series([True, False]), None),
        (pd.Series(["a", "b"]), pd.Series(["b", "a"]), None),
        (pd.Series(["a", "b"]), pd.Series([True, False]), "text labels"),
    ],
)
def test_kind_mismatch(reference, candidate, expected):
    assert kind_mismatch(reference, candidate) == expected


def _competition(tmp_path, labels):
    ids = [f"p{i}" for i in range(len(labels))]
    pd.DataFrame({"id": ids, "label": labels}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": ids, "fold": [i % 2 for i in range(len(ids))]}).to_csv(
        tmp_path / "folds.csv", index=False
    )
    return ids


def _oof(tmp_path, ids, values):
    path = tmp_path / "oof.csv"
    pd.DataFrame({"id": ids, "label": values}).to_csv(path, index=False)
    return path


def test_bool_target_accuracy_scores_bool_oof(tmp_path):
    labels = [True, False, True, True, False, False, True, False]
    ids = _competition(tmp_path, labels)
    score, _, errors = score_oof(tmp_path, _oof(tmp_path, ids, labels), "accuracy", "id", "label")
    assert (score, errors) == (1.0, [])


def test_bool_target_rejects_probability_oof(tmp_path):
    labels = [True, False, True, True, False, False, True, False]
    ids = _competition(tmp_path, labels)
    path = _oof(tmp_path, ids, [0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4])
    errors = score_oof(tmp_path, path, "accuracy", "id", "label")[2]
    assert any("True/False" in e for e in errors)


def test_text_class_labels_are_scored_not_rejected(tmp_path):
    labels = ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"]
    ids = _competition(tmp_path, labels)
    flipped = ["dog" if x == "cat" else "cat" for x in labels[:2]] + labels[2:]
    score, _, errors = score_oof(tmp_path, _oof(tmp_path, ids, flipped), "accuracy", "id", "label")
    assert errors == [] and score == pytest.approx(0.75)
