"""Cross-validation score from out-of-fold predictions.

The script supplies predictions; labels, folds and the metric stay on the grader side, so
the agent cannot report a score it did not earn on the shared folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS
from .validate import MAX_ERRORS


def score_oof(
    public: Path, oof_path: Path, metric: str, id_column: str, target: str
) -> tuple[float | None, float | None, list[str]]:
    """Return (cv_score, cv_std over folds, errors)."""
    try:
        train = pd.read_csv(public / "train.csv", usecols=[id_column, target])
        folds = pd.read_csv(public / "folds.csv")
        oof = pd.read_csv(oof_path)
    except Exception as exc:  # noqa: BLE001  unreadable input is reported, not raised
        return None, None, [f"cannot read oof inputs: {exc}"]

    errors: list[str] = []
    if list(oof.columns) != [id_column, target]:
        return (
            None,
            None,
            [f"oof.csv columns must be {[id_column, target]}, got {list(oof.columns)}"],
        )
    if len(oof) != len(train):
        errors.append(f"oof.csv needs {len(train)} rows, got {len(oof)}")
    if oof[id_column].duplicated().any():
        errors.append(f"duplicate values in oof '{id_column}'")
    if set(oof[id_column].astype(str)) != set(train[id_column].astype(str)):
        errors.append(f"oof '{id_column}' values differ from train.csv")
    if oof[target].isna().any():
        errors.append(f"missing values in oof '{target}'")
    if not pd.api.types.is_numeric_dtype(oof[target]):
        errors.append(f"oof '{target}' must be numeric")
    if errors:
        return None, None, errors[:MAX_ERRORS]

    df = train.merge(folds, on=id_column).merge(oof, on=id_column, suffixes=("_true", "_pred"))
    fn = METRICS[metric].fn
    true, pred = df[f"{target}_true"], df[f"{target}_pred"]
    try:
        overall = fn(true.to_numpy(), pred.to_numpy())
        per_fold = [
            fn(true[df["fold"] == k].to_numpy(), pred[df["fold"] == k].to_numpy())
            for k in sorted(df["fold"].unique())
        ]
    except ValueError as exc:  # e.g. a fold with a single class for AUC
        return None, None, [f"cannot score oof: {exc}"]
    return overall, float(np.std(per_fold)), []
