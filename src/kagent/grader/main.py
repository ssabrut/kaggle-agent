"""Grader entry point. Runs in its own container with the private labels mounted.

    python -m grader.main --public /data/public --private /data/private \
        --submission /submission/submission.csv --metric auc --id-column id \
        [--oof /submission/oof.csv]

Prints one JSON line (GradeResult) on stdout. Imports only siblings, so the same
package works on the host (as kagent.grader) and in the image (as grader).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .cv import score_oof
from .metrics import METRICS
from .result import GradeResult
from .validate import validate_submission


def grade(
    public: Path,
    private: Path,
    submission: Path,
    metric: str,
    id_column: str,
    oof: Path | None = None,
) -> GradeResult:
    if metric not in METRICS:
        return GradeResult(valid=False, errors=[f"unknown metric '{metric}'"])
    try:
        sample = pd.read_csv(public / "sample_submission.csv")
        sub = pd.read_csv(submission)
    except Exception as exc:  # noqa: BLE001  unreadable or malformed CSV is a submission error
        return GradeResult(valid=False, errors=[f"cannot read submission: {exc}"], metric=metric)

    errors = validate_submission(sub, sample, id_column)
    if errors:
        return GradeResult(valid=False, errors=errors, metric=metric)

    result = GradeResult(valid=True, metric=metric)
    target_cols = [c for c in sample.columns if c != id_column]
    if len(target_cols) != 1:
        result.errors.append("scoring supports one target column only")
        return result
    target = target_cols[0]

    if oof is not None:
        result.cv_score, result.cv_std, result.cv_errors = score_oof(
            public, oof, metric, id_column, target
        )

    labels_path = private / "labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        merged = labels[[id_column, target]].merge(
            sub[[id_column, target]], on=id_column, suffixes=("_true", "_pred")
        )
        if len(merged) != len(labels):
            result.errors.append("labels and submission ids differ")
        else:
            result.score = METRICS[metric].fn(
                merged[f"{target}_true"].to_numpy(), merged[f"{target}_pred"].to_numpy()
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--id-column", required=True)
    args = parser.parse_args(argv)

    try:
        result = grade(
            args.public, args.private, args.submission, args.metric, args.id_column, args.oof
        )
    except Exception as exc:  # noqa: BLE001  grader must always emit a result
        result = GradeResult(valid=False, errors=[f"grader error: {exc}"], metric=args.metric)
    print(json.dumps(asdict(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
