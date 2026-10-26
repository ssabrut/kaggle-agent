"""Submission format checks against the competition's sample_submission.csv."""

import pandas as pd

MAX_ERRORS = 10


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, id_column: str) -> list[str]:
    if list(sub.columns) != list(sample.columns):
        return [f"columns {list(sub.columns)} do not match expected {list(sample.columns)}"]

    errors: list[str] = []
    if len(sub) != len(sample):
        errors.append(f"expected {len(sample)} rows, got {len(sub)}")
    if sub[id_column].duplicated().any():
        errors.append(f"duplicate values in '{id_column}'")
    if set(sub[id_column].astype(str)) != set(sample[id_column].astype(str)):
        errors.append(f"'{id_column}' values differ from sample_submission")

    for col in sample.columns:
        if col == id_column:
            continue
        if sub[col].isna().any():
            errors.append(f"missing values in '{col}'")
        if pd.api.types.is_numeric_dtype(sample[col]) and not pd.api.types.is_numeric_dtype(
            sub[col]
        ):
            errors.append(f"'{col}' must be numeric")
    return errors[:MAX_ERRORS]
