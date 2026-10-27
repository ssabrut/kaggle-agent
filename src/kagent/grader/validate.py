"""Submission format checks against the competition's sample_submission.csv."""

import pandas as pd

MAX_ERRORS = 10


def kind_mismatch(reference: pd.Series, candidate: pd.Series) -> str | None:
    """What `candidate` should look like when its dtype cannot stand in for `reference`.

    Booleans are checked first because pandas counts them as numeric, and a 0/1 file is not
    acceptable where the competition expects True/False. Text labels must stay text.
    """
    if pd.api.types.is_bool_dtype(reference):
        return None if pd.api.types.is_bool_dtype(candidate) else "True/False"
    if pd.api.types.is_numeric_dtype(reference):
        return None if pd.api.types.is_numeric_dtype(candidate) else "numeric"
    is_text = not pd.api.types.is_numeric_dtype(candidate)  # bools count as numeric here
    return None if is_text else "text labels"


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
        if expected := kind_mismatch(sample[col], sub[col]):
            errors.append(f"'{col}' must be {expected}, like sample_submission")
    return errors[:MAX_ERRORS]
