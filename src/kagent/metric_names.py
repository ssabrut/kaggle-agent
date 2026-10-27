"""Map the metric name a competition page shows onto one of our scoring metrics.

Matching is by whole normalized phrase, never by substring. A looser rule would turn
"Root Mean Squared Logarithmic Error" into RMSE, which scores a different quantity.
A name we do not recognise returns None so the caller can ask a human.
"""

import re

_PHRASES: dict[str, tuple[str, ...]] = {
    "accuracy": (
        "accuracy",
        "categorization accuracy",
        "classification accuracy",
        "multiclass classification accuracy",
        "multi class accuracy",
    ),
    "auc": (
        "auc",
        "roc auc",
        "roc auc score",
        "auroc",
        "area under the roc curve",
        "area under the receiver operating characteristic curve",
    ),
    "logloss": (
        "log loss",
        "logloss",
        "logarithmic loss",
        "binary log loss",
        "multiclass log loss",
        "multi class log loss",
    ),
    "rmse": ("rmse", "root mean squared error", "root mean square error"),
    "mae": ("mae", "mean absolute error"),
    "f1_macro": (
        "macro f1",
        "macro f1 score",
        "f1 macro",
        "macro averaged f1",
        "macro averaged f1 score",
    ),
}

_BY_PHRASE = {phrase: metric for metric, phrases in _PHRASES.items() for phrase in phrases}


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def map_metric(text: str) -> str | None:
    """The registry key for a metric name from a competition page, or None."""
    return _BY_PHRASE.get(normalize_name(text))
