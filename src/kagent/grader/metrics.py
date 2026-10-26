"""Metric registry. The agent never sees or imports this code; only the grader does."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


@dataclass(frozen=True)
class Metric:
    fn: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


METRICS: dict[str, Metric] = {
    "accuracy": Metric(lambda t, p: float(accuracy_score(t, p)), True),
    "auc": Metric(lambda t, p: float(roc_auc_score(t, p)), True),
    "f1_macro": Metric(lambda t, p: float(f1_score(t, p, average="macro")), True),
    "logloss": Metric(lambda t, p: float(log_loss(t, p)), False),
    "mae": Metric(_mae, False),
    "rmse": Metric(_rmse, False),
}


def available() -> list[str]:
    return sorted(METRICS)
