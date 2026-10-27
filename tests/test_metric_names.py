import pytest

from kagent.grader.metrics import METRICS
from kagent.metric_names import map_metric, normalize_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Categorization Accuracy", "accuracy"),  # the tag on spaceship-titanic
        ("classification accuracy", "accuracy"),
        ("Area under the ROC curve", "auc"),
        ("ROC-AUC", "auc"),
        ("Log Loss", "logloss"),
        ("Multi-Class Log Loss", "logloss"),
        ("Root Mean Squared Error", "rmse"),
        ("RMSE", "rmse"),
        ("Mean Absolute Error", "mae"),
        ("Macro F1 Score", "f1_macro"),
    ],
)
def test_known_names_map_to_our_metrics(name, expected):
    assert map_metric(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "Root Mean Squared Logarithmic Error",  # a different quantity from RMSE
        "RMSLE",
        "Quadratic Weighted Kappa",
        "Mean Average Precision @ 5",
        "F1 Score",  # averaging unspecified, so it is not macro
        "Area Under the Curve",  # could be precision-recall
        "Tabular",
        "Binary Classification",
        "",
    ],
)
def test_unknown_or_ambiguous_names_return_none(name):
    assert map_metric(name) is None


def test_normalization_ignores_case_and_punctuation():
    assert normalize_name("  Multi-Class  LOG_Loss! ") == "multi class log loss"


def test_every_target_is_a_real_metric():
    from kagent.metric_names import _PHRASES

    assert set(_PHRASES) <= set(METRICS)
