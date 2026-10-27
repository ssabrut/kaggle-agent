import pytest

from kagent.deps.detect import dist_name, missing_module, needs_package

INSTALLED = {"numpy", "pandas", "scikit-learn", "lightgbm"}


def trace(module, kind="ModuleNotFoundError"):
    return f"Traceback (most recent call last):\n  File x.py\n{kind}: No module named '{module}'\n"


def test_finds_the_missing_top_level_module():
    assert missing_module(trace("catboost")) == "catboost"
    assert missing_module(trace("torch.nn.functional")) == "torch"


def test_last_error_wins_and_double_quotes_work():
    text = trace("first") + 'ModuleNotFoundError: No module named "second"\n'
    assert missing_module(text) == "second"


@pytest.mark.parametrize("text", ["", "KeyError: 'zip'", "ImportError: cannot import name 'x'"])
def test_other_failures_are_not_missing_modules(text):
    assert missing_module(text) is None


def test_import_names_map_to_distribution_names():
    assert dist_name("sklearn") == "scikit-learn"
    assert dist_name("cv2") == "opencv-python"
    assert dist_name("catboost") == "catboost"


def test_request_the_distribution_when_it_is_not_installed():
    assert needs_package(trace("catboost"), INSTALLED) == "catboost"
    assert needs_package(trace("PIL"), INSTALLED) == "pillow"


def test_submodule_typo_in_an_installed_package_is_not_a_dependency():
    assert needs_package(trace("sklearn.bogus"), INSTALLED) is None
    assert needs_package(trace("pandas.core.nope"), INSTALLED) is None
