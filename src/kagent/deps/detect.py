"""Spot a missing package in a failed run's output and name the distribution to ask for."""

import re

from kagent.deps.lockfile import normalize

_MISSING = re.compile(r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]")

# Import names that differ from the name on the package index.
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "Bio": "biopython",
}


def missing_module(stderr: str) -> str | None:
    """Top-level import name from the last ModuleNotFoundError, or None."""
    found = _MISSING.findall(stderr)
    return found[-1].split(".")[0] if found else None


def dist_name(module: str) -> str:
    return IMPORT_TO_DIST.get(module, module)


def needs_package(stderr: str, installed: set[str]) -> str | None:
    """The distribution to request, or None when the failure is not a missing package.

    A submodule typo inside an installed package (`No module named 'sklearn.foo'`) is an API
    error for the debug step, not a dependency.
    """
    module = missing_module(stderr)
    if module is None:
        return None
    dist = dist_name(module)
    return None if normalize(dist) in installed else dist
