"""Automatic checks run before a human sees a dependency request.

Network access happens here, on the host, never in the sandbox. The index lookup and the
throwaway resolve are injectable so tests (and offline runs) need no network.
"""

import difflib
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from kagent.deps.lockfile import add_requirement, normalize, parse_lock
from kagent.deps.models import PreCheck

# Well-known names a typosquat would imitate. Installed packages are added at check time.
POPULAR = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "transformers",
    "datasets",
    "tokenizers",
    "sentence-transformers",
    "matplotlib",
    "seaborn",
    "plotly",
    "statsmodels",
    "optuna",
    "hyperopt",
    "shap",
    "lime",
    "imbalanced-learn",
    "category-encoders",
    "feature-engine",
    "featuretools",
    "tsfresh",
    "prophet",
    "sktime",
    "pmdarima",
    "nltk",
    "spacy",
    "gensim",
    "opencv-python",
    "pillow",
    "scikit-image",
    "albumentations",
    "timm",
    "requests",
    "httpx",
    "flask",
    "django",
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "pytest",
    "joblib",
    "tqdm",
    "pyyaml",
    "beautifulsoup4",
    "lxml",
    "openpyxl",
    "polars",
    "pyarrow",
    "dask",
    "numba",
    "cython",
    "networkx",
    "umap-learn",
]

NEAR_MISS_CUTOFF = 0.85


class IndexClient(Protocol):
    def fetch(self, package: str) -> dict | None:
        """The index's JSON document for the package, or None when it does not exist."""
        ...


class PyPIClient:
    def __init__(self, base_url: str = "https://pypi.org/pypi", timeout_s: float = 20) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def fetch(self, package: str) -> dict | None:
        resp = self._http.get(f"/{package}/json")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


Resolver = Callable[[str], dict[str, str]]  # requirements text -> {package: version}


def uv_resolver(cooldown: str, python_version: str = "3.11") -> Resolver:
    """Resolve requirements text with a throwaway `uv pip compile`; returns pinned versions."""

    def resolve(requirements: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "r.in", Path(tmp) / "r.txt"
            src.write_text(requirements)
            proc = subprocess.run(
                [
                    "uv", "pip", "compile", str(src), "--universal", "--exclude-newer", cooldown,
                    "--python-version", python_version, "-o", str(out), "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )  # fmt: skip
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip()[-400:])
            return parse_lock(out.read_text())

    return resolve


def _days_since(timestamp: str, now: datetime) -> int:
    return (now - datetime.fromisoformat(timestamp)).days


def _upload_times(data: dict) -> list[str]:
    return [
        f["upload_time_iso_8601"]
        for files in data.get("releases", {}).values()
        for f in files
        if f.get("upload_time_iso_8601")
    ]


def near_miss(package: str, known: set[str]) -> str | None:
    """A known name this one is close to but not equal to."""
    name = normalize(package)
    if name in known:
        return None
    close = difflib.get_close_matches(name, sorted(known), n=1, cutoff=NEAR_MISS_CUTOFF)
    return close[0] if close else None


def run_prechecks(
    package: str,
    constraint: str,
    *,
    index: IndexClient,
    resolver: Resolver,
    sandbox_in_text: str,
    installed: set[str],
    now: datetime | None = None,
) -> PreCheck:
    now = now or datetime.now(UTC)
    try:
        data = index.fetch(package)
    except httpx.HTTPError:
        return PreCheck(checked=False)
    if data is None:
        return PreCheck(exists_on_index=False)

    times = sorted(_upload_times(data))
    check = PreCheck(
        exists_on_index=True,
        age_days=_days_since(times[0], now) if times else None,
        latest_release_age_days=_days_since(times[-1], now) if times else None,
        near_miss_of=near_miss(package, {normalize(p) for p in POPULAR} | installed),
        vulns=[v.get("id", "?") for v in data.get("vulnerabilities", [])],
    )
    try:
        resolved = resolver(add_requirement(sandbox_in_text, package, constraint))
    except RuntimeError as exc:
        check.resolve_error = str(exc)
        return check
    check.resolved_version = resolved.get(normalize(package))
    check.new_transitive = sorted(set(resolved) - installed - {normalize(package)})
    return check
