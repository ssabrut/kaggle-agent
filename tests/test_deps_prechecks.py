from datetime import UTC, datetime

import httpx

from kagent.deps.models import PreCheck
from kagent.deps.prechecks import near_miss, run_prechecks

NOW = datetime(2026, 10, 27, tzinfo=UTC)
SANDBOX_IN = "lightgbm\nnumpy\npandas>=2.2,<3\nscikit-learn\n"
INSTALLED = {"lightgbm", "numpy", "pandas", "scikit-learn", "scipy", "joblib"}


def doc(first="2019-01-10T00:00:00Z", latest="2026-10-01T00:00:00Z", vulns=()):
    return {
        "releases": {
            "1.0": [{"upload_time_iso_8601": first}],
            "2.0": [{"upload_time_iso_8601": latest}],
            "0.0-yanked": [],
        },
        "vulnerabilities": [{"id": v} for v in vulns],
    }


class FakeIndex:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.asked = data, error, []

    def fetch(self, package):
        self.asked.append(package)
        if self.error:
            raise self.error
        return self.data


def check(package, index, resolver=None, constraint=""):
    resolver = resolver or (lambda text: {n: "1" for n in INSTALLED} | {package.lower(): "1.2.8"})
    return run_prechecks(
        package, constraint, index=index, resolver=resolver,
        sandbox_in_text=SANDBOX_IN, installed=INSTALLED, now=NOW,
    )  # fmt: skip


def test_established_package_passes_cleanly():
    result = check("catboost", FakeIndex(doc()))
    assert result.exists_on_index and result.age_days > 2000
    assert result.latest_release_age_days == 26
    assert result.resolved_version == "1.2.8"
    assert result.new_transitive == [] and result.flags() == []


def test_unknown_name_is_reported_as_not_on_the_index():
    result = check("catboostt", FakeIndex(None))
    assert not result.exists_on_index and result.flags() == ["NOT ON INDEX"]


def test_new_package_and_advisories_are_flagged():
    result = check(
        "shinynew",
        FakeIndex(
            doc(
                first="2026-10-10T00:00:00Z",
                latest="2026-10-20T00:00:00Z",
                vulns=["PYSEC-1", "GHSA-2"],
            )
        ),
    )
    assert result.age_days == 17 and result.vulns == ["PYSEC-1", "GHSA-2"]
    assert "NEW PACKAGE (17d)" in result.flags() and "2 ADVISORIES" in result.flags()


def test_typosquat_names_are_flagged_as_near_misses():
    assert near_miss("catboots", {"catboost", "numpy"}) == "catboost"
    assert near_miss("numpyy", {"numpy"}) == "numpy"
    assert near_miss("catboost", {"catboost"}) is None  # exact match is not a near-miss
    assert near_miss("zzzzzz", {"catboost"}) is None
    assert "NEAR-MISS OF catboost" in check("catboots", FakeIndex(doc())).flags()


def test_transitive_additions_are_listed_without_the_package_itself():
    def resolver(text):
        assert "catboost" in text and "lightgbm" in text  # candidate added to the current set
        return {n: "1" for n in INSTALLED} | {"catboost": "1.2.8", "graphviz": "0.2", "plotly": "6"}

    result = check("catboost", FakeIndex(doc()), resolver)
    assert result.new_transitive == ["graphviz", "plotly"]
    assert "+2 TRANSITIVE" in result.flags()


def test_unresolvable_request_carries_the_resolver_message():
    def resolver(text):
        raise RuntimeError("no versions of catboost older than 7 days")

    result = check("catboost", FakeIndex(doc()), resolver)
    assert result.resolve_error and "DOES NOT RESOLVE" in result.flags()


def test_unreachable_index_marks_the_checks_incomplete():
    result = check("catboost", FakeIndex(error=httpx.ConnectError("offline")))
    assert result.checked is False and result.flags() == ["PRE-CHECKS INCOMPLETE"]


def test_precheck_defaults_are_inert():
    assert PreCheck().flags() == ["NOT ON INDEX"]
