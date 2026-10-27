"""Page parsing and the init flow, using text shaped like a real Kaggle page (no browser)."""

import re
from contextlib import nullcontext

import pandas as pd
import pytest
from typer.testing import CliRunner

from kagent.cli import app
from kagent.competition_page import (
    FetchedPages,
    PageFetchError,
    clean_text,
    fetch_pages,
    parse_overview,
    strip_tab_chrome,
)
from kagent.config import CompetitionConfig

CONTAINER = """Description
link
keyboard_arrow_up
📣  Recommended Competition

We highly recommend Titanic - Machine Learning from Disaster to get familiar with the basics.

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery.

Evaluation
link
keyboard_arrow_up
Metric

Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct.

Submission Format

PassengerId,Transported
0013_01,False
etc.

Frequently Asked Questions
link
keyboard_arrow_up
What is a Getting Started competition?

Getting Started competitions were created by Kaggle data scientists.

Citation
link
keyboard_arrow_up

Addison Howard, Ashley Chow, and Ryan Holbrook. Spaceship Titanic. Kaggle.
"""

HEADER = """menu
Skip to
content

Login
Sign Up
Kaggle uses cookies from Google to deliver and enhance the quality of its services.
OK, Got it.
Kaggle · Getting Started Prediction Competition · Ongoing
Join Competition
more_horiz
Spaceship Titanic

Predict which passengers are transported to an alternate dimension

Overview
Data
Code
Models
Discussion
Leaderboard
Rules
Overview
all_inclusive
This competition runs indefinitely with a rolling leaderboard.
"""

SIDEBAR = """Competition Host

Kaggle

Prizes & Awards

Knowledge

Does not award Points or Medals

Participation

150,311 Entrants

Tags
Beginner
Tabular
Binary Classification
Categorization Accuracy
Table of Contents
collapse_all
Description
Evaluation
Frequently Asked Questions
Citation
"""

BODY = HEADER + CONTAINER + SIDEBAR


def test_overview_keeps_description_and_evaluation_but_not_faq_or_icons():
    info = parse_overview(CONTAINER, BODY)
    assert info.description.startswith("Description\n")
    assert "Welcome to the year 2912" in info.description
    assert "classification accuracy" in info.description
    assert "Frequently Asked Questions" not in info.description
    assert "Addison Howard" not in info.description
    assert "keyboard_arrow_up" not in info.description
    assert info.evaluation.startswith("Evaluation\n")
    assert "Welcome to the year" not in info.evaluation


def test_sidebar_facts_are_read_from_the_body():
    info = parse_overview(CONTAINER, BODY)
    assert info.tags == ["Beginner", "Tabular", "Binary Classification", "Categorization Accuracy"]
    assert info.kind == "Kaggle · Getting Started Prediction Competition · Ongoing"
    assert info.prizes == ["Knowledge", "Does not award Points or Medals"]
    assert info.prize_warning is None


def test_metric_comes_from_the_tag_that_maps():
    info = parse_overview(CONTAINER, BODY)
    assert (info.metric, info.metric_text) == ("accuracy", "Categorization Accuracy")


def test_unmapped_metric_tag_leaves_metric_empty():
    body = BODY.replace("Categorization Accuracy", "Quadratic Weighted Kappa")
    info = parse_overview(CONTAINER, body)
    assert info.metric is None and "Quadratic Weighted Kappa" in info.tags


def test_prize_competitions_and_missing_prize_info_raise_a_warning():
    prized = BODY.replace("Does not award Points or Medals", "$25,000\n\nAwards Points & Medals")
    assert "may award prizes" in parse_overview(CONTAINER, prized).prize_warning
    missing = BODY.replace("Prizes & Awards", "Something else")
    assert "could not read" in parse_overview(CONTAINER, missing).prize_warning


def test_page_without_the_expected_sections_still_parses():
    info = parse_overview("Just some text", "nothing useful here")
    assert info.description == "Just some text"
    assert info.evaluation == "" and info.tags == [] and info.kind is None and info.metric is None


def test_clean_text_drops_icon_lines_and_collapses_blank_runs():
    assert clean_text("a\nlink\n\n\n\nb\ncollapse_all\n") == "a\n\nb"


def test_strip_tab_chrome_removes_header_and_tab_bar():
    text = HEADER + "Data Description\n\nPassengerId - a unique Id"
    stripped = strip_tab_chrome(text)
    assert stripped.startswith("Overview\nThis competition runs indefinitely")
    assert "Kaggle uses cookies" not in stripped and "Join Competition" not in stripped
    assert "Data Description" in stripped and "PassengerId - a unique Id" in stripped


def test_strip_tab_chrome_leaves_text_without_a_tab_bar():
    assert strip_tab_chrome("plain\nlink\ntext") == "plain\ntext"


# --- fetch_pages with a fake browser page -----------------------------------------------------


class FakeLocator:
    def __init__(self, page, chain):
        self.page, self.chain = page, chain

    def locator(self, selector):
        return FakeLocator(self.page, [*self.chain, selector])

    def inner_text(self):
        return self.page.locator_texts[tuple(self.chain)]


class FakePage:
    def __init__(self):
        self.visited, self.waited, self.current = [], [], ""
        self.locator_texts = {("#description", "xpath=.."): CONTAINER}
        self.texts = {
            ("https://www.kaggle.com/competitions/spaceship-titanic/overview", "body"): BODY,
            ("https://www.kaggle.com/competitions/spaceship-titanic/data", "#site-content"): (
                HEADER + "Data Description\n\nPassengerId - a unique Id"
            ),
            ("https://www.kaggle.com/competitions/spaceship-titanic/rules", "#site-content"): (
                HEADER + "Rules\n\nAutomated tools are not allowed"
            ),
        }

    def goto(self, url, **kwargs):
        self.visited.append(url)
        self.current = url

    def wait_for_selector(self, selector, **kwargs):
        self.waited.append(selector)

    def wait_for_timeout(self, ms):
        pass

    def locator(self, selector):
        return FakeLocator(self, [selector])

    def inner_text(self, selector):
        return self.texts[(self.current, selector)]


def test_fetch_pages_reads_overview_container_and_both_tabs():
    page = FakePage()
    seen = []

    def factory(headless):
        seen.append(headless)
        return nullcontext(page)

    fetched = fetch_pages("spaceship-titanic", headless=True, page_factory=factory)

    assert seen == [True]
    assert [u.rsplit("/", 1)[1] for u in page.visited] == ["overview", "data", "rules"]
    assert page.waited == ["#description", "#site-content", "#site-content"]
    assert fetched.overview_container == CONTAINER and fetched.overview_body == BODY
    assert "PassengerId - a unique Id" in fetched.data_text
    assert "Kaggle uses cookies" not in fetched.data_text
    assert "Automated tools are not allowed" in fetched.rules_text


# --- kagent init ------------------------------------------------------------------------------

runner = CliRunner()


def flat(result) -> str:
    """Output with Rich's box characters and line wrapping removed, so phrases can be matched."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())


@pytest.fixture
def data_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rows = pd.DataFrame({"PassengerId": range(40), "x": range(40), "Transported": [0, 1] * 20})
    rows.to_csv(src / "train.csv", index=False)
    rows.drop(columns="Transported").to_csv(src / "test.csv", index=False)
    pd.DataFrame({"PassengerId": range(40), "Transported": 0}).to_csv(
        src / "sample_submission.csv", index=False
    )
    return src


def init(settings, data_dir, *args):
    env = {"KAGENT_WORKSPACES_DIR": str(settings.workspaces_dir)}
    return runner.invoke(
        app, ["init", "-c", "titanic", "--data-dir", str(data_dir), *args], env=env
    )


def fake_fetch(monkeypatch, error=None):
    def fetch(slug, headless=False):
        if error:
            raise error
        return FetchedPages(CONTAINER, BODY, data_text="data dictionary", rules_text="the rules")

    monkeypatch.setattr("kagent.cli.fetch_pages", fetch)


def load(settings) -> CompetitionConfig:
    return CompetitionConfig.model_validate_json(
        (settings.workspaces_dir / "titanic" / "config.json").read_text()
    )


def test_fetch_page_supplies_the_metric_description_and_saved_pages(settings, data_dir, monkeypatch):
    fake_fetch(monkeypatch)
    result = init(settings, data_dir, "--fetch-page")

    assert result.exit_code == 0, result.output
    config = load(settings)
    assert (config.metric, config.id_column) == ("accuracy", "PassengerId")  # both inferred
    root = settings.workspaces_dir / "titanic"
    assert "Welcome to the year 2912" in (root / "description.md").read_text()
    assert (root / "page" / "rules.txt").read_text().strip() == "the rules"
    assert (root / "page" / "data.txt").read_text().strip() == "data dictionary"
    assert "metric from page: accuracy" in flat(result)
    assert "read rules.txt" in flat(result)


def test_explicit_metric_and_description_beat_the_page(settings, data_dir, tmp_path, monkeypatch):
    fake_fetch(monkeypatch)
    mine = tmp_path / "mine.md"
    mine.write_text("my notes")

    result = init(settings, data_dir, "--fetch-page", "--metric", "auc", "--description", str(mine))

    assert result.exit_code == 0, result.output
    assert load(settings).metric == "auc"
    assert (settings.workspaces_dir / "titanic" / "description.md").read_text() == "my notes"


def test_failed_page_fetch_is_not_fatal_when_a_metric_is_given(settings, data_dir, monkeypatch):
    fake_fetch(monkeypatch, error=PageFetchError("blocked"))
    result = init(settings, data_dir, "--fetch-page", "--metric", "accuracy")

    assert result.exit_code == 0, result.output
    assert "could not read the competition page: blocked" in flat(result)
    assert not (settings.workspaces_dir / "titanic" / "description.md").exists()


def test_no_metric_anywhere_is_a_clear_error(settings, data_dir, monkeypatch):
    fake_fetch(monkeypatch, error=PageFetchError("blocked"))
    result = init(settings, data_dir, "--fetch-page")
    assert result.exit_code != 0 and "pass --metric" in flat(result)

    plain = init(settings, data_dir)
    assert plain.exit_code != 0 and "pass --metric" in flat(plain)


def test_unmapped_page_metric_lists_the_tags_it_saw(settings, data_dir, monkeypatch):
    def fetch(slug, headless=False):
        return FetchedPages(CONTAINER, BODY.replace("Categorization Accuracy", "Weighted Kappa"))

    monkeypatch.setattr("kagent.cli.fetch_pages", fetch)
    result = init(settings, data_dir, "--fetch-page")
    assert result.exit_code != 0 and "Weighted Kappa" in flat(result)


def test_unknown_metric_is_rejected_before_anything_is_created(settings, data_dir):
    result = init(settings, data_dir, "--metric", "bogus")
    assert result.exit_code != 0 and "unknown metric" in flat(result)
    assert not (settings.workspaces_dir / "titanic").exists()


def test_id_column_is_inferred_without_a_page(settings, data_dir):
    assert init(settings, data_dir, "--metric", "accuracy").exit_code == 0
    assert load(settings).id_column == "PassengerId"


def test_explicit_id_column_wins(settings, data_dir):
    # rmse uses plain K-fold, so the folds still build when the two columns swap roles
    result = init(settings, data_dir, "--metric", "rmse", "--id-column", "Transported")
    assert result.exit_code == 0, result.output
    assert load(settings).id_column == "Transported"


def test_missing_sample_submission_cannot_infer_the_id_column(settings, data_dir):
    (data_dir / "sample_submission.csv").unlink()
    result = init(settings, data_dir, "--metric", "accuracy")
    assert result.exit_code != 0 and result.exception is not None
