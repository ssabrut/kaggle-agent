"""Read a competition's public pages for what the Kaggle download does not carry.

Opt-in (`kagent init --fetch-page`). Playwright drives a real browser because the pages are
rendered by JavaScript. The Overview selectors were checked against a real page
(spaceship-titanic): the Overview text sits in the parent of `div#description`, and the sidebar
tags and prize lines are in the page body. The Data and Rules tabs use `#site-content` with the
navigation stripped; that part has not been checked against a live page, so those texts are saved
for a human to read and are not fed to the models.

Kaggle's Terms of Use govern automated access, so this fetches a handful of pages once per
competition and nothing more. It does not try to get past a login, a consent wall or a bot check.
"""

import json
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from kagent.metric_names import map_metric
from kagent.workspace import Workspace

# Material icon names that the page renders as text next to section headings.
ICON_WORDS = frozenset(
    {
        "link", "keyboard_arrow_up", "keyboard_arrow_down", "all_inclusive", "collapse_all",
        "expand_all", "more_horiz", "content_copy", "menu", "search", "close",
    }
)  # fmt: skip
TAB_BAR = re.compile(r"Overview\s+Data\s+Code\s+Models\s+Discussion\s+Leaderboard\s+Rules\s*")
KIND_LINE = re.compile(r"(?m)^Kaggle\s+·.*Competition.*$")
NO_PRIZE = "does not award points or medals"
OVERVIEW_END_MARKERS = ("\nFrequently Asked Questions", "\nCitation")

PageFactory = Callable[[bool], AbstractContextManager[Any]]


class PageFetchError(RuntimeError):
    """The page could not be read; init carries on without it."""


@dataclass
class FetchedPages:
    overview_container: str
    overview_body: str
    data_text: str = ""
    rules_text: str = ""


@dataclass
class PageInfo:
    description: str  # Overview text up to the FAQ, Evaluation section included
    evaluation: str
    # for example "Kaggle · Getting Started Prediction Competition · Ongoing"
    kind: str | None = None
    tags: list[str] = field(default_factory=list)
    prizes: list[str] = field(default_factory=list)
    metric: str | None = None  # our registry key, when a tag maps onto one
    metric_text: str | None = None  # the tag it came from
    prize_warning: str | None = None


def clean_text(text: str) -> str:
    """Drop icon-name lines and collapse runs of blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() not in ICON_WORDS]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i : j if j >= 0 else None]


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _cut_at_first(text: str, markers: tuple[str, ...]) -> str:
    cuts = [i for m in markers if (i := text.find(m)) > 0]
    return text[: min(cuts)] if cuts else text


def parse_overview(container_text: str, body_text: str) -> PageInfo:
    main = _cut_at_first(clean_text(container_text), OVERVIEW_END_MARKERS)
    evaluation = main[main.find("\nEvaluation\n") :].strip() if "\nEvaluation\n" in main else ""

    body = clean_text(body_text)
    tags = _lines(_between(body, "\nTags\n", "\nTable of Contents"))
    prizes = _lines(_between(body, "Prizes & Awards", "\nParticipation"))
    kind = KIND_LINE.search(body)

    metric = metric_text = None
    for tag in reversed(tags):
        if (found := map_metric(tag)) is not None:
            metric, metric_text = found, tag
            break

    if not prizes:
        warning = "could not read the prize information"
    elif NO_PRIZE not in " ".join(prizes).lower():
        warning = "this competition may award prizes or medals"
    else:
        warning = None

    return PageInfo(
        description=main,
        evaluation=evaluation,
        kind=kind.group(0).strip() if kind else None,
        tags=tags,
        prizes=prizes,
        metric=metric,
        metric_text=metric_text,
        prize_warning=warning,
    )


def strip_tab_chrome(text: str) -> str:
    """Remove the site header and tab bar that precede a tab's own content."""
    cleaned = clean_text(text)
    bar = TAB_BAR.search(cleaned)
    return cleaned[bar.end() :].strip() if bar else cleaned


@contextmanager
def playwright_page(headless: bool) -> Iterator[Any]:
    """A fresh browser page. Playwright is imported here so it stays an optional tool."""
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as exc:
        raise PageFetchError(
            "playwright is not installed; run `uv sync` and `uv run playwright install chromium`"
        ) from exc

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless)
        except Error as exc:
            raise PageFetchError(
                f"could not start the browser ({str(exc).splitlines()[0]}); "
                "try `uv run playwright install chromium`"
            ) from exc
        try:
            yield browser.new_page()
        except Error as exc:
            raise PageFetchError(f"{type(exc).__name__}: {str(exc).splitlines()[0]}") from exc
        finally:
            browser.close()


def fetch_pages(
    slug: str,
    *,
    headless: bool = False,
    timeout_ms: int = 30_000,
    page_factory: PageFactory = playwright_page,
) -> FetchedPages:
    base = f"https://www.kaggle.com/competitions/{slug}"
    with page_factory(headless) as page:
        page.goto(f"{base}/overview", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_selector("#description", timeout=timeout_ms)
        container = page.locator("#description").locator("xpath=..").inner_text()
        body = page.inner_text("body")

        tabs = []
        for tab in ("data", "rules"):  # unverified selectors; see the module docstring
            page.goto(f"{base}/{tab}", wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_selector("#site-content", timeout=timeout_ms)
            page.wait_for_timeout(3000)  # the tab body loads after the page shell
            tabs.append(strip_tab_chrome(page.inner_text("#site-content")))
    return FetchedPages(container, body, data_text=tabs[0], rules_text=tabs[1])


def save_pages(ws: Workspace, fetched: FetchedPages, info: PageInfo) -> None:
    """Keep everything fetched under page/ so a human can read it."""
    ws.page_dir.mkdir(parents=True, exist_ok=True)
    (ws.page_dir / "overview.md").write_text(info.description + "\n")
    (ws.page_dir / "data.txt").write_text(fetched.data_text + "\n")
    (ws.page_dir / "rules.txt").write_text(fetched.rules_text + "\n")
    (ws.page_dir / "info.json").write_text(json.dumps(asdict(info), indent=2) + "\n")
