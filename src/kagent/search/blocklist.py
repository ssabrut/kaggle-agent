"""Approaches that already failed or did not help, shown to the planner so it moves on.

Built from the tree on every step (spec 13.2), so there is nothing to store and an entry cannot
outlive its cause: it vanishes with its node, and a missing-package failure is dropped as soon as
the package is installed. Only the newest few entries are shown, which keeps the prompt short and
lets old ones fade.
"""

import re
from collections.abc import Collection

from kagent.deps.detect import dist_name, missing_module
from kagent.deps.lockfile import normalize
from kagent.search.tree import Node, SolutionTree

MAX_ENTRIES = 8
ENTRY_CHARS = 220


def _clip(text: str, limit: int = ENTRY_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _plan(node: Node) -> str:
    """The `# PLAN:` comment block at the top of the script, which names its approach."""
    try:
        lines = node.code_path.read_text().splitlines()
    except OSError:
        return ""
    header: list[str] = []
    for line in lines:
        if not line.startswith("#"):
            break
        header.append(line.lstrip("#").strip())
    return re.sub(r"^PLAN:?\s*", "", " ".join(h for h in header if h), flags=re.IGNORECASE)


def _approach(node: Node) -> str:
    return _clip(node.summary) or _clip(_plan(node)) or "an earlier attempt"


def _last_line(error: str) -> str:
    lines = [ln.strip() for ln in error.splitlines() if ln.strip()]
    return _clip(lines[-1]) if lines else "no error recorded"


def _stale(node: Node, installed: Collection[str]) -> bool:
    """A missing-package failure no longer applies once that package is installed."""
    module = missing_module(node.error)
    return module is not None and normalize(dist_name(module)) in installed


def _entry(tree: SolutionTree, node: Node, installed: Collection[str]) -> str | None:
    if node.status == "dead":
        if _stale(node, installed):
            return None
        return f"{_approach(node)} -> failed with {_last_line(node.error)}"

    if node.status == "ok" and node.operator == "improve" and node.parent is not None:
        parent = tree.get(node.parent)
        if (
            node.cv_score is not None
            and parent.cv_score is not None
            and tree.rank_value(node) <= tree.rank_value(parent)
        ):
            return (
                f"{_approach(node)} -> cv {node.cv_score:.5f} against "
                f"{parent.cv_score:.5f} before: no gain"
            )
    return None


def build_blocklist(
    tree: SolutionTree,
    installed: Collection[str] = (),
    *,
    skip_children_of: str | None = None,
    limit: int = MAX_ENTRIES,
) -> list[str]:
    """Newest first. `installed` holds normalized distribution names.

    Nodes still being debugged are left out, since their branch is not over. The children of
    `skip_children_of` are left out because the improve prompt already lists them.
    """
    entries: list[str] = []
    for node in reversed(tree.nodes):
        if skip_children_of is not None and node.parent == skip_children_of:
            continue
        entry = _entry(tree, node, installed)
        if entry is not None and entry not in entries:
            entries.append(entry)
            if len(entries) == limit:
                break
    return entries
