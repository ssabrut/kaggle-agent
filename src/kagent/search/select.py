"""Final selection: pick the node to submit."""

import json

from kagent.search.tree import Node, SolutionTree
from kagent.workspace import Workspace

CHOICES = ("robust_cv", "cv")


def select_best(ws: Workspace, tree: SolutionTree, by: str = "robust_cv") -> Node:
    """Choose by cross-validation, never by the held-out score, and record the choice.

    robust_cv penalizes fold-to-fold spread; cv uses the mean alone.
    """
    if by not in CHOICES:
        raise ValueError(f"unknown selection rule '{by}', choose from {CHOICES}")
    node = tree.best(robust=by == "robust_cv")
    if node is None:
        raise ValueError("no node has a cross-validation score yet; run `kagent search` first")
    ws.selected_path.write_text(
        json.dumps(
            {
                "node": node.id,
                "run_id": node.run_id,
                "cv_score": node.cv_score,
                "cv_std": node.cv_std,
                "by": by,
            },
            indent=2,
        )
        + "\n"
    )
    return node


def selected_run_id(ws: Workspace) -> str:
    if not ws.selected_path.exists():
        raise FileNotFoundError("nothing selected yet; run `kagent select` or pass --run")
    return json.loads(ws.selected_path.read_text())["run_id"]
