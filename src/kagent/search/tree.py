"""Solution tree: each node is a standalone script plus how it scored.

Persisted as JSONL, one line per node version (last line for an id wins), so a crashed
search can resume and the history stays inspectable.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Status = Literal["ok", "buggy", "dead", "blocked"]  # blocked: waiting on a dependency request

ROBUST_STD_WEIGHT = 0.5  # [S] penalty for fold-to-fold spread when selecting by robust CV


@dataclass
class Node:
    id: str
    parent: str | None
    operator: str  # draft | debug | improve
    code_path: Path
    status: Status
    cv_score: float | None = None
    cv_std: float | None = None
    summary: str = ""  # compressed log, used as sibling context
    error: str = ""  # short failure description for buggy/dead nodes
    run_id: str | None = None
    debug_depth: int = 0  # consecutive debug ancestors, including this node
    stage: str = ""  # furthest step its run reached (kagent.stages); empty on older nodes
    dep_request: str | None = None  # the dependency request a blocked node waits on
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SolutionTree:
    def __init__(self, path: Path, higher_is_better: bool) -> None:
        self.path = path
        self.higher_is_better = higher_is_better
        self._nodes: dict[str, Node] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    raw = json.loads(line)
                    raw["code_path"] = Path(raw["code_path"])
                    self._nodes[raw["id"]] = Node(**raw)

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def next_id(self) -> str:
        return f"n{len(self._nodes) + 1:03d}"

    def add(self, node: Node) -> Node:
        self._nodes[node.id] = node
        self._append(node)
        return node

    def update(self, node_id: str, **changes) -> Node:
        node = replace(self._nodes[node_id], **changes)
        self._nodes[node_id] = node
        self._append(node)
        return node

    def roots(self) -> list[Node]:
        return [n for n in self.nodes if n.parent is None]

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes if n.parent == node_id]

    def ok_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.status == "ok" and n.cv_score is not None]

    def blocked_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.status == "blocked"]

    def buggy_leaves(self) -> list[Node]:
        """Failed nodes nobody has tried to fix yet."""
        return [n for n in self.nodes if n.status == "buggy" and not self.children(n.id)]

    def best(self, robust: bool = False) -> Node | None:
        ok = self.ok_nodes()
        if not ok:
            return None
        return max(ok, key=lambda n: self.rank_value(n, robust))

    def incumbent(self, noise_k: float = 0.0) -> Node | None:
        """The node the search builds on: the best one, ignoring gains that are only noise.

        Nodes are visited in creation order. A node replaces the incumbent only when it beats it
        by more than `noise_k` times the larger of the two fold spreads (spec 13.2). Comparing
        with the incumbent, not the previous node, lets several small gains add up to a real one.
        With noise_k = 0 this is the plain best node, earliest first on a tie.
        """
        trail = self.incumbent_trail(noise_k)
        return trail[-1][1] if trail else None

    def incumbent_trail(self, noise_k: float = 0.0) -> list[tuple[Node, Node]]:
        """For each ok node in creation order, the incumbent once that node has been considered."""
        trail: list[tuple[Node, Node]] = []
        incumbent: Node | None = None
        for node in self.ok_nodes():
            if incumbent is None:
                incumbent = node
            else:
                floor = noise_k * max(node.cv_std or 0.0, incumbent.cv_std or 0.0)
                if self.rank_value(node) - self.rank_value(incumbent) > floor:
                    incumbent = node
            trail.append((node, incumbent))
        return trail

    def rank_value(self, node: Node, robust: bool = False) -> float:
        """Larger is better regardless of metric direction."""
        assert node.cv_score is not None
        score = node.cv_score
        if robust and node.cv_std is not None:
            score += (
                -ROBUST_STD_WEIGHT * node.cv_std
                if self.higher_is_better
                else ROBUST_STD_WEIGHT * node.cv_std
            )
        return score if self.higher_is_better else -score

    def _append(self, node: Node) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(node), default=str) + "\n")
