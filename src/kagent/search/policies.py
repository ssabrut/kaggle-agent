"""Search policies: decide which operator to apply to which node next."""

import random
from dataclasses import dataclass
from typing import Protocol

from kagent.search.tree import Node, SolutionTree
from kagent.stages import stage_rank


@dataclass(frozen=True)
class Action:
    operator: str  # draft | debug | improve
    parent: Node | None  # None for a draft: it starts a new root


class SearchPolicy(Protocol):
    def select(self, tree: SolutionTree) -> Action: ...


def _furthest(nodes: list[Node]) -> list[Node]:
    """The failures that got furthest: closest to a working script, so cheapest to fix first."""
    top = max(stage_rank(n.stage) for n in nodes)
    return [n for n in nodes if stage_rank(n.stage) == top]


class GreedyPolicy:
    """AIDE-style: build a few diverse drafts, then fix failures or improve the best node."""

    def __init__(
        self, num_drafts: int = 3, debug_prob: float = 0.5, seed: int = 0, noise_k: float = 0.0
    ) -> None:
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob
        self.noise_k = noise_k  # a node must beat the incumbent by this many fold std devs
        self._rng = random.Random(seed)

    def select(self, tree: SolutionTree) -> Action:
        if len(tree.roots()) < self.num_drafts:
            return Action("draft", None)

        buggy = tree.buggy_leaves()
        best = tree.incumbent(self.noise_k)
        if buggy and (best is None or self._rng.random() < self.debug_prob):
            return Action("debug", self._rng.choice(_furthest(buggy)))
        if best is not None:
            return Action("improve", best)
        return Action("draft", None)  # every draft died: start over
