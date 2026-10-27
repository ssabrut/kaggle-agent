"""Search policies: decide which operator to apply to which node next."""

import random
from dataclasses import dataclass
from typing import Protocol

from kagent.search.tree import Node, SolutionTree


@dataclass(frozen=True)
class Action:
    operator: str  # draft | debug | improve
    parent: Node | None  # None for a draft: it starts a new root


class SearchPolicy(Protocol):
    def select(self, tree: SolutionTree) -> Action: ...


class GreedyPolicy:
    """AIDE-style: build a few diverse drafts, then fix failures or improve the best node."""

    def __init__(self, num_drafts: int = 3, debug_prob: float = 0.5, seed: int = 0) -> None:
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob
        self._rng = random.Random(seed)

    def select(self, tree: SolutionTree) -> Action:
        if len(tree.roots()) < self.num_drafts:
            return Action("draft", None)

        buggy = tree.buggy_leaves()
        best = tree.best()
        if buggy and (best is None or self._rng.random() < self.debug_prob):
            return Action("debug", self._rng.choice(buggy))
        if best is not None:
            return Action("improve", best)
        return Action("draft", None)  # every draft died: start over
