"""Operators turn a node (or nothing, for a draft) into new code.

Planner and coder are separate roles (R&D-Agent style): the planner decides what to try,
the coder writes the script. Debug skips the planner.
"""

import ast
import re
from dataclasses import dataclass
from typing import Protocol

from kagent.llm.client import ChatClient, Message
from kagent.search import prompts
from kagent.search.tree import Node, SolutionTree

SUMMARY_CHARS = 600
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class OperatorError(RuntimeError):
    """The model did not produce usable code."""


@dataclass
class Context:
    description: str
    eda: str
    metric: str
    higher_is_better: bool
    id_column: str
    target: str
    tree: SolutionTree
    step_timeout_s: int
    remaining_s: float
    sandbox_cpus: float = 4
    sandbox_memory: str = "8g"
    error_log: str = ""  # failure output of the node being debugged

    def contract(self) -> str:
        return prompts.contract(
            metric=self.metric,
            id_column=self.id_column,
            target=self.target,
            step_timeout_s=self.step_timeout_s,
            cpus=self.sandbox_cpus,
            memory=self.sandbox_memory,
        )

    @property
    def remaining_min(self) -> float:
        return max(self.remaining_s, 0) / 60


class Operator(Protocol):
    name: str

    def apply(self, node: Node | None, ctx: Context) -> str:
        """Return the new script."""
        ...


def extract_code(text: str) -> str:
    """Pull the script out of a reply and check that it parses."""
    blocks = _FENCE.findall(text)
    code = max(blocks, key=len) if blocks else text
    code = code.strip()
    if not code:
        raise OperatorError("reply contained no code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise OperatorError(f"syntax error on line {exc.lineno}: {exc.msg}") from exc
    # A one-word prose reply such as "nope" is valid Python; real scripts do something.
    if all(isinstance(s, ast.Expr) and not isinstance(s.value, ast.Call) for s in tree.body):
        raise OperatorError("reply is not a script")
    return code + "\n"


def _write_code(coder: ChatClient, messages: list[Message]) -> str:
    """Ask the coder for code; if the reply is unusable, give it one chance to repair."""
    reply = coder.chat(messages)
    try:
        return extract_code(reply)
    except OperatorError as first:
        retry = [
            *messages,
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": f"That reply was unusable ({first}). Reply with exactly one complete "
                "```python block and nothing else.",
            },
        ]
        return extract_code(coder.chat(retry))


def _tried(ctx: Context, node: Node) -> list[tuple[str, str]]:
    out = []
    for child in ctx.tree.children(node.id):
        if child.status == "ok" and child.cv_score is not None:
            out.append((child.id, f"cv {child.cv_score:.5f}. {child.summary}"))
        else:
            out.append((child.id, f"failed. {(child.error or child.summary)[-300:]}"))
    return out


class DraftOperator:
    name = "draft"

    def __init__(self, planner: ChatClient, coder: ChatClient) -> None:
        self.planner, self.coder = planner, coder

    def apply(self, node: Node | None, ctx: Context) -> str:
        roots = ctx.tree.roots()
        tried = [(n.id, n.summary or n.error) for n in roots]
        plan = self.planner.chat(
            prompts.plan_draft(
                description=ctx.description,
                eda=ctx.eda,
                metric=ctx.metric,
                higher_is_better=ctx.higher_is_better,
                tried=tried,
                first=not roots,
                remaining_min=ctx.remaining_min,
            )
        )
        return _write_code(
            self.coder, prompts.code_from_plan(contract_text=ctx.contract(), plan=plan)
        )


class ImproveOperator:
    name = "improve"

    def __init__(self, planner: ChatClient, coder: ChatClient) -> None:
        self.planner, self.coder = planner, coder

    def apply(self, node: Node | None, ctx: Context) -> str:
        assert node is not None and node.cv_score is not None
        code = node.code_path.read_text()
        idea = self.planner.chat(
            prompts.plan_improvement(
                code=code,
                metric=ctx.metric,
                higher_is_better=ctx.higher_is_better,
                cv_score=node.cv_score,
                cv_std=node.cv_std,
                summary=node.summary,
                tried=_tried(ctx, node),
                remaining_min=ctx.remaining_min,
            )
        )
        return _write_code(
            self.coder, prompts.code_improve(contract_text=ctx.contract(), code=code, idea=idea)
        )


class DebugOperator:
    name = "debug"

    def __init__(self, coder: ChatClient) -> None:
        self.coder = coder

    def apply(self, node: Node | None, ctx: Context) -> str:
        assert node is not None
        messages = prompts.code_debug(
            contract_text=ctx.contract(), code=node.code_path.read_text(), error=ctx.error_log
        )
        return _write_code(self.coder, messages)


def summarize(
    planner: ChatClient, *, code: str, log: str, metric: str, cv_score: float, cv_std: float | None
) -> str:
    text = planner.chat(
        prompts.summarize_run(code=code, log=log, metric=metric, cv_score=cv_score, cv_std=cv_std),
        max_tokens=400,
        temperature=0.2,
    )
    return text[:SUMMARY_CHARS]
