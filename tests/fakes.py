"""Scripted stand-ins for the LLM clients."""


class FakeChat:
    """Replays canned replies and records every message list it receives."""

    def __init__(self, replies, role="fake"):
        self.replies = list(replies)
        self.role = role
        self.calls: list[list[dict]] = []
        self.options: list[dict] = []  # the keyword arguments of each call

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.options.append(kwargs)
        if not self.replies:
            raise AssertionError(f"{self.role} was called more often than scripted")
        return self.replies.pop(0)

    def user_text(self, index: int) -> str:
        return self.calls[index][-1]["content"]


def code_block(body: str) -> str:
    return f"Here you go:\n```python\n{body}\n```\n"
