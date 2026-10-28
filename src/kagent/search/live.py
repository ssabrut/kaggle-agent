"""The model reply as it is being written, for the dashboard's live view.

search/live/<role>.json holds the call in flight, or the last one finished, and is rewritten
atomically as text arrives. search/live/<role>.prompt.json holds the messages that call was sent,
written once, so the frequent rewrites stay small. Both are best effort: a search never fails
because the live view could not be written.
"""

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_CHARS = 120_000  # a reply is capped near 8k tokens; this only guards against a runaway
ROLE = re.compile(r"^[a-z]{1,16}$")


class LiveFeed:
    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        self.directory = directory
        self._clock = clock
        self._calls: dict[str, dict[str, Any]] = {}
        self._count = 0

    def start(
        self,
        role: str,
        *,
        model: str,
        purpose: str,
        node: str | None,
        operator: str | None,
        parent: str | None,
        messages: list[dict[str, str]],
    ) -> None:
        if not ROLE.match(role):
            return
        self._count += 1
        now = self._clock()
        record = {
            "id": f"{int(now * 1000)}-{self._count}",
            "role": role,
            "model": model,
            "purpose": purpose,
            "node": node,
            "operator": operator,
            "parent": parent,
            "started_at": now,
            "first_at": None,  # when the first token arrived, to tell waiting from writing
            "updated_at": now,
            "done": False,
            "ok": None,
            "chunks": 0,
            "thinking": "",
            "answer": "",
            "truncated": False,
            "prompt_chars": sum(len(m.get("content") or "") for m in messages),
        }
        self._calls[role] = record
        self._write(f"{role}.prompt.json", {"id": record["id"], "messages": messages})
        self._write(f"{role}.json", record)

    def progress(self, role: str, *, thinking: str, answer: str, chunks: int) -> None:
        record = self._calls.get(role)
        if record is None:
            return
        now = self._clock()
        if chunks and record["first_at"] is None:
            record["first_at"] = now
        record.update(updated_at=now, chunks=chunks)
        record["truncated"] = len(thinking) > MAX_CHARS or len(answer) > MAX_CHARS
        record["thinking"] = thinking[-MAX_CHARS:]
        record["answer"] = answer[-MAX_CHARS:]
        self._write(f"{role}.json", record)

    def finish(self, role: str, *, ok: bool) -> None:
        record = self._calls.get(role)
        if record is None:
            return
        record.update(updated_at=self._clock(), done=True, ok=ok)
        self._write(f"{role}.json", record)

    def _write(self, name: str, payload: dict[str, Any]) -> None:
        path = self.directory / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)  # a reader never sees a half-written file
        except OSError:
            pass  # the live view is a convenience
