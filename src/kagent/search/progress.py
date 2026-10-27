"""Live status of a running search, for the dashboard or anyone tailing the files.

status.json is rewritten at every phase change; events.jsonl is append-only. Both sit in the
workspace's search/ directory, so another process can follow a search without talking to it.
"""

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kagent.workspace import Workspace

KEEP: Any = object()  # passed for node/operator to leave the field as it is


class Progress:
    def __init__(
        self,
        ws: Workspace,
        *,
        budget_s: float,
        params: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        ws.search_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = ws.search_dir / "status.json"
        self.events_path = ws.search_dir / "events.jsonl"
        self._clock = clock
        now = clock()
        self._status: dict[str, Any] = {
            "state": "running",  # running | finished | stopped
            "pid": os.getpid(),
            "started_at": now,
            "deadline_at": now + budget_s,
            "budget_s": budget_s,
            "params": params or {},
            "phase": "starting",
            "node": None,
            "operator": None,
            "detail": "",
            "phase_since": now,
            "llm": None,  # {"role", "since"} while a model call is in flight
            "message": "",
            "updated_at": now,
        }
        self._seq = self._count_events()
        self._write()
        self.event("search started")

    def _count_events(self) -> int:
        if not self.events_path.exists():
            return 0
        with self.events_path.open() as fh:
            return sum(1 for line in fh if line.strip())

    def _write(self) -> None:
        self._status["updated_at"] = self._clock()
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._status))
        os.replace(tmp, self.status_path)  # a reader never sees a half-written file

    def event(self, text: str, *, node: str | None = None) -> None:
        self._seq += 1
        entry = {"seq": self._seq, "ts": datetime.now(UTC).isoformat(), "node": node, "text": text}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def phase(self, name: str, *, node: Any = KEEP, op: Any = KEEP, detail: str = "") -> None:
        self._status["phase"] = name
        self._status["phase_since"] = self._clock()
        self._status["detail"] = detail
        if node is not KEEP:
            self._status["node"] = node
        if op is not KEEP:
            self._status["operator"] = op
        self._write()

    # ChatClient observer: which model the search is waiting on right now.
    def call_started(self, role: str) -> None:
        self._status["llm"] = {"role": role, "since": self._clock()}
        self._write()

    def call_finished(self, role: str) -> None:
        self._status["llm"] = None
        self._write()

    def finish(self, state: str, message: str = "") -> None:
        self._status.update(state=state, phase="done", node=None, operator=None, llm=None)
        self._status["message"] = message
        self._write()
        self.event(f"search {state}" + (f": {message}" if message else ""))
