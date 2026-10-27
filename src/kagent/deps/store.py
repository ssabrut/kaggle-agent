"""Persistent dependency requests and an append-only event log.

requests.jsonl keeps one line per request version (the last line for an id wins), so the
history of each decision stays on disk. events.jsonl records everything that touched the
environment: requests, pre-checks, decisions, resolved versions and failed installs.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from kagent.deps.lockfile import normalize
from kagent.deps.models import DependencyRequest, Status


class DependencyStore:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.requests_path = directory / "requests.jsonl"
        self.events_path = directory / "events.jsonl"
        self._requests: dict[str, DependencyRequest] = {}
        self.reload()

    def reload(self) -> None:
        """Re-read the file, to pick up decisions a human made from another terminal."""
        self._requests = {}
        if self.requests_path.exists():
            for line in self.requests_path.read_text().splitlines():
                if line.strip():
                    request = DependencyRequest.model_validate_json(line)
                    self._requests[request.id] = request

    def all(self, status: Status | None = None) -> list[DependencyRequest]:
        return [r for r in self._requests.values() if status is None or r.status == status]

    def get(self, request_id: str) -> DependencyRequest:
        try:
            return self._requests[request_id]
        except KeyError:
            raise KeyError(f"no dependency request '{request_id}'") from None

    def find(self, package: str, status: Status | None = None) -> list[DependencyRequest]:
        wanted = normalize(package)
        return [r for r in self.all(status) if normalize(r.package) == wanted]

    def next_id(self) -> str:
        return f"dep-{len(self._requests) + 1:03d}"

    def save(self, request: DependencyRequest) -> DependencyRequest:
        self._requests[request.id] = request
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.requests_path.open("a") as fh:
            fh.write(request.model_dump_json() + "\n")
        return request

    def log(self, event: str, **fields) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
