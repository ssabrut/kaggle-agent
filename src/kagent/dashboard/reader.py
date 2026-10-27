"""What the dashboard shows, read from a workspace's files.

Strictly read-only. Nothing here writes, and nothing reads data/private or the grader source.
Paths are built from validated ids, never from strings stored in the files.
"""

import json
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kagent.config import Settings
from kagent.deps.lockfile import normalize, sandbox_versions
from kagent.deps.store import DependencyStore
from kagent.grader import metrics
from kagent.integrity.hashing import is_regular_file
from kagent.search.blocklist import build_blocklist
from kagent.search.tree import Node, SolutionTree
from kagent.stages import STAGES
from kagent.workspace import Workspace

NODE_ID = re.compile(r"^n\d{3,}$")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SNIPPET_CHARS = 200
TAIL_BYTES = 6_000
CODE_CHARS = 60_000
MESSAGE_CHARS = 20_000
RECENT_LATENCIES = 80
MAX_EVENTS = 300
MAX_CALLS = 100


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def json_safe(value: Any) -> Any:
    """Replace NaN and infinity, which JSON cannot carry, with null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clip(lines[-1], SNIPPET_CHARS) if lines else ""


def _plan_header(code: str) -> list[str]:
    """The leading comment lines of a script, which hold its `# PLAN:`."""
    header: list[str] = []
    for line in code.splitlines():
        if not line.startswith("#"):
            break
        header.append(line.lstrip("#").strip())
    return [h for h in header if h]


def _tail(path: Path, limit: int = TAIL_BYTES) -> str:
    if not is_regular_file(path):
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(fh.tell() - limit, 0))
        return fh.read().decode(errors="replace")


class JsonlTail:
    """Follows a JSONL file that another process appends to; each complete line is parsed once."""

    def __init__(self, path: Path, parse: Callable[[int, dict], dict]) -> None:
        self.path = path
        self._parse = parse
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self.starts: list[int] = []  # byte offset where each kept row begins
        self.rows: list[dict] = []

    def refresh(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            self._reset()
            return
        if size < self.offset:  # truncated or replaced
            self._reset()
        if size == self.offset:
            return
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n")
        if end < 0:  # a partial line, still being written
            return
        position = self.offset
        for line in chunk[: end + 1].split(b"\n")[:-1]:
            start, position = position, position + len(line) + 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                self.starts.append(start)
                self.rows.append(self._parse(len(self.rows), row))
        self.offset += end + 1

    def read_raw(self, index: int) -> dict | None:
        if not 0 <= index < len(self.starts):
            return None
        with self.path.open("rb") as fh:
            fh.seek(self.starts[index])
            try:
                return json.loads(fh.readline())
            except ValueError:
                return None


def _compact_call(index: int, row: dict) -> dict:
    """One model call without its prompt and reply, which can be large."""
    usage = row.get("usage") or {}
    messages = row.get("messages") or []
    return {
        "i": index,
        "ts": row.get("ts"),
        "role": row.get("role"),
        "model": row.get("model"),
        "seconds": row.get("seconds"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": row.get("finish_reason"),
        "chars_in": sum(len(m.get("content") or "") for m in messages if isinstance(m, dict)),
        "chars_out": len(row.get("response") or ""),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


class WorkspaceReader:
    def __init__(self, ws: Workspace, settings: Settings | None = None) -> None:
        self.ws = ws
        self.settings = settings or Settings()
        self._lock = threading.Lock()
        self._events = JsonlTail(ws.search_dir / "events.jsonl", lambda _i, row: row)
        self._calls = JsonlTail(ws.search_dir / "trajectory.jsonl", _compact_call)
        self._runs: dict[str, dict] = {}

    # -- change detection ------------------------------------------------------------------

    def _watched(self) -> list[Path]:
        return [
            self.ws.search_dir / "nodes.jsonl",
            self.ws.search_dir / "status.json",
            self.ws.search_dir / "events.jsonl",
            self.ws.search_dir / "trajectory.jsonl",
            self.ws.root / "deps" / "requests.jsonl",
            self.ws.selected_path,
        ]

    def fingerprint(self) -> str:
        """Changes whenever something the page shows could have changed."""
        parts = []
        for path in self._watched():
            try:
                stat = path.stat()
                parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append("-")
        status = self._status()
        parts.append(str(pid_alive(status.get("pid")) if status else None))
        return "|".join(parts)

    # -- small readers ---------------------------------------------------------------------

    def _status(self) -> dict | None:
        try:
            data = json.loads((self.ws.search_dir / "status.json").read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _json_file(self, path: Path) -> dict:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _run(self, run_id: str | None) -> dict | None:
        if not run_id or not RUN_ID.match(run_id):
            return None
        if run_id in self._runs:
            return self._runs[run_id]
        path = self.ws.run_dir(run_id) / "run.json"
        if not is_regular_file(path):
            return None
        record = self._json_file(path)
        if record:
            self._runs[run_id] = record  # a record is written once and never changes
        return record or None

    def _tree(self) -> SolutionTree:
        higher = metrics.METRICS[self.ws.load_config().metric].higher_is_better
        return SolutionTree(self.ws.search_dir / "nodes.jsonl", higher_is_better=higher)

    def _noise_k(self, status: dict | None) -> float:
        params = (status or {}).get("params") or {}
        try:
            return float(params.get("noise_k", self.settings.promotion_noise_k))
        except (TypeError, ValueError):
            return self.settings.promotion_noise_k

    def _installed(self) -> set[str]:
        try:
            return {normalize(name) for name in sandbox_versions(self.settings.docker_dir)}
        except (OSError, ValueError):  # the lock files are optional for a read-only view
            return set()

    # -- the state the page polls ----------------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return self._state()

    def _state(self) -> dict:
        config = self.ws.load_config()
        higher = metrics.METRICS[config.metric].higher_is_better
        status = self._status()
        noise_k = self._noise_k(status)
        tree = self._tree()
        trail = tree.incumbent_trail(noise_k)
        incumbent = trail[-1][1] if trail else None
        best = tree.best(robust=True)
        incumbent_after = {node.id: inc.id for node, inc in trail}

        items = [self._item(tree, n, noise_k, incumbent_after.get(n.id)) for n in tree.nodes]
        by_id = {item["id"]: item for item in items}

        durations = [i["duration_s"] for i in items if i["duration_s"] is not None]
        counts = Counter(n.status for n in tree.nodes)
        operators: dict[str, dict[str, int]] = {}
        for item in items:
            stats = operators.setdefault(
                item["operator"], {"n": 0, "ok": 0, "better": 0, "noise": 0, "worse": 0}
            )
            stats["n"] += 1
            stats["ok"] += item["status"] == "ok"
            if item["verdict"]:
                stats[item["verdict"]] += 1

        stages = Counter(n.stage for n in tree.nodes if n.stage)
        deps = self._deps()

        self._events.refresh()
        self._calls.refresh()
        state = self._run_state(status)

        return {
            "server_time": time.time(),
            "competition": {
                "slug": config.slug,
                "metric": config.metric,
                "higher_is_better": higher,
                "id_column": config.id_column,
                "page": self._json_file(self.ws.page_dir / "info.json"),
            },
            "run": state,
            "noise_k": noise_k,
            "selected": self._json_file(self.ws.selected_path).get("node"),
            "summary": {
                "total": len(items),
                "ok": counts["ok"],
                "buggy": counts["buggy"],
                "dead": counts["dead"],
                "blocked": counts["blocked"],
                "best": by_id.get(best.id) if best else None,
                "incumbent": by_id.get(incumbent.id) if incumbent else None,
                "nodes_per_hour": self._rate(tree, state),
                "mean_run_s": sum(durations) / len(durations) if durations else None,
                "pending_deps": sum(d["status"] == "pending" for d in deps),
            },
            "nodes": items,
            "operators": operators,
            "stages": {s: stages.get(s, 0) for s in STAGES},
            "blocklist": build_blocklist(tree, self._installed()),
            "deps": deps,
            "llm": self._llm_stats(),
            "event_count": len(self._events.rows),
            "call_count": len(self._calls.rows),
        }

    def _run_state(self, status: dict | None) -> dict:
        if status is None:
            return {"state": "never"}
        alive = pid_alive(status.get("pid"))
        state = status.get("state", "stopped")
        if state == "running" and not alive:
            state = "lost"  # the process is gone without having said so
        return {**status, "state": state, "alive": alive}

    def _rate(self, tree: SolutionTree, run: dict) -> float | None:
        """Nodes per hour since the first node, up to now while a search is running."""
        stamps = [_parse_time(n.created_at) for n in tree.nodes]
        stamps = [s for s in stamps if s is not None]
        if len(stamps) < 2:
            return None
        end = time.time() if run.get("state") == "running" else max(stamps)
        hours = (end - min(stamps)) / 3600
        return len(stamps) / hours if hours > 60 / 3600 else None

    def _item(
        self, tree: SolutionTree, node: Node, noise_k: float, incumbent_id: str | None
    ) -> dict:
        parent = None
        if node.parent:
            try:
                parent = tree.get(node.parent)
            except KeyError:
                parent = None
        gain = floor = verdict = None
        if node.cv_score is not None and parent is not None and parent.cv_score is not None:
            gain = tree.rank_value(node) - tree.rank_value(parent)
            floor = noise_k * max(node.cv_std or 0.0, parent.cv_std or 0.0)
            verdict = "better" if gain > floor else "worse" if gain < -floor else "noise"
        run = self._run(node.run_id)
        return {
            "id": node.id,
            "parent": node.parent,
            "operator": node.operator,
            "status": node.status,
            "cv": node.cv_score,
            "std": node.cv_std,
            "stage": node.stage,
            "debug_depth": node.debug_depth,
            "created_at": node.created_at,
            "run_id": node.run_id,
            "dep_request": node.dep_request,
            "gain": gain,
            "floor": floor,
            "verdict": verdict,
            "incumbent_after": incumbent_id,
            "duration_s": run.get("duration_s") if run else None,
            "summary": _clip(node.summary, SNIPPET_CHARS),
            "error": _last_line(node.error),
        }

    def _deps(self) -> list[dict]:
        store = DependencyStore(self.ws.root / "deps")
        return [
            {
                "id": r.id,
                "package": r.package,
                "constraint": r.constraint,
                "status": r.status,
                "node_id": r.node_id,
                "reason": _clip(r.reason, SNIPPET_CHARS),
                "flags": r.precheck.flags(),
                "resolved_version": r.resolved_version,
                "created_at": r.created_at,
                "decided_at": r.decided_at,
                "decision_reason": _clip(r.decision_reason, SNIPPET_CHARS),
            }
            for r in store.all()
        ]

    def _llm_stats(self) -> dict:
        by_role: dict[str, dict[str, Any]] = {}
        for call in self._calls.rows:
            role = call.get("role") or "?"
            stats = by_role.setdefault(
                role, {"calls": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0, "latency": []}
            )
            stats["calls"] += 1
            seconds = call.get("seconds") or 0.0
            stats["seconds"] += seconds
            stats["tokens_in"] += call.get("prompt_tokens") or 0
            stats["tokens_out"] += call.get("completion_tokens") or 0
            stats["latency"].append(seconds)
        for stats in by_role.values():
            seconds, latency = stats["seconds"], stats["latency"]
            stats["mean_s"] = seconds / stats["calls"]
            stats["p95_s"] = _percentile(latency, 0.95)
            stats["tokens_per_s"] = stats["tokens_out"] / seconds if seconds else None
            stats["latency"] = latency[-RECENT_LATENCIES:]
        return by_role

    # -- the pieces the page fetches on demand ---------------------------------------------

    def events(self, after: int = 0) -> dict:
        with self._lock:
            self._events.refresh()
            rows = [r for r in self._events.rows if (r.get("seq") or 0) > after]
            return {"events": rows[-MAX_EVENTS:], "last": rows[-1]["seq"] if rows else after}

    def calls(self, after: int = -1) -> dict:
        with self._lock:
            self._calls.refresh()
            rows = [r for r in self._calls.rows if r["i"] > after]
            return {"calls": rows[-MAX_CALLS:], "last": rows[-1]["i"] if rows else after}

    def call(self, index: int) -> dict | None:
        with self._lock:
            self._calls.refresh()
            raw = self._calls.read_raw(index)
        if raw is None:
            return None
        messages = [
            {"role": m.get("role"), "content": _truncate(m.get("content") or "")}
            for m in raw.get("messages") or []
            if isinstance(m, dict)
        ]
        return {
            **_compact_call(index, raw),
            "messages": messages,
            "response": _truncate(raw.get("response") or ""),
        }

    def node(self, node_id: str) -> dict | None:
        if not NODE_ID.match(node_id):
            return None
        with self._lock:
            tree = self._tree()
            try:
                node = tree.get(node_id)
            except KeyError:
                return None
            status = self._status()
            noise_k = self._noise_k(status)
            trail = {n.id: inc.id for n, inc in tree.incumbent_trail(noise_k)}
            item = self._item(tree, node, noise_k, trail.get(node.id))
            children = [c.id for c in tree.children(node.id)]

            # The script always lives at this path, so no path stored in the file is trusted.
            code_path = self.ws.search_dir / "nodes" / f"{node.id}.py"
            code = ""
            if is_regular_file(code_path):
                code = code_path.read_text(errors="replace")[:CODE_CHARS]
            run = self._run(node.run_id)
            run_dir = None
            if node.run_id and RUN_ID.match(node.run_id):
                run_dir = self.ws.run_dir(node.run_id)
            return {
                **item,
                "full_summary": node.summary,
                "full_error": node.error[-TAIL_BYTES:],
                "children": children,
                "plan": _plan_header(code),
                "code": code,
                "run": _run_view(run),
                "stdout": _tail(run_dir / "stdout.log") if run_dir else "",
                "stderr": _tail(run_dir / "stderr.log") if run_dir else "",
            }


def _truncate(text: str) -> str:
    if len(text) <= MESSAGE_CHARS:
        return text
    half = MESSAGE_CHARS // 2
    return f"{text[:half]}\n... [{len(text) - MESSAGE_CHARS} characters cut] ...\n{text[-half:]}"


def _run_view(run: dict | None) -> dict | None:
    """The parts of a run record worth showing. The held-out score is for display only."""
    if not run:
        return None
    grade = run.get("grade") or {}
    return {
        "status": run.get("status"),
        "exit_code": run.get("exit_code"),
        "duration_s": run.get("duration_s"),
        "stage_reached": run.get("stage_reached"),
        "integrity_changes": run.get("integrity_changes") or [],
        "valid": grade.get("valid"),
        "grade_errors": grade.get("errors") or [],
        "cv_errors": grade.get("cv_errors") or [],
        "heldout_score": grade.get("score"),
    }


def _parse_time(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()
