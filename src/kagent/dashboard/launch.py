"""Start and stop a search from the dashboard.

The page sends a fixed set of numbers and URLs, which are validated here and turned into an argument
list (never a shell string) plus a few KAGENT_* environment overrides for a `kagent search`
subprocess. Nothing else can be set from the page. The search gets its own process group, so Stop can
end the sandbox container with it, and only a process this dashboard started can be stopped.
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kagent.config import REPO_ROOT, Settings
from kagent.dashboard.reader import pid_alive
from kagent.deps.store import DependencyStore
from kagent.docker import docker_available
from kagent.images import stale_images
from kagent.workspace import Workspace

SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MODEL = re.compile(r"^[\w.\-:/@+]{1,200}$")


class LaunchError(RuntimeError):
    """The request is valid but cannot be carried out right now."""


def _http_url(value: str) -> str:
    value = value.strip()
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("must be an http or https URL")
    if parts.username or parts.password:
        raise ValueError("must not contain a username or password")
    if len(value) > 300 or any(ch.isspace() for ch in value):
        raise ValueError("must be a single URL under 300 characters")
    return value.rstrip("/").removesuffix("/chat/completions")


class RunRequest(BaseModel):
    """Everything the page may set. Anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    budget_hours: float = Field(1.0, ge=0.05, le=48)
    max_nodes: int | None = Field(None, ge=1, le=1000)
    num_drafts: int = Field(3, ge=1, le=10)
    max_debug: int = Field(3, ge=0, le=10)
    noise_k: float = Field(0.5, ge=0, le=5)
    planner_base_url: str
    planner_model: str
    coder_base_url: str
    coder_model: str
    step_timeout_s: int = Field(600, ge=30, le=7200)
    sandbox_cpus: float = Field(4, ge=0.5, le=64)
    sandbox_memory: str = Field("8g", pattern=r"^\d{1,3}[mg]$")
    llm_timeout_s: float = Field(900, ge=30, le=7200)
    llm_max_tokens: int = Field(8192, ge=256, le=65536)
    planner_thinking: Literal["default", "off", "low", "medium", "xhigh"] = "xhigh"
    planner_temperature: float = Field(1.0, ge=0, le=2)
    planner_top_p: float = Field(0.95, gt=0, le=1)
    planner_top_k: int = Field(20, ge=0, le=1000)
    coder_temperature: float = Field(1.0, ge=0, le=2)
    coder_top_p: float = Field(0.95, gt=0, le=1)
    coder_top_k: int = Field(40, ge=0, le=1000)

    @field_validator("planner_base_url", "coder_base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        return _http_url(value)

    @field_validator("planner_model", "coder_model")
    @classmethod
    def _model(cls, value: str) -> str:
        value = value.strip()
        if not MODEL.match(value):
            raise ValueError("must be a model name of letters, digits and . - _ : / @ +")
        return value

    @classmethod
    def defaults(cls, settings: Settings) -> "RunRequest":
        return cls(
            planner_base_url=settings.planner_base_url,
            planner_model=settings.planner_model,
            coder_base_url=settings.coder_base_url,
            coder_model=settings.coder_model,
            step_timeout_s=settings.step_timeout_s,
            sandbox_cpus=settings.sandbox_cpus,
            sandbox_memory=settings.sandbox_memory,
            llm_timeout_s=settings.llm_timeout_s,
            llm_max_tokens=settings.llm_max_tokens,
            planner_thinking=settings.planner_thinking,
            planner_temperature=settings.planner_temperature,
            planner_top_p=settings.planner_top_p,
            planner_top_k=settings.planner_top_k,
            coder_temperature=settings.coder_temperature,
            coder_top_p=settings.coder_top_p,
            coder_top_k=settings.coder_top_k,
        )


def build_command(slug: str, req: RunRequest, python: str = sys.executable) -> list[str]:
    if not SLUG.match(slug):
        raise ValueError(f"unsafe competition slug: {slug!r}")
    argv = [
        python, "-m", "kagent", "search", "-c", slug,
        "--budget-hours", str(req.budget_hours),
        "--num-drafts", str(req.num_drafts),
        "--max-debug", str(req.max_debug),
        "--noise-k", str(req.noise_k),
    ]  # fmt: skip
    if req.max_nodes is not None:
        argv += ["--max-nodes", str(req.max_nodes)]
    return argv


def build_env(req: RunRequest, settings: Settings, base: dict[str, str]) -> dict[str, str]:
    """The parent environment plus the settings the page chose."""
    return {
        **base,
        "KAGENT_WORKSPACES_DIR": str(settings.workspaces_dir),
        "KAGENT_DOCKER_DIR": str(settings.docker_dir),
        "KAGENT_PLANNER_BASE_URL": req.planner_base_url,
        "KAGENT_PLANNER_MODEL": req.planner_model,
        "KAGENT_CODER_BASE_URL": req.coder_base_url,
        "KAGENT_CODER_MODEL": req.coder_model,
        "KAGENT_STEP_TIMEOUT_S": str(req.step_timeout_s),
        "KAGENT_SANDBOX_CPUS": str(req.sandbox_cpus),
        "KAGENT_SANDBOX_MEMORY": req.sandbox_memory,
        "KAGENT_LLM_TIMEOUT_S": str(req.llm_timeout_s),
        "KAGENT_LLM_MAX_TOKENS": str(req.llm_max_tokens),
        "KAGENT_PLANNER_THINKING": req.planner_thinking,
        "KAGENT_PLANNER_TEMPERATURE": str(req.planner_temperature),
        "KAGENT_PLANNER_TOP_P": str(req.planner_top_p),
        "KAGENT_PLANNER_TOP_K": str(req.planner_top_k),
        "KAGENT_CODER_TEMPERATURE": str(req.coder_temperature),
        "KAGENT_CODER_TOP_P": str(req.coder_top_p),
        "KAGENT_CODER_TOP_K": str(req.coder_top_k),
    }


def _spawn(argv: list[str], env: dict[str, str], log: BinaryIO) -> Any:
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # its own process group, so Stop reaches the sandbox container
    )
    threading.Thread(target=proc.wait, daemon=True).start()  # reap it, so it never lingers
    return proc


def _reachable(url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{url}/models", timeout=4.0)
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"
    return response.status_code < 500, f"answered HTTP {response.status_code}"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _check(name: str, label: str, ok: bool, good: str, bad: str, *, warn: bool = False) -> dict:
    status = "ok" if ok else "warn" if warn else "fail"
    return {"id": name, "label": label, "status": status, "message": good if ok else bad}


class Launcher:
    def __init__(
        self,
        ws: Workspace,
        settings: Settings,
        *,
        spawn: Callable[[list[str], dict[str, str], BinaryIO], Any] = _spawn,
        getpgid: Callable[[int], int] = os.getpgid,
        killpg: Callable[[int, int], None] = os.killpg,
        docker_ok: Callable[[], bool] = docker_available,
        stale: Callable[[Settings], list[str]] = stale_images,
        reach: Callable[[str], tuple[bool, str]] = _reachable,
    ) -> None:
        if not SLUG.match(ws.root.name):
            raise ValueError(f"unsafe competition slug: {ws.root.name!r}")
        self.ws, self.settings = ws, settings
        self._spawn, self._getpgid, self._killpg = spawn, getpgid, killpg
        self._docker_ok, self._stale, self._reach = docker_ok, stale, reach
        self._lock = threading.Lock()
        self._proc: Any = None
        self.launch_path = ws.search_dir / "launch.json"
        self.log_path = ws.search_dir / "run.log"
        self.last_path = ws.root / "run_config.json"

    # -- what is running -------------------------------------------------------------------

    def active_pid(self) -> int | None:
        """The search this dashboard started, if it is still alive."""
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc.pid
            self._proc = None
        pid = _read_json(self.launch_path).get("pid")
        return pid if isinstance(pid, int) and pid_alive(pid) else None

    def _owned(self, pid: int) -> bool:
        """True for a process we spawned, or one whose status file says it is the live search."""
        if self._proc is not None and self._proc.pid == pid:
            return True
        status = _read_json(self.ws.search_dir / "status.json")
        return status.get("pid") == pid and status.get("state") == "running"

    def defaults(self) -> dict:
        base = RunRequest.defaults(self.settings).model_dump()
        try:
            return RunRequest.model_validate({**base, **_read_json(self.last_path)}).model_dump()
        except ValidationError:
            return base

    def state(self) -> dict:
        pid = self.active_pid()
        active = pid is not None
        return {"enabled": True, "active": active, "pid": pid, "defaults": self.defaults()}

    # -- checks before starting ------------------------------------------------------------

    def preflight(self, req: RunRequest) -> dict:
        with ThreadPoolExecutor(max_workers=3) as pool:
            docker = pool.submit(self._docker_ok)
            planner = pool.submit(self._reach, req.planner_base_url)
            coder = pool.submit(self._reach, req.coder_base_url)
            docker_up = docker.result()
            planner_ok, planner_msg = planner.result()
            coder_ok, coder_msg = coder.result()
        stale = self._stale(self.settings) if docker_up else []
        resumed = self._node_count()
        pending = len(DependencyStore(self.ws.root / "deps").all("pending"))

        checks = [
            _check("config", "Competition set up", self.ws.config_path.exists(),
                   "config.json found", "run `kagent init` for this competition first"),
            _check("folds", "Cross-validation folds", (self.ws.public / "folds.csv").exists(),
                   "folds.csv found", "folds.csv missing; run `kagent init` with the data"),
            _check("idle", "No search running", self.active_pid() is None,
                   "nothing is running", "a search is already running"),
            _check("docker", "Docker daemon", docker_up,
                   "reachable", "not reachable; start Docker Desktop"),
            _check("images", "Sandbox and grader images", docker_up and not stale,
                   "built from the current lock files",
                   "need building: run `kagent build-images`" if docker_up else "needs Docker"),
            _check("planner", "Planner model server", planner_ok, planner_msg, planner_msg),
            _check("coder", "Coder model server", coder_ok, coder_msg, coder_msg),
            _check("packages", "Package requests", pending == 0, "none waiting",
                   f"{pending} waiting for your decision (nodes needing them stay parked)",
                   warn=True),
            _check("resume", "Existing tree", resumed == 0, "a fresh start",
                   f"resumes from {resumed} existing nodes", warn=True),
        ]  # fmt: skip
        return {"checks": checks, "ready": all(c["status"] != "fail" for c in checks)}

    def _node_count(self) -> int:
        path = self.ws.search_dir / "nodes.jsonl"
        try:
            with path.open() as fh:
                return len({json.loads(line)["id"] for line in fh if line.strip()})
        except (OSError, ValueError, KeyError):
            return 0

    # -- start and stop --------------------------------------------------------------------

    def start(self, req: RunRequest) -> dict:
        with self._lock:
            if self.active_pid() is not None:
                raise LaunchError("a search is already running")
            if not self.ws.config_path.exists():
                raise LaunchError("run `kagent init` for this competition first")
            if not (self.ws.public / "folds.csv").exists():
                raise LaunchError("folds.csv is missing; run `kagent init` with the data")

            argv = build_command(self.ws.root.name, req)
            env = build_env(req, self.settings, dict(os.environ))
            self.ws.search_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("ab") as log:
                stamp = datetime.now(UTC).isoformat(timespec="seconds")
                log.write(f"\n=== {stamp} starting: {shlex.join(argv)}\n".encode())
                log.flush()
                try:
                    self._proc = self._spawn(argv, env, log)
                except OSError as exc:
                    raise LaunchError(f"could not start the search: {exc}") from exc
            record = {
                "pid": self._proc.pid,
                "started_at": time.time(),
                "argv": argv,
                "request": req.model_dump(),
            }
            self.launch_path.write_text(json.dumps(record))
            self.last_path.write_text(json.dumps(req.model_dump()))
            return {"pid": self._proc.pid}

    def stop(self, *, force: bool = False) -> None:
        with self._lock:
            pid = self.active_pid()
            if pid is None:
                raise LaunchError("no search is running")
            if not self._owned(pid):
                raise LaunchError("that process is not a search this dashboard started")
            try:
                group = self._getpgid(pid)
            except ProcessLookupError as exc:
                raise LaunchError("the search has already ended") from exc
            if group != pid:  # never signal a group we did not create
                raise LaunchError("refusing to signal a group this dashboard did not create")
            self._killpg(group, signal.SIGKILL if force else signal.SIGTERM)
