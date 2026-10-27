"""Thin wrapper over the docker CLI. Every container started here is locked down."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

TAIL_BYTES = 4000
HEAD_BYTES = 1000

# What the docker CLI prints when it cannot start the container at all. Docker writes these
# before the container exists, so they sit at the very top of stderr, ahead of any script output.
_INFRA_MARKERS = (
    "failed to connect to the docker api",
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "unable to find image",
    "docker: error response from daemon",
    "docker: invalid",
)


class DockerError(RuntimeError):
    """Docker itself failed (daemon down, image or mount problem); the script never ran."""


@dataclass(frozen=True)
class Mount:
    host: Path
    container: str
    read_only: bool = True


@dataclass
class ContainerResult:
    exit_code: int
    timed_out: bool
    duration_s: float
    stdout_tail: str
    stderr_tail: str


def docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, check=False
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _head(path: Path) -> str:
    with path.open("rb") as fh:
        return fh.read(HEAD_BYTES).decode(errors="replace")


def _docker_failure(stderr_path: Path) -> str | None:
    """The docker CLI's own error, if it is what ended the run; None for an ordinary failure."""
    lines = [ln.strip() for ln in _head(stderr_path).splitlines() if ln.strip()][:2]
    if any(marker in ln.lower() for ln in lines for marker in _INFRA_MARKERS):
        return " ".join(lines)[:300]
    return None


def _tail(path: Path) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - TAIL_BYTES))
        return fh.read().decode(errors="replace")


def run_container(
    image: str,
    command: list[str],
    *,
    name: str,
    mounts: list[Mount],
    timeout_s: int,
    memory: str,
    cpus: float,
    stdout_path: Path,
    stderr_path: Path,
    workdir: str | None = None,
) -> ContainerResult:
    """Run `command` in `image` with no network, a read-only root, no capabilities and no env.

    stdout and stderr go to files so a chatty script cannot exhaust memory here.
    """
    argv = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=1g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "512",
        "--memory", memory, "--cpus", str(cpus),
        "--user", "65534:65534",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
    ]  # fmt: skip
    for mount in mounts:
        mode = "ro" if mount.read_only else "rw"
        argv += ["-v", f"{mount.host.resolve()}:{mount.container}:{mode}"]
    if workdir:
        argv += ["-w", workdir]
    argv += [image, *command]

    start = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(argv, stdout=out, stderr=err)
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            exit_code = proc.wait()
    if not timed_out and exit_code != 0 and (failure := _docker_failure(stderr_path)):
        raise DockerError(failure)
    return ContainerResult(
        exit_code=exit_code,
        timed_out=timed_out,
        duration_s=time.monotonic() - start,
        stdout_tail=_tail(stdout_path),
        stderr_tail=_tail(stderr_path),
    )
