"""Run one agent-written script in the sandbox container.

The script sees the public data read-only at /data and writes to /work/out. It has no
network, no env vars, no grader code and no private labels.
"""

from pathlib import Path

from kagent.config import Settings
from kagent.docker import ContainerResult, Mount, run_container
from kagent.workspace import Workspace


def execute(ws: Workspace, run_dir: Path, settings: Settings) -> ContainerResult:
    out_dir = run_dir / "out"
    out_dir.mkdir(exist_ok=True)
    out_dir.chmod(0o777)  # the container runs as an unprivileged uid
    return run_container(
        settings.sandbox_image,
        ["python", "/work/script.py"],
        name=f"kagent-{run_dir.name}",
        mounts=[
            Mount(ws.public, "/data"),
            Mount(run_dir / "script.py", "/work/script.py"),
            Mount(out_dir, "/work/out", read_only=False),
        ],
        timeout_s=settings.step_timeout_s,
        memory=settings.sandbox_memory,
        cpus=settings.sandbox_cpus,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        workdir="/work",
    )
