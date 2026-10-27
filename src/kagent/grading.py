"""Host-side launcher for the grader container."""

import json
from pathlib import Path

from kagent.config import CompetitionConfig, Settings
from kagent.docker import Mount, run_container
from kagent.grader.result import GradeResult
from kagent.integrity.hashing import is_regular_file
from kagent.workspace import Workspace


def grade_submission(
    ws: Workspace, config: CompetitionConfig, run_dir: Path, settings: Settings
) -> GradeResult:
    """Validate and score run_dir/out/submission.csv (and oof.csv, if any) in a container.

    Only the validity flag, errors and scores come back; nothing else crosses the boundary.
    """
    out = run_dir / "out"
    mounts = [
        Mount(ws.public, "/data/public"),
        Mount(ws.private, "/data/private"),
        Mount(out / "submission.csv", "/submission/submission.csv"),
    ]
    args = [
        "--public", "/data/public",
        "--private", "/data/private",
        "--submission", "/submission/submission.csv",
        "--metric", config.metric,
        "--id-column", config.id_column,
    ]  # fmt: skip
    if is_regular_file(out / "oof.csv"):  # never mount a symlink: it resolves on the host
        mounts.append(Mount(out / "oof.csv", "/submission/oof.csv"))
        args += ["--oof", "/submission/oof.csv"]

    result = run_container(
        settings.grader_image,
        args,
        name=f"kagent-grade-{run_dir.name}",
        mounts=mounts,
        timeout_s=settings.grader_timeout_s,
        memory=settings.grader_memory,
        cpus=1,
        stdout_path=run_dir / "grader.stdout.log",
        stderr_path=run_dir / "grader.stderr.log",
    )

    lines = [ln for ln in result.stdout_tail.splitlines() if ln.strip()]
    try:
        return GradeResult(**json.loads(lines[-1]))
    except (IndexError, ValueError, TypeError):
        detail = "timed out" if result.timed_out else result.stderr_tail[-300:]
        return GradeResult(valid=False, errors=[f"grader gave no result: {detail}"])
