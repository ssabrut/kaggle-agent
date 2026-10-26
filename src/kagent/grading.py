"""Host-side launcher for the grader container."""

import json
from pathlib import Path

from kagent.config import CompetitionConfig, Settings
from kagent.docker import Mount, run_container
from kagent.grader.result import GradeResult
from kagent.workspace import Workspace


def grade_submission(
    ws: Workspace, config: CompetitionConfig, run_dir: Path, settings: Settings
) -> GradeResult:
    """Validate and score run_dir/out/submission.csv in an isolated container.

    Only the validity flag, errors and score come back; nothing else crosses the boundary.
    """
    result = run_container(
        settings.grader_image,
        [
            "--public", "/data/public",
            "--private", "/data/private",
            "--submission", "/submission/submission.csv",
            "--metric", config.metric,
            "--id-column", config.id_column,
        ],
        name=f"kagent-grade-{run_dir.name}",
        mounts=[
            Mount(ws.public, "/data/public"),
            Mount(ws.private, "/data/private"),
            Mount(run_dir / "out" / "submission.csv", "/submission/submission.csv"),
        ],
        timeout_s=settings.grader_timeout_s,
        memory=settings.grader_memory,
        cpus=1,
        stdout_path=run_dir / "grader.stdout.log",
        stderr_path=run_dir / "grader.stderr.log",
    )  # fmt: skip

    lines = [ln for ln in result.stdout_tail.splitlines() if ln.strip()]
    try:
        return GradeResult(**json.loads(lines[-1]))
    except (IndexError, ValueError, TypeError):
        detail = "timed out" if result.timed_out else result.stderr_tail[-300:]
        return GradeResult(valid=False, errors=[f"grader gave no result: {detail}"])
