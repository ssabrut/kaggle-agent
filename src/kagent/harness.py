"""Run one script end to end: sandbox, integrity check, grade, record."""

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import kagent.grader
from kagent import grading
from kagent.config import Settings
from kagent.grader.result import GradeResult
from kagent.integrity.hashing import find_changes, hash_file, is_regular_file, snapshot
from kagent.sandbox import runner
from kagent.stages import stage_reached
from kagent.workspace import Workspace

GRADER_SRC = Path(kagent.grader.__file__).parent

Status = Literal["ok", "failed", "timeout", "invalid_submission", "integrity_violation"]


class RunRecord(BaseModel):
    run_id: str
    slug: str
    status: Status
    exit_code: int
    duration_s: float
    script_sha256: str
    submission_sha256: str | None = None
    grade: GradeResult | None = None
    integrity_changes: list[str] = []
    stderr_tail: str = ""
    stage_reached: str = "start"  # see kagent.stages


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def load_record(ws: Workspace, run_id: str) -> RunRecord:
    path = ws.run_dir(run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"no run record at {path}")
    return RunRecord.model_validate_json(path.read_text())


def run_script(ws: Workspace, script: Path, settings: Settings) -> RunRecord:
    config = ws.load_config()
    run_id = new_run_id()
    run_dir = ws.run_dir(run_id)
    run_dir.mkdir(parents=True)
    shutil.copyfile(script, run_dir / "script.py")

    watched = {"public": ws.public, "private": ws.private, "grader": GRADER_SRC}
    before = snapshot(watched)

    sandbox = runner.execute(ws, run_dir, settings)

    submission = run_dir / "out" / "submission.csv"
    has_submission = is_regular_file(submission)
    grade: GradeResult | None = None
    if not sandbox.timed_out and sandbox.exit_code == 0:
        if has_submission:
            grade = grading.grade_submission(ws, config, run_dir, settings)
        else:
            grade = GradeResult(valid=False, errors=["submission.csv missing or not a plain file"])

    changes = find_changes(before, snapshot(watched))

    status: Status
    if changes:
        status = "integrity_violation"
    elif sandbox.timed_out:
        status = "timeout"
    elif sandbox.exit_code != 0:
        status = "failed"
    elif grade is None or not grade.valid:
        status = "invalid_submission"
    else:
        status = "ok"

    record = RunRecord(
        run_id=run_id,
        slug=config.slug,
        status=status,
        exit_code=sandbox.exit_code,
        duration_s=round(sandbox.duration_s, 2),
        script_sha256=hash_file(run_dir / "script.py"),
        submission_sha256=hash_file(submission) if has_submission else None,
        grade=grade,
        integrity_changes=changes,
        stderr_tail=sandbox.stderr_tail,
        stage_reached=stage_reached(
            run_dir, graded=grade is not None and grade.valid and grade.cv_score is not None
        ),
    )
    (run_dir / "run.json").write_text(record.model_dump_json(indent=2) + "\n")
    return record
