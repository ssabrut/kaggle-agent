"""Submit a graded run to Kaggle. Real submissions need explicit approval."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kagent.harness import RunRecord, load_record
from kagent.integrity.hashing import hash_file, is_regular_file
from kagent.workspace import Workspace

_RUN_ID = re.compile(r"^[\w.-]+$")


class SubmitRefused(Exception):
    pass


@dataclass
class SubmitResult:
    dry_run: bool
    command: list[str]
    output: str = ""


def _vet(ws: Workspace, run_id: str) -> tuple[RunRecord, Path]:
    if not _RUN_ID.match(run_id) or run_id.startswith("."):
        raise SubmitRefused(f"bad run id '{run_id}'")
    record = load_record(ws, run_id)
    if record.status != "ok":
        raise SubmitRefused(
            f"run {run_id} has status '{record.status}', only 'ok' can be submitted"
        )
    path = ws.run_dir(run_id) / "out" / "submission.csv"
    if not is_regular_file(path):
        raise SubmitRefused("submission.csv is missing or not a plain file")
    if hash_file(path) != record.submission_sha256:
        raise SubmitRefused("submission.csv changed after it was graded")
    return record, path


def submit(ws: Workspace, run_id: str, message: str, approve: bool = False) -> SubmitResult:
    """Dry run unless approve is set. Both paths re-check the graded file first."""
    record, path = _vet(ws, run_id)
    command = [
        "kaggle", "competitions", "submit",
        "-c", record.slug, "-f", str(path), "-m", message,
    ]  # fmt: skip
    if not approve:
        return SubmitResult(dry_run=True, command=command)

    if shutil.which("kaggle") is None:
        raise SubmitRefused("kaggle CLI not found on PATH (run via `uv run`)")
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SubmitRefused(f"kaggle submit failed: {(proc.stderr or proc.stdout).strip()}")
    return SubmitResult(dry_run=False, command=command, output=proc.stdout.strip())
