"""How far a run got before it stopped (spec 13.2 "sparse signal", 13.3 `stage_reached`).

A script that loads the data and builds features before crashing is closer to working than one
that fails on line one, though both have no score. The stage tells debugging where to look and
tells the policy which failure is worth fixing first.

Evidence, strongest first: the grade, the files the script left in out/, then `STAGE: <name>`
lines the script prints. The script writes those lines itself, so they only ever add detail
and are never used for scoring or selection.
"""

import re
from pathlib import Path

from kagent.integrity.hashing import is_regular_file

STAGES = ("start", "load", "features", "cv", "predict", "validate")
MARKED = STAGES[1:5]  # the stages a script reports itself
MAX_LOG_BYTES = 5_000_000

_MARKER = re.compile(rf"^[ \t]*STAGE:[ \t]*({'|'.join(MARKED)})\b", re.IGNORECASE | re.MULTILINE)


def stage_rank(stage: str) -> int:
    """Position in the pipeline; an unknown or empty stage counts as the start."""
    return STAGES.index(stage) if stage in STAGES else 0


def _stdout(run_dir: Path) -> str:
    log = run_dir / "stdout.log"
    if not is_regular_file(log):
        return ""
    with log.open("rb") as fh:
        return fh.read(MAX_LOG_BYTES).decode(errors="replace")


def stage_reached(run_dir: Path, *, graded: bool) -> str:
    """The furthest stage the evidence supports.

    graded means the grader accepted the submission and computed a cross-validation score.
    """
    reached = ["start", *(m.lower() for m in _MARKER.findall(_stdout(run_dir)))]
    if is_regular_file(run_dir / "out" / "oof.csv"):
        reached.append("cv")
    if is_regular_file(run_dir / "out" / "submission.csv"):
        reached.append("predict")
    if graded:
        reached.append("validate")
    return max(reached, key=stage_rank)
