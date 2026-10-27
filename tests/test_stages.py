import os

import pytest

from conftest import write_oof, write_perfect_submission
from kagent.harness import load_record, run_script
from kagent.stages import MAX_LOG_BYTES, STAGES, stage_rank, stage_reached


def _run_dir(tmp_path, stdout=None):
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    if stdout is not None:
        (run_dir / "stdout.log").write_text(stdout)
    return run_dir


def test_a_run_with_no_evidence_is_at_the_start(tmp_path):
    assert stage_reached(_run_dir(tmp_path), graded=False) == "start"


def test_the_furthest_marker_wins_whatever_the_order(tmp_path):
    early = _run_dir(tmp_path / "a", "STAGE: load\nfitting...\nSTAGE: features\n")
    shuffled = _run_dir(tmp_path / "b", "STAGE: cv\nSTAGE: load\n")
    assert stage_reached(early, graded=False) == "features"
    assert stage_reached(shuffled, graded=False) == "cv"


def test_markers_ignore_case_and_indentation(tmp_path):
    assert stage_reached(_run_dir(tmp_path, "  stage: LOAD\n"), graded=False) == "load"


@pytest.mark.parametrize(
    "line", ["info: STAGE: cv", "STAGE: banana", "STAGE: loaded", "# STAGE: cv", "STAGE cv"]
)
def test_lines_that_are_not_markers_are_ignored(tmp_path, line):
    assert stage_reached(_run_dir(tmp_path, line + "\n"), graded=False) == "start"


def test_output_files_show_progress_without_any_marker(tmp_path):
    run_dir = _run_dir(tmp_path)
    (run_dir / "out" / "oof.csv").write_text("id,target\n")
    assert stage_reached(run_dir, graded=False) == "cv"
    (run_dir / "out" / "submission.csv").write_text("id,target\n")
    assert stage_reached(run_dir, graded=False) == "predict"


def test_files_and_the_grade_override_a_lower_marker(tmp_path):
    run_dir = _run_dir(tmp_path, "STAGE: load\n")
    (run_dir / "out" / "submission.csv").write_text("id,target\n")
    assert stage_reached(run_dir, graded=False) == "predict"
    assert stage_reached(run_dir, graded=True) == "validate"


def test_a_symlinked_output_file_is_not_evidence(tmp_path):
    run_dir = _run_dir(tmp_path)
    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_text("x")
    os.symlink(elsewhere, run_dir / "out" / "oof.csv")
    assert stage_reached(run_dir, graded=False) == "start"


def test_only_the_start_of_a_huge_log_is_read(tmp_path):
    run_dir = _run_dir(tmp_path, "x" * MAX_LOG_BYTES + "\nSTAGE: cv\n")
    assert stage_reached(run_dir, graded=False) == "start"


def test_stage_rank_follows_the_pipeline_and_treats_unknowns_as_the_start():
    ranks = [stage_rank(stage) for stage in STAGES]
    assert ranks == sorted(set(ranks)) and ranks[0] == 0
    assert stage_rank("") == stage_rank("nonsense") == 0


# --- the harness keeps the stage on the run record ---------------------------------------------


def _script(ws):
    path = ws.root / "solution.py"
    path.write_text("print('hi')\n")
    return path


def test_failed_run_record_keeps_how_far_it_got(ws, settings, fake_sandbox):
    def behaviour(ws_, run_dir):
        (run_dir / "stdout.log").write_text("STAGE: load\nSTAGE: features\n")

    fake_sandbox(behaviour, exit_code=1)

    record = run_script(ws, _script(ws), settings)

    assert (record.status, record.stage_reached) == ("failed", "features")
    assert load_record(ws, record.run_id).stage_reached == "features"


def test_successful_run_reaches_validate(ws, settings, fake_sandbox):
    def behaviour(ws_, run_dir):
        write_perfect_submission(ws_, run_dir)
        write_oof(ws_, run_dir)

    fake_sandbox(behaviour)

    record = run_script(ws, _script(ws), settings)

    assert (record.status, record.stage_reached) == ("ok", "validate")


def test_submission_without_a_cv_score_stops_at_predict(ws, settings, fake_sandbox):
    fake_sandbox(lambda ws_, run_dir: write_perfect_submission(ws_, run_dir))

    record = run_script(ws, _script(ws), settings)

    assert record.stage_reached == "predict"
