import os
import shutil

from conftest import write_perfect_submission
from kagent import harness


def _script(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text("print('hi')\n")
    return path


def test_ok_run_is_graded_and_recorded(ws, settings, tmp_path, fake_sandbox):
    fake_sandbox(write_perfect_submission)
    record = harness.run_script(ws, _script(tmp_path), settings)

    assert record.status == "ok"
    assert record.grade.valid and record.grade.score == 1.0
    assert record.submission_sha256 and record.integrity_changes == []
    assert harness.load_record(ws, record.run_id) == record


def test_modifying_public_data_is_an_integrity_violation(ws, settings, tmp_path, fake_sandbox):
    def tamper(ws, run_dir):
        write_perfect_submission(ws, run_dir)
        (ws.public / "train.csv").write_text("id,x1,x2,target\n")

    fake_sandbox(tamper)
    record = harness.run_script(ws, _script(tmp_path), settings)

    assert record.status == "integrity_violation"
    assert record.integrity_changes == ["public/train.csv modified"]


def test_modifying_grader_source_is_an_integrity_violation(
    ws, settings, tmp_path, fake_sandbox, monkeypatch
):
    grader_copy = tmp_path / "grader"
    grader_copy.mkdir()
    (grader_copy / "metrics.py").write_text("SCORE = 1\n")
    monkeypatch.setattr(harness, "GRADER_SRC", grader_copy)

    def tamper(ws, run_dir):
        write_perfect_submission(ws, run_dir)
        (grader_copy / "metrics.py").write_text("SCORE = 999\n")

    fake_sandbox(tamper)
    record = harness.run_script(ws, _script(tmp_path), settings)

    assert record.status == "integrity_violation"
    assert record.integrity_changes == ["grader/metrics.py modified"]


def test_adding_a_file_to_private_data_is_an_integrity_violation(
    ws, settings, tmp_path, fake_sandbox
):
    def plant(ws, run_dir):
        write_perfect_submission(ws, run_dir)
        shutil.copy(ws.private / "labels.csv", ws.private / "copy.csv")

    fake_sandbox(plant)
    record = harness.run_script(ws, _script(tmp_path), settings)
    assert record.integrity_changes == ["private/copy.csv added"]


def test_symlinked_submission_is_rejected_not_followed(ws, settings, tmp_path, fake_sandbox):
    def link(ws, run_dir):
        os.symlink("/etc/passwd", run_dir / "out" / "submission.csv")

    fake_sandbox(link)
    record = harness.run_script(ws, _script(tmp_path), settings)

    assert record.status == "invalid_submission"
    assert record.submission_sha256 is None
    assert "plain file" in record.grade.errors[0]


def test_missing_submission(ws, settings, tmp_path, fake_sandbox):
    fake_sandbox(lambda ws, run_dir: None)
    assert harness.run_script(ws, _script(tmp_path), settings).status == "invalid_submission"


def test_invalid_submission_format(ws, settings, tmp_path, fake_sandbox):
    def bad(ws, run_dir):
        (run_dir / "out" / "submission.csv").write_text("id,target\n1,0.5\n")

    fake_sandbox(bad)
    record = harness.run_script(ws, _script(tmp_path), settings)
    assert record.status == "invalid_submission" and not record.grade.valid


def test_crash_skips_grading(ws, settings, tmp_path, fake_sandbox):
    fake_sandbox(write_perfect_submission, exit_code=1)
    record = harness.run_script(ws, _script(tmp_path), settings)
    assert record.status == "failed" and record.grade is None
    assert record.stderr_tail == "boom"


def test_timeout_skips_grading(ws, settings, tmp_path, fake_sandbox):
    fake_sandbox(write_perfect_submission, exit_code=137, timed_out=True)
    record = harness.run_script(ws, _script(tmp_path), settings)
    assert record.status == "timeout" and record.grade is None
