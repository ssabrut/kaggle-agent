import os
import subprocess

import pytest

from conftest import write_perfect_submission
from kagent import harness
from kagent import submit as submit_mod
from kagent.submit import SubmitRefused, submit


@pytest.fixture
def ok_run(ws, settings, tmp_path, fake_sandbox):
    script = tmp_path / "solution.py"
    script.write_text("print('hi')\n")
    fake_sandbox(write_perfect_submission)
    return harness.run_script(ws, script, settings)


@pytest.fixture
def no_network(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Successfully submitted", stderr="")

    monkeypatch.setattr(submit_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(submit_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    return calls


def test_default_is_dry_run(ws, ok_run, no_network):
    result = submit(ws, ok_run.run_id, "msg")
    assert result.dry_run
    assert result.command[:4] == ["kaggle", "competitions", "submit", "-c"]
    assert no_network == []


def test_approve_runs_the_kaggle_command(ws, ok_run, no_network):
    result = submit(ws, ok_run.run_id, "msg", approve=True)
    assert not result.dry_run and result.output == "Successfully submitted"
    assert no_network == [result.command]


def test_refuses_run_that_is_not_ok(ws, settings, tmp_path, fake_sandbox, no_network):
    script = tmp_path / "solution.py"
    script.write_text("")
    fake_sandbox(write_perfect_submission, exit_code=1)
    record = harness.run_script(ws, script, settings)
    with pytest.raises(SubmitRefused, match="status 'failed'"):
        submit(ws, record.run_id, "msg", approve=True)
    assert no_network == []


def test_refuses_file_changed_after_grading(ws, ok_run, no_network):
    path = ws.run_dir(ok_run.run_id) / "out" / "submission.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(SubmitRefused, match="changed after"):
        submit(ws, ok_run.run_id, "msg", approve=True)
    assert no_network == []


def test_refuses_symlink_swapped_in_after_grading(ws, ok_run, no_network):
    path = ws.run_dir(ok_run.run_id) / "out" / "submission.csv"
    path.unlink()
    os.symlink("/etc/passwd", path)
    with pytest.raises(SubmitRefused, match="plain file"):
        submit(ws, ok_run.run_id, "msg", approve=True)
    assert no_network == []


@pytest.mark.parametrize("run_id", ["../x", ".hidden", "a/b", ""])
def test_refuses_bad_run_ids(ws, no_network, run_id):
    with pytest.raises(SubmitRefused, match="bad run id"):
        submit(ws, run_id, "msg")
