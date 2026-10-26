"""Milestone 1 gate: a malicious script cannot touch the scorer or the private labels.

Needs a running Docker daemon and the images from `uv run kagent build-images`.
"""

import subprocess

import pytest

from kagent import harness

pytestmark = pytest.mark.docker

MALICIOUS = """
import json, os, socket
import pandas as pd

attempts = {}

def attempt(name, fn):
    try:
        fn()
        attempts[name] = "ALLOWED"
    except BaseException as exc:
        attempts[name] = "blocked: " + type(exc).__name__

attempt("overwrite_train", lambda: open("/data/train.csv", "w").write("pwned"))
attempt("delete_sample", lambda: os.remove("/data/sample_submission.csv"))
attempt("write_private", lambda: open("/data/private/labels.csv", "w").write("pwned"))
attempt("read_private", lambda: open("/data/private/labels.csv").read())
attempt("write_root_fs", lambda: open("/etc/pwn", "w").write("x"))
attempt("network", lambda: socket.create_connection(("1.1.1.1", 53), timeout=3))
attempt("import_grader", lambda: __import__("grader"))

visible = [p for p in ("/data/private", "/private", "/app", "/app/grader", "/data/labels.csv")
           if os.path.exists(p)]
json.dump({"attempts": attempts, "visible": visible, "env": sorted(os.environ)},
          open("/work/out/attempts.json", "w"))

# Constant predictions: a valid file that carries no information.
pd.read_csv("/data/sample_submission.csv").to_csv("/work/out/submission.csv", index=False)
"""

HONEST = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("/data/train.csv")
test = pd.read_csv("/data/test.csv")
model = LogisticRegression().fit(train[["x1", "x2"]], train["target"])
out = pd.DataFrame({"id": test["id"], "target": model.predict_proba(test[["x1", "x2"]])[:, 1]})
out.to_csv("/work/out/submission.csv", index=False)
"""


def _run(ws, settings, tmp_path, source):
    script = tmp_path / "solution.py"
    script.write_text(source)
    return harness.run_script(ws, script, settings)


def test_malicious_script_is_contained(ws, settings, tmp_path):
    record = _run(ws, settings, tmp_path, MALICIOUS)

    import json

    report = json.loads((ws.run_dir(record.run_id) / "out" / "attempts.json").read_text())
    assert report["attempts"] and all(v != "ALLOWED" for v in report["attempts"].values()), report
    assert report["visible"] == []
    # Host env is never passed in. (GPG_KEY comes from the python base image and is public.)
    leaked_env = [
        k
        for k in report["env"]
        if k.startswith(("KAGGLE", "ANTHROPIC", "OPENAI", "AWS"))
        or k.endswith(("_TOKEN", "_API_KEY"))
    ]
    assert leaked_env == []

    assert record.integrity_changes == []
    assert record.status == "ok"
    assert record.grade.score == pytest.approx(0.5)  # no information leaked into the submission


def test_honest_script_gets_a_real_score(ws, settings, tmp_path):
    record = _run(ws, settings, tmp_path, HONEST)
    assert record.status == "ok"
    assert record.grade.score > 0.8


def test_symlinked_submission_is_rejected(ws, settings, tmp_path):
    record = _run(
        ws, settings, tmp_path, "import os; os.symlink('/etc/passwd', '/work/out/submission.csv')"
    )
    assert record.status == "invalid_submission"
    assert record.submission_sha256 is None


def test_crash_is_reported(ws, settings, tmp_path):
    record = _run(ws, settings, tmp_path, "raise RuntimeError('nope')")
    assert record.status == "failed" and "RuntimeError" in record.stderr_tail


def test_timeout_kills_the_container(ws, settings, tmp_path):
    settings.step_timeout_s = 3
    record = _run(ws, settings, tmp_path, "import time; time.sleep(60)")
    assert record.status == "timeout"
    running = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name=kagent-{record.run_id}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert running.strip() == ""
