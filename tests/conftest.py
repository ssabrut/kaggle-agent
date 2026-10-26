import subprocess

import numpy as np
import pandas as pd
import pytest

from kagent.config import CompetitionConfig, Settings
from kagent.docker import docker_available
from kagent.workspace import Workspace


@pytest.fixture
def settings(tmp_path):
    return Settings(workspaces_dir=tmp_path / "workspaces")


@pytest.fixture
def ws(settings) -> Workspace:
    """Synthetic binary-classification competition with held-out test labels."""
    rng = np.random.default_rng(0)
    n_train, n_test = 200, 100

    def make(n: int, start: int) -> pd.DataFrame:
        x = rng.normal(size=(n, 2))
        y = (x.sum(axis=1) + rng.normal(scale=0.5, size=n) > 0).astype(int)
        return pd.DataFrame(
            {"id": range(start, start + n), "x1": x[:, 0], "x2": x[:, 1], "target": y}
        )

    train, test = make(n_train, 0), make(n_test, n_train)
    workspace = Workspace.for_slug(settings, "toy")
    workspace.create()
    train.to_csv(workspace.public / "train.csv", index=False)
    test.drop(columns="target").to_csv(workspace.public / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.5}).to_csv(
        workspace.public / "sample_submission.csv", index=False
    )
    test[["id", "target"]].to_csv(workspace.private / "labels.csv", index=False)
    workspace.save_config(CompetitionConfig(slug="toy", metric="auc"))
    return workspace


@pytest.fixture
def fake_sandbox(monkeypatch):
    """Replace the Docker sandbox and grader with host-side stand-ins.

    install(behaviour) makes `behaviour(ws, run_dir)` play the role of the agent script.
    Grading still runs the real grader code, just not inside a container.
    """
    from kagent import grading
    from kagent.docker import ContainerResult
    from kagent.grader.main import grade
    from kagent.sandbox import runner

    def install(behaviour, exit_code=0, timed_out=False):
        def execute(ws, run_dir, settings):
            out = run_dir / "out"
            out.mkdir(exist_ok=True)
            behaviour(ws, run_dir)
            return ContainerResult(exit_code, timed_out, 0.1, "", "boom" if exit_code else "")

        def grade_submission(ws, config, run_dir, settings):
            path = run_dir / "out" / "submission.csv"
            return grade(ws.public, ws.private, path, config.metric, config.id_column)

        monkeypatch.setattr(runner, "execute", execute)
        monkeypatch.setattr(grading, "grade_submission", grade_submission)

    return install


def write_perfect_submission(ws, run_dir):
    labels = pd.read_csv(ws.private / "labels.csv")
    labels.astype({"target": float}).to_csv(run_dir / "out" / "submission.csv", index=False)


def _images_present(settings: Settings) -> bool:
    for image in (settings.sandbox_image, settings.grader_image):
        if subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, check=False
        ).returncode:
            return False
    return True


def pytest_collection_modifyitems(config, items):
    if not any("docker" in item.keywords for item in items):
        return
    if not docker_available():
        reason = "docker daemon not running"
    elif not _images_present(Settings()):
        reason = "images missing; run `uv run kagent build-images`"
    else:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)
