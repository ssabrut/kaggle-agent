"""Docker CLI failures must not be mistaken for bugs in the agent's script."""

import stat

import pytest

from fakes import FakeChat, code_block
from kagent.docker import DockerError, run_container
from kagent.search.loop import open_tree, run_search
from kagent.search.operators import DebugOperator, DraftOperator, ImproveOperator
from kagent.search.policies import GreedyPolicy


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """Put a `docker` on PATH that prints canned stderr and exits with a canned code."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def install(stderr: str, exit_code: int):
        script = bin_dir / "docker"
        script.write_text(f"#!/bin/sh\ncat >&2 <<'EOF'\n{stderr}\nEOF\nexit {exit_code}\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    return install


def _run(tmp_path):
    return run_container(
        "img",
        ["python", "x.py"],
        name="t",
        mounts=[],
        timeout_s=10,
        memory="1g",
        cpus=1,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )


@pytest.mark.parametrize(
    "stderr",
    [
        "failed to connect to the docker API at unix:///x.sock; is the daemon running?",
        "Cannot connect to the Docker daemon at unix:///x.sock. Is the docker daemon running?",
        (
            "Unable to find image 'img:latest' locally\n"
            "docker: Error response from daemon: pull access denied for img"
        ),
        "docker: Error response from daemon: Mounts denied: path is not shared",
    ],
)
def test_docker_cli_failures_raise(fake_docker, tmp_path, stderr):
    fake_docker(stderr, 1)
    with pytest.raises(DockerError):
        _run(tmp_path)


def test_ordinary_script_failure_is_still_a_result(fake_docker, tmp_path):
    fake_docker("Traceback (most recent call last):\nKeyError: 'zip'", 1)
    result = _run(tmp_path)
    assert result.exit_code == 1 and "KeyError" in result.stderr_tail


def test_script_output_that_mentions_docker_later_is_not_infrastructure(fake_docker, tmp_path):
    fake_docker(
        "Traceback (most recent call last):\n  File x.py\nRuntimeError: "
        "failed to connect to the docker API (inside a script)",
        1,
    )
    assert _run(tmp_path).exit_code == 1


def test_success_with_docker_words_in_stderr_is_not_an_error(fake_docker, tmp_path):
    fake_docker("warning: failed to connect to the docker API (printed by script)", 0)
    assert _run(tmp_path).exit_code == 0


def test_search_stops_without_recording_a_node_when_docker_fails(ws, settings):
    config = ws.load_config()
    coder = FakeChat([code_block("print(1)")], role="coder")
    planner = FakeChat(["- plan"], role="planner")
    ops = {
        "draft": DraftOperator(planner, coder),
        "improve": ImproveOperator(planner, coder),
        "debug": DebugOperator(coder),
    }
    tree = open_tree(ws, config)

    def broken_run(ws, script, settings):
        raise DockerError("failed to connect to the docker API")

    with pytest.raises(DockerError):
        run_search(
            ws, config, settings, tree, GreedyPolicy(num_drafts=1), ops, planner,
            description="", eda="", budget_s=600, max_nodes=2, run=broken_run,
        )  # fmt: skip

    assert len(open_tree(ws, config)) == 0
