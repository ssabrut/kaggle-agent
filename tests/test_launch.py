"""Starting and stopping a search from the dashboard, with every process and probe faked."""

import json
import os
import signal
import sys

import pytest
from pydantic import ValidationError

from kagent.config import Settings
from kagent.dashboard.launch import (
    Launcher,
    LaunchError,
    RunRequest,
    _spawn,
    build_command,
    build_env,
)
from kagent.deps.models import DependencyRequest
from kagent.deps.store import DependencyStore

NO_PROCESS = 2**30  # a pid nothing is running as


class FakeProc:
    def __init__(self, pid=NO_PROCESS):
        self.pid, self.code = pid, None

    def poll(self):
        return self.code

    def wait(self):
        return self.code


class Rig:
    """A launcher whose spawn, signals and probes are all recorded, never real."""

    def __init__(self, ws, settings, **overrides):
        self.spawned, self.signals = [], []
        self.group = NO_PROCESS  # what getpgid reports for the search
        self.proc = FakeProc()

        def spawn(argv, env, log):
            self.spawned.append((argv, env))
            return self.proc

        options = {
            "spawn": spawn,
            "getpgid": lambda pid: self.group,
            "killpg": lambda group, sig: self.signals.append((group, sig)),
            "docker_ok": lambda: True,
            "stale": lambda settings: [],
            "reach": lambda url: (True, "answered HTTP 200"),
        }
        self.launcher = Launcher(ws, settings, **{**options, **overrides})
        self.request = RunRequest.defaults(settings)


@pytest.fixture
def rig(ws, settings):
    return Rig(ws, settings)


def payload(settings, **changes):
    return {**RunRequest.defaults(settings).model_dump(), **changes}


# --- what the page may set ----------------------------------------------------------------------


def test_defaults_come_from_settings(settings):
    req = RunRequest.defaults(settings)
    assert req.planner_model == settings.planner_model
    assert req.coder_base_url == settings.coder_base_url
    assert (req.step_timeout_s, req.sandbox_memory) == (600, "8g")


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_hours": 0},
        {"budget_hours": 100},
        {"max_nodes": 0},
        {"num_drafts": 0},
        {"max_debug": 11},
        {"noise_k": -1},
        {"step_timeout_s": 5},
        {"sandbox_cpus": 0},
        {"sandbox_memory": "8gb"},
        {"sandbox_memory": "8g; rm -rf /"},
        {"planner_base_url": "ftp://models.test/v1"},
        {"planner_base_url": "http://user:secret@models.test/v1"},
        {"planner_base_url": "http://models test/v1"},
        {"planner_base_url": "not a url"},
        {"coder_model": "two words"},
        {"coder_model": "model;rm"},
        {"coder_model": ""},
        {"planner_thinking": "max"},
        {"planner_thinking": "on; rm"},
        {"planner_temperature": -0.1},
        {"coder_temperature": 3},
        {"planner_top_p": 0},
        {"coder_top_p": 1.5},
        {"planner_top_k": -1},
        {"coder_top_k": 5000},
        {"surprise": 1},
    ],
)
def test_out_of_range_or_unknown_settings_are_rejected(settings, changes):
    with pytest.raises(ValidationError):
        RunRequest.model_validate(payload(settings, **changes))


def test_urls_and_models_are_tidied(settings):
    req = RunRequest.model_validate(
        payload(
            settings,
            planner_base_url=" http://models.test:11000/v1/chat/completions/ ",
            coder_model=" mlx-community/Qwen3-Coder-Next-4bit ",
        )
    )
    assert req.planner_base_url == "http://models.test:11000/v1"
    assert req.coder_model == "mlx-community/Qwen3-Coder-Next-4bit"


# --- the command and environment ----------------------------------------------------------------


def test_command_is_an_argument_list_for_search(settings):
    req = RunRequest.model_validate(payload(settings, max_nodes=12, budget_hours=0.5))
    assert build_command("toy", req, python="py") == [
        "py", "-m", "kagent", "search", "-c", "toy",
        "--budget-hours", "0.5", "--num-drafts", "3", "--max-debug", "3", "--noise-k", "0.5",
        "--max-nodes", "12",
    ]  # fmt: skip


def test_no_node_limit_leaves_the_option_out(settings):
    req = RunRequest.defaults(settings)
    assert "--max-nodes" not in build_command("toy", req)


@pytest.mark.parametrize("slug", ["--help", "a b", "-c", "x;y", "", "../x", "a/b"])
def test_an_unsafe_slug_is_refused(settings, slug):
    with pytest.raises(ValueError, match="unsafe"):
        build_command(slug, RunRequest.defaults(settings))


def test_environment_adds_only_the_chosen_kagent_settings(settings):
    base = {"PATH": "/bin", "HOME": "/home/me"}
    req = RunRequest.model_validate(payload(settings, sandbox_cpus=2, sandbox_memory="4g"))

    env = build_env(req, settings, base)

    assert {k: env[k] for k in base} == base
    added = set(env) - set(base)
    assert added and all(key.startswith("KAGENT_") for key in added)
    assert env["KAGENT_SANDBOX_CPUS"] == "2.0" and env["KAGENT_SANDBOX_MEMORY"] == "4g"
    assert env["KAGENT_PLANNER_BASE_URL"] == req.planner_base_url
    assert env["KAGENT_WORKSPACES_DIR"] == str(settings.workspaces_dir)


def test_sampling_and_thinking_reach_the_search_through_the_environment(settings):
    req = RunRequest.model_validate(
        payload(settings, planner_thinking="low", coder_temperature=0.6, coder_top_k=20)
    )

    env = build_env(req, settings, {})

    assert env["KAGENT_PLANNER_THINKING"] == "low"
    assert (env["KAGENT_CODER_TEMPERATURE"], env["KAGENT_CODER_TOP_K"]) == ("0.6", "20")
    assert env["KAGENT_PLANNER_TOP_P"] == "0.95" and env["KAGENT_PLANNER_TOP_K"] == "20"


def test_the_environment_values_are_what_the_search_reads_back(settings, monkeypatch):
    """The page's values must survive the trip through KAGENT_* into Settings."""
    req = RunRequest.model_validate(
        payload(settings, planner_thinking="off", planner_temperature=0.7, coder_top_p=0.9)
    )
    for key, value in build_env(req, settings, {}).items():
        monkeypatch.setenv(key, value)

    read = Settings()

    assert read.planner_thinking == "off"
    assert read.planner_template_kwargs() == {"enable_thinking": False}
    assert (read.planner_temperature, read.coder_top_p) == (0.7, 0.9)


# --- starting -----------------------------------------------------------------------------------


def test_start_spawns_search_and_records_the_launch(rig, ws):
    result = rig.launcher.start(rig.request)

    assert result == {"pid": NO_PROCESS}
    argv, env = rig.spawned[0]
    assert argv[1:6] == ["-m", "kagent", "search", "-c", "toy"]
    assert env["KAGENT_CODER_MODEL"] == rig.request.coder_model
    launch = json.loads((ws.search_dir / "launch.json").read_text())
    assert launch["pid"] == NO_PROCESS and launch["argv"] == argv
    assert launch["request"]["budget_hours"] == rig.request.budget_hours
    assert "starting: " in (ws.search_dir / "run.log").read_text()


def test_a_second_start_is_refused_while_one_is_running(rig):
    rig.launcher.start(rig.request)
    with pytest.raises(LaunchError, match="already running"):
        rig.launcher.start(rig.request)
    assert len(rig.spawned) == 1


def test_start_works_again_once_the_search_has_ended(rig):
    rig.launcher.start(rig.request)
    rig.proc.code = 0  # it exited
    rig.launcher.start(rig.request)
    assert len(rig.spawned) == 2


def test_start_needs_a_set_up_competition(rig, ws):
    (ws.public / "folds.csv").unlink()
    with pytest.raises(LaunchError, match="folds.csv"):
        rig.launcher.start(rig.request)
    ws.config_path.unlink()
    with pytest.raises(LaunchError, match="kagent init"):
        rig.launcher.start(rig.request)
    assert rig.spawned == []


def test_a_spawn_that_fails_is_reported_not_raised_raw(ws, settings):
    def broken(argv, env, log):
        raise OSError("no such interpreter")

    launcher = Rig(ws, settings, spawn=broken).launcher
    with pytest.raises(LaunchError, match="could not start"):
        launcher.start(RunRequest.defaults(settings))


def test_settings_are_remembered_for_next_time(rig, ws, settings):
    rig.launcher.start(RunRequest.model_validate(payload(settings, max_nodes=7, noise_k=1.5)))

    fresh = Rig(ws, settings).launcher.defaults()

    assert (fresh["max_nodes"], fresh["noise_k"]) == (7, 1.5)


def test_unreadable_remembered_settings_fall_back_to_defaults(ws, settings):
    (ws.root / "run_config.json").write_text(json.dumps({"budget_hours": 9999}))
    defaults = Rig(ws, settings).launcher.defaults()
    assert defaults["budget_hours"] == 1.0


def test_a_search_started_by_an_earlier_dashboard_still_counts_as_running(ws, settings):
    ws.search_dir.mkdir(parents=True, exist_ok=True)
    (ws.search_dir / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    assert Rig(ws, settings).launcher.active_pid() == os.getpid()


# --- stopping -----------------------------------------------------------------------------------


def test_stop_asks_the_search_to_end_in_its_own_process_group(rig):
    rig.launcher.start(rig.request)
    rig.launcher.stop()
    assert rig.signals == [(NO_PROCESS, signal.SIGTERM)]


def test_force_stop_kills(rig):
    rig.launcher.start(rig.request)
    rig.launcher.stop(force=True)
    assert rig.signals == [(NO_PROCESS, signal.SIGKILL)]


def test_stop_without_a_running_search_says_so(rig):
    with pytest.raises(LaunchError, match="no search is running"):
        rig.launcher.stop()


def test_stop_never_signals_a_group_it_did_not_create(rig):
    rig.launcher.start(rig.request)
    rig.group = 4242  # the pid is not a session leader of its own group
    with pytest.raises(LaunchError, match="refusing"):
        rig.launcher.stop()
    assert rig.signals == []


def test_a_recorded_pid_is_not_stopped_unless_its_own_status_says_it_is_the_search(ws, settings):
    ws.search_dir.mkdir(parents=True, exist_ok=True)
    me = os.getpid()
    (ws.search_dir / "launch.json").write_text(json.dumps({"pid": me}))
    rig = Rig(ws, settings)
    rig.group = me

    with pytest.raises(LaunchError, match="not a search"):
        rig.launcher.stop()  # an unrelated process that happens to have this pid
    assert rig.signals == []

    (ws.search_dir / "status.json").write_text(json.dumps({"pid": me, "state": "running"}))
    rig.launcher.stop()
    assert rig.signals == [(me, signal.SIGTERM)]


# --- preflight ----------------------------------------------------------------------------------


def statuses(result):
    return {c["id"]: c["status"] for c in result["checks"]}


def test_preflight_passes_when_everything_is_in_place(rig):
    result = rig.launcher.preflight(rig.request)
    assert result["ready"] is True
    assert set(statuses(result).values()) == {"ok"}


def test_preflight_blocks_on_docker_images_and_models(ws, settings):
    down = Rig(ws, settings, docker_ok=lambda: False).launcher.preflight(
        RunRequest.defaults(settings)
    )
    assert down["ready"] is False
    assert statuses(down)["docker"] == statuses(down)["images"] == "fail"

    stale = Rig(ws, settings, stale=lambda s: ["sandbox"]).launcher.preflight(
        RunRequest.defaults(settings)
    )
    assert statuses(stale)["images"] == "fail" and stale["ready"] is False

    def one_model_down(url):
        return (False, "ConnectError: refused") if ":10000" in url else (True, "answered HTTP 200")

    models = Rig(ws, settings, reach=one_model_down).launcher.preflight(
        RunRequest.defaults(settings)
    )
    assert statuses(models)["coder"] == "fail" and statuses(models)["planner"] == "ok"
    assert models["ready"] is False


def test_preflight_blocks_on_a_missing_setup_and_a_running_search(rig, ws):
    rig.launcher.start(rig.request)
    (ws.public / "folds.csv").unlink()

    result = rig.launcher.preflight(rig.request)

    assert statuses(result)["idle"] == statuses(result)["folds"] == "fail"
    assert result["ready"] is False


def test_pending_packages_and_an_existing_tree_only_warn(rig, ws):
    DependencyStore(ws.root / "deps").save(
        DependencyRequest(id="dep-001", package="catboost", reason="x", node_id="n001")
    )
    ws.search_dir.mkdir(parents=True, exist_ok=True)
    (ws.search_dir / "nodes.jsonl").write_text('{"id": "n001"}\n{"id": "n002"}\n{"id": "n001"}\n')

    result = rig.launcher.preflight(rig.request)

    assert statuses(result)["packages"] == statuses(result)["resume"] == "warn"
    assert result["ready"] is True
    messages = {c["id"]: c["message"] for c in result["checks"]}
    assert "resumes from 2 existing nodes" in messages["resume"]  # a node's versions count once


# --- the real spawn -----------------------------------------------------------------------------


def test_the_real_spawn_gives_the_search_its_own_process_group(tmp_path):
    log_path = tmp_path / "out.log"
    code = "import os; print(os.getpgrp() == os.getpid())"
    with log_path.open("wb") as log:
        proc = _spawn([sys.executable, "-c", code], dict(os.environ), log)
        proc.wait(timeout=30)
    assert proc.returncode == 0 and log_path.read_text().strip() == "True"
