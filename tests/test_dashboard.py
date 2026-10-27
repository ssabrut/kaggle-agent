"""The dashboard's data layer and server, driven by the invented search in kagent.dashboard.demo."""

import http.client
import json
import urllib.error
import urllib.request

import pytest
from typer.testing import CliRunner

from kagent.cli import app
from kagent.dashboard.demo import SCRIPT, DemoRun
from kagent.dashboard.reader import JsonlTail, WorkspaceReader, json_safe, pid_alive
from kagent.dashboard.server import make_server, start_in_background
from kagent.workspace import Workspace


@pytest.fixture
def demo(settings):
    run = DemoRun(Workspace.for_slug(settings, "demo"))
    run.populate(len(SCRIPT))
    return run


@pytest.fixture
def reader(demo, settings):
    return WorkspaceReader(demo.ws, settings)


def nodes_by_id(state):
    return {n["id"]: n for n in state["nodes"]}


# --- state ------------------------------------------------------------------------------------


def test_state_describes_the_competition_and_the_run(reader):
    state = reader.state()
    assert state["competition"]["metric"] == "accuracy"
    assert state["competition"]["higher_is_better"] is True
    assert state["run"]["state"] == "running" and state["run"]["alive"] is True
    assert state["noise_k"] == 0.5  # from the search's own parameters
    assert state["summary"]["total"] == len(SCRIPT) == 15


def test_status_counts_and_the_two_best_nodes(reader):
    summary = reader.state()["summary"]
    assert (summary["ok"], summary["buggy"], summary["dead"], summary["blocked"]) == (9, 4, 1, 1)
    assert summary["incumbent"]["id"] == "n009"  # what the search builds on, noise-gated
    assert summary["best"]["id"] == "n012"  # best by robust CV, which can differ


def test_each_improvement_is_judged_against_the_noise_floor(reader):
    nodes = nodes_by_id(reader.state())
    assert nodes["n005"]["verdict"] == "better"
    assert nodes["n006"]["verdict"] == "noise"
    assert nodes["n008"]["verdict"] == "worse"
    assert nodes["n012"]["verdict"] == "noise"
    assert nodes["n005"]["gain"] == pytest.approx(0.0056)
    assert nodes["n005"]["floor"] == pytest.approx(0.5 * 0.0071)
    # No score to compare with: a draft, and a debug of a node that never scored.
    assert nodes["n001"]["verdict"] is None and nodes["n011"]["verdict"] is None


def test_incumbent_after_each_node_traces_the_step_line(reader):
    nodes = nodes_by_id(reader.state())
    assert [nodes[i]["incumbent_after"] for i in ("n001", "n003", "n006", "n009", "n012")] == [
        "n001", "n003", "n005", "n009", "n009",
    ]  # fmt: skip
    assert nodes["n002"]["incumbent_after"] is None  # a failed node is not considered


def test_stage_counts_cover_every_stage(reader):
    stages = reader.state()["stages"]
    assert list(stages) == ["start", "load", "features", "cv", "predict", "validate"]
    assert (stages["validate"], stages["features"], stages["cv"]) == (9, 4, 2)


def test_blocklist_dependency_and_model_sections(reader):
    state = reader.state()
    assert any("MemoryError" in entry for entry in state["blocklist"])
    assert any("no gain" in entry for entry in state["blocklist"])
    dep = state["deps"][0]
    assert (dep["id"], dep["package"], dep["status"], dep["node_id"]) == (
        "dep-001", "catboost", "pending", "n007",
    )  # fmt: skip
    assert "+2 TRANSITIVE" in dep["flags"] and state["summary"]["pending_deps"] == 1
    assert state["llm"]["coder"]["calls"] == 15 and state["llm"]["planner"]["calls"] == 20
    assert state["llm"]["coder"]["tokens_per_s"] > 0


def test_a_search_that_died_without_saying_so_is_reported_as_lost(reader, demo):
    path = demo.ws.search_dir / "status.json"
    status = json.loads(path.read_text())
    status["pid"] = 2**30  # no such process
    path.write_text(json.dumps(status))
    assert reader.state()["run"]["state"] == "lost"


def test_a_finished_search_stays_finished(reader, demo):
    demo.progress.finish("finished", "time budget used up")
    run = reader.state()["run"]
    assert (run["state"], run["message"]) == ("finished", "time budget used up")


def test_a_workspace_that_never_ran_a_search(settings):
    ws = Workspace.for_slug(settings, "fresh")
    DemoRun(ws)  # config and an empty tree, then remove the status to look never-started
    (ws.search_dir / "status.json").unlink()
    state = WorkspaceReader(ws, settings).state()
    assert state["run"] == {"state": "never"} and state["nodes"] == []


def test_fingerprint_moves_when_something_changes(settings):
    run = DemoRun(Workspace.for_slug(settings, "other"))
    run.populate(3)
    reader = WorkspaceReader(run.ws, settings)
    before = reader.fingerprint()
    assert reader.fingerprint() == before
    run.populate(4)
    assert reader.fingerprint() != before


# --- node detail ------------------------------------------------------------------------------


def test_node_detail_has_code_plan_logs_and_run(reader):
    detail = reader.node("n002")
    assert detail["status"] == "buggy" and detail["children"] == ["n003"]
    assert detail["plan"] == ["PLAN: split Cabin into deck, number and side, then LightGBM"]
    assert "import pandas" in detail["code"]
    assert "STAGE: features" in detail["stdout"] and "ValueError" in detail["stderr"]
    assert (detail["run"]["status"], detail["run"]["exit_code"]) == ("failed", 1)


@pytest.mark.parametrize("node_id", ["n999", "../etc/passwd", "n01", "n001/../x", "", "N001"])
def test_unknown_or_malformed_node_ids_give_nothing(reader, node_id):
    assert reader.node(node_id) is None


# --- following the logs -----------------------------------------------------------------------


def test_events_and_calls_arrive_incrementally(settings):
    run = DemoRun(Workspace.for_slug(settings, "inc"))
    reader = WorkspaceReader(run.ws, settings)
    run.populate(3)
    first = reader.events(0)
    assert first["last"] == len(first["events"]) == 4  # "search started" plus three nodes
    run.populate(5)
    more = reader.events(first["last"])
    assert len(more["events"]) == 2 and more["last"] == 6
    assert reader.events(more["last"]) == {"events": [], "last": 6}
    assert reader.calls(-1)["last"] == len(reader.calls(-1)["calls"]) - 1


def test_call_detail_includes_the_prompt_and_reply_without_trusting_size(reader):
    call = reader.call(0)
    assert call["role"] == "planner" and call["messages"][-1]["role"] == "user"
    assert call["response"].startswith("(demo reply)")
    assert reader.call(10_000) is None and reader.call(-1) is None


def test_tail_waits_for_a_complete_line(tmp_path):
    path = tmp_path / "log.jsonl"
    tail = JsonlTail(path, lambda _i, row: row)
    path.write_text('{"a": 1}\n{"a": ')
    tail.refresh()
    assert tail.rows == [{"a": 1}]
    with path.open("a") as fh:
        fh.write("2}\n")
    tail.refresh()
    assert tail.rows == [{"a": 1}, {"a": 2}] and tail.read_raw(1) == {"a": 2}


def test_tail_skips_bad_lines_and_restarts_when_the_file_is_replaced(tmp_path):
    path = tmp_path / "log.jsonl"
    tail = JsonlTail(path, lambda _i, row: row)
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    tail.refresh()
    assert tail.rows == [{"a": 1}, {"a": 2}]
    path.write_text('{"b": 1}\n')
    tail.refresh()
    assert tail.rows == [{"b": 1}]


def test_pid_alive_and_json_safe():
    import os

    assert pid_alive(os.getpid()) and not pid_alive(None) and not pid_alive(0) and not pid_alive(-3)
    assert json_safe({"a": [float("nan"), 1.5, float("inf")], "b": "x"}) == {
        "a": [None, 1.5, None], "b": "x",
    }  # fmt: skip


def test_playing_the_demo_to_the_end_finishes_the_search(settings):
    run = DemoRun(Workspace.for_slug(settings, "play"))
    run.populate(2)
    run.play(sleep=lambda _s: None)
    status = json.loads((run.ws.search_dir / "status.json").read_text())
    assert (status["state"], status["message"]) == ("finished", "demo complete")
    assert len(run.tree) == len(SCRIPT) and run.done


# --- the HTTP server --------------------------------------------------------------------------


@pytest.fixture
def served(reader):
    server = make_server(reader)
    start_in_background(server)
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(port, path, **kwargs):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", **kwargs)
    return NO_PROXY.open(request, timeout=5)


def test_state_is_served_with_security_headers_and_an_etag(served):
    with fetch(served, "/api/state") as resp:
        body = json.load(resp)
        headers = resp.headers
    assert len(body["nodes"]) == 15
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["ETag"]


def test_unchanged_state_answers_not_modified(served):
    with fetch(served, "/api/state") as resp:
        etag = resp.headers["ETag"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch(served, "/api/state", headers={"If-None-Match": etag})
    assert caught.value.code == 304


def test_the_page_and_its_assets_are_served(served):
    for path, kind in (
        ("/", "text/html"),
        ("/app.js", "text/javascript"),
        ("/style.css", "text/css"),
    ):
        with fetch(served, path) as resp:
            assert resp.headers["Content-Type"].startswith(kind) and resp.read()


def test_node_events_and_call_endpoints(served):
    with fetch(served, "/api/node/n002") as resp:
        assert json.load(resp)["plan"]
    with fetch(served, "/api/events?after=0") as resp:
        assert json.load(resp)["events"]
    with fetch(served, "/api/calls?after=-1") as resp:
        assert json.load(resp)["calls"]
    with fetch(served, "/api/call/0") as resp:
        assert json.load(resp)["messages"]


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/nope", 404),
        ("/api/node/not-a-node", 404),
        ("/api/call/9999", 404),
        ("/api/events?after=x", 400),
    ],
)
def test_bad_requests_get_a_json_error(served, path, code):
    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch(served, path)
    assert caught.value.code == code
    assert "error" in json.load(caught.value)


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_nothing_can_be_changed_through_the_server(served, method):
    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch(served, "/api/state", data=b"x", method=method)
    assert caught.value.code == 405


def test_a_foreign_host_header_is_refused(served):
    """DNS rebinding: a web page on another name must not reach the server."""
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=5)
    conn.request("GET", "/api/state", headers={"Host": "evil.example"})
    assert conn.getresponse().status == 421


@pytest.mark.parametrize("args", [["--no-open"], ["-c", "nothing-here", "--no-open"]])
def test_the_command_refuses_to_start_without_a_workspace(settings, args):
    env = {"KAGENT_WORKSPACES_DIR": str(settings.workspaces_dir)}
    result = CliRunner().invoke(app, ["dashboard", *args], env=env)
    assert result.exit_code == 2  # a usage error, and nothing was served


def test_a_workspace_without_config_explains_itself(settings):
    server = make_server(WorkspaceReader(Workspace.for_slug(settings, "empty"), settings))
    start_in_background(server)
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            fetch(server.server_address[1], "/api/state")
        assert caught.value.code == 404 and "kagent init" in json.load(caught.value)["error"]
    finally:
        server.shutdown()
        server.server_close()
