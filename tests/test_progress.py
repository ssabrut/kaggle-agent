import json

import httpx
import pytest

from kagent.llm.client import ChatClient, LLMError
from kagent.search.progress import Progress


def read_status(ws):
    return json.loads((ws.search_dir / "status.json").read_text())


def read_events(ws):
    return [json.loads(line) for line in (ws.search_dir / "events.jsonl").read_text().splitlines()]


def test_starting_writes_a_running_status_and_the_first_event(ws):
    Progress(ws, budget_s=600, params={"noise_k": 0.5}, clock=lambda: 1000.0)

    status = read_status(ws)
    assert status["state"] == "running" and status["params"] == {"noise_k": 0.5}
    assert (status["started_at"], status["deadline_at"], status["budget_s"]) == (
        1000.0,
        1600.0,
        600,
    )
    assert status["pid"] > 0 and status["llm"] is None
    assert [(e["seq"], e["text"]) for e in read_events(ws)] == [(1, "search started")]


def test_phase_keeps_node_and_operator_unless_given(ws):
    clock = [1000.0]
    progress = Progress(ws, budget_s=600, clock=lambda: clock[0])

    clock[0] = 1010.0
    progress.phase("writing", node="n003", op="improve", detail="from n001")
    clock[0] = 1020.0
    progress.phase("running")  # same step, next phase
    status = read_status(ws)
    assert (status["phase"], status["node"], status["operator"]) == ("running", "n003", "improve")
    assert status["phase_since"] == 1020.0 and status["detail"] == ""

    progress.phase("selecting", node=None, op=None)
    status = read_status(ws)
    assert (status["node"], status["operator"]) == (None, None)


def test_the_model_call_in_flight_is_shown_and_cleared(ws):
    progress = Progress(ws, budget_s=600, clock=lambda: 5.0)
    progress.call_started("coder")
    assert read_status(ws)["llm"] == {"role": "coder", "since": 5.0}
    progress.call_finished("coder")
    assert read_status(ws)["llm"] is None


def test_finishing_records_the_outcome_and_an_event(ws):
    progress = Progress(ws, budget_s=600)
    progress.phase("running", node="n002", op="draft")
    progress.finish("stopped", "KeyboardInterrupt: ")

    status = read_status(ws)
    assert (status["state"], status["phase"], status["node"]) == ("stopped", "done", None)
    assert read_events(ws)[-1]["text"] == "search stopped: KeyboardInterrupt: "


def test_events_keep_counting_across_sessions(ws):
    first = Progress(ws, budget_s=600)
    first.event("one", node="n001")
    second = Progress(ws, budget_s=600)  # a resumed search
    second.event("two")
    assert [e["seq"] for e in read_events(ws)] == [1, 2, 3, 4]
    assert read_events(ws)[1]["node"] == "n001"


def test_no_temporary_file_is_left_behind(ws):
    progress = Progress(ws, budget_s=600)
    progress.phase("running")
    assert sorted(p.name for p in ws.search_dir.iterdir() if "status" in p.name) == ["status.json"]


# --- the model client reports each call to an observer ------------------------------------------


class Recorder:
    def __init__(self):
        self.seen = []

    def call_started(self, role):
        self.seen.append(("start", role))

    def call_finished(self, role):
        self.seen.append(("end", role))


def client(recorder, status):
    def handler(request):
        reply = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        return httpx.Response(status, json=reply)

    return ChatClient(
        "http://models.test/v1", "m", role="planner", observer=recorder,
        transport=httpx.MockTransport(handler), sleep=lambda _s: None,
    )  # fmt: skip


def test_observer_sees_a_call_start_and_end():
    recorder = Recorder()
    assert client(recorder, 200).chat([{"role": "user", "content": "q"}]) == "hi"
    assert recorder.seen == [("start", "planner"), ("end", "planner")]


def test_observer_hears_the_end_of_a_call_that_fails():
    recorder = Recorder()
    with pytest.raises(LLMError):
        client(recorder, 400).chat([{"role": "user", "content": "q"}])
    assert recorder.seen == [("start", "planner"), ("end", "planner")]
