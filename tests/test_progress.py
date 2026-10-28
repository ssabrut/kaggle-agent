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

    def call_started(self, role, **info):
        self.seen.append(("start", role))

    def call_progress(self, role, **info):
        pass

    def call_finished(self, role, *, ok=True):
        self.seen.append(("end", role, ok))


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
    assert recorder.seen == [("start", "planner"), ("end", "planner", True)]


def test_observer_hears_the_end_of_a_call_that_fails():
    recorder = Recorder()
    with pytest.raises(LLMError):
        client(recorder, 400).chat([{"role": "user", "content": "q"}])
    assert recorder.seen == [("start", "planner"), ("end", "planner", False)]


# --- the reply as it is written -----------------------------------------------------------------


def read_live(ws, name):
    return json.loads((ws.search_dir / "live" / name).read_text())


def test_a_call_writes_its_text_as_it_arrives(ws):
    clock = [100.0]
    progress = Progress(ws, budget_s=600, clock=lambda: clock[0])
    progress.phase("writing", node="n004", op="improve", parent="n002")
    messages = [{"role": "user", "content": "do it"}]
    progress.call_started("coder", model="m", purpose="apply the idea", messages=messages)

    started = read_live(ws, "coder.json")
    assert (started["done"], started["answer"], started["first_at"]) == (False, "", None)
    assert (started["node"], started["operator"], started["parent"]) == ("n004", "improve", "n002")
    assert (started["model"], started["purpose"], started["prompt_chars"]) == (
        "m",
        "apply the idea",
        5,
    )

    clock[0] = 101.0
    progress.call_progress("coder", thinking="hm", answer="def f", chunks=3)
    midway = read_live(ws, "coder.json")
    assert (midway["thinking"], midway["answer"], midway["chunks"]) == ("hm", "def f", 3)
    assert midway["first_at"] == 101.0 and midway["done"] is False

    clock[0] = 103.0
    progress.call_progress("coder", thinking="hm", answer="def f(): pass", chunks=7)
    progress.call_finished("coder", ok=True)
    finished = read_live(ws, "coder.json")
    assert (finished["done"], finished["ok"], finished["answer"]) == (True, True, "def f(): pass")
    assert (finished["first_at"], finished["updated_at"]) == (101.0, 103.0)


def test_the_prompt_is_kept_apart_from_the_text_that_is_rewritten(ws):
    progress = Progress(ws, budget_s=600)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    progress.call_started("planner", messages=messages)
    progress.call_progress("planner", thinking="", answer="plan", chunks=1)

    prompt = read_live(ws, "planner.prompt.json")
    assert prompt["messages"] == messages
    assert prompt["id"] == read_live(ws, "planner.json")["id"]
    assert "messages" not in read_live(ws, "planner.json")


def test_each_role_has_its_own_file_and_a_new_call_replaces_the_last(ws):
    progress = Progress(ws, budget_s=600)
    progress.call_started("planner", purpose="plan")
    progress.call_started("coder", purpose="code")
    first = read_live(ws, "coder.json")["id"]
    progress.call_started("coder", purpose="repair")

    assert read_live(ws, "planner.json")["purpose"] == "plan"
    assert read_live(ws, "coder.json")["purpose"] == "repair"
    assert read_live(ws, "coder.json")["id"] != first


def test_a_role_that_is_not_a_plain_word_gets_no_file(ws):
    progress = Progress(ws, budget_s=600)
    progress.call_started("../escape")
    progress.call_progress("../escape", thinking="", answer="x", chunks=1)
    assert not (ws.search_dir / "live").exists()


def test_a_live_view_that_cannot_be_written_does_not_stop_the_search(ws):
    progress = Progress(ws, budget_s=600)
    (ws.search_dir / "live").write_text("a file where the directory should be")

    progress.call_started("coder", purpose="code")
    progress.call_progress("coder", thinking="", answer="x", chunks=1)
    progress.call_finished("coder", ok=True)

    assert read_status(ws)["llm"] is None
    assert list(ws.search_dir.glob("live/*")) == []


def test_no_temporary_file_is_left_in_the_live_directory(ws):
    progress = Progress(ws, budget_s=600)
    progress.call_started("coder")
    progress.call_progress("coder", thinking="", answer="x", chunks=1)
    names = sorted(p.name for p in (ws.search_dir / "live").iterdir())
    assert names == ["coder.json", "coder.prompt.json"]
