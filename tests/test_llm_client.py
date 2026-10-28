import json

import httpx
import pytest

from kagent.llm.client import ChatClient, LLMError, split_thinking, strip_thinking


def _client(handler, **kwargs):
    sleeps: list[float] = []
    client = ChatClient(
        "http://llm.test/v1",
        "model-x",
        role="coder",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def _reply(text, status=200):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {}}
    return httpx.Response(status, json=body)


def test_returns_content_and_sends_openai_payload():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return _reply("hello")

    client, _ = _client(handler, max_tokens=123, temperature=0.4)
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "model-x"
    assert seen["body"]["max_tokens"] == 123
    assert seen["body"]["temperature"] == 0.4


def test_sampling_options_and_template_kwargs_are_sent_only_when_set():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _reply("ok")

    plain, _ = _client(handler)
    plain.chat([{"role": "user", "content": "q"}])
    assert not {"top_p", "top_k", "chat_template_kwargs"} & set(bodies[0])

    tuned, _ = _client(handler, top_p=0.95, top_k=40, template_kwargs={"reasoning_effort": "low"})
    tuned.chat([{"role": "user", "content": "q"}])
    assert (bodies[1]["top_p"], bodies[1]["top_k"]) == (0.95, 40)
    assert bodies[1]["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_a_call_can_override_the_clients_template_kwargs():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _reply("ok")

    client, _ = _client(handler, template_kwargs={"reasoning_effort": "medium"})
    client.chat([{"role": "user", "content": "q"}], template_kwargs={"enable_thinking": False})
    client.chat([{"role": "user", "content": "q"}])

    assert bodies[0]["chat_template_kwargs"] == {
        "reasoning_effort": "medium",
        "enable_thinking": False,
    }
    assert bodies[1]["chat_template_kwargs"] == {"reasoning_effort": "medium"}  # not remembered


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>plan it</think>answer", "answer"),
        ("<think>\nmulti\nline\n</think>\n\nanswer", "answer"),
        ("reasoning that never opened</think>answer", "answer"),
        ("answer<think>cut off mid-thought", "answer"),
        ("plain", "plain"),
    ],
)
def test_strip_thinking(raw, clean):
    assert strip_thinking(raw) == clean


def test_retries_transient_errors_with_backoff():
    calls = []

    def handler(request):
        calls.append(1)
        return _reply("ok") if len(calls) == 3 else httpx.Response(503)

    client, sleeps = _client(handler)
    assert client.chat([{"role": "user", "content": "x"}]) == "ok"
    assert sleeps == [2.0, 4.0]


def test_gives_up_after_all_attempts():
    client, _ = _client(lambda request: httpx.Response(503), attempts=2)
    with pytest.raises(LLMError, match="unreachable after 2 tries"):
        client.chat([{"role": "user", "content": "x"}])


def test_connection_errors_are_retried_then_reported():
    def handler(request):
        raise httpx.ConnectError("refused")

    client, _ = _client(handler)
    with pytest.raises(LLMError, match="ConnectError"):
        client.chat([{"role": "user", "content": "x"}])


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad model")

    client, _ = _client(handler)
    with pytest.raises(LLMError, match="HTTP 400: bad model"):
        client.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_unexpected_shape_is_an_llm_error():
    client, _ = _client(lambda request: httpx.Response(200, json={"oops": 1}))
    with pytest.raises(LLMError, match="unexpected response shape"):
        client.chat([{"role": "user", "content": "x"}])


def test_trajectory_is_logged(tmp_path):
    log = tmp_path / "logs" / "trajectory.jsonl"
    client, _ = _client(lambda request: _reply("<think>hmm</think>done"), log_path=log)
    client.chat([{"role": "user", "content": "q"}])
    client.chat([{"role": "user", "content": "q2"}])

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 2
    assert entries[0]["role"] == "coder"
    assert entries[0]["model"] == "model-x"
    assert entries[0]["messages"] == [{"role": "user", "content": "q"}]
    assert entries[0]["response"] == "done"


# --- streaming ----------------------------------------------------------------------------------


STREAM = {"content-type": "text/event-stream"}


def _frames(*events):
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)


def _sse(*events):
    return httpx.Response(200, content=_frames(*events).encode(), headers=STREAM)


def _delta(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


class Watcher:
    """An observer that keeps everything it is told."""

    def __init__(self):
        self.started, self.updates, self.ended = [], [], []

    def call_started(self, role, **info):
        self.started.append((role, info))

    def call_progress(self, role, **update):
        self.updates.append(update)

    def call_finished(self, role, *, ok=True):
        self.ended.append(ok)


def _stream_client(handler, watcher, **kwargs):
    return _client(handler, observer=watcher, push_every=0, **kwargs)[0]


def test_a_streamed_reply_is_assembled_and_shown_as_it_grows(tmp_path):
    log = tmp_path / "trajectory.jsonl"
    usage = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        events = [_delta("Hel"), _delta("lo"), _delta("!", finish="stop"), usage, "[DONE]"]
        return _sse(*events)

    watcher = Watcher()
    client = _stream_client(handler, watcher, log_path=log)

    assert client.chat([{"role": "user", "content": "q"}], purpose="greet") == "Hello!"

    answers = [u["answer"] for u in watcher.updates]
    assert answers[0] == "Hel" and answers[-1] == "Hello!"
    assert [len(a) for a in answers] == sorted(len(a) for a in answers)
    role, info = watcher.started[0]
    assert (role, info["model"], info["purpose"]) == ("coder", "model-x", "greet")
    assert info["messages"] == [{"role": "user", "content": "q"}]
    assert watcher.ended == [True]
    entry = json.loads(log.read_text())
    assert entry["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    assert (entry["response"], entry["finish_reason"]) == ("Hello!", "stop")


def test_reasoning_is_shown_apart_from_the_answer():
    events = [_delta(reasoning="weigh "), _delta(reasoning="options"), _delta("pick B"), "[DONE]"]
    watcher = Watcher()
    client = _stream_client(lambda request: _sse(*events), watcher)

    assert client.chat([{"role": "user", "content": "q"}]) == "pick B"

    last = watcher.updates[-1]
    assert (last["thinking"], last["answer"]) == ("weigh options", "pick B")


def test_inline_think_tags_are_split_while_streaming():
    events = [_delta("<think>plan"), _delta(" it</think>"), _delta("done"), "[DONE]"]
    watcher = Watcher()
    client = _stream_client(lambda request: _sse(*events), watcher)

    assert client.chat([{"role": "user", "content": "q"}]) == "done"

    assert (watcher.updates[0]["thinking"], watcher.updates[0]["answer"]) == ("plan", "")
    assert (watcher.updates[-1]["thinking"], watcher.updates[-1]["answer"]) == ("plan it", "done")


def test_without_a_usage_report_the_chunk_count_stands_in(tmp_path):
    log = tmp_path / "trajectory.jsonl"
    client, _ = _client(lambda r: _sse(_delta("a"), _delta("b"), "[DONE]"), log_path=log)
    client.chat([{"role": "user", "content": "q"}])
    assert json.loads(log.read_text())["usage"] == {"completion_tokens": 2}


def test_an_error_event_in_the_stream_is_an_llm_error():
    watcher = Watcher()
    events = [_delta("a"), {"error": {"message": "out of memory"}}]
    client = _stream_client(lambda request: _sse(*events), watcher)
    with pytest.raises(LLMError, match="out of memory"):
        client.chat([{"role": "user", "content": "q"}])
    assert watcher.ended == [False]


def test_events_that_are_not_data_are_ignored():
    noise = ": keep-alive\n\nevent: ping\n\ndata: not json\n\n"
    content = (noise + _frames(_delta("ok"), "[DONE]")).encode()
    response = httpx.Response(200, content=content, headers=STREAM)
    client, _ = _client(lambda request: response)
    assert client.chat([{"role": "user", "content": "q"}]) == "ok"


def test_a_server_that_ignores_stream_still_shows_its_reply():
    watcher = Watcher()
    client = _stream_client(lambda request: _reply("whole reply"), watcher)

    assert client.chat([{"role": "user", "content": "q"}]) == "whole reply"

    assert watcher.updates[-1]["answer"] == "whole reply"
    assert watcher.ended == [True]


def test_an_observer_is_told_when_the_call_fails():
    watcher = Watcher()
    client = _stream_client(lambda request: httpx.Response(400, text="bad model"), watcher)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "q"}])
    assert watcher.ended == [False] and watcher.updates == []


@pytest.mark.parametrize(
    ("raw", "thinking", "answer"),
    [
        ("<think>plan it</think>answer", "plan it", "answer"),
        ("a<think>one</think>b<think>two</think>c", "one\ntwo", "abc"),
        ("reasoning that never opened</think>answer", "reasoning that never opened", "answer"),
        ("answer<think>cut off mid-thought", "cut off mid-thought", "answer"),
        ("plain", "", "plain"),
        ("", "", ""),
    ],
)
def test_split_thinking(raw, thinking, answer):
    assert split_thinking(raw) == (thinking, answer)
