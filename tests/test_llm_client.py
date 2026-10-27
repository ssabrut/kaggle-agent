import json

import httpx
import pytest

from kagent.llm.client import ChatClient, LLMError, strip_thinking


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
