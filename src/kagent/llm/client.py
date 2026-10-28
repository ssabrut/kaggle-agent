"""OpenAI-compatible chat client (mlx_lm.server). One instance per role.

Replies are streamed, so a display can show them as they are written. Every call is appended to a
JSONL trajectory log: prompts and responses are the raw material for later SFT or a skill library.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

Message = dict[str, str]

_RETRY_STATUS = {429, 500, 502, 503, 504}
PUSH_EVERY_S = 0.2  # how often a streaming call tells its observer what has arrived so far
ERROR_CHARS = 300


class CallObserver(Protocol):
    """Told how a call is going, so a display can show what is being waited on and read the reply
    while it is still being written."""

    def call_started(
        self,
        role: str,
        *,
        model: str = "",
        purpose: str = "",
        messages: list[Message] | None = None,
    ) -> None: ...

    def call_progress(self, role: str, *, thinking: str, answer: str, chunks: int) -> None: ...

    def call_finished(self, role: str, *, ok: bool = True) -> None: ...


class LLMError(RuntimeError):
    pass


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer) of a reply that may still be arriving.

    Reasoning models may put <think> blocks into the content field.
    """
    thoughts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    rest = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in rest:  # unterminated: generation was cut off mid-thought
        rest, tail = rest.split("<think>", 1)
        thoughts.append(tail)
    elif "</think>" in rest:  # chat template already opened the block
        head, rest = rest.rsplit("</think>", 1)
        thoughts.insert(0, head)
    return "\n".join(t.strip() for t in thoughts if t.strip()), rest.strip()


def strip_thinking(text: str) -> str:
    """Drop <think> blocks that reasoning models may emit into the content field."""
    return split_thinking(text)[1]


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        role: str,
        timeout_s: float = 900,
        max_tokens: int = 8192,
        temperature: float = 0.3,
        top_p: float | None = None,
        top_k: int | None = None,
        template_kwargs: dict[str, Any] | None = None,
        log_path: Path | None = None,
        attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        observer: CallObserver | None = None,
        clock: Callable[[], float] = time.monotonic,
        push_every: float = PUSH_EVERY_S,
    ) -> None:
        self.observer = observer
        self.model = model
        self.role = role
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.template_kwargs = dict(template_kwargs or {})
        self.log_path = log_path
        self._attempts = attempts
        self._sleep = sleep
        self._clock = clock
        self._push_every = push_every
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        purpose: str = "",
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """`purpose` says what the call is for, in a few words, and is only shown to a watcher.

        `template_kwargs` go to the model's chat template, over the client's own (for example
        {"enable_thinking": False} for a call that must answer within a few tokens).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if kwargs := {**self.template_kwargs, **(template_kwargs or {})}:
            payload["chat_template_kwargs"] = kwargs
        start = time.monotonic()
        ok = False
        if self.observer:
            self.observer.call_started(
                self.role, model=self.model, purpose=purpose, messages=messages
            )
        try:
            data = self._post(payload)
            ok = True
        finally:
            if self.observer:
                self.observer.call_finished(self.role, ok=ok)
        try:
            choice = data["choices"][0]
            text = strip_thinking(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.role}: unexpected response shape: {str(data)[:300]}") from exc
        self._log(messages, text, choice.get("finish_reason"), data.get("usage"), start)
        return text

    def _post(self, payload: dict) -> dict:
        last = "no attempt made"
        for attempt in range(self._attempts):
            if attempt:
                self._sleep(2.0**attempt)
            try:
                with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code in _RETRY_STATUS:
                        last = f"HTTP {resp.status_code}"
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise LLMError(
                            f"{self.role}: HTTP {resp.status_code}: {resp.text[:ERROR_CHARS]}"
                        )
                    if "text/event-stream" in resp.headers.get("content-type", ""):
                        return self._read_stream(resp)
                    resp.read()  # a server that ignores `stream` answers with one JSON body
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise LLMError(
                            f"{self.role}: response is not JSON: {resp.text[:ERROR_CHARS]}"
                        ) from exc
                    self._push_reply(data)
                    return data
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
        raise LLMError(
            f"{self.role} ({self.model}) unreachable after {self._attempts} tries: {last}"
        )

    def _read_stream(self, resp: httpx.Response) -> dict:
        """Assemble server-sent events into the shape of a non-streaming reply."""
        pieces: list[str] = []
        reasoning: list[str] = []
        finish = usage = None
        chunks = 0
        pushed = self._clock()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise LLMError(f"{self.role}: {str(event['error'])[:ERROR_CHARS]}")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                content = delta.get("content")
                thought = delta.get("reasoning") or delta.get("reasoning_content")
                if isinstance(content, str) and content:
                    pieces.append(content)
                    chunks += 1
                if isinstance(thought, str) and thought:
                    reasoning.append(thought)
                    chunks += 1
                finish = choice.get("finish_reason") or finish
            now = self._clock()
            if now - pushed >= self._push_every:
                pushed = now
                self._push("".join(pieces), "".join(reasoning), chunks)
        self._push("".join(pieces), "".join(reasoning), chunks)
        return {
            "choices": [{"message": {"content": "".join(pieces)}, "finish_reason": finish}],
            "usage": usage or {"completion_tokens": chunks},  # one chunk is about one token
        }

    def _push(self, content: str, reasoning: str, chunks: int) -> None:
        if not self.observer:
            return
        thinking, answer = split_thinking(content)
        thinking = "\n".join(t for t in (reasoning.strip(), thinking) if t)
        self.observer.call_progress(self.role, thinking=thinking, answer=answer, chunks=chunks)

    def _push_reply(self, data: dict) -> None:
        """Show a reply that arrived whole, so a watcher sees it too."""
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return
        if isinstance(content, str):
            self._push(content, "", 1)

    def _log(self, messages, text, finish_reason, usage, start) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "role": self.role,
            "model": self.model,
            "messages": messages,
            "response": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "seconds": round(time.monotonic() - start, 2),
        }
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
