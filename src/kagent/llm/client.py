"""OpenAI-compatible chat client (mlx_lm.server). One instance per role.

Every call is appended to a JSONL trajectory log: prompts and responses are the raw
material for later SFT or a skill library.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

Message = dict[str, str]

_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def strip_thinking(text: str) -> str:
    """Drop <think> blocks that reasoning models may emit into the content field."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:  # unterminated: generation was cut off mid-thought
        text = text.split("<think>", 1)[0]
    elif "</think>" in text:  # chat template already opened the block
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


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
        log_path: Path | None = None,
        attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.role = role
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.log_path = log_path
        self._attempts = attempts
        self._sleep = sleep
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        start = time.monotonic()
        data = self._post(payload)
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
                resp = self._http.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            if resp.status_code in _RETRY_STATUS:
                last = f"HTTP {resp.status_code}"
                continue
            if resp.status_code >= 400:
                raise LLMError(f"{self.role}: HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError(f"{self.role}: response is not JSON: {resp.text[:300]}") from exc
        raise LLMError(
            f"{self.role} ({self.model}) unreachable after {self._attempts} tries: {last}"
        )

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
