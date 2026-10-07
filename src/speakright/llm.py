"""Optional LLM tip writer. It only rewrites tips from the computed result. It never gives a score.

The reply must be JSON `{"tips": [str, ...]}` with 1 to 5 tips of at most 240 characters each.
Any other reply, an error or a missing key gives the deterministic tips. Nothing is logged.
"""
from __future__ import annotations

import json
from typing import Protocol

SYSTEM = ("You are a pronunciation coach. You get a JSON assessment that a program computed. "
          "Write 1 to 5 short, concrete practice tips for the listed errors only. "
          "Do not give a score. Reply only with JSON: {\"tips\": [\"...\"]}.")


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:  # pragma: no cover - network
    """Any chat-completions server: OpenAI, DeepSeek, a local Ollama or vLLM server."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout_s: float = 20.0):
        import httpx

        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s,
                                   headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        self.model = model
        self.name = f"openai-compatible:{model}"

    def complete(self, system: str, user: str) -> str:
        r = self.client.post("/chat/completions", json={
            "model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}")
        return r.json()["choices"][0]["message"]["content"]


class FakeLLM:
    """Returns a fixed reply and counts the calls (tests)."""

    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def parse_tips(reply: str) -> list[str]:
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMError("reply is not JSON") from exc
    tips = data.get("tips") if isinstance(data, dict) else None
    if not isinstance(tips, list) or not 1 <= len(tips) <= 5:
        raise LLMError("reply must have 1 to 5 tips")
    if not all(isinstance(t, str) and 0 < len(t.strip()) <= 240 for t in tips):
        raise LLMError("each tip must be a non-empty string of at most 240 characters")
    return [t.strip() for t in tips]


def build_llm(base_url: str | None, api_key: str | None, model: str | None) -> LLM | None:
    if not base_url or not model:
        return None
    return OpenAICompatibleLLM(base_url, api_key, model)
