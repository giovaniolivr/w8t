"""LLM providers behind one tiny interface, so the provider is configuration, not code.

Gemini (Google AI Studio free tier) via its REST API with ``requests`` - no SDK dependency.
The Anthropic API was the original plan (spec) but has no free tier; switched on 2026-09-24.
"""

from __future__ import annotations

from typing import Protocol

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class ProviderError(Exception):
    """The provider couldn't produce text (network, quota, auth, blocked...)."""


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...


class GeminiProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL, timeout: float = 60):
        if not api_key:
            raise ProviderError("GEMINI_API_KEY não configurada no .env.")
        self._key = api_key
        self.model = model
        self.name = f"Gemini ({model})"
        self._timeout = timeout

    def generate(self, system: str, prompt: str) -> str:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
                # Narrating precomputed numbers needs no reasoning; thinking tokens would also
                # eat the output budget (observed: empty reply with a small maxOutputTokens).
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            r = requests.post(
                GEMINI_URL.format(model=self.model),
                headers={"x-goog-api-key": self._key},
                json=body,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Falha de rede ao chamar o Gemini: {type(exc).__name__}") from exc
        if r.status_code == 429:
            raise ProviderError("Limite gratuito do Gemini atingido; tente mais tarde.")
        if not r.ok:
            detail = r.text[:200].replace(self._key, "***")
            raise ProviderError(f"Gemini respondeu {r.status_code}: {detail}")
        candidates = r.json().get("candidates") or []
        if not candidates:
            raise ProviderError("Gemini não retornou texto (resposta vazia ou bloqueada).")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            reason = candidates[0].get("finishReason", "desconhecido")
            raise ProviderError(f"Gemini não retornou texto (motivo: {reason}).")
        return text


class FakeProvider:
    """Deterministic stand-in for tests: returns a canned text and records what it was sent."""

    name = "fake"

    def __init__(self, reply: str = "Resumo de teste."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply
