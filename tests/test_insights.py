import json
from datetime import date, timedelta

import numpy as np
import pytest
import requests

from w8t.core import metrics
from w8t.data.models import GoalDirection, Period
from w8t.insights import narrator, providers
from w8t.insights.summary import INSUFFICIENT, build_summary

TODAY = date(2026, 9, 24)


def series_days_ago(values_by_ago: dict[int, float]):
    return metrics.to_series((TODAY - timedelta(days=a), v) for a, v in values_by_ago.items())


def cutting_series(n=90, seed=0):
    rng = np.random.default_rng(seed)
    return series_days_ago(
        {n - 1 - i: round(90 - 0.1 * i + rng.normal(0, 0.3), 1) for i in range(n)}
    )


def period(**kw):
    base = {
        "label": "Cut", "goal_direction": GoalDirection.LOSS,
        "start_date": TODAY - timedelta(days=60), "end_date": None, "target_weight_kg": None,
    }
    return Period(**(base | kw))


# --- summary ----------------------------------------------------------------------------------


def test_summary_uses_relative_days_and_never_the_raw_series():
    s = cutting_series()
    summary = build_summary(s, [], None, TODAY)

    assert summary["registros"]["medicoes"] == len(s)
    assert summary["registros"]["ultima_medicao_ha_dias"] == 0
    assert summary["registros"]["primeira_medicao_ha_dias"] == 89
    assert summary["tendencia_atual"]["direcao"] == "descendo"
    assert [h["dias_apos_a_ultima_medicao"] for h in summary["previsao"]["horizontes"]] == [7, 30]
    # No list in the summary is anywhere near the length of the series.
    text = json.dumps(summary)
    assert "2026-" not in text  # no calendar dates
    assert len(narrator._summary_numbers(summary)) < len(s)


def test_short_history_is_reported_as_insufficient():
    s = series_days_ago({3: 80.0, 2: 80.2, 1: 80.1, 0: 80.0})
    summary = build_summary(s, [], None, TODAY)

    assert summary["tendencia_atual"] == INSUFFICIENT
    assert summary["previsao"].startswith(INSUFFICIENT)
    assert summary["peso"]["media_movel_7d_kg"] == 80.1  # 4 points >= the 3 required


def test_closed_period_has_no_forecast_and_target_is_reported():
    s = cutting_series()
    closed = period(end_date=TODAY - timedelta(days=10), target_weight_kg=83.0)
    summary = build_summary(s, [closed], closed, TODAY)

    assert summary["escopo"]["objetivo"] == "perda de peso"
    assert summary["escopo"]["situacao"] == "encerrado há 10 dias"
    assert summary["previsao"] == "não se aplica: período encerrado"
    assert summary["meta"]["meta_kg"] == 83.0


def test_empty_scope():
    s = cutting_series()
    future = period(start_date=TODAY + timedelta(days=5))
    assert build_summary(s, [], future, TODAY)["registros"] == INSUFFICIENT


# --- grounding check --------------------------------------------------------------------------


def test_unverified_numbers_catch_invented_values_only():
    summary = {"peso": {"atual_kg": 81.5, "variacao_kg": -6.6}, "tendencia": {"ritmo": 0.27}}
    text = (
        "O peso atual é 81,5 kg, uma variação de -6,6 kg; ritmo de 0,3 kg/semana "
        "(intervalo de 95%). Em 12 semanas seriam 3,2 kg."
    )
    assert narrator.unverified_numbers(text, summary) == ["12", "3,2"]


def test_numbers_inside_summary_strings_count_as_known():
    summary = {"situacao": "encerrado há 10 dias"}
    assert narrator.unverified_numbers("Encerrado há 10 dias.", summary) == []


def test_narrate_sends_rules_and_only_the_summary():
    summary = {"peso": {"atual_kg": 81.5}}
    fake = providers.FakeProvider("O peso atual é 81,5 kg.")
    result = narrator.narrate(summary, fake)

    system, prompt = fake.calls[0]
    assert "Nunca afirme ou sugira causas" in system
    assert json.dumps(summary, ensure_ascii=False, indent=2) in prompt
    assert result.unverified_numbers == []
    assert result.provider == "fake"


# --- Gemini provider (network mocked) ---------------------------------------------------------


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


def _patch_post(monkeypatch, response):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(providers.requests, "post", fake_post)
    return sent


def test_gemini_success_and_request_shape(monkeypatch):
    payload = {"candidates": [{"content": {"parts": [{"text": "Olá "}, {"text": "mundo"}]}}]}
    sent = _patch_post(monkeypatch, _Resp(200, payload))

    text = providers.GeminiProvider("SECRET", "m-1").generate("sys", "prompt")

    assert text == "Olá mundo"
    assert sent["url"].endswith("/models/m-1:generateContent")
    assert sent["headers"]["x-goog-api-key"] == "SECRET"
    assert sent["json"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert sent["json"]["system_instruction"]["parts"][0]["text"] == "sys"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_Resp(429, text="quota"), "Limite gratuito"),
        (_Resp(403, text="bad key SECRET here"), "403"),
        (_Resp(200, {"candidates": []}), "não retornou texto"),
        (_Resp(200, {"candidates": [{"finishReason": "SAFETY", "content": {}}]}), "SAFETY"),
        (requests.ConnectionError("down"), "Falha de rede"),
    ],
)
def test_gemini_errors_become_provider_errors(monkeypatch, response, message):
    _patch_post(monkeypatch, response)
    with pytest.raises(providers.ProviderError, match=message) as exc:
        providers.GeminiProvider("SECRET").generate("s", "p")
    assert "SECRET" not in str(exc.value)  # the key never leaks into messages


def test_gemini_requires_key():
    with pytest.raises(providers.ProviderError, match="GEMINI_API_KEY"):
        providers.GeminiProvider("")


def test_demo_text_exists_and_is_grounded_in_the_demo_summary():
    from w8t.data import demo

    s = metrics.to_series(demo.generate_series(TODAY))
    summary = build_summary(s, [], None, TODAY)
    text = narrator.DEMO_TEXT_PATH.read_text(encoding="utf-8")

    assert len(text) > 200
    assert narrator.unverified_numbers(text, summary) == []
