"""Turn the structured summary into a short natural-language text - and check it.

The LLM only narrates: the system instructions forbid new numbers, causal claims, advice and
certainty about forecasts. Because instructions alone aren't a guarantee, every number in the
reply is checked against the numbers in the summary; anything that doesn't match is reported as
unverified so the UI can flag it instead of silently trusting the text.

``python -m w8t.insights.narrator --demo`` regenerates the fixed demo text (the public demo never
calls the API).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from w8t.insights.providers import LLMProvider

DEMO_TEXT_PATH = Path(__file__).with_name("demo_summary.md")

SYSTEM_PROMPT = """\
Você escreve um resumo curto sobre a evolução do peso de uma pessoa, em português do Brasil, a \
partir EXCLUSIVAMENTE de um JSON com números já calculados por modelos estatísticos.

Regras obrigatórias:
1. Use somente números que aparecem no JSON. Não calcule números novos (nem somas, diferenças, \
médias, conversões ou arredondamentos diferentes). Se não houver o número, não o mencione.
2. Nunca afirme ou sugira causas (dieta, treino, sono, água, hormônios etc.) nem dê conselhos \
médicos, nutricionais ou de treino. Descreva o que os números mostram, não por quê.
3. Previsões são estimativas: sempre apresente com o intervalo de 95% fornecido e sem tom de \
certeza.
4. Se um campo diz "dados insuficientes" ou "não se aplica", diga isso de forma breve; não \
preencha a lacuna com suposições.
5. Refira-se ao tempo como "há N dias" (campos terminados em "_ha_dias") ou "N dias após a \
última medição" (previsões). Não escreva datas.
6. Interprete platôs à luz do objetivo do período, quando houver: em manutenção, estabilidade é o \
esperado; em perda ou ganho, indica estagnação. Sem julgamento de valor.
7. Medições atípicas não são erros; foram mantidas como registradas.
8. Uma tendência "indefinida" significa que o ruído não permite concluir a direção; não force uma \
direção.
9. Formato: 2 a 4 parágrafos curtos, tom neutro e informativo, sem títulos, listas ou emojis. \
Use vírgula decimal (81,5 kg). "Há 0 dias" é "hoje".
10. Priorize: variação no período e tendência atual primeiro; depois platôs, medições atípicas e \
previsão. Detalhes de registro (número de medições, lacunas) só se forem relevantes, em uma frase. \
Não é preciso citar todos os campos, e não mencione listas vazias ou campos ausentes.
"""

NUMBER = re.compile(r"(?<![\w.,])[-+−]?\d+(?:[.,]\d+)?")
TOLERANCE = 0.05  # the text may round 0.27 to 0.3


@dataclass(frozen=True)
class Narrative:
    text: str
    provider: str
    unverified_numbers: list[str]


def build_prompt(summary: dict) -> str:
    return (
        "Escreva o resumo a partir deste JSON (é a única fonte de informação):\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )


def narrate(summary: dict, provider: LLMProvider) -> Narrative:
    text = provider.generate(SYSTEM_PROMPT, build_prompt(summary))
    return Narrative(text, provider.name, unverified_numbers(text, summary))


def _summary_numbers(obj) -> set[float]:
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, int | float):
        found.add(abs(float(obj)))
    elif isinstance(obj, str):
        found |= {abs(_parse(m)) for m in NUMBER.findall(obj)}
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found |= _summary_numbers(k) | _summary_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _summary_numbers(v)
    return found


def _parse(token: str) -> float:
    return float(token.replace("−", "-").replace(",", "."))


def unverified_numbers(text: str, summary: dict) -> list[str]:
    """Numbers in ``text`` that don't match (within rounding) any number in the summary."""
    allowed = _summary_numbers(summary) | {95.0}  # "intervalo de 95%" is part of the vocabulary
    bad = []
    for token in NUMBER.findall(text):
        value = abs(_parse(token))
        if not any(abs(value - a) <= TOLERANCE + 1e-9 for a in allowed):
            bad.append(token)
    return bad


def main() -> None:  # pragma: no cover - CLI, calls the real API
    from w8t.config import settings
    from w8t.core import metrics
    from w8t.data import demo
    from w8t.insights.providers import GeminiProvider
    from w8t.insights.summary import build_summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="regenera o texto fixo da demo")
    args = parser.parse_args()
    if not args.demo:
        parser.error("use --demo")

    today = date.today()  # noqa: DTZ011 - demo data is anchored to the local calendar day
    series = metrics.to_series(demo.generate_series(today))
    summary = build_summary(series, [], None, today)
    result = narrate(summary, GeminiProvider(settings.gemini_api_key))
    DEMO_TEXT_PATH.write_text(result.text + "\n", encoding="utf-8")
    print(result.text)
    print("\nNúmeros não verificados:", result.unverified_numbers or "nenhum")


if __name__ == "__main__":  # pragma: no cover
    main()
