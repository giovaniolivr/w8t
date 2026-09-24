# Avaliação rotulada: detectores baseline vs. Kalman

Gerado por `python -m w8t.patterns.evaluation`. 6 cenários × 8 sementes = 48 séries sintéticas de 180 dias (ruído 0.35 kg, 15% de dias faltando, 3 anomalias de 1,5-2,5 kg plantadas por série). Tendência avaliada só com dados até cada ponto de checagem (a cada 7 dias); platô por dia medido; anomalia por ponto plantado. Faixa de estabilidade: ±0,25 kg/semana.

## Geral

| method | trend_correct | trend_wrong | trend_undetermined | plateau_precision | plateau_recall | plateau_f1 | anomaly_recall | anomaly_fp_per_100 |
|---|---|---|---|---|---|---|---|---|
| baseline | 59% | 4% | 38% | 54% | 100% | 70% | 82% | 1.36 |
| kalman | 74% | 3% | 22% | 100% | 99% | 100% | 85% | 0.11 |

## Por cenário

| scenario | method | trend_correct | trend_wrong | trend_undetermined | plateau_precision | plateau_recall | plateau_f1 | anomaly_recall | anomaly_fp_per_100 |
|---|---|---|---|---|---|---|---|---|---|
| bulk | baseline | 51% | 2% | 47% | 0% | — | — | 92% | 1.09 |
| bulk | kalman | 75% | 0% | 25% | — | — | — | 100% | 0.08 |
| cutting rápido | baseline | 100% | 0% | 0% | — | — | — | 88% | 1.51 |
| cutting rápido | kalman | 99% | 0% | 1% | — | — | — | 100% | 0.08 |
| cutting→bulk | baseline | 75% | 8% | 17% | 0% | — | — | 83% | 1.51 |
| cutting→bulk | kalman | 63% | 6% | 31% | — | — | — | 83% | 0.24 |
| cutting→platô | baseline | 49% | 9% | 42% | 87% | 99% | 93% | 83% | 1.42 |
| cutting→platô | kalman | 55% | 10% | 36% | 100% | 99% | 99% | 83% | 0.08 |
| manutenção semanal | baseline | 10% | 2% | 88% | 100% | 100% | 100% | 62% | 1.09 |
| manutenção semanal | kalman | 82% | 1% | 17% | 100% | 100% | 100% | 67% | 0.08 |
| manutenção→cutting | baseline | 67% | 2% | 32% | 76% | 100% | 86% | 83% | 1.51 |
| manutenção→cutting | kalman | 71% | 5% | 24% | 99% | 98% | 99% | 79% | 0.08 |
