# Benchmark multi-série de previsão

Gerado por `python -m w8t.forecasting.benchmark`. 6 cenários sintéticos × 5 sementes = 30 séries independentes de 180 dias; walk-forward a cada 3 dias; métricas nos casos comuns a todos os modelos em cada série. Cobertura = fração dos valores reais dentro do intervalo de 95%.

## Geral (média entre séries)

| horizon | model | mae | coverage | mean_rank | series |
|---|---|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.350 | 95% | 2.63 | 30 |
| 1 | GPR (linear + RBF, 60d) | 0.356 | 94% | 2.30 | 30 |
| 1 | Regressão linear 28d | 0.359 | 96% | 3.07 | 30 |
| 1 | Holt amortecido | 0.360 | 95% | 3.03 | 30 |
| 1 | Média móvel 7d | 0.390 | 93% | 4.07 | 30 |
| 1 | Último valor | 0.488 | 94% | 5.90 | 30 |
| 7 | Kalman (tendência suave) | 0.396 | 93% | 2.17 | 30 |
| 7 | Regressão linear 28d | 0.415 | 93% | 2.80 | 30 |
| 7 | GPR (linear + RBF, 60d) | 0.417 | 92% | 2.60 | 30 |
| 7 | Holt amortecido | 0.418 | 95% | 2.93 | 30 |
| 7 | Média móvel 7d | 0.561 | 80% | 4.97 | 30 |
| 7 | Último valor | 0.576 | 100% | 5.53 | 30 |
| 14 | Kalman (tendência suave) | 0.420 | 93% | 2.20 | 30 |
| 14 | Regressão linear 28d | 0.435 | 93% | 2.40 | 30 |
| 14 | Holt amortecido | 0.456 | 95% | 2.97 | 30 |
| 14 | GPR (linear + RBF, 60d) | 0.465 | 92% | 2.93 | 30 |
| 14 | Último valor | 0.731 | 100% | 5.17 | 30 |
| 14 | Média móvel 7d | 0.768 | 67% | 5.33 | 30 |
| 30 | Kalman (tendência suave) | 0.671 | 87% | 2.47 | 30 |
| 30 | Regressão linear 28d | 0.709 | 86% | 2.63 | 30 |
| 30 | Holt amortecido | 0.737 | 93% | 2.63 | 30 |
| 30 | GPR (linear + RBF, 60d) | 0.763 | 82% | 2.83 | 30 |
| 30 | Último valor | 1.240 | 100% | 5.17 | 30 |
| 30 | Média móvel 7d | 1.305 | 42% | 5.27 | 30 |

## Melhor modelo vs. demais (Wilcoxon pareado entre séries, Holm por horizonte)

| horizon | best | other | series | mean_mae_diff | best_wins_share | p_value | p_holm |
|---|---|---|---|---|---|---|---|
| 1 | Kalman (tendência suave) | GPR (linear + RBF, 60d) | 30 | -0.006 | 43% | 0.7766 | 0.7766 |
| 1 | Kalman (tendência suave) | Holt amortecido | 30 | -0.010 | 60% | 0.0577 | 0.1154 |
| 1 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.040 | 73% | 0.0001 | 0.0004 |
| 1 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.009 | 63% | 0.0310 | 0.0930 |
| 1 | Kalman (tendência suave) | Último valor | 30 | -0.138 | 97% | 0.0000 | 0.0000 |
| 7 | Kalman (tendência suave) | GPR (linear + RBF, 60d) | 30 | -0.021 | 60% | 0.0473 | 0.0473 |
| 7 | Kalman (tendência suave) | Holt amortecido | 30 | -0.021 | 67% | 0.0137 | 0.0273 |
| 7 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.164 | 93% | 0.0000 | 0.0000 |
| 7 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.019 | 63% | 0.0032 | 0.0097 |
| 7 | Kalman (tendência suave) | Último valor | 30 | -0.180 | 100% | 0.0000 | 0.0000 |
| 14 | Kalman (tendência suave) | GPR (linear + RBF, 60d) | 30 | -0.045 | 63% | 0.0054 | 0.0108 |
| 14 | Kalman (tendência suave) | Holt amortecido | 30 | -0.036 | 70% | 0.0030 | 0.0090 |
| 14 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.348 | 93% | 0.0000 | 0.0000 |
| 14 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.015 | 57% | 0.1840 | 0.1840 |
| 14 | Kalman (tendência suave) | Último valor | 30 | -0.311 | 97% | 0.0000 | 0.0000 |
| 30 | Kalman (tendência suave) | GPR (linear + RBF, 60d) | 30 | -0.092 | 67% | 0.0093 | 0.0279 |
| 30 | Kalman (tendência suave) | Holt amortecido | 30 | -0.067 | 50% | 0.2286 | 0.3412 |
| 30 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.634 | 83% | 0.0000 | 0.0000 |
| 30 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.038 | 53% | 0.1706 | 0.3412 |
| 30 | Kalman (tendência suave) | Último valor | 30 | -0.570 | 100% | 0.0000 | 0.0000 |

## Por cenário

### cutting→platô

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Regressão linear 28d | 0.293 | 96% |
| 1 | GPR (linear + RBF, 60d) | 0.296 | 94% |
| 1 | Kalman (tendência suave) | 0.300 | 95% |
| 1 | Holt amortecido | 0.302 | 95% |
| 1 | Média móvel 7d | 0.350 | 92% |
| 1 | Último valor | 0.415 | 94% |
| 7 | GPR (linear + RBF, 60d) | 0.361 | 92% |
| 7 | Regressão linear 28d | 0.366 | 91% |
| 7 | Holt amortecido | 0.377 | 94% |
| 7 | Kalman (tendência suave) | 0.381 | 90% |
| 7 | Último valor | 0.529 | 100% |
| 7 | Média móvel 7d | 0.539 | 73% |
| 14 | Regressão linear 28d | 0.405 | 91% |
| 14 | GPR (linear + RBF, 60d) | 0.413 | 91% |
| 14 | Holt amortecido | 0.425 | 96% |
| 14 | Kalman (tendência suave) | 0.430 | 89% |
| 14 | Último valor | 0.688 | 100% |
| 14 | Média móvel 7d | 0.742 | 60% |
| 30 | GPR (linear + RBF, 60d) | 0.728 | 78% |
| 30 | Holt amortecido | 0.735 | 93% |
| 30 | Regressão linear 28d | 0.837 | 69% |
| 30 | Kalman (tendência suave) | 0.887 | 72% |
| 30 | Último valor | 1.144 | 100% |
| 30 | Média móvel 7d | 1.222 | 51% |

### bulk

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.320 | 95% |
| 1 | GPR (linear + RBF, 60d) | 0.325 | 94% |
| 1 | Holt amortecido | 0.328 | 94% |
| 1 | Regressão linear 28d | 0.330 | 95% |
| 1 | Média móvel 7d | 0.341 | 93% |
| 1 | Último valor | 0.467 | 95% |
| 7 | Kalman (tendência suave) | 0.364 | 94% |
| 7 | Holt amortecido | 0.381 | 96% |
| 7 | Regressão linear 28d | 0.391 | 95% |
| 7 | GPR (linear + RBF, 60d) | 0.398 | 92% |
| 7 | Média móvel 7d | 0.479 | 84% |
| 7 | Último valor | 0.555 | 100% |
| 14 | Kalman (tendência suave) | 0.336 | 97% |
| 14 | Regressão linear 28d | 0.346 | 97% |
| 14 | Holt amortecido | 0.371 | 96% |
| 14 | GPR (linear + RBF, 60d) | 0.381 | 94% |
| 14 | Último valor | 0.642 | 100% |
| 14 | Média móvel 7d | 0.663 | 75% |
| 30 | Kalman (tendência suave) | 0.407 | 96% |
| 30 | Regressão linear 28d | 0.446 | 97% |
| 30 | GPR (linear + RBF, 60d) | 0.594 | 85% |
| 30 | Holt amortecido | 0.643 | 97% |
| 30 | Último valor | 1.179 | 100% |
| 30 | Média móvel 7d | 1.292 | 20% |

### manutenção semanal

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | GPR (linear + RBF, 60d) | 0.300 | 95% |
| 1 | Holt amortecido | 0.305 | 95% |
| 1 | Média móvel 7d | 0.306 | 94% |
| 1 | Kalman (tendência suave) | 0.307 | 95% |
| 1 | Regressão linear 28d | 0.321 | 97% |
| 1 | Último valor | 0.404 | 96% |
| 7 | GPR (linear + RBF, 60d) | 0.307 | 94% |
| 7 | Kalman (tendência suave) | 0.312 | 96% |
| 7 | Holt amortecido | 0.312 | 95% |
| 7 | Média móvel 7d | 0.320 | 95% |
| 7 | Regressão linear 28d | 0.331 | 96% |
| 7 | Último valor | 0.382 | 100% |
| 14 | GPR (linear + RBF, 60d) | 0.307 | 95% |
| 14 | Média móvel 7d | 0.309 | 97% |
| 14 | Kalman (tendência suave) | 0.311 | 96% |
| 14 | Holt amortecido | 0.314 | 95% |
| 14 | Regressão linear 28d | 0.319 | 99% |
| 14 | Último valor | 0.340 | 100% |
| 30 | GPR (linear + RBF, 60d) | 0.313 | 95% |
| 30 | Média móvel 7d | 0.318 | 97% |
| 30 | Holt amortecido | 0.330 | 97% |
| 30 | Kalman (tendência suave) | 0.360 | 96% |
| 30 | Regressão linear 28d | 0.421 | 97% |
| 30 | Último valor | 0.470 | 100% |

### esparso c/ gaps longos

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.305 | 94% |
| 1 | GPR (linear + RBF, 60d) | 0.313 | 92% |
| 1 | Regressão linear 28d | 0.320 | 96% |
| 1 | Holt amortecido | 0.327 | 96% |
| 1 | Média móvel 7d | 0.351 | 94% |
| 1 | Último valor | 0.412 | 93% |
| 7 | Kalman (tendência suave) | 0.262 | 96% |
| 7 | Regressão linear 28d | 0.292 | 95% |
| 7 | GPR (linear + RBF, 60d) | 0.293 | 93% |
| 7 | Holt amortecido | 0.317 | 96% |
| 7 | Último valor | 0.466 | 100% |
| 7 | Média móvel 7d | 0.604 | 73% |
| 14 | Kalman (tendência suave) | 0.296 | 94% |
| 14 | GPR (linear + RBF, 60d) | 0.379 | 91% |
| 14 | Regressão linear 28d | 0.384 | 93% |
| 14 | Holt amortecido | 0.392 | 95% |
| 14 | Último valor | 0.857 | 100% |
| 14 | Média móvel 7d | 1.000 | 36% |
| 30 | Kalman (tendência suave) | 0.323 | 95% |
| 30 | Regressão linear 28d | 0.442 | 97% |
| 30 | GPR (linear + RBF, 60d) | 0.489 | 88% |
| 30 | Holt amortecido | 0.662 | 97% |
| 30 | Último valor | 1.720 | 100% |
| 30 | Média móvel 7d | 1.912 | 0% |

### cutting→bulk

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | GPR (linear + RBF, 60d) | 0.301 | 94% |
| 1 | Kalman (tendência suave) | 0.309 | 95% |
| 1 | Holt amortecido | 0.310 | 95% |
| 1 | Regressão linear 28d | 0.314 | 96% |
| 1 | Média móvel 7d | 0.382 | 93% |
| 1 | Último valor | 0.414 | 94% |
| 7 | Holt amortecido | 0.425 | 92% |
| 7 | Kalman (tendência suave) | 0.426 | 90% |
| 7 | GPR (linear + RBF, 60d) | 0.433 | 89% |
| 7 | Regressão linear 28d | 0.439 | 88% |
| 7 | Último valor | 0.613 | 100% |
| 7 | Média móvel 7d | 0.717 | 64% |
| 14 | Regressão linear 28d | 0.552 | 83% |
| 14 | Holt amortecido | 0.555 | 90% |
| 14 | Kalman (tendência suave) | 0.557 | 87% |
| 14 | GPR (linear + RBF, 60d) | 0.622 | 84% |
| 14 | Último valor | 0.965 | 100% |
| 14 | Média móvel 7d | 1.129 | 42% |
| 30 | Holt amortecido | 1.178 | 80% |
| 30 | Regressão linear 28d | 1.332 | 62% |
| 30 | Kalman (tendência suave) | 1.335 | 68% |
| 30 | GPR (linear + RBF, 60d) | 1.587 | 57% |
| 30 | Último valor | 1.795 | 100% |
| 30 | Média móvel 7d | 1.975 | 11% |

### perda lenta ruidosa

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.560 | 95% |
| 1 | Regressão linear 28d | 0.575 | 95% |
| 1 | Holt amortecido | 0.590 | 95% |
| 1 | GPR (linear + RBF, 60d) | 0.601 | 94% |
| 1 | Média móvel 7d | 0.608 | 94% |
| 1 | Último valor | 0.816 | 94% |
| 7 | Kalman (tendência suave) | 0.633 | 94% |
| 7 | Regressão linear 28d | 0.676 | 95% |
| 7 | Holt amortecido | 0.695 | 94% |
| 7 | Média móvel 7d | 0.705 | 92% |
| 7 | GPR (linear + RBF, 60d) | 0.712 | 93% |
| 7 | Último valor | 0.912 | 100% |
| 14 | Kalman (tendência suave) | 0.589 | 97% |
| 14 | Regressão linear 28d | 0.604 | 97% |
| 14 | Holt amortecido | 0.677 | 95% |
| 14 | GPR (linear + RBF, 60d) | 0.687 | 94% |
| 14 | Média móvel 7d | 0.764 | 92% |
| 14 | Último valor | 0.893 | 100% |
| 30 | Kalman (tendência suave) | 0.712 | 96% |
| 30 | Regressão linear 28d | 0.775 | 97% |
| 30 | GPR (linear + RBF, 60d) | 0.867 | 89% |
| 30 | Holt amortecido | 0.876 | 92% |
| 30 | Média móvel 7d | 1.110 | 73% |
| 30 | Último valor | 1.134 | 100% |
