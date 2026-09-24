# Benchmark multi-série de previsão

Gerado por `python -m w8t.forecasting.benchmark`. 6 cenários sintéticos × 5 sementes = 30 séries independentes de 180 dias; walk-forward a cada 3 dias; métricas nos casos comuns a todos os modelos em cada série. Cobertura = fração dos valores reais dentro do intervalo de 95%.

## Geral (média entre séries)

| horizon | model | mae | coverage | mean_rank | series |
|---|---|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.345 | 95% | 2.70 | 30 |
| 1 | Combinação Kalman + Holt | 0.347 | 95% | 2.70 | 30 |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.352 | 94% | 3.20 | 30 |
| 1 | Regressão linear 28d | 0.359 | 96% | 3.67 | 30 |
| 1 | Holt amortecido | 0.360 | 95% | 3.87 | 30 |
| 1 | Média móvel 7d | 0.390 | 93% | 4.97 | 30 |
| 1 | Último valor | 0.488 | 94% | 6.90 | 30 |
| 7 | Kalman (tendência suave) | 0.392 | 93% | 2.33 | 30 |
| 7 | Combinação Kalman + Holt | 0.396 | 95% | 2.23 | 30 |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.411 | 93% | 3.23 | 30 |
| 7 | Regressão linear 28d | 0.415 | 93% | 3.73 | 30 |
| 7 | Holt amortecido | 0.418 | 95% | 3.90 | 30 |
| 7 | Média móvel 7d | 0.561 | 80% | 6.03 | 30 |
| 7 | Último valor | 0.576 | 100% | 6.53 | 30 |
| 14 | Combinação Kalman + Holt | 0.413 | 96% | 2.00 | 30 |
| 14 | Kalman (tendência suave) | 0.413 | 94% | 2.43 | 30 |
| 14 | Regressão linear 28d | 0.435 | 93% | 3.23 | 30 |
| 14 | Holt amortecido | 0.456 | 95% | 4.00 | 30 |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.459 | 92% | 3.80 | 30 |
| 14 | Último valor | 0.731 | 100% | 6.20 | 30 |
| 14 | Média móvel 7d | 0.768 | 67% | 6.33 | 30 |
| 30 | Combinação Kalman + Holt | 0.643 | 95% | 2.30 | 30 |
| 30 | Kalman (tendência suave) | 0.668 | 86% | 2.87 | 30 |
| 30 | Regressão linear 28d | 0.709 | 86% | 3.43 | 30 |
| 30 | Holt amortecido | 0.737 | 93% | 3.50 | 30 |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.738 | 85% | 3.47 | 30 |
| 30 | Último valor | 1.240 | 100% | 6.17 | 30 |
| 30 | Média móvel 7d | 1.305 | 42% | 6.27 | 30 |

## Melhor modelo vs. demais (Wilcoxon pareado entre séries, Holm por horizonte)

| horizon | best | other | series | mean_mae_diff | best_wins_share | p_value | p_holm |
|---|---|---|---|---|---|---|---|
| 1 | Kalman (tendência suave) | Combinação Kalman + Holt | 30 | -0.002 | 57% | 0.5699 | 0.5699 |
| 1 | Kalman (tendência suave) | GPR (linear + Matérn 3/2, 90d) | 30 | -0.007 | 57% | 0.1706 | 0.3412 |
| 1 | Kalman (tendência suave) | Holt amortecido | 30 | -0.015 | 73% | 0.0043 | 0.0174 |
| 1 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.044 | 83% | 0.0000 | 0.0000 |
| 1 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.014 | 63% | 0.0128 | 0.0385 |
| 1 | Kalman (tendência suave) | Último valor | 30 | -0.143 | 97% | 0.0000 | 0.0000 |
| 7 | Kalman (tendência suave) | Combinação Kalman + Holt | 30 | -0.004 | 53% | 0.5838 | 0.5838 |
| 7 | Kalman (tendência suave) | GPR (linear + Matérn 3/2, 90d) | 30 | -0.019 | 73% | 0.0007 | 0.0020 |
| 7 | Kalman (tendência suave) | Holt amortecido | 30 | -0.025 | 73% | 0.0032 | 0.0064 |
| 7 | Kalman (tendência suave) | Média móvel 7d | 30 | -0.168 | 97% | 0.0000 | 0.0000 |
| 7 | Kalman (tendência suave) | Regressão linear 28d | 30 | -0.023 | 70% | 0.0004 | 0.0017 |
| 7 | Kalman (tendência suave) | Último valor | 30 | -0.184 | 100% | 0.0000 | 0.0000 |
| 14 | Combinação Kalman + Holt | GPR (linear + Matérn 3/2, 90d) | 30 | -0.046 | 83% | 0.0000 | 0.0001 |
| 14 | Combinação Kalman + Holt | Holt amortecido | 30 | -0.043 | 93% | 0.0000 | 0.0000 |
| 14 | Combinação Kalman + Holt | Kalman (tendência suave) | 30 | -0.000 | 57% | 0.6850 | 0.6850 |
| 14 | Combinação Kalman + Holt | Média móvel 7d | 30 | -0.355 | 97% | 0.0000 | 0.0000 |
| 14 | Combinação Kalman + Holt | Regressão linear 28d | 30 | -0.022 | 70% | 0.0145 | 0.0291 |
| 14 | Combinação Kalman + Holt | Último valor | 30 | -0.318 | 100% | 0.0000 | 0.0000 |
| 30 | Combinação Kalman + Holt | GPR (linear + Matérn 3/2, 90d) | 30 | -0.095 | 70% | 0.0310 | 0.0620 |
| 30 | Combinação Kalman + Holt | Holt amortecido | 30 | -0.094 | 77% | 0.0004 | 0.0015 |
| 30 | Combinação Kalman + Holt | Kalman (tendência suave) | 30 | -0.025 | 57% | 0.2801 | 0.2801 |
| 30 | Combinação Kalman + Holt | Média móvel 7d | 30 | -0.662 | 93% | 0.0000 | 0.0000 |
| 30 | Combinação Kalman + Holt | Regressão linear 28d | 30 | -0.066 | 73% | 0.0006 | 0.0018 |
| 30 | Combinação Kalman + Holt | Último valor | 30 | -0.597 | 100% | 0.0000 | 0.0000 |

## Por cenário

### cutting→platô

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Regressão linear 28d | 0.293 | 96% |
| 1 | Combinação Kalman + Holt | 0.295 | 95% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.299 | 95% |
| 1 | Kalman (tendência suave) | 0.301 | 95% |
| 1 | Holt amortecido | 0.302 | 95% |
| 1 | Média móvel 7d | 0.350 | 92% |
| 1 | Último valor | 0.415 | 94% |
| 7 | Regressão linear 28d | 0.366 | 91% |
| 7 | Combinação Kalman + Holt | 0.372 | 94% |
| 7 | Holt amortecido | 0.377 | 94% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.379 | 91% |
| 7 | Kalman (tendência suave) | 0.381 | 90% |
| 7 | Último valor | 0.529 | 100% |
| 7 | Média móvel 7d | 0.539 | 73% |
| 14 | Regressão linear 28d | 0.405 | 91% |
| 14 | Combinação Kalman + Holt | 0.410 | 96% |
| 14 | Holt amortecido | 0.425 | 96% |
| 14 | Kalman (tendência suave) | 0.431 | 89% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.449 | 90% |
| 14 | Último valor | 0.688 | 100% |
| 14 | Média móvel 7d | 0.742 | 60% |
| 30 | Holt amortecido | 0.735 | 93% |
| 30 | Combinação Kalman + Holt | 0.745 | 95% |
| 30 | Regressão linear 28d | 0.837 | 69% |
| 30 | Kalman (tendência suave) | 0.888 | 72% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.922 | 73% |
| 30 | Último valor | 1.144 | 100% |
| 30 | Média móvel 7d | 1.222 | 51% |

### bulk

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.321 | 95% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.322 | 95% |
| 1 | Combinação Kalman + Holt | 0.323 | 95% |
| 1 | Holt amortecido | 0.328 | 94% |
| 1 | Regressão linear 28d | 0.330 | 95% |
| 1 | Média móvel 7d | 0.341 | 93% |
| 1 | Último valor | 0.467 | 95% |
| 7 | Kalman (tendência suave) | 0.364 | 94% |
| 7 | Combinação Kalman + Holt | 0.366 | 95% |
| 7 | Holt amortecido | 0.381 | 96% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.385 | 93% |
| 7 | Regressão linear 28d | 0.391 | 95% |
| 7 | Média móvel 7d | 0.479 | 84% |
| 7 | Último valor | 0.555 | 100% |
| 14 | Combinação Kalman + Holt | 0.332 | 97% |
| 14 | Kalman (tendência suave) | 0.337 | 97% |
| 14 | Regressão linear 28d | 0.346 | 97% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.364 | 95% |
| 14 | Holt amortecido | 0.371 | 96% |
| 14 | Último valor | 0.642 | 100% |
| 14 | Média móvel 7d | 0.663 | 75% |
| 30 | Kalman (tendência suave) | 0.408 | 95% |
| 30 | Regressão linear 28d | 0.446 | 97% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.456 | 90% |
| 30 | Combinação Kalman + Holt | 0.472 | 99% |
| 30 | Holt amortecido | 0.643 | 97% |
| 30 | Último valor | 1.179 | 100% |
| 30 | Média móvel 7d | 1.292 | 20% |

### manutenção semanal

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Combinação Kalman + Holt | 0.273 | 95% |
| 1 | Kalman (tendência suave) | 0.274 | 94% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.299 | 94% |
| 1 | Holt amortecido | 0.305 | 95% |
| 1 | Média móvel 7d | 0.306 | 94% |
| 1 | Regressão linear 28d | 0.321 | 97% |
| 1 | Último valor | 0.404 | 96% |
| 7 | Kalman (tendência suave) | 0.286 | 94% |
| 7 | Combinação Kalman + Holt | 0.286 | 96% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.308 | 95% |
| 7 | Holt amortecido | 0.312 | 95% |
| 7 | Média móvel 7d | 0.320 | 95% |
| 7 | Regressão linear 28d | 0.331 | 96% |
| 7 | Último valor | 0.382 | 100% |
| 14 | Kalman (tendência suave) | 0.267 | 97% |
| 14 | Combinação Kalman + Holt | 0.277 | 97% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.307 | 96% |
| 14 | Média móvel 7d | 0.309 | 97% |
| 14 | Holt amortecido | 0.314 | 95% |
| 14 | Regressão linear 28d | 0.319 | 99% |
| 14 | Último valor | 0.340 | 100% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.316 | 96% |
| 30 | Média móvel 7d | 0.318 | 97% |
| 30 | Combinação Kalman + Holt | 0.321 | 97% |
| 30 | Holt amortecido | 0.330 | 97% |
| 30 | Kalman (tendência suave) | 0.337 | 92% |
| 30 | Regressão linear 28d | 0.421 | 97% |
| 30 | Último valor | 0.470 | 100% |

### esparso c/ gaps longos

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.304 | 94% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.305 | 92% |
| 1 | Combinação Kalman + Holt | 0.313 | 97% |
| 1 | Regressão linear 28d | 0.320 | 96% |
| 1 | Holt amortecido | 0.327 | 96% |
| 1 | Média móvel 7d | 0.351 | 94% |
| 1 | Último valor | 0.412 | 93% |
| 7 | Kalman (tendência suave) | 0.263 | 95% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.272 | 95% |
| 7 | Combinação Kalman + Holt | 0.275 | 96% |
| 7 | Regressão linear 28d | 0.292 | 95% |
| 7 | Holt amortecido | 0.317 | 96% |
| 7 | Último valor | 0.466 | 100% |
| 7 | Média móvel 7d | 0.604 | 73% |
| 14 | Kalman (tendência suave) | 0.296 | 94% |
| 14 | Combinação Kalman + Holt | 0.308 | 96% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.327 | 92% |
| 14 | Regressão linear 28d | 0.384 | 93% |
| 14 | Holt amortecido | 0.392 | 95% |
| 14 | Último valor | 0.857 | 100% |
| 14 | Média móvel 7d | 1.000 | 36% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.330 | 92% |
| 30 | Kalman (tendência suave) | 0.330 | 95% |
| 30 | Combinação Kalman + Holt | 0.407 | 100% |
| 30 | Regressão linear 28d | 0.442 | 97% |
| 30 | Holt amortecido | 0.662 | 97% |
| 30 | Último valor | 1.720 | 100% |
| 30 | Média móvel 7d | 1.912 | 0% |

### cutting→bulk

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Combinação Kalman + Holt | 0.307 | 94% |
| 1 | Kalman (tendência suave) | 0.310 | 95% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.310 | 95% |
| 1 | Holt amortecido | 0.310 | 95% |
| 1 | Regressão linear 28d | 0.314 | 96% |
| 1 | Média móvel 7d | 0.382 | 93% |
| 1 | Último valor | 0.414 | 94% |
| 7 | Combinação Kalman + Holt | 0.420 | 92% |
| 7 | Holt amortecido | 0.425 | 92% |
| 7 | Kalman (tendência suave) | 0.427 | 90% |
| 7 | Regressão linear 28d | 0.439 | 88% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.469 | 91% |
| 7 | Último valor | 0.613 | 100% |
| 7 | Média móvel 7d | 0.717 | 64% |
| 14 | Combinação Kalman + Holt | 0.539 | 91% |
| 14 | Regressão linear 28d | 0.552 | 83% |
| 14 | Holt amortecido | 0.555 | 90% |
| 14 | Kalman (tendência suave) | 0.557 | 87% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.679 | 86% |
| 14 | Último valor | 0.965 | 100% |
| 14 | Média móvel 7d | 1.129 | 42% |
| 30 | Holt amortecido | 1.178 | 80% |
| 30 | Combinação Kalman + Holt | 1.187 | 84% |
| 30 | Regressão linear 28d | 1.332 | 62% |
| 30 | Kalman (tendência suave) | 1.335 | 67% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 1.667 | 66% |
| 30 | Último valor | 1.795 | 100% |
| 30 | Média móvel 7d | 1.975 | 11% |

### perda lenta ruidosa

| horizon | model | mae | coverage |
|---|---|---|---|
| 1 | Kalman (tendência suave) | 0.563 | 95% |
| 1 | Combinação Kalman + Holt | 0.571 | 95% |
| 1 | Regressão linear 28d | 0.575 | 95% |
| 1 | GPR (linear + Matérn 3/2, 90d) | 0.579 | 95% |
| 1 | Holt amortecido | 0.590 | 95% |
| 1 | Média móvel 7d | 0.608 | 94% |
| 1 | Último valor | 0.816 | 94% |
| 7 | Kalman (tendência suave) | 0.633 | 94% |
| 7 | GPR (linear + Matérn 3/2, 90d) | 0.656 | 94% |
| 7 | Combinação Kalman + Holt | 0.657 | 95% |
| 7 | Regressão linear 28d | 0.676 | 95% |
| 7 | Holt amortecido | 0.695 | 94% |
| 7 | Média móvel 7d | 0.705 | 92% |
| 7 | Último valor | 0.912 | 100% |
| 14 | Kalman (tendência suave) | 0.590 | 97% |
| 14 | Regressão linear 28d | 0.604 | 97% |
| 14 | Combinação Kalman + Holt | 0.610 | 97% |
| 14 | GPR (linear + Matérn 3/2, 90d) | 0.625 | 95% |
| 14 | Holt amortecido | 0.677 | 95% |
| 14 | Média móvel 7d | 0.764 | 92% |
| 14 | Último valor | 0.893 | 100% |
| 30 | Kalman (tendência suave) | 0.713 | 95% |
| 30 | Combinação Kalman + Holt | 0.728 | 98% |
| 30 | GPR (linear + Matérn 3/2, 90d) | 0.739 | 92% |
| 30 | Regressão linear 28d | 0.775 | 97% |
| 30 | Holt amortecido | 0.876 | 92% |
| 30 | Média móvel 7d | 1.110 | 73% |
| 30 | Último valor | 1.134 | 100% |
