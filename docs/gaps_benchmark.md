# Avaliação rotulada: reconstrução de lacunas

Gerado por `python -m w8t.patterns.gaps_evaluation`. 48 séries sintéticas (6 cenários × 8 sementes); em cada uma, 2 buracos artificiais de 3, 7, 14 dias (medições reais apagadas). Cobertura = fração das medições apagadas dentro do intervalo de 95% (alvo ~95%). *mae_vs_true_level* compara com o peso verdadeiro sem ruído. Anomalias plantadas não são pontuadas.

## Por método

| method | points | mae_vs_measurement | coverage | mean_width | mae_vs_true_level | failed_gaps |
|---|---|---|---|---|---|---|
| gpr | 1974 | 0.286 | 94% | 1.485 | 0.109 | 0 |
| kalman | 1974 | 0.277 | 95% | 1.435 | 0.084 | 0 |
| linear | 1974 | 0.355 | 96% | 1.793 | 0.235 | 0 |

## Por tamanho do buraco

| hole_days | method | points | mae_vs_measurement | coverage | mean_width | mae_vs_true_level | failed_gaps |
|---|---|---|---|---|---|---|---|
| 3 | gpr | 240 | 0.281 | 95% | 1.439 | 0.111 | 0 |
| 3 | kalman | 240 | 0.266 | 95% | 1.412 | 0.077 | 0 |
| 3 | linear | 240 | 0.379 | 95% | 1.748 | 0.280 | 0 |
| 7 | gpr | 606 | 0.298 | 94% | 1.492 | 0.106 | 0 |
| 7 | kalman | 606 | 0.286 | 94% | 1.430 | 0.081 | 0 |
| 7 | linear | 606 | 0.400 | 94% | 1.787 | 0.249 | 0 |
| 14 | gpr | 1128 | 0.281 | 95% | 1.492 | 0.111 | 0 |
| 14 | kalman | 1128 | 0.274 | 95% | 1.443 | 0.086 | 0 |
| 14 | linear | 1128 | 0.326 | 98% | 1.806 | 0.219 | 0 |

## Por cenário

| scenario | method | points | mae_vs_measurement | coverage | mean_width | mae_vs_true_level | failed_gaps |
|---|---|---|---|---|---|---|---|
| bulk | gpr | 329 | 0.265 | 95% | 1.392 | 0.044 | 0 |
| bulk | kalman | 329 | 0.263 | 94% | 1.370 | 0.040 | 0 |
| bulk | linear | 329 | 0.341 | 96% | 1.707 | 0.215 | 0 |
| cutting rápido | gpr | 329 | 0.265 | 94% | 1.402 | 0.047 | 0 |
| cutting rápido | kalman | 329 | 0.265 | 94% | 1.379 | 0.041 | 0 |
| cutting rápido | linear | 329 | 0.346 | 95% | 1.727 | 0.220 | 0 |
| cutting→bulk | gpr | 329 | 0.293 | 93% | 1.487 | 0.129 | 0 |
| cutting→bulk | kalman | 329 | 0.289 | 95% | 1.491 | 0.129 | 0 |
| cutting→bulk | linear | 329 | 0.340 | 97% | 1.767 | 0.214 | 0 |
| cutting→platô | gpr | 329 | 0.278 | 95% | 1.471 | 0.103 | 0 |
| cutting→platô | kalman | 329 | 0.274 | 95% | 1.464 | 0.096 | 0 |
| cutting→platô | linear | 329 | 0.340 | 97% | 1.767 | 0.220 | 0 |
| manutenção semanal | gpr | 329 | 0.326 | 95% | 1.647 | 0.201 | 0 |
| manutenção semanal | kalman | 329 | 0.275 | 94% | 1.413 | 0.072 | 0 |
| manutenção semanal | linear | 329 | 0.426 | 96% | 1.997 | 0.339 | 0 |
| manutenção→cutting | gpr | 329 | 0.292 | 95% | 1.512 | 0.132 | 0 |
| manutenção→cutting | kalman | 329 | 0.296 | 95% | 1.493 | 0.123 | 0 |
| manutenção→cutting | linear | 329 | 0.337 | 97% | 1.795 | 0.205 | 0 |
