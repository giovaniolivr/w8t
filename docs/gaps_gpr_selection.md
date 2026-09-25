# Reconstrução de lacunas: seleção do kernel do GPR (2026-09-25)

Mesmo protocolo de `docs/gpr_selection.md`: seleção em séries **reservadas**, confirmação nas
séries padrão com teste pareado. Harness: `w8t.patterns.gaps_evaluation` (buracos artificiais de
3, 7 e 14 dias; critério principal = erro vs. peso verdadeiro sem ruído; cobertura do IC95% para
uma medição).

## Seleção — 18 séries reservadas (6 cenários × sementes 100-102)

| variante | erro vs. medição | cobertura | largura IC | erro vs. peso verdadeiro |
|---|---|---|---|---|
| linear + RBF, 45 d (**atual**) | 0,285 | 94,6% | 1,40 | 0,104 |
| linear + RBF, 90 d | 0,289 | 95,2% | 1,41 | 0,105 |
| linear + Matérn 3/2, 45 d | 0,281 | 95,2% | 1,42 | 0,103 |
| linear + Matérn 3/2, 90 d | 0,284 | 95,3% | 1,42 | 0,094 |
| linear + RQ, 45 d | 0,286 | 94,6% | 1,40 | 0,108 |
| linear + RQ, 90 d | 0,287 | 95,2% | 1,41 | 0,100 |
| linear + RBF + semanal, 45 d | 0,280 | 94,0% | 1,35 | 0,095 |
| **linear + RBF + semanal, 90 d** | 0,286 | 94,2% | 1,37 | **0,088** |

## Confirmação — 48 séries padrão (6 cenários × sementes 0-7)

Erro vs. peso verdadeiro, média por série:

| cenário | RBF 45 d (atual) | Matérn 3/2 90 d | RBF + semanal 90 d |
|---|---|---|---|
| bulk | 0,044 | 0,039 | 0,044 |
| cutting rápido | 0,047 | 0,044 | 0,043 |
| cutting→bulk | 0,129 | 0,133 | 0,148 |
| cutting→platô | 0,103 | 0,100 | 0,111 |
| manutenção semanal | 0,201 | 0,201 | **0,075** |
| manutenção→cutting | 0,132 | 0,123 | 0,151 |
| **geral** | 0,109 | 0,107 | 0,095 |

Wilcoxon pareado entre séries (semanal 90 d vs. atual): vence em 48% das séries, p = 0,49.
Vs. Matérn 90 d: vence em 38%, p = 0,72.

**Decisão: manter o kernel atual.** O ganho agregado da variante semanal vem inteiro de um
cenário (o semanal) e é pago nos cenários com mudança de regime — média melhor, mas não um
método melhor. O Kalman (0,084 nas mesmas séries, `docs/gaps_benchmark.md`) continua o método
recomendado e já trata padrão semanal com seleção automática.
