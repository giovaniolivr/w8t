# Seleção do kernel do GPR (sem viés de seleção)

**Primeira tentativa (erro registrado):** o kernel `linear + RBF, 60d` foi escolhido olhando só a
série da demo. No benchmark multi-série ele ficou em último entre os quatro principais modelos em
14 e 30 dias — viés de seleção.

**Segunda tentativa (2026-09-24), com separação entre seleção e confirmação:**

1. **Seleção** em 18 séries *reservadas* (6 cenários × sementes 100-102), walk-forward a cada 5
   dias — nunca nas sementes do benchmark.
2. **Confirmação** da vencedora contra a variante antiga nas 30 séries do benchmark padrão
   (sementes 0-4), walk-forward a cada 3 dias, Wilcoxon pareado entre séries.

## Seleção (18 séries reservadas)

| variante | MAE h=1 | h=7 | h=14 | h=30 | MAE médio | cobertura h=30 |
|---|---|---|---|---|---|---|
| **linear + Matérn 3/2, 90d** | 0.362 | 0.408 | 0.461 | 0.735 | **0.492** | 85% |
| linear + Matérn 3/2, 60d | 0.367 | 0.422 | 0.460 | 0.766 | 0.504 | 83% |
| linear + RQ, 90d | 0.365 | 0.428 | 0.487 | 0.774 | 0.513 | 84% |
| linear + RQ, 60d | 0.368 | 0.435 | 0.479 | 0.792 | 0.518 | 81% |
| linear + RBF, 90d | 0.365 | 0.430 | 0.501 | 0.803 | 0.525 | 82% |
| linear + RBF, 60d (antiga) | 0.369 | 0.444 | 0.494 | 0.795 | 0.526 | 81% |
| linear + RBF + semanal, 60d | 0.365 | 0.443 | 0.491 | 0.804 | 0.526 | 80% |
| linear + RBF + semanal, 90d | 0.355 | 0.434 | 0.498 | 0.816 | 0.526 | 81% |

O kernel periódico de 7 dias não ajudou no agregado (só 1 dos 6 cenários tem padrão semanal; o
Kalman já trata isso com seleção automática).

## Confirmação (30 séries do benchmark)

| horizonte | Matérn 3/2 90d | RBF 60d (antiga) | cobertura (nova / antiga) | p (Wilcoxon) |
|---|---|---|---|---|
| 1 | 0.351 | 0.355 | 94% / 94% | 0.36 |
| 7 | 0.410 | 0.419 | 93% / 92% | 0.27 |
| 14 | 0.460 | 0.466 | 92% / 92% | 0.37 |
| 30 | 0.738 | 0.762 | 85% / 82% | 0.38 |

**Conclusão:** a direção se repete na seleção e na confirmação (menor erro em todos os
horizontes, melhor cobertura em 30 dias), mas a melhora **não é estatisticamente demonstrada**.
Por cenário aparece o mesmo trade-off de regime do resto do projeto: a janela maior ganha quando a
tendência continua (bulk h=30: 0.46 vs 0.59; esparso: 0.33 vs 0.48) e perde logo após viradas
(cutting→platô h=30: 0.92 vs 0.73). Adotada como o GPR do app; continua atrás do Kalman e da
combinação Kalman + Holt (o modelo recomendado).
