# CLAUDE.md — W8T

Este arquivo é a fonte da verdade do projeto entre sessões. Atualize-o sempre que uma decisão de
arquitetura mudar ou uma fase for concluída — não deixe a memória de longo prazo divergir do
estado real do código.

## O que é o W8T

App de acompanhamento inteligente de peso: registro diário → métricas/tendências → detecção de
padrões (platô, anomalia) → previsão estatística/ML com intervalos de incerteza → backtesting e
comparação de modelos → (opcional) narrativa em linguagem natural via IA generativa.

**Projeto de estudo de Machine Learning/estatística, não comercial.** O foco de esforço é quase
100% na camada analítica/estatística — a interface existe para tornar o projeto apresentável
(é também peça de portfólio), não é objeto de estudo em si.

Especificação funcional completa (33 áreas de features) foi definida no kickoff do projeto e deve
ser buscada no histórico da conversa/commits quando necessário — este arquivo documenta *decisões
e estado*, não repete a spec inteira.

## Princípios inegociáveis (da spec original)

- Nunca fabricar dado ausente.
- Nunca apresentar previsão como certeza — sempre com intervalo de incerteza.
- Nunca tratar correlação como causalidade.
- Cada feature analítica deve verificar se há dado suficiente antes de rodar, e informar
  claramente quando não há, em vez de produzir resultado artificial.
- Diferenciar sempre: valor registrado / métrica calculada / tendência estimada / anomalia
  detectada / previsão / insight gerado por IA.
- A camada de insights (LLM) só narra números já calculados pelas outras camadas — nunca calcula
  nada sozinha.

## Decisões de arquitetura (fechadas em 2026-09-21)

### Modelagem
Sem deep learning / rede treinada do zero — dado de um único usuário (centenas de pontos/ano) é
regime de "small data", onde modelos estatísticos clássicos com quantificação de incerteza nativa
ganham de modelos data-hungry. Progressão planejada:

1. Baseline (último valor / média móvel)
2. Regressão linear sobre janela móvel
3. Holt / Holt-Winters (exponential smoothing) — `statsmodels`
4. Modelo de espaço de estados / Filtro de Kalman (`statsmodels.tsa.UnobservedComponents`) —
   modela peso como tendência + ruído, dá incerteza nativamente
5. Gaussian Process Regression (`scikit-learn`)
6. Prophet como benchmark externo no backtesting (não obrigatório, comparação)

"Treinar" aqui significa reajustar os parâmetros de cada modelo leve a cada novo dado, não treinar
uma rede do zero.

### Camadas
```
data layer          → modelos SQLAlchemy, fonte da verdade dos registros
stats engine (core)  → cálculos determinísticos: médias, tendência, platô, consistência (sem ML)
forecasting engine   → modelos plugáveis (interface fit/predict/uncertainty), cada um se auto-gate
                       conforme dado disponível
backtesting/eval     → roda modelos contra o passado, métricas por horizonte, comparação
insights layer       → opcional, chama Claude API só para narrar números já calculados
```

### Stack
Python 3.12 + `uv` · Streamlit (UI) · SQLAlchemy 2.0 + Alembic · pandas/numpy · statsmodels +
scikit-learn · pytest · anthropic SDK isolado na camada de insights.

### Dois modos de execução (privacidade)
Peso corporal é dado sensível — não deve ser exposto publicamente.

- **`local`** (default): dados reais do usuário, persistentes, SQLite local. Nunca hospedado
  publicamente.
- **`demo`**: dados sintéticos gerados (trajetória plausível com tendência, ruído, um platô, uma
  anomalia proposital), resetáveis a qualquer momento. Backend Postgres via **Neon** (free tier,
  branching combina com "resetar"). Usado na versão pública hospedada.

Selecionado via `APP_ENV` em `.env` (ver `src/w8t/config.py`). Troca de banco é só `DATABASE_URL`
— SQLAlchemy abstrai, não deve exigir mudança de código.

### Hospedagem
**Streamlit Community Cloud** — gratuito, feito para Streamlit, redeploy automático a partir do
GitHub. (Alternativa descartada: Render, que já hospeda outro site do usuário, mas exigiria
Dockerfile/build manual para Streamlit.)

### Git / GitHub
Commits devem contar no dashboard de contribuições de `Giovaniolivr`. Email de commit configurado
neste repo (não global): `Giovaniolivr@gmail.com` — precisa ser um email verificado na conta
GitHub para os quadrados verdes contarem.

## Estado atual

Fase: **esqueleto inicial**. Nenhuma feature da spec foi implementada ainda.

Feito:
- Repo git inicializado, identidade de commit configurada.
- Estrutura de pastas (`src/w8t/{app,core,data,forecasting,insights}`, `tests/`).
- `pyproject.toml` com dependências via `uv` (grupo dev: pytest/ruff; extras: `postgres`,
  `insights`).
- `src/w8t/config.py`: settings via `pydantic-settings`, modo `local`/`demo`, `.env.example`.
- Entrypoint Streamlit mínimo (`src/w8t/app/Home.py`) só provando que a stack sobe.
- Smoke test (`tests/test_config.py`).

Próximo passo (não iniciado): implementar o registro diário de peso (item 1 da spec) — modelo
SQLAlchemy de `WeightEntry`, repositório em `data/`, e a primeira tela de CRUD no Streamlit.
Depois disso, dashboard principal (item 2) só faz sentido quando já houver dado para calcular em
cima.

## Convenções de trabalho

- Construir incrementalmente, uma feature por vez, sem pular fases (ex.: não implementar
  forecasting antes de ter o registro básico e o stats engine funcionando).
- Manter este arquivo atualizado a cada mudança relevante de arquitetura ou fase concluída.
- Toda lógica de cálculo (stats/forecasting) deve ter teste — é código numérico fácil de quebrar
  silenciosamente.
