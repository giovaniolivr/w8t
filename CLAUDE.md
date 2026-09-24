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
- Diferenciar sempre quatro categorias de valor, nunca misturadas na mesma tabela/estrutura:
  **medição real** (`weight_entries`, o que o usuário de fato registrou) / **estado
  filtrado-suavizado** (saída do Kalman, descreve tendência subjacente a partir de medições
  ruidosas) / **valor reconstruído-interpolado** (preenchimento de gap via Kalman/GPR) /
  **previsão futura**. As três últimas nunca são gravadas em `weight_entries` nem em qualquer
  lugar que possa ser confundido com medição real — são sempre derivadas/computadas (ou, se
  cacheadas por performance, numa tabela própria e claramente marcada como estimativa).
- A camada de insights (LLM) só narra números já calculados pelas outras camadas — nunca calcula
  nada sozinha.
- **Dados temporais nunca são embaralhados.** Nenhum split aleatório (`shuffle=True`,
  `train_test_split` clássico, k-fold aleatório) em cima da série de peso. Treino/validação/
  avaliação sempre respeitam a ordem cronológica: split cronológico simples ou walk-forward/
  rolling. Nenhuma avaliação pode usar informação que só existiria "no futuro" em relação ao
  ponto sendo avaliado.
- A referência temporal de cada registro é sempre `entry_date` (a data a que o peso se refere),
  nunca `created_at`/ordem de inserção no banco. Um registro pode ser inserido fora de ordem
  (ex.: registrar segunda-feira na terça) e isso não pode afetar nenhum cálculo — já suportado
  pelo modelo atual (`WeightEntry.entry_date` vs. `created_at`).

## Decisões de arquitetura (fechadas em 2026-09-21, com adições em 2026-09-23)

### Modelagem
Sem deep learning / rede treinada do zero — dado de um único usuário (centenas de pontos/ano) é
regime de "small data", onde modelos estatísticos clássicos com quantificação de incerteza nativa
ganham de modelos data-hungry. Progressão planejada:

1. Baseline (último valor / média móvel)
2. Regressão linear sobre janela móvel
3. Holt com tendência amortecida — **implementação própria** (ver "Holt amortecido" em Estado
   atual: `statsmodels` falha silenciosamente com dias faltantes)
4. Modelo de espaço de estados / Filtro de Kalman (`statsmodels.tsa.UnobservedComponents`) —
   modela peso como tendência + ruído, dá incerteza nativamente
5. Gaussian Process Regression (`scikit-learn`)
6. Prophet como benchmark externo no backtesting (não obrigatório, comparação)

"Treinar" aqui significa reajustar os parâmetros de cada modelo leve a cada novo dado, não treinar
uma rede do zero.

**Kalman: filter vs. smoother.** Para exibir "qual era a tendência num ponto do passado" (uso
descritivo, olhando pra trás com toda a série disponível), usar o *smoother* (RTS). Para qualquer
coisa que alimente avaliação/backtesting de previsão, usar só o *filter* (forward-only, só passado
até aquele ponto) — usar o smoother nesse caso seria vazamento de informação futura.

**GPR como referência probabilística de trajetória, não só "prever amanhã".** A posterior do GPR
(média + variância) é candidata a virar fonte única para tendência, platô e anomalia (desvio =
distância da observação em relação à média esperada, ponderada pela incerteza — diferença pequena
importa quando a incerteza é baixa, diferença igual importa pouco quando a incerteza é alta). Isso
é um *refactor futuro*, não a implementação inicial: cada detector (tendência, platô, anomalia)
nasce como baseline simples (inclinação de janela móvel, z-score/MAD) para servir de comparação
via backtesting antes de qualquer versão baseada em GPR — para poder aplicar de fato o princípio
"não considerar um modelo melhor só por ser mais complexo".

### Períodos/ciclos (bulking, cutting, manutenção) — implementado em 2026-09-23

- `src/w8t/data/models.py`: `Period` (label livre, `goal_direction` — enum controlado
  `loss`/`gain`/`maintenance`, `start_date`, `end_date` opcional = em andamento,
  `target_weight_kg` opcional). Sem overlap constraint no schema (SQLite não tem exclusion
  constraint) — validado em `src/w8t/data/periods.py`.
- `src/w8t/data/periods.py`: `create_period`/`update_period`/`delete_period`/`list_periods`/
  `get_period`/`get_period_for_date`. Overlap (incluindo período em andamento tratado como
  data-fim infinita) levanta `OverlappingPeriodError`; `update_period` exclui o próprio período
  da checagem e usa sentinela (`...`) para distinguir "não passei esse campo" de "quero setar
  `end_date=None`" (reabrir um período).
  Associação de `WeightEntry` a um período continua sendo por intervalo de data na consulta
  (`get_period_for_date`), não por FK — período pode ser editado sem tocar nos registros.
- Migração `86da55a7e986` cria a tabela `periods`.
- `src/w8t/app/pages/2_Periodos.py`: formulário de criação (com toggle "em andamento" e meta
  opcional), listagem, edição e exclusão — mesmo padrão da tela de registro.
- Testes: `tests/test_periods.py` (overlap, período em andamento bloqueando qualquer início
  posterior, update reabrindo período, `get_period_for_date`) e `tests/test_periodos_page.py`
  (AppTest, incluindo o caso de sobreposição rejeitada na UI).
- **Opcional por design**: app continua funcionando sem nenhum período definido (spec item 32) —
  coberto pelo teste `test_page_loads_with_no_periods`.
- **Ainda não feito**: nenhuma tela hoje *usa* `get_period_for_date` para escopar análises —
  isso só faz sentido a partir do dashboard (próximo passo).

### Reconstrução de gaps (Kalman + GPR)

Decidido em 2026-09-23; **implementado em 2026-09-24** (ver "Reconstrução de lacunas" em
Estado atual).

- Funcionalidade opt-in (usuário solicita explicitamente para um gap específico), nunca automática.
- Kalman estima o estado/trajetória subjacente a partir das medições imperfeitas disponíveis;
  GPR modela a trajetória de forma probabilística e estima valores (com incerteza) nos dias sem
  observação.
- Resultado da reconstrução **nunca** é gravado em `weight_entries` — ver princípio da taxonomia
  de 4 categorias acima. Sempre marcado visualmente como estimativa, nunca confundido com medição.

### Camadas
```
data layer          → modelos SQLAlchemy, fonte da verdade dos registros
stats engine (core)  → cálculos determinísticos: médias, tendência, platô, consistência (sem ML)
forecasting engine   → modelos plugáveis (interface fit/predict/uncertainty), cada um se auto-gate
                       conforme dado disponível
backtesting/eval     → roda modelos contra o passado, métricas por horizonte, comparação
insights layer       → opcional, chama um LLM (Gemini) só para narrar números já calculados
```

Fluxo da camada de insights (LLM), quando existir: dados brutos → processamento estatístico/
modelos → análise estruturada (inclui contexto de período/objetivo quando houver: bulking/cutting/
manutenção, duração, peso inicial/final, tendência, estabilidade, desvios, gaps, reconstruções,
previsão + incerteza, MAE/RMSE) → LLM → explicação em linguagem natural. A LLM nunca recebe a
série bruta como fonte principal de interpretação. **Última fase do projeto**, condicionada a
viabilidade — na prática o custo esperado é desprezível mesmo sem tier gratuito dado o volume de
uso (single-user, poucas chamadas), então é mais questão de prioridade do que de bloqueio técnico.

### Stack
Python 3.12 + `uv` · Streamlit (UI) · SQLAlchemy 2.0 + Alembic · pandas/numpy · statsmodels +
scikit-learn · pytest · Gemini (REST via `requests`, plano gratuito) isolado na camada de
insights — trocado do SDK da Anthropic em 2026-09-24 (a API da Anthropic não tem plano
gratuito; Workers AI descartado para não dividir a cota diária de 10k neurons, que é por conta
Cloudflare, com outro projeto do usuário).

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
Confirmado de novo em 2026-09-24. Sem autenticação: a versão pública (modo `demo`) é editável
por qualquer visitante, o que é aceitável porque só contém dados sintéticos; o botão "Resetar
dados de demonstração" (sidebar do `Home.py`, visível só com `APP_ENV=demo`) restaura o estado.

### Dados de demonstração
`src/w8t/data/demo.py`: `generate_series(today)` gera ~180 dias determinísticos (seed fixa,
datas ancoradas em `today`): tendência de perda → platô → manutenção com leve reganho, ruído
gaussiano, ~15% de dias faltando e uma anomalia proposital (+2,5 kg no dia 60).
`reset_demo_data` apaga todos os registros/períodos e grava a série + 2 períodos ("Cutting" com
meta, "Manutenção" em andamento). Levanta `NotDemoModeError` se `APP_ENV` não for `demo` — guarda
contra apagar dados reais. Testes em `tests/test_demo.py`.

### Identidade visual
Paleta verde / cinza escuro / preto, tema escuro. Tema do Streamlit em `.streamlit/config.toml`;
cores de gráfico em `src/w8t/app/theme.py` (manter em sincronia). Convenção: **medição real em
cinza neutro, valores derivados (médias, tendência, futuramente previsão) em verde** — reforça
visualmente a taxonomia de 4 categorias de valor.

### Git / GitHub
Commits devem contar no dashboard de contribuições de `Giovaniolivr`. Email de commit configurado
neste repo (não global): `Giovaniolivr@gmail.com` — precisa ser um email verificado na conta
GitHub para os quadrados verdes contarem.

## Estado atual

Fase: **itens 1, 2, 8-10 (baselines) e 15 da spec concluídos** (registro diário, dashboard,
tendência/platô/anomalia, backtesting) + forecasting engine completo (baselines, Holt amortecido, Kalman, GPR, combinação Kalman+Holt) **+ `Period` no data layer** (entidade nova, fora da numeração da spec original, ver seção "Períodos/ciclos"
acima). Repo
público em `github.com/giovaniolivr/w8t`.

Feito:
- Repo git inicializado, identidade de commit configurada (`Giovaniolivr@gmail.com`), remoto no
  GitHub (`giovaniolivr/w8t`, público) com push funcionando via GitHub CLI autenticado localmente.
- Estrutura de pastas (`src/w8t/{app,core,data,forecasting,insights}`, `tests/`).
- `pyproject.toml` com dependências via `uv` (grupo dev: pytest/ruff; extras: `postgres`,
  `insights`).
- `src/w8t/config.py`: settings via `pydantic-settings`, modo `local`/`demo`, `.env.example`.
- **Registro diário de peso (spec item 1) — completo:**
  - `src/w8t/data/models.py`: `WeightEntry` (data, peso, horário opcional, unique constraint por
    `entry_date` — um registro por dia).
  - `src/w8t/data/db.py`: engine/sessão SQLAlchemy (`get_session()` context manager).
  - `src/w8t/data/repository.py`: `create_entry`/`update_entry`/`delete_entry`/`list_entries`/
    `get_entry_by_date`; duplicata na mesma data levanta `DuplicateEntryError` (a UI trata isso
    oferecendo sobrescrever em vez de deixar o usuário criar um segundo registro no mesmo dia).
  - Alembic configurado (`alembic/`, `env.py` lê `DATABASE_URL` de `w8t.config.settings`
    dinamicamente); primeira migração `409e08cf7c09` cria `weight_entries`.
  - `src/w8t/app/pages/1_Registro_de_Peso.py`: formulário de novo registro (permite data
    retroativa, bloqueia data futura), alerta + opção de sobrescrever em caso de duplicata,
    histórico com diferença vs. registro anterior, edição e exclusão inline.
  - Testes: `tests/test_repository.py` (CRUD, duplicata, retroativo) e
    `tests/test_registro_de_peso_page.py` (via `streamlit.testing.v1.AppTest` — carrega a página
    de verdade e simula preencher/enviar o formulário, sem navegador).
- **`Period` (períodos/ciclos) — completo**, ver seção "Períodos/ciclos" acima para detalhes de
  arquivo. Migração `86da55a7e986`. Consumido pelo dashboard como escopo de análise.
- **Dashboard principal (spec item 2) — completo (2026-09-24):**
  - `src/w8t/core/metrics.py` (stats engine, puro, sem acesso a banco): `to_series` (série
    indexada e ordenada por `entry_date`, rejeita data duplicada), `slice_series` (escopo
    inclusivo), `rolling_mean` (janela de calendário `7D`/`30D` só sobre medições reais — sem
    reamostrar/interpolar; `NaN` se a janela tiver menos que `ROLLING_MIN_OBS` = 3 em 7d, 8 em
    30d), `pace_kg_per_week` (inclinação OLS sobre datas reais; exige ≥3 medições cobrindo ≥7
    dias), `summarize` → `Summary` (atual/inicial/mín/máx com datas, variação kg/%, ritmo, MM7/
    MM30 na última data). Métrica sem dado suficiente vira `None` → UI mostra "dados
    insuficientes".
  - `src/w8t/app/Home.py` é o dashboard (primeira tela, decisão do usuário): seletor de escopo
    (histórico completo ou um `Period`; período em andamento vai até hoje), cartões de KPI,
    meta do período quando houver, gráfico Plotly com medição real em pontos e médias móveis em
    linha (derivadas, visualmente distintas). Médias do gráfico e dos cartões usam a mesma série
    escopada, para os números baterem.
  - Escopo por período é feito por intervalo de datas (`slice_series`), não por
    `get_period_for_date`.
  - Testes: `tests/test_metrics.py` (valores calculados à mão, gaps, janela, ordem de inserção,
    espaçamento irregular) e `tests/test_home_page.py` (AppTest: vazio, KPIs, escopo por
    período, período sem registros).
- **Tendência / platô / anomalia — baselines (spec itens 8-10) — completo (2026-09-24).** São
  os baselines simples contra os quais a versão futura baseada na posterior do GPR será comparada
  via backtesting. Parâmetros são constantes no topo de cada módulo.
  - `src/w8t/core/trend.py`: `linear_fit` (OLS + erro-padrão da inclinação), `window_slope`,
    `classify`, `current_trend` sobre os últimos 21 dias (≥6 medições). Direção só é declarada
    se o IC95% da inclinação exclui zero **e** |inclinação| ≥ 0,25 kg/sem; "estável" se o IC
    inteiro cabe em ±0,25; senão "indefinido" (nunca força uma direção).
  - `src/w8t/core/plateau.py`: `detect_plateaus` — data "plana" quando a inclinação OLS da
    janela de 21 dias terminando nela tem |·| < 0,25 kg/sem (≥8 medições); datas planas
    consecutivas viram um platô se durarem ≥21 dias. Agnóstico ao objetivo: a UI mostra os
    períodos sobrepostos para o usuário interpretar (manutenção = esperado; perda/ganho =
    estagnação). Limitação conhecida: início do platô sai adiantado até ~1 janela.
  - `src/w8t/core/anomaly.py`: `detect_anomalies` — para cada medição, reta **Theil-Sen** sobre
    as medições *anteriores* dos últimos 21 dias (≥8), extrapolada até a data; z = resíduo /
    erro-padrão de previsão (sigma via MAD dos resíduos × fator de intervalo de previsão OLS);
    |z| ≥ 3,5 marca. Só usa passado (acrescentar pontos futuros não muda veredito passado —
    testado). Nunca altera nem remove a medição.
    **Lição registrada:** a 1ª versão (OLS excluindo pontos já marcados) entrava em cascata —
    um falso positivo cedo congelava a reta e gerava dezenas de falsos positivos seguidos.
    Theil-Sen resolve sem exclusão.
    **Números do baseline** (simulação, ruído 0,35 kg, 15% de dias faltando): ~1,6% de falsos
    positivos; detecta +1,5 kg em ~73% e +2,0 kg em ~100% (limiar 4,0: 0,9% / 60% / 87%).
  - `Home.py`: seção "Padrões" (tendência com IC95%, nº de platôs, nº de atípicas), platôs como
    faixas cinza e atípicas como círculos âmbar no gráfico, tabelas com período(s) sobreposto(s)
    e valor esperado/z. Detecção em `st.cache_data` (~0,4 s para 150 pontos).
  - Testes: `tests/test_patterns.py` (OLS vs. `np.polyfit`, janela, IC, platô após perda,
    anomalia plantada, só-passado, não-contaminação, série não modificada) e caso de UI em
    `tests/test_home_page.py`.

- **Forecasting engine — interface + baselines (roadmap passos 1-2) — completo (2026-09-24).**
  Na UI a partir do backtesting (ver abaixo).
  - `src/w8t/forecasting/base.py`: `ForecastModel` (`fit(series)` → `predict(horizons, level)`
    → `Forecast` com média + `lower`/`upper`; não existe método só-ponto). Horizonte em dias de
    calendário após a última medição. Auto-gate: `fit` levanta `InsufficientDataError` com
    motivo legível; rejeita série fora de ordem. Modelo só vê a série que recebe — quem chama
    (backtesting) garante que é só passado.
  - `src/w8t/forecasting/baselines.py`: `NaiveLastValue` (passeio aleatório; σ²/dia estimado de
    (Δy)²/Δt, respeita espaçamento irregular; intervalo ∝ √h), `MovingAverage(7)` (nível = média
    da janela; intervalo = dp dos próprios erros one-step passados, **constante no horizonte** —
    ingenuidade proposital, backtesting deve mostrar subcobertura em horizonte longo),
    `LinearTrend(28)` (OLS na janela; intervalo de predição t-Student, alarga com a distância;
    assume tendência continuando — quebra em platô). `baseline_models()` lista os três.
  - `core/trend.LinearFit` ganhou `residual_sd`/`x_mean`/`sxx`/`prediction_se()`.
  - Testes: `tests/test_forecasting.py` (contrato, gates, valores exatos, espaçamento irregular,
    janela, alargamento, e **cobertura empírica ~95%** do intervalo da regressão quando as
    premissas valem — 400 simulações).
  - Observação na demo: origem no meio do cutting, `LinearTrend` prevê 79,5 kg em h=30 (IC
    78,0–81,1) enquanto o peso real estabilizou ~80,7 — exemplo concreto do que o backtesting
    precisa quantificar.

- **Backtesting walk-forward (spec item 15) — completo (2026-09-24).**
  - `src/w8t/forecasting/backtest.py`: `forecast_origins` (datas reais, após aquecimento de 28
    dias, espaçadas ≥ `step_days`), `walk_forward` (cópia nova do modelo por origem, ajustada só
    com dados ≤ origem; pontuada só se existe medição real exatamente na data-alvo — gap nunca
    vira "verdade" interpolada; modelo que recusa ajuste vai para `skipped` com o motivo),
    `summarize` (MAE, RMSE, viés = real − previsto, cobertura empírica do IC, largura média,
    `skill_vs_ref` = 1 − MAE/MAE_ref). **Por padrão compara só casos que todos os modelos
    previram** (`common_only`) — senão um modelo que pula origens difíceis ganharia de graça.
  - Testes: `tests/test_backtest.py` (modelo espião prova que nunca vê dado após a origem; alvo
    faltante não é pontuado; métricas com valores conhecidos; gate + casos comuns; skill; série
    de entrada intacta).
  - `src/w8t/app/pages/3_Previsao.py`: tabela de backtesting (passo diário, cacheada), seletor
    de modelo com o de menor MAE médio pré-selecionado, gráfico com medições (cinza) + previsão
    tracejada + faixa do IC (verde translúcido, cores em `theme.FORECAST*`), tabela h=1/7/14/30
    e legenda com a **cobertura real medida** do modelo escolhido ao lado do intervalo nominal.
    Estados: sem registros; histórico curto (explica por que não há avaliação, mas prevê se o
    modelo aceitar); modelo que não pode ajustar (mostra o motivo). Testes em
    `tests/test_previsao_page.py`. Escopo: sempre histórico completo (sem seletor de período
    por ora).
  - **Resultados na demo (passo 1 dia, ~80-100 casos por horizonte):** regressão 28d vence em
    todos os horizontes (ganho vs. último valor 16% em h=1 a 32% em h=14), mas cobertura cai
    para 75% em h=30 com viés +0,53 kg (continua prevendo perda depois do platô). Média móvel:
    cobertura 86% → 53% conforme o horizonte (intervalo constante). Último valor: cobertura
    ~100% às custas de IC de 9,5 kg em h=30 — alta cobertura com intervalo inútil não é mérito.

- **Holt amortecido — ETS(A,Ad,N) (roadmap passo 3) — completo (2026-09-24).**
  - `src/w8t/forecasting/holt.py`, implementação própria. **Por quê:** `ExponentialSmoothing` e
    `ETSModel` do `statsmodels` exigem série regular; com NaN nos dias faltantes falham em
    silêncio (não convergem / log-verossimilhança NaN / todos os ajustados NaN, só com warning)
    — verificado. Reamostrar + interpolar fabricaria medições.
  - Grade diária; dia sem medição só propaga o estado (`l ← l+φb`, `b ← φb`), nada é imputado.
    `holt_filter` é a recursão pura (testada contra `statsmodels.ETSModel.smooth` numa série sem
    faltas: erros idênticos até 1e-10). Erro observado após gap de k dias tem variância
    σ²·v(k); parâmetros (α, β=α·β*, φ∈[0,8; 0,98]) por máxima verossimilhança gaussiana com σ²
    concentrado, 2 pontos de partida L-BFGS-B. Previsão: média `l + (φ+…+φ^h)·b`, variância
    σ²·v(h). Gate: ≥14 medições cobrindo ≥21 dias. Estado inicial: 1ª medição + inclinação OLS
    dos primeiros 14 dias.
  - `src/w8t/forecasting/registry.py`: `all_models()` = baselines + Holt; a página de previsão e
    o backtesting usam o registro (modelo novo entra na comparação automaticamente).
  - Testes: `tests/test_holt.py` (equivalência com statsmodels, propagação em gap, v(k),
    gate, amortecimento limitado por φ/(1−φ)·b, recuperação de α/σ e cobertura ~95% em dados
    simulados do próprio processo ETS com 15% de dias faltando).
  - **Resultado na demo (mesmos casos):** único modelo que melhora MAE *e* calibra a cobertura
    em horizonte longo — h=30: MAE 0,79 (regressão 0,83), cobertura 94% (regressão 75%), viés
    −0,03 (regressão +0,53); h=14: MAE 0,38 vs 0,43, cobertura 94%. Horizontes curtos ≈ empate
    com a regressão. Ressalvas: uma única série sintética; φ estimado bateu no limite 0,98.
  - Custo: backtesting passo 1 dia com 4 modelos na demo ~10 s (cacheado por série) — observar
    quando entrarem Kalman/GPR.

- **Kalman — tendência suave (roadmap passo 4) — completo (2026-09-24).**
  - `src/w8t/forecasting/kalman.py`: `KalmanSmoothTrend` = `UnobservedComponents(level="strend")`
    sobre grade diária com NaN nos dias faltantes (Kalman trata nativamente — **verificado**
    antes de usar, após o problema do ETS). Intervalo de `get_forecast` inclui o ruído de
    medição (é intervalo para uma medição futura, comparável aos demais). Só o *filter* é usado
    para previsão/backtesting. Expõe `noise_sd` (dp do ruído) e `slope_change_sd`
    (volatilidade do ritmo) — a parte interpretável.
  - **Escolha strend vs. lltrend (local linear trend):** na demo mesma log-verossimilhança
    (−68,4; variância própria do nível estimada ≈ 0), AIC melhor (140,8 vs 142,7) e ~6x mais
    rápido (0,07 s vs 0,41 s por ajuste) — relevante porque o backtesting reajusta a cada origem.
  - **Armadilha de convergência:** L-BFGS reporta "não convergiu" quando o ótimo está na
    fronteira (`sigma2.trend` → 0, tendência determinística) — 4 origens da demo eram
    descartadas sem motivo. Powell/Nelder-Mead chegam à mesma verossimilhança e parâmetros e
    reportam convergência, então há fallback para Powell; só se ainda assim falhar vira
    `InsufficientDataError`.
  - Testes: `tests/test_kalman.py` (gate, ajuste com 30% de dias faltando sem preencher,
    recupera dp do ruído no próprio processo, intervalo h=1 ≥ ruído, cobertura ~95% com gaps,
    ótimo de fronteira aceito). Registrado em `all_models()`.
  - **Resultado na demo (mesmos casos):** MAE empata com Holt em todos os horizontes (h=30:
    0,794 vs 0,793), mas cobertura h=30 de 80% com viés +0,43 (Holt: 94%, −0,03) — tendência
    não amortecida extrapola a perda para dentro do platô, mesma falha da regressão. Veredito:
    **não supera o Holt como previsor**; seu valor aqui é interpretativo (ruído estimado
    0,34 kg vs 0,35 real da demo) e o *smoother* para descrição do passado / reconstrução de
    gaps. Backtesting na página agora ~17 s no primeiro carregamento (cacheado).

- **GPR (roadmap passo 5) — completo (2026-09-24).**
  - `src/w8t/forecasting/gpr.py`: `GPRLinearRBF(60)` — `GaussianProcessRegressor` sobre os
    últimos 60 dias, tempo contínuo (dia faltante só não é ponto de treino). Kernel
    `C·DotProduct + C·RBF + WhiteKernel`, `normalize_y`, 1 restart. Desvio preditivo inclui o
    WhiteKernel → intervalo de medição. Expõe `noise_sd`.
  - **Seleção de kernel/janela** (demo, walk-forward passo 3 dias, 30-37 casos/horizonte, vs.
    Holt; MAE h=14 / h=30, cobertura h=30): RBF 60d 0,44/1,09, 77% · RBF 120d 0,54/1,46, 67% ·
    Matérn-3/2 60d 0,41/0,91, 90% · Matérn-3/2 120d 0,41/1,02, 83% · linear+RBF 120d 0,46/1,15,
    77% · **linear+RBF 60d 0,37/0,85, 80%** · Holt 0,34/0,76, 97%. Kernels estacionários revertem
    à média da janela (viés ≈ −0,5 kg em h=30: preveem reganho). **Viés de seleção:** escolhido
    na mesma série em que é avaliado.
  - Na demo o RBF aprendido bate no limite inferior (length_scale=3 dias) → captura oscilação de
    curto prazo; ruído estimado 0,345 kg (real 0,35).
  - Testes: `tests/test_gpr.py` (gate por janela, só janela recente, gaps, ruído aprendido e
    incluído no intervalo, segue tendência linear em vez de reverter à média, cobertura).
  - **Resultado final na demo (passo 1 dia, 6 modelos, mesmos casos):**
    | h | melhor MAE | MAE Holt | cobertura GPR / Holt / Kalman |
    |---|---|---|---|
    | 1 | GPR/regressão 0,290 | 0,294 | 93% / 90% / 92% |
    | 7 | GPR 0,350 | 0,360 | 89% / 89% / 89% |
    | 14 | GPR 0,356 | 0,378 | 90% / 94% / 90% |
    | 30 | GPR 0,787 | 0,793 | 81% / 94% / 80% |
    Diferenças de MAE entre GPR/Holt/Kalman/regressão são de centésimos de kg com ~90 casos
    sobrepostos (autocorrelacionados) — **não há vencedor estatisticamente demonstrado**. Holt é
    o único calibrado em horizonte longo; modelos com tendência não amortecida (regressão,
    Kalman, GPR com DotProduct) ficam confiantes demais em h=30 e com viés positivo (seguem
    prevendo perda dentro do platô).
  - Página de previsão: backtesting agora a cada 2 dias (`BACKTEST_STEP_DAYS`) — 6 modelos em
    passo diário levavam ~33 s no primeiro carregamento.

- **Rigor da comparação — significância + benchmark multi-série — completo (2026-09-24).**
  - `src/w8t/forecasting/significance.py`: Diebold-Mariano com correção HLN, variância de longo
    prazo com defasagens `L = ceil(h/step) − 1` (sobreposição das previsões), pesos uniformes;
    `versus_best` (menor MAE vs. cada outro, Holm por horizonte). **Limites medidos por
    simulação:** teste t pareado ingênuo rejeita >25% sob H0 com sobreposição; DM ainda é
    liberal (~6-10% ao nível 5%; pesos de Bartlett piores, 12-14%) → só p < 0,01 conta como
    "diferença demonstrada". Abaixo de 10 casos *efetivamente independentes* (`n/(L+1)`) não
    testa e reporta "não testável" — na demo, h=14 e h=30 não são testáveis.
  - Na demo (passo 1 dia): única diferença significativa é regressão vs. média móvel em h=1;
    GPR/Holt/Kalman/regressão indistinguíveis em h=1 e h=7 (p 0,40-0,98).
  - `src/w8t/forecasting/benchmark.py` + `python -m w8t.forecasting.benchmark` → `docs/
    benchmark.md` e `.csv`. 6 cenários sintéticos (cutting→platô, bulk, manutenção com padrão
    semanal, esparso com gaps de 12 dias e 40% faltando, cutting→bulk, perda lenta ruidosa) × 5
    sementes = 30 séries independentes; walk-forward passo 3; Wilcoxon pareado entre séries
    (séries são independentes → teste válido), Holm por horizonte.
  - **Resultado — muda a conclusão tirada só da demo:**
    - Kalman tem o menor MAE médio em todos os horizontes; demonstradamente melhor que Holt e
      GPR em h=14 (p_holm 0,009 / 0,011); em h=30 Kalman vs. Holt não se distinguem (0,34), e
      Holt tem a melhor cobertura (93% vs 87%).
    - **Não há vencedor universal — depende do regime.** Tendência que continua (bulk, perda
      lenta, esparso): Kalman vence com folga, o amortecimento do Holt atrapalha (bulk h=30: MAE
      0,64 vs 0,41, viés +0,49). Mudança de regime (cutting→platô, cutting→bulk): Holt vence em
      h=30 e é o melhor calibrado. Manutenção: empate. O amortecimento é uma *aposta* de que a
      tendência acaba.
    - A demo é justamente cutting→platô, por isso lá o Holt parecia o melhor em horizonte longo.
    - **GPR, com kernel escolhido olhando a demo, foi o pior dos 4 principais em h=14/30** —
      viés de seleção confirmado na prática.
    - Valida o desenho da página: escolher o modelo pelo backtesting *da série do usuário*, não
      fixar um vencedor global.
  - Página de previsão: tabela "O menor erro é de fato menor?" (DM sobre os casos comuns do
    backtesting da página) com conclusão por par: diferença demonstrada (p_holm < 0,01) /
    evidência fraca (< 0,05) / sem diferença demonstrada / não testável.
  - Testes: `tests/test_significance.py` (vencedor claro, "não testável", perdas idênticas → NaN,
    taxa de falso positivo sob MA(6) vs. t ingênuo, Holm), `tests/test_benchmark.py` (cenários
    determinísticos/realistas, gaps longos, relatórios, Wilcoxon detecta vencedor consistente) e
    caso de UI.

- **Combinação Kalman + Holt — completo (2026-09-24).**
  - `src/w8t/forecasting/ensemble.py`: `EqualWeightEnsemble` — média das previsões; intervalo
    `"mixture"` (quantis da mistura 50/50 das preditivas gaussianas, resolvidos com `brentq`:
    discordância entre membros vira incerteza) ou `"quantile_avg"` (média dos limites). Gate:
    só ajusta se *todos* os membros ajustam (não degrada em silêncio para um modelo só).
    `fit_from_fitted` combina membros já ajustados na mesma série; `walk_forward` usa isso para
    não reajustar Kalman/Holt dentro da combinação (resultado idêntico — testado; página ~27 s →
    ~20 s). `registry.kalman_holt_ensemble()` (intervalo mistura) está em `all_models()`.
  - **Benchmark (30 séries, passo 3; Kalman / Holt / mistura / média-dos-limites):** mistura
    tem o menor MAE médio em h=14 e h=30 (h=30: 0,645 vs Kalman 0,672, Holt 0,739) e é o único
    **calibrado em todos os horizontes** (cobertura 95,1 / 95,0 / 95,8 / 95,1%; Kalman h=30 87%,
    Holt 93%). Vs. Holt: demonstradamente melhor em h=14/30 (p_holm < 0,003); vs. Kalman: MAE
    indistinguível (h=30 p=0,19) mas cobertura muito melhor. Por cenário nunca é o pior e fica
    perto do melhor em cada regime. Média-dos-limites tem a mesma média mas subcobre em h=30
    (92%) → mistura escolhida.
  - Página de previsão: **combinação é o padrão recomendado** (com justificativa na legenda);
    o de menor MAE no histórico do usuário aparece rotulado, não pré-selecionado — numa série
    só essa diferença raramente é demonstrada. Se o histórico é curto demais para a combinação,
    cai para o primeiro modelo que ajusta.
  - Testes: `tests/test_ensemble.py` (média, membros idênticos reproduzem o intervalo, mistura
    ≥ média-dos-limites quando discordam, quantil da mistura vs. Monte Carlo, gate por membro,
    protótipos intactos, reuso no backtesting ≡ reajuste).

- **Tendência / platô / anomalia via Kalman + avaliação rotulada — completo (2026-09-24).**
  - `src/w8t/patterns/kalman.py` (pacote novo `patterns/`: detectores *baseados em modelo*; os
    determinísticos continuam em `core/` como referência). Mesmos tipos de saída dos baselines.
    `smoothed_states` (nível e inclinação suavizados diários com faixa 95% e coluna `observed` —
    linhas não observadas são reconstrução do modelo), `current_trend` (inclinação *filtrada* no
    último dia, ajuste só nos últimos 60 dias, mesma regra de decisão do baseline),
    `detect_plateaus` (inclinação *suavizada* |·| < 0,25 kg/sem por ≥ 21 dias — uso descritivo,
    usar dados posteriores para localizar platô passado é legítimo), `detect_anomalies`
    (inovações do *filtro* padronizadas, |z| ≥ 3,5; ponto marcado vira faltante e o modelo é
    reajustado, maior |z| primeiro — evita contaminar os dias seguintes; variâncias estimadas na
    série toda). `forecasting/kalman.fit_smooth_trend` extraído para reuso (fallback Powell
    incluso); variâncias ~−1e-18 do suavizador são truncadas em 0.
  - `src/w8t/patterns/evaluation.py` + `python -m w8t.patterns.evaluation` →
    `docs/patterns_benchmark.md`/`.csv`: 6 cenários (cutting→platô, bulk, manutenção semanal,
    cutting→bulk, cutting rápido, manutenção→cutting) × 8 sementes = 48 séries com **gabarito**
    (inclinação verdadeira por dia + 3 anomalias plantadas de 1,5-2,5 kg por série). Tendência
    avaliada só com dados até cada checkpoint (a cada 7 dias); platô por dia medido; anomalia
    por ponto plantado.
  - **Resultado (geral, baseline → Kalman):** tendência correta 59% → 74%, direção errada
    4% → 3%, indefinido 38% → 22%; platô precisão 54% → 100%, F1 70% → 100% (baseline chegava a
    declarar platô no meio de bulk); anomalia recall 82% → 85%, falsos positivos 1,36 → 0,11 por
    100 medições (12x menos). Na demo: só a anomalia plantada (baseline: +2 falsas); início do
    platô a 2 dias do real (baseline 9 dias adiantado); ritmo final +0,07 kg/sem = valor
    verdadeiro.
  - **Ponto fraco honesto — tendência após mudança de regime:** ajustado na série toda, o
    Kalman estima variância de inclinação ~0 e reage devagar a viradas (cutting→bulk 45% vs 75%
    do baseline). Janela de 60 dias resolve boa parte (63%; em cutting→platô passa o baseline,
    55% vs 49%) — escolhida entre {toda, 90, 60} na mesma avaliação (viés de seleção). Em
    cutting→bulk o baseline de 21 dias ainda reage melhor.
  - `Home.py`: detectores Kalman por padrão com **fallback para os baselines** quando o histórico
    é curto (< 14 medições / 21 dias) — o rótulo/ajuda dizem qual método rodou. Gráfico ganhou a
    **tendência estimada (Kalman) com faixa 95%** (verde contínuo, categoria "estado
    filtrado-suavizado"; legenda avisa que em dias sem registro é estimativa do modelo); médias
    móveis passam a começar ocultas (clicáveis na legenda) quando a tendência do modelo está
    presente. Conferido no navegador.
  - Testes: `tests/test_patterns_kalman.py` (insuficiente, grade diária com `observed`, ritmo
    recuperado, direção, janela de 60 dias, fronteira do platô ±7 dias, sem platô em ganho
    contínuo, anomalia sem alarmes em cascata, série intacta, colunas iguais ao baseline, séries
    rotuladas determinísticas, contagens consistentes) + casos de UI (Kalman e fallback).

- **Reconstrução de lacunas — completo (2026-09-24).**
  - `src/w8t/patterns/gaps.py`: `find_gaps` (só lacunas internas, com medição antes e depois —
    depois do último registro seria previsão), `reconstruct(series, gap, method)` → por dia
    faltante `estimate_kg` + intervalo 95% **para uma medição** (incerteza do nível + ruído da
    balança). Métodos: `linear` (referência; ruído estimado em pares de dias consecutivos),
    `kalman` (nível suavizado RTS — usa os dois lados da lacuna — + variância do ruído), `gpr`
    (kernel da previsão, ±45 dias de contexto). **Antes de qualquer método, pontos marcados pelo
    detector de anomalias do Kalman são deixados de fora como evidência** (não apagados).
  - `src/w8t/patterns/gaps_evaluation.py` + `python -m w8t.patterns.gaps_evaluation` →
    `docs/gaps_benchmark.md`/`.csv`: nas 48 séries rotuladas, 2 buracos artificiais de 3, 7 e
    14 dias por série (medições reais apagadas); erro vs. medição apagada, cobertura do IC e erro
    vs. **peso verdadeiro sem ruído** (`LabeledSeries.true_level`, campo novo). Anomalias
    plantadas não são pontuadas.
  - **Resultado:** Kalman MAE vs. peso verdadeiro 0,105 kg (GPR 0,109; linear 0,236 — mais que o
    dobro), cobertura 95% em todos os tamanhos e cenários (linear 97%, intervalo 22% mais
    largo). **Achado:** sem mascarar anomalias, as 3 anomalias plantadas inflavam o ruído
    estimado e a cobertura ia a 98% (intervalos largos demais); com a máscara, 94,8% e intervalo
    20% mais estreito, mesmo erro. Limitação: padrão semanal (fim de semana) não é modelado —
    todos pioram nesse cenário (Kalman 0,20 vs 0,04-0,13 nos demais).
  - `src/w8t/app/pages/4_Lacunas.py`: **opt-in** — escolhe a lacuna (mais recentes primeiro) e
    o método (Kalman recomendado), só calcula ao clicar; gráfico com medições (cinza),
    estimativas como losango verde vazado + faixa, tabela com coluna "Tipo: estimativa — não é
    medição". Nada é gravado (teste confirma contagem de `weight_entries` inalterada).
    Conferido no navegador.
  - Testes: `tests/test_gaps.py` (lacunas internas/min_days, borda e método inválido,
    interpolação exata, Kalman/GPR recuperam a tendência com IC cobrindo, saída = exatamente os
    dias faltantes e série intacta, flanco anômalo não usado como evidência, linhas da
    avaliação) e `tests/test_lacunas_page.py`.

- **Camada de insights via LLM — completo (2026-09-24).**
  - Provedor: **Gemini** (Google AI Studio, plano gratuito), `gemini-3.5-flash` configurável via
    `GEMINI_MODEL`; chave em `GEMINI_API_KEY` **só no `.env`** (ignorado pelo git; o
    `.env.example` tem o campo vazio). Chaves novas do AI Studio podem começar com `AQ.` (não só
    `AIza`) — verificado que funcionam no endpoint `generativelanguage.googleapis.com`.
    `thinkingConfig.thinkingBudget = 0`: sem isso o modelo gastava o limite de saída
    "pensando" e devolvia texto vazio (observado); narrar números não precisa de raciocínio.
  - `src/w8t/insights/summary.py`: `build_summary(series, periods, scope, today)` — **única
    entrada do LLM**; números já calculados e arredondados (peso, tendência com IC e método,
    platôs com períodos sobrepostos, atípicas, lacunas, previsão da combinação Kalman+Holt com
    IC, ruído estimado da balança, meta). Tempo como "há N dias" / "N dias após a última
    medição", **sem datas** (mantém o texto fixo da demo válido, já que a demo é ancorada em
    hoje). Nunca inclui a série bruta. Período encerrado → sem previsão.
  - `src/w8t/insights/providers.py`: interface `LLMProvider.generate(system, prompt)`;
    `GeminiProvider` (REST, erros viram `ProviderError` sem vazar a chave; 429 → mensagem de
    cota), `FakeProvider` para testes.
  - `src/w8t/insights/narrator.py`: instruções (só números do JSON, sem causas, sem conselhos,
    previsão sempre com IC, "dados insuficientes" dito como tal, sem datas, vírgula decimal,
    prioridades) + **checagem de grounding**: todo número do texto é comparado (tolerância 0,05)
    com os números do resumo; os que não batem são listados e a UI avisa. `python -m
    w8t.insights.narrator --demo` regenera `src/w8t/insights/demo_summary.md` (texto fixo da
    demo, gerado uma vez pelo Gemini; todos os números verificados — teste garante).
  - `src/w8t/patterns/pipeline.py`: `detect(series)` (Kalman com fallback para baselines)
    extraído do `Home.py` e compartilhado com o resumo, para os dois descreverem o mesmo.
  - `src/w8t/app/pages/5_Resumo.py`: escopo como no dashboard, JSON enviado visível num
    expander, texto **só ao clicar** (cada clique usa cota). Demo: texto fixo, nunca chama a
    API. Sem chave: explica como configurar. Testado de ponta a ponta com o Gemini real em modo
    local sobre dados sintéticos (não sobre os dados reais do usuário).
  - Testes: `tests/test_insights.py` (resumo sem datas/série bruta, insuficiente, período
    encerrado + meta, grounding pega números inventados, prompt contém regras e só o resumo,
    Gemini com rede simulada: formato do request, erros, chave nunca vaza; texto da demo
    verificado contra o resumo) e `tests/test_resumo_page.py` (sem chave, só gera ao clicar,
    números inventados sinalizados, demo nunca chama a API). Nenhum teste chama a API real.
  - Pendência: `pyproject.toml` trocou o extra `insights` de `anthropic` para `requests`;
    **`uv.lock` precisa ser regenerado com `uv lock`** (uv não estava no PATH da sessão).
  - Privacidade: no plano gratuito os termos do Google permitem usar o conteúdo enviado para
    melhorar produtos; só números agregados são enviados, e só quando o usuário clica.

- **Padrão semanal no Kalman (fine-tuning, item 1 de 3) — completo (2026-09-24).**
  - `forecasting/kalman.fit_smooth_trend(series, weekly=None)`: com `None` (padrão) ajusta o
    modelo sem e com componente semanal fixo (`seasonal=7`, determinístico) e só mantém o
    semanal se a **log-verossimilhança preditiva um-passo-à-frente**, nas mesmas observações
    após 21 dias de burn-in, for maior (`WEEKLY_MIN_GAIN = 0`; o escore preditivo já penaliza
    os estados extras). Só tenta com ≥ 56 dias. (AIC/llf do statsmodels não servem aqui: o
    burn-in difuso difere entre os modelos, então compararia observações diferentes.)
  - **Calibração do critério:** séries planas sem padrão, 0/30 seleções falsas; efeito de fim
    de semana 0,5 / 0,35 / 0,25 / 0,15 kg (ruído 0,35) selecionado 97% / 80% / 37% / 10%; nos 6
    cenários rotulados × 8 sementes, 8/8 no semanal e 0/40 nos demais.
  - Helpers: `has_weekly`, `smoothed_signal` (nível + efeito semanal e variância, via vetor de
    observação `design`), `weekly_effect_range`. O estado 0 continua sendo o nível
    **dessazonalizado** e o 1 a inclinação — consumidores não mudam de significado. Bug pego pelo
    teste: `smoothed_states` desempacotava exatamente 2 estados.
  - Consumidores: previsão (`get_forecast` inclui o efeito), detectores (inovações já descontam
    o efeito → fim de semana deixa de parecer anomalia), lacunas (estimativa = nível + efeito do
    dia), resumo LLM (`padrao_semanal` com amplitude), dashboard (aviso "Padrão semanal
    detectado", linha verde = tendência sem o efeito).
  - **Resultado (antes → depois, só o cenário semanal muda; demais idênticos, variação máx.
    0,002 kg):** lacunas MAE vs. peso verdadeiro 0,203 → 0,072 kg (2,8x), intervalo 17% mais
    estreito, cobertura 94%; anomalias recall 67% → 92% (FP 0,08 → 0,17 /100); previsão Kalman
    MAE −9% a −14% por horizonte (h=14: 0,311 → 0,267), combinação h=14 0,311 → 0,277;
    tendência 82% → 79% (leve piora). Custo: Kalman ajusta 2 modelos em séries ≥ 56 dias
    (benchmark de previsão ~2x mais lento).
  - Testes novos em `tests/test_patterns_kalman.py` (detecta só quando existe, amplitude, não
    tenta em série curta, detectores com modelo semanal) e `tests/test_home_page.py` (aviso).

Armadilha de teste já resolvida (documentada para não reintroduzir): `w8t.config.settings` é um
singleton resolvido no primeiro import do módulo. Se outro arquivo de teste importar
`w8t.config`/`w8t.data.db` antes de um teste tentar trocar `DATABASE_URL` via `monkeypatch.setenv`,
a troca chega tarde demais e o teste acaba usando o banco local real. A correção em
`test_registro_de_peso_page.py` faz `monkeypatch.setattr` direto em `w8t.data.db.engine` e
`w8t.data.db.SessionLocal` (que `get_session()` sempre relê no momento da chamada), em vez de mexer
em variável de ambiente. Qualquer novo teste que precise de um banco isolado deve seguir o mesmo
padrão.

Próximo passo (fine-tuning, ordem combinada com o usuário em 2026-09-24): (2) revisitar o GPR
com seleção de kernel pelo benchmark multi-série (não pela demo) → seguir ajustando os modelos até
o produto estar "bem amarrado" → só então polimento de UI (animações/CSS) e deploy da demo.

Sem autenticação/login por decisão (2026-09-24): não é foco do projeto; pode ser adicionado
depois, se necessário.

Roadmap de mais longo prazo, na ordem recomendada (debate de 2026-09-23; `Period`, dashboard,
baselines de tendência/platô/anomalia, forecasting baselines e backtesting, Holt, Kalman, GPR, combinação, detectores via Kalman e reconstrução de lacunas já
feitos) → camada de insights via LLM (feita em 2026-09-24, Gemini gratuito).

## Convenções de trabalho

- Construir incrementalmente, uma feature por vez, sem pular fases (ex.: não implementar
  forecasting antes de ter o registro básico e o stats engine funcionando).
- Manter este arquivo atualizado a cada mudança relevante de arquitetura ou fase concluída.
- Toda lógica de cálculo (stats/forecasting) deve ter teste — é código numérico fácil de quebrar
  silenciosamente.
