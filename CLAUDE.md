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
3. Holt / Holt-Winters (exponential smoothing) — `statsmodels`
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

Decidido em 2026-09-23, ainda não implementado. Depende do forecasting engine (Kalman/GPR) já
existir e estar validado — não é a próxima coisa a construir.

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
insights layer       → opcional, chama Claude API só para narrar números já calculados
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

Fase: **item 1 da spec concluído** (registro diário de peso) **+ `Period` no data layer**
(entidade nova, fora da numeração da spec original, ver seção "Períodos/ciclos" acima). Repo
público em `github.com/giovaniolivr/w8t`.

Feito:
- Repo git inicializado, identidade de commit configurada (`Giovaniolivr@gmail.com`), remoto no
  GitHub (`giovaniolivr/w8t`, público) com push funcionando via GitHub CLI autenticado localmente.
- Estrutura de pastas (`src/w8t/{app,core,data,forecasting,insights}`, `tests/`).
- `pyproject.toml` com dependências via `uv` (grupo dev: pytest/ruff; extras: `postgres`,
  `insights`).
- `src/w8t/config.py`: settings via `pydantic-settings`, modo `local`/`demo`, `.env.example`.
- Entrypoint Streamlit mínimo (`src/w8t/app/Home.py`).
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
  arquivo. Migração `86da55a7e986`. Ainda não consumido por nenhuma análise (não há análise além
  do CRUD ainda).

Armadilha de teste já resolvida (documentada para não reintroduzir): `w8t.config.settings` é um
singleton resolvido no primeiro import do módulo. Se outro arquivo de teste importar
`w8t.config`/`w8t.data.db` antes de um teste tentar trocar `DATABASE_URL` via `monkeypatch.setenv`,
a troca chega tarde demais e o teste acaba usando o banco local real. A correção em
`test_registro_de_peso_page.py` faz `monkeypatch.setattr` direto em `w8t.data.db.engine` e
`w8t.data.db.SessionLocal` (que `get_session()` sempre relê no momento da chamada), em vez de mexer
em variável de ambiente. Qualquer novo teste que precise de um banco isolado deve seguir o mesmo
padrão.

Próximo passo (não iniciado): dashboard principal (spec item 2) — médias móveis, peso
atual/inicial/mín/máx, variação, ritmo, com opção de escopo por período (usando
`periods.get_period_for_date`) ou histórico inteiro. A lógica de cálculo deve nascer em `core/`
(stats engine, determinístico, sem ML) com testes próprios antes de virar tela.

Roadmap de mais longo prazo, na ordem recomendada (debate de 2026-09-23; `Period` já feito) →
dashboard (item 2) → tendência/platô/anomalia com baselines simples (itens 8-10) → forecasting
engine com Kalman/GPR (itens 6, 11-14) → backtesting (item 15) → revisão opcional de
tendência/platô/anomalia usando a posterior do GPR, comparada contra o baseline via backtesting →
reconstrução de gaps (reusa Kalman/GPR já validados) → camada de insights via LLM (última fase,
condicionada a viabilidade).

## Convenções de trabalho

- Construir incrementalmente, uma feature por vez, sem pular fases (ex.: não implementar
  forecasting antes de ter o registro básico e o stats engine funcionando).
- Manter este arquivo atualizado a cada mudança relevante de arquitetura ou fase concluída.
- Toda lógica de cálculo (stats/forecasting) deve ter teste — é código numérico fácil de quebrar
  silenciosamente.
