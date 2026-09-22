# W8T

Acompanhamento inteligente de perda/ganho de peso: registro diário, métricas de tendência e
previsões estatísticas/ML com intervalos de incerteza. Projeto de estudo (ML/estatística), não
comercial — construído incrementalmente, feature por feature. Veja `CLAUDE.md` para o estado
atual do projeto e as decisões de arquitetura.

## Stack

- Python 3.12, gerenciado com [uv](https://docs.astral.sh/uv/)
- [Streamlit](https://streamlit.io/) para a interface
- SQLAlchemy 2.0 + Alembic para persistência
- pandas / numpy / statsmodels / scikit-learn para a camada estatística e de forecasting
- pytest para testes

## Modos de execução

| Modo | Uso | Dados | Banco |
|---|---|---|---|
| `local` (default) | uso pessoal | reais, persistentes | SQLite local |
| `demo` | deploy público (portfólio) | sintéticos, resetáveis | Postgres (Neon) |

Configurado via `.env` (copie `.env.example`). Dados reais nunca são expostos na versão hospedada.

## Rodando localmente

```bash
uv sync --extra postgres --extra insights   # ou só `uv sync` se não precisar dessas extras
uv run alembic upgrade head                 # cria/atualiza o schema do banco configurado em .env
uv run streamlit run src/w8t/app/Home.py
```

## Migrações

Schema versionado com Alembic, alvo determinado por `DATABASE_URL`/`APP_ENV` (ver `alembic/env.py`).

```bash
uv run alembic revision --autogenerate -m "descricao da mudanca"
uv run alembic upgrade head
```

## Testes

```bash
uv run pytest
```

## Estrutura

```
src/w8t/
  app/           # Streamlit UI (entrypoint: Home.py, páginas em app/pages/)
  core/          # stats engine — cálculos determinísticos (médias, tendência, platô...)
  forecasting/   # modelos de previsão plugáveis (estatísticos/ML) + backtesting
  data/          # modelos SQLAlchemy, sessão de banco, repositórios
  insights/      # camada opcional de narrativa via IA generativa (Claude API)
alembic/         # migrações de schema
tests/
```
