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

Com [uv](https://docs.astral.sh/uv/) instalado:

```bash
uv sync --extra postgres --extra insights   # ou só `uv sync` se não precisar dessas extras
uv run python -m w8t migrate                # cria/atualiza o schema do banco configurado em .env
uv run python -m w8t                        # sobe o app em http://localhost:8501
```

Sem uv, usando o ambiente virtual já criado (Windows / PowerShell):

```powershell
.venv\Scripts\Activate.ps1      # ativa o ambiente (uma vez por terminal)
python -m w8t migrate            # equivalente ao "manage.py migrate"
python -m w8t                    # equivalente ao "manage.py runserver"; Ctrl+C para parar
```

Argumentos extras vão para o Streamlit, ex.: `python -m w8t --server.port 8600`. Para ver o modo
demonstração localmente, use `APP_ENV=demo` no `.env` **com um `DATABASE_URL` diferente do seu
banco real** (o botão de reset apaga todos os registros do banco configurado).

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
  insights/      # camada opcional de narrativa via IA generativa (Gemini, plano gratuito)
alembic/         # migrações de schema
tests/
```
