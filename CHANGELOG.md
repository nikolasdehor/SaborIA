# Changelog

Todas as mudanças relevantes do projeto SaborAI estão documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [0.2.0] — 2026-02-22

Refatoração completa focada em produção: agentes assíncronos, resiliência,
observabilidade, testes abrangentes e framework de experimentação.

### Adicionado

- **Agentes assíncronos (`arun`)**: todos os 3 agentes especialistas
  (`NutritionAgent`, `RecommendationAgent`, `QualityAgent`) agora possuem
  método `arun()` para execução não-bloqueante via `chain.ainvoke()`.
- **Execução paralela no Supervisor**: `SupervisorAgent.arun()` invoca
  agentes selecionados em paralelo via `asyncio.gather`, reduzindo latência
  em ~1/N quando N agentes são acionados.
- **Retry com exponential backoff** (`agents/retry.py`): decorators
  `@retry_with_backoff` (sync) e `@async_retry_with_backoff` (async) com
  jitter, detecção de erros transientes (rate limit, timeout, 502/503/529)
  e logging estruturado de tentativas.
- **Endpoint SSE streaming** (`POST /query/stream`): Server-Sent Events com
  eventos tipados (`routing`, `agent`, `response`, `done`) para UI
  progressiva em tempo real.
- **Middleware de observabilidade** (`api/middleware.py`):
  `RequestTrackingMiddleware` atribui `X-Request-ID` a cada request,
  `StructuredLogFormatter` emite logs em JSON, e header
  `X-Response-Time-Ms` mede latência.
- **Framework de experimentação** (`experiments/`): `run_experiment()` roda
  grid de modelos × temperaturas × chunk sizes, usa LLM-as-judge para
  scoring (relevância, coerência, completude) e salva resultados em JSON.
  CLI via `python -m experiments.compare_models`.
- **30+ testes novos** em 4 arquivos:
  - `tests/test_retry.py` — 8 testes (sync/async, retry, max exceeded,
    backoff delay).
  - `tests/test_agents.py` — testes de inicialização, temperaturas,
    settings e eval suite.
  - `tests/test_supervisor.py` — 7 testes de routing (nutrition, multi,
    markdown-fenced, parse failure, unknown agents, run completo, falha
    de agente).
  - `tests/test_pipeline.py` — 6 testes (load docs, metadata, IDs
    determinísticos, deduplicação por hash, retriever config).
- **Auto-skip de testes de integração**: `conftest.py` detecta
  `sk-test-*` e pula testes `@pytest.mark.requires_api` automaticamente.
- Campo `latency_ms` no retorno de `supervisor.run()` e `arun()`.
- `retriever_k_full_menu` extraído para `settings.py` (antes hardcoded
  como `k=50`).

### Alterado

- **`POST /query`** agora é `async def` e usa `supervisor.arun()` internamente.
- Versão do projeto unificada em `0.2.0` (constante `APP_VERSION` no
  `api/main.py`, sincronizada com `pyproject.toml`).
- `.env.example` alinhado com defaults reais do `settings.py`
  (`CHUNK_SIZE=1024`, `CHUNK_OVERLAP=128`).
- Prompt do `QualityAgent`: removida formatação markdown (`**...**`) que
  contradizia a instrução "Use plain text only".
- Testes async agora usam `asyncio.run()` em vez do deprecated
  `asyncio.get_event_loop().run_until_complete()`.
- `test_api.py` expandido de 3 para 8 testes (validação 422, bad file type,
  latency_ms).
- Comentário explicativo no `evals/runner.py` sobre uso intencional de
  modelo fixo (`gpt-4o-mini`) para o judge de avaliações.

### Removido

- Dependência `posthog` do `requirements.txt` (nunca era importada).
- Imports não utilizados: `import asyncio` removido dos 3 agentes
  especialistas, `import time` removido de `api/main.py`.

### Corrigido

- `.gitignore` atualizado para incluir `data/experiment_results/` e
  `.venv_test/`.
- Árvore de diretórios no README corrigida de `saborai/` para `CardapIA/`.

### Documentação

- README atualizado com:
  - Diagrama de arquitetura com anotação `asyncio.gather (paralelo)`.
  - Tabela de decisões de design (4 novas entradas: execução paralela,
    retry backoff, SSE streaming, logging estruturado).
  - Referência de API atualizada (`/query` async, `/query/stream` novo).
  - Estrutura de projeto completa (retry.py, middleware.py, experiments/,
    novos testes).
  - Seções "Experimentação com LLMs" e "Observabilidade".
  - Roadmap atualizado (5 itens concluídos).

---

## [0.1.0] — 2026-02-18

### Release inicial

- Sistema multi-agente (Supervisor + 3 especialistas) com RAG via
  LangChain + ChromaDB.
- API FastAPI com endpoints de ingestão (file/text), query e avaliação.
- Pipeline de ingestão com chunking, deduplicação por hash e embeddings
  OpenAI.
- Framework de avaliação LLM-as-judge (relevância, fundamentação, routing
  accuracy, keyword coverage).
- Dashboard Streamlit com upload de cardápio, consultas interativas,
  radar chart de métricas e tendências históricas.
- CI/CD via GitHub Actions (lint, test, build Docker).
- Docker + Docker Compose para deploy local.
