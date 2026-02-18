# SaborAI

> Sistema multi-agente com RAG para analise inteligente de cardapios de restaurantes.

[![CI](https://github.com/nikolasdehor/CardapIA/actions/workflows/ci.yml/badge.svg)](https://github.com/nikolasdehor/CardapIA/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-purple.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## O que e o SaborAI?

O SaborAI responde perguntas sobre cardapios de restaurantes usando uma **arquitetura multi-agente com RAG**. Um agente supervisor roteia cada query para um ou mais agentes especialistas, agrega as respostas e entrega um resultado estruturado.

**Casos de uso:**
- *"Quais pratos sao adequados para veganos com intolerancia a gluten?"*
- *"Monte um combo completo por ate R$60 para um casal."*
- *"Avalie a qualidade das descricoes do cardapio e sugira melhorias para aumentar a conversao."*

---

## Arquitetura

```
Query do Usuario
      |
      v
SupervisorAgent  <- roteamento via LLM (GPT-4o-mini)
      |
      |---> NutritionAgent       (restricoes alimentares, alergenicos, calorias)
      |---> RecommendationAgent  (combos, filtro por budget, harmonizacoes)
      \---> QualityAgent         (score de qualidade descritiva, otimizacao de conversao)
               |
               v
         ChromaDB (vector store local)
               ^
               |
      Pipeline de Ingestao
      (PDF/TXT -> chunking -> deduplicacao -> embedding -> persist)
```

### Decisoes de design

| Decisao | Justificativa |
|---|---|
| Supervisor + especialistas | Separacao de responsabilidades; cada agente tem prompts focados e tunados |
| Roteamento via LLM | Flexivel — lida com queries ambiguas que abrangem multiplos dominios |
| Deduplicacao por hash de conteudo | Evita drift de embedding ao re-ingerir o mesmo conteudo |
| LLM-as-judge nos evals | Metricas de qualidade escalaveis sem necessidade de dataset anotado manualmente |
| Dados sinteticos via LLM | Aumenta cobertura de dados para cuisines/faixas de preco com baixa representacao |

---

## Framework de Avaliacao

O suite de evals mede 4 dimensoes por caso de teste:

| Metrica | Metodo |
|---|---|
| **Relevancia** | GPT-4o-mini como juiz (0-1) |
| **Groundedness** | GPT-4o-mini verifica se a resposta esta ancorada no contexto recuperado |
| **Routing Accuracy** | Agentes esperados vs. agentes selecionados |
| **Keyword Coverage** | Heuristica — termos esperados presentes na resposta |

Os resultados sao persistidos em `data/eval_results/` como JSON com timestamp para rastreamento de experimentos.

### Dashboard de Avaliacao

Painel Streamlit para visualizacao interativa de metricas:

```bash
streamlit run dashboard.py
```

Funcionalidades:
- Metricas agregadas do ultimo run com delta vs run anterior
- Drill-down por caso de teste
- Grafico radar de scores por caso
- Trends de metricas ao longo do tempo
- Execucao de novas avaliacoes via UI

---

## Geracao de Dados Sinteticos

Script que gera cardapios realistas via LLM para aumentar a cobertura de dados:

```bash
# Gerar 5 cardapios de cuisines diversas
python -m scripts.generate_synthetic_menus --count 5

# Gerar para cozinha especifica e ingerir automaticamente
python -m scripts.generate_synthetic_menus --cuisine Japonesa --count 2 --ingest

# Gerar faixa premium
python -m scripts.generate_synthetic_menus --price-tier premium --count 3 --output data/sample_menus/
```

O script gera um manifesto JSON com metadata (Pydantic) para rastreamento de qualidade dos dados gerados.

---

## Referencia da API

| Metodo | Endpoint | Descricao |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest/file` | Upload de cardapio em PDF ou TXT |
| `POST` | `/ingest/text` | Ingestao de texto puro |
| `POST` | `/query` | Query multi-agente |
| `POST` | `/evaluate` | Executa o suite de evals |

Documentacao interativa disponivel em `http://localhost:8000/docs` (Swagger UI).

---

## Quickstart

### 1. Clone e configure

```bash
git clone https://github.com/nikolasdehor/CardapIA.git
cd CardapIA
cp .env.example .env
# Adicione sua OPENAI_API_KEY no .env
```

### 2. Rode com Docker (recomendado)

```bash
docker compose up --build
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### 3. Ou rode localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 4. Ingira o cardapio de exemplo

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@data/sample_menus/bella_terra.txt" \
  -F "menu_name=Bella Terra"
```

### 5. Faca queries

```bash
# Recomendacao com budget
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Monte um combo completo por ate R$60 para um casal vegano."}'

# Avaliacao de qualidade
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Avalie a qualidade das descricoes do cardapio e sugira melhorias."}'

# Restricoes alimentares
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quais pratos nao contem gluten nem laticinios?"}'
```

### 6. Gere dados sinteticos

```bash
python -m scripts.generate_synthetic_menus --count 3 --ingest
```

### 7. Execute os evals

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"suite": "default"}'
```

### 8. Visualize no dashboard

```bash
streamlit run dashboard.py
```

---

## Stack

- **Runtime:** Python 3.11
- **API:** FastAPI + Uvicorn
- **Orquestracao LLM:** LangChain (agents, RAG, prompts)
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** ChromaDB (persistencia local)
- **Dashboard:** Streamlit + Plotly
- **Containerizacao:** Docker + Docker Compose
- **CI/CD:** GitHub Actions (lint, test, build, deploy)

---

## Estrutura do Projeto

```
saborai/
├── agents/
│   ├── supervisor.py              # Roteamento e consolidacao
│   ├── nutrition.py               # Especialista em dietas e alergenicos
│   ├── recommendation.py         # Especialista em combos e harmonizacoes
│   └── quality.py                 # Especialista em qualidade e conversao
├── api/
│   ├── main.py                    # App FastAPI e endpoints
│   └── settings.py                # Configuracao via Pydantic-settings
├── ingestion/
│   └── pipeline.py                # Load -> chunk -> deduplica -> embed
├── evals/
│   └── runner.py                  # Framework LLM-as-judge
├── scripts/
│   └── generate_synthetic_menus.py  # Geracao de dados sinteticos via LLM
├── data/
│   └── sample_menus/              # Cardapios de exemplo e sinteticos
├── tests/
│   ├── conftest.py                # Fixtures e configuracao de testes
│   └── test_api.py                # Testes de integracao
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint, test, build Docker
│       └── cd.yml                 # Deploy staging/production
├── dashboard.py                   # Dashboard Streamlit de evals
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## CI/CD

O pipeline de CI/CD roda no GitHub Actions:

**CI (`ci.yml`):**
1. **Lint & Format** — Ruff check + format
2. **Tests** — Unit tests (sem API key)
3. **Build Docker** — Build e push para GitHub Container Registry
4. **Integration Tests** — Testes com API key real (apenas na main)

**CD (`cd.yml`):**
1. **Build Release** — Imagem com tag semver
2. **Deploy Staging** — Deploy automatico
3. **Smoke Tests & Evals** — Health check + eval suite no staging
4. **Deploy Production** — Apenas apos smoke tests passarem

---

## Roadmap

- [x] Sistema multi-agente com RAG (Supervisor + 3 especialistas)
- [x] Pipeline de ingestao com deduplicacao
- [x] Framework de avaliacao LLM-as-judge
- [x] Geracao de dados sinteticos via LLM
- [x] Dashboard de evals (Streamlit)
- [x] CI/CD com GitHub Actions
- [ ] Execucao assincrona dos agentes (chamadas paralelas)
- [ ] Streaming de respostas via SSE
- [ ] Suporte a cardapios em imagem (vision + OCR)
- [ ] Suporte multilingue

---

## Contribuindo

PRs sao bem-vindos! Abra uma issue descrevendo o que deseja implementar antes de comecar.

---

## Licenca

MIT
