# RAG Observability

[![CI](https://github.com/agaleaniket10/rag-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/agaleaniket10/rag-observability/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)


End-to-end observability stack for RAG pipelines — tracing, metrics, cost tracking, and automated quality gating.

## Stack

| Layer | Tool |
|---|---|
| Tracing | OpenTelemetry + LangSmith |
| Metrics | Prometheus + Grafana |
| Cost tracking | Custom token-to-USD tracker |
| Evaluation | RAGAS + LLM-as-a-judge |
| CI quality gate | GitHub Actions |

## Project Structure

```
rag-observability/
├── .github/workflows/regression-gate.yml   # CI quality gate
├── src/
│   ├── monitor/
│   │   ├── tracing.py       # OpenTelemetry / LangSmith setup
│   │   ├── metrics.py       # Prometheus metrics definitions
│   │   └── cost_tracker.py  # Token-to-USD accounting
│   ├── engine/
│   │   └── rag_pipeline.py  # RAG pipeline with observability hooks
│   └── main.py              # Entry point
├── tests/evaluation/
│   └── test_quality.py      # RAGAS + LLM-as-a-judge regression tests
├── data/
│   └── golden_dataset.json  # Ground truth Q&A pairs
├── docker-compose.yml        # Prometheus + Grafana stack
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
```

### 3. Start the observability stack

```bash
docker compose up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default credentials: admin / admin)

### 4. Run the pipeline

```bash
python -m src.main
```

### 5. Run quality evaluation tests

```bash
pytest tests/evaluation/test_quality.py -v
```

## Metrics Reference

| Metric | Type | Description |
|---|---|---|
| `rag_requests_total` | Counter | Total requests, labelled by status |
| `rag_latency_seconds` | Histogram | End-to-end pipeline latency |
| `retrieval_latency_seconds` | Histogram | Vector store retrieval latency |
| `llm_latency_seconds` | Histogram | LLM generation latency |
| `retrieved_docs_count` | Histogram | Documents retrieved per query |
| `rag_active_requests` | Gauge | In-flight requests |

## Quality Thresholds (CI Gate)

Thresholds are configurable via environment variables:

| Metric | Default |
|---|---|
| `FAITHFULNESS_THRESHOLD` | 0.80 |
| `ANSWER_RELEVANCY_THRESHOLD` | 0.75 |
| `CONTEXT_RECALL_THRESHOLD` | 0.70 |

## Wiring Up Your Own RAG Logic

Replace the stub methods in `src/engine/rag_pipeline.py`:

- `_retrieve()` — connect to your vector store (Pinecone, Chroma, pgvector, etc.)
- `_generate()` — call your LLM and return the answer + token usage dict

All observability instrumentation is already in place and will activate automatically.