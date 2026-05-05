# Financial RAG Assistant

> A production-style **Retrieval-Augmented Generation (RAG)** system for querying Peruvian bank annual reports — built end-to-end from document ingestion to a streaming chat UI, with an LLM-as-judge evaluation framework.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/tests-137%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Project Demonstrates

This project was built **phase by phase** as an AI Engineering portfolio piece. Every component was written from scratch to show understanding of the underlying mechanics — not just gluing frameworks together.

| Area | Implementation |
|---|---|
| **RAG Pipeline** | Full ingest → chunk → embed → retrieve → generate loop |
| **Document Parsing** | PDF (pypdf) and plain-text ingestion with metadata |
| **Chunking Optimization** | Recursive character splitting; chunk size tuned via empirical evaluation |
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` (local, no API key) |
| **Vector Search** | FAISS with cosine similarity, retrieval scores, and source filtering |
| **LLM Generation** | Ollama (qwen3 family) with pluggable backend via Abstract Base Class |
| **Streaming API** | FastAPI + Server-Sent Events; per-request model selection |
| **Streaming UI** | Next.js 16 chat interface with token streaming and citation display |
| **LLM-as-Judge Eval** | Custom faithfulness scorer (0–3 scale) with score extraction heuristics |
| **Benchmark Runner** | Multi-config evaluation across models, top-k, and think-mode variants |
| **Chunking Experiments** | Automated grid search over chunk sizes with index rebuild |
| **Containerization** | Docker Compose (API + Frontend + Ollama) for one-command deploy |
| **Testing** | 137 unit tests; mock pipeline injection pattern for API tests |

---

## Key Results

Evaluated on a 12-question benchmark covering factual retrieval, risk analysis, and out-of-scope detection across two real bank annual reports.

| Model | think | Faithfulness | Source Hit Rate | Avg. Generation |
|---|---|---|---|---|
| qwen3:4b | off | 58% | 100% | ~35s |
| qwen3:8b | off | 58% | 100% | ~5s |
| **qwen3:14b** | **off** | **94%** | **100%** | **~8s** |
| qwen3:14b | on | 94% | 100% | ~22s |

> Faithfulness improved **+30 percentage points** (63.9% → 94%) after two rounds of optimization: chunk size tuning (800/100 tokens) and fixing the judge prompt to read actual chunk content instead of citation filenames.

> **Extended thinking (think=True) did not improve faithfulness** — 94% in both modes — but added 2.7× latency overhead (~22s vs ~8s). This confirms that for grounded RAG the bottleneck is retrieval quality, not reasoning depth. `qwen3:14b think=off` is the optimal production config.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Annual Reports (PDF)         │
                        │   Interbank 2025 · Scotiabank 2025   │
                        └──────────────────┬──────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Ingestion & Chunking   │
                              │  RecursiveCharSplitter   │
                              │  chunk_size=800  ovlp=100│
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │       Embeddings         │
                              │  all-MiniLM-L6-v2        │
                              │  384-dim · local         │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │      FAISS Index         │
                              │  943 chunks · cosine sim │
                              └────────────┬────────────┘
                                           │
          User Question ──────►  ┌─────────▼──────────┐
                                 │    RAG Pipeline      │
                                 │  retrieve top-k      │
                                 │  build context       │
                                 │  generate answer     │
                                 └─────────┬──────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │              FastAPI (uvicorn)               │
                    │  POST /ask · POST /ask/stream · GET /models  │
                    └──────────────────────┬──────────────────────┘
                                           │ SSE tokens
                              ┌────────────▼────────────┐
                              │     Next.js 16 Chat UI   │
                              │  Model selector · Filter │
                              │  Citations · Scores      │
                              └─────────────────────────┘
```

### Design Decisions

**Why FAISS over a managed vector DB?**
FAISS keeps the project fully local — no cloud dependencies, no API keys, reproducible on any machine. The `BaseVectorStore` ABC makes it trivially swappable for Pinecone or Qdrant in production.

**Why Ollama?**
Privacy-preserving inference for financial documents. All computation stays on-device. The `BaseGenerator` ABC allows substituting OpenAI or any other backend.

**Why a custom LLM-as-judge instead of RAGAS?**
RAGAS requires an OpenAI key for grading; this project scores faithfulness with the same local models used for generation. The judge parses scores from the *tail* of the response (not the first digit) to avoid false matches on citation numbers like `[2]`.

**Why chunk_size=800?**
Empirically determined via `scripts/chunk_experiment.py`: the 800/100 config reduced chunk count from 1,421 to 943 (fewer, denser chunks) while maintaining retrieval quality, improving faithfulness from 63.9% to 83.3%.

---

## Quick Start — Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/financial-rag-assistant
cd financial-rag-assistant

# Start everything (API + Frontend + Ollama)
docker compose up

# Pull the default model (first time only — ~3 GB)
docker compose exec ollama ollama pull qwen3:8b
```

- **Chat UI** → http://localhost:3000
- **API docs** → http://localhost:8000/docs

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- Node.js 18+ (for the frontend)

### Backend

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy environment config
cp .env.example .env

# 3. Pull a model
ollama pull qwen3:8b

# 4. Build the vector index from sample documents
python scripts/build_index.py --chunk-size 800 --chunk-overlap 100

# 5. Start the API
uvicorn financial_rag.api.app:app --reload
# → http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Project Structure

```
financial-rag-assistant/
├── src/financial_rag/
│   ├── ingestion/           # BaseLoader, TextLoader, PDFLoader, factory
│   ├── chunking/            # RecursiveCharacterSplitter, Chunk dataclass
│   ├── embeddings/          # BaseEmbedder, SentenceTransformerEmbedder, MockEmbedder
│   ├── retrieval/           # BaseRetriever, VectorRetriever, RetrievalResult
│   ├── generation/          # BaseGenerator, OllamaGenerator (think mode), MockGenerator
│   ├── pipeline/            # RAGPipeline, RAGResponse, factory
│   ├── api/                 # FastAPI app, routes, schemas (SSE streaming)
│   ├── evaluation/          # LLM-as-judge, BenchmarkRunner, BenchmarkConfig
│   └── config.py            # pydantic-settings config
├── frontend/                # Next.js 16 chat UI
│   └── src/
│       ├── app/             # Root page layout
│       ├── components/      # Sidebar (model/filter/top-k), ChatInput, MessageBubble
│       ├── hooks/           # useChat (streaming state machine)
│       └── lib/             # API client, TypeScript types
├── scripts/
│   ├── build_index.py       # Build FAISS index with configurable chunk params
│   ├── chunk_experiment.py  # Grid search over chunk sizes
│   └── evaluate.py          # CLI for running benchmark configs
├── tests/
│   └── unit/                # 137 tests — API, pipeline, chunking, retrieval, eval
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── init-ollama.sh
├── data/
│   ├── samples/             # Interbank 2025 (162pp) · Scotiabank Perú 2025 (120pp)
│   ├── processed/           # FAISS index (943 chunks · 384-dim)
│   └── eval/                # Benchmark results (CSV + JSON)
├── docs/
│   ├── architecture.md
│   └── learning_notes.md
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Evaluation Framework

The benchmark runs a 12-question set across three categories against any pipeline configuration:

| Category | Questions | Purpose |
|---|---|---|
| **Factual** (`ib_*`, `sb_*`) | 8 | Test grounded retrieval of specific financial data |
| **Analytical** | 2 | Test multi-chunk synthesis (e.g., risk management) |
| **Out-of-scope** (`oos_*`) | 2 | Test graceful refusal (PBI data, Bitcoin) |

**Faithfulness scoring (LLM-as-judge, 0–3 scale):**

```
3 — Fully grounded: every claim traces to the retrieved chunks
2 — Mostly grounded: minor unsupported additions
1 — Partially grounded: unsupported claims present
0 — Hallucination: contradicts sources or invents information
```

**Run the benchmark:**

```bash
# Compare 3 models at top_k=5
python scripts/evaluate.py --models qwen3:4b qwen3:8b qwen3:14b --top-k 5 --out data/eval/final

# Run with extended thinking mode (qwen3 think=True)
python scripts/evaluate.py --models qwen3:14b --top-k 5 --think --out data/eval/final
```

Results are saved as both CSV (per-question detail) and JSON (config-level summary), merging with previous runs non-destructively.

---

## Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=financial_rag --cov-report=term-missing
```

137 unit tests covering: document ingestion · chunking · embeddings · vector store · retrieval · generation · RAG pipeline · FastAPI endpoints (mock pipeline injection) · evaluation runner.

The API tests use a **mock pipeline injection pattern** — no Ollama or FAISS required to run the test suite.

---

## Development Commands

```bash
make help          # List all targets
make test          # Run pytest
make lint          # ruff + mypy
make format        # ruff format
make api           # Start FastAPI server
make evaluate      # Run full benchmark
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (faiss-cpu) |
| LLM Backend | Ollama (qwen3:4b / 8b / 14b) |
| API | FastAPI + uvicorn (SSE streaming) |
| Frontend | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui |
| Testing | pytest (137 tests) |
| Containerization | Docker Compose |
| Config | pydantic-settings |

---

## License

MIT
