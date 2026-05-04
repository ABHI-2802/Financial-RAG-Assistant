# Financial RAG Assistant

> A production-style Retrieval-Augmented Generation system for financial documents — built for learning and portfolio demonstration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| RAG Architecture | End-to-end pipeline: ingest → chunk → embed → retrieve → generate |
| Document Ingestion | PDF and plain-text parsing with metadata extraction |
| Chunking Strategies | Recursive character splitting with overlap |
| Embeddings | `sentence-transformers` (local, no API key required) |
| Vector Search | FAISS with cosine similarity and retrieval scores |
| Grounded Generation | LLM answers with source citations and confidence |
| LLM Abstraction | Pluggable backend: Ollama (local), OpenAI, or Mock |
| API Serving | FastAPI with async endpoints, request validation, OpenAPI docs |
| UI | Streamlit interface with document upload and Q&A |
| Evaluation | RAGAS-style metrics: faithfulness, answer relevancy, context recall |
| Testing | pytest with unit + integration test separation |
| Containerization | Docker Compose for one-command local deployment |
| Code Quality | ruff, mypy, pre-commit hooks |

## Architecture

```
Documents (PDF/TXT)
      ↓
  Ingestion & Chunking
      ↓
  Embeddings (sentence-transformers)
      ↓
  Vector Store (FAISS / Chroma)
      ↓
  Query → Retrieve Top-K Chunks
      ↓
  LLM Generation (Ollama / OpenAI / Mock)
      ↓
  Answer + Citations + Retrieval Scores
      ↓
  FastAPI ←→ Streamlit UI
```

See [`docs/architecture.md`](docs/architecture.md) for detailed diagrams and design decisions.

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/financial-rag-assistant
cd financial-rag-assistant

# 2. Install dependencies
make install

# 3. Copy and configure environment
cp .env.example .env

# 4. Run the API
make api

# 5. Run the UI (separate terminal)
make ui
```

## Project Structure

```
financial-rag-assistant/
├── src/financial_rag/       # Core library
│   ├── ingestion/           # Document loading and parsing
│   ├── chunking/            # Text splitting strategies
│   ├── embeddings/          # Embedding model wrappers
│   ├── retrieval/           # Vector store and search
│   ├── generation/          # LLM backends and prompting
│   ├── api/                 # FastAPI application
│   ├── ui/                  # Streamlit application
│   └── evaluation/          # RAG evaluation metrics
├── data/
│   ├── raw/                 # Original documents (gitignored)
│   ├── processed/           # Chunked + embedded data (gitignored)
│   └── samples/             # Sample documents for testing
├── tests/
│   ├── unit/                # Fast, isolated tests
│   └── integration/         # End-to-end pipeline tests
├── docs/                    # Architecture and learning notes
├── notebooks/               # Exploratory analysis
├── scripts/                 # One-off utility scripts
├── docker/                  # Dockerfiles
├── pyproject.toml           # Project metadata and dependencies
├── Makefile                 # Common commands
└── .env.example             # Environment variable template
```

## Development

```bash
make help          # Show all available commands
make test          # Run test suite
make lint          # Run ruff + mypy
make format        # Auto-format code
make ingest        # Ingest sample documents
make evaluate      # Run evaluation pipeline
```

## Learning Notes

This project was built phase by phase as a learning exercise. See [`docs/learning_notes.md`](docs/learning_notes.md) for notes on each decision, trade-off, and concept encountered.

## License

MIT
