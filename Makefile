.PHONY: help install install-dev format lint test test-unit test-integration \
        api ui ingest evaluate clean docker-up docker-down

# ── Colors ───────────────────────────────────
CYAN  := \033[0;36m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "  Financial RAG Assistant — available commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install all dependencies including dev tools
	pip install -e ".[dev,eval,openai]"
	pre-commit install

# ── Code quality ──────────────────────────────
format: ## Auto-format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint: ## Run linter and type checker
	ruff check src/ tests/
	mypy src/

# ── Testing ───────────────────────────────────
test: ## Run full test suite with coverage
	pytest tests/ --cov=src --cov-report=term-missing

test-unit: ## Run only unit tests (fast)
	pytest tests/unit/ -v

test-integration: ## Run only integration tests (slower)
	pytest tests/integration/ -v

# ── Application ───────────────────────────────
api: ## Start FastAPI development server
	uvicorn financial_rag.api.main:app --reload \
		--host $${API_HOST:-0.0.0.0} \
		--port $${API_PORT:-8000}

ui: ## Start Streamlit UI
	streamlit run src/financial_rag/ui/app.py

ingest: ## Ingest sample documents into vector store
	python scripts/ingest.py --source data/samples/

evaluate: ## Run RAG evaluation pipeline
	python scripts/evaluate.py

# ── Docker ────────────────────────────────────
docker-up: ## Start all services with Docker Compose
	docker compose up --build

docker-down: ## Stop all Docker services
	docker compose down

# ── Utilities ─────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	@echo "Cleaned."
