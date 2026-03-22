.PHONY: install ingest serve eval ui test lint clean

install:
	pip install -e ".[dev]"
	pre-commit install

# build the FAISS index from documents in data/corpus/
ingest:
	python -m edadvisor.ingestion.pipeline

# start the FastAPI backend
serve:
	uvicorn src.edadvisor.serving.app:app --reload --host 0.0.0.0 --port 8000

# start the Streamlit chat UI
ui:
	streamlit run ui/app.py --server.port 8501

# run the ragas evaluation suite
eval:
	python -m edadvisor.evaluation.runner

# run tests
test:
	pytest tests/ -v --cov=src/edadvisor --cov-report=term-missing

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage build dist
