.PHONY: dev
dev:
	@echo ">>> Installing development environment..."
	uv sync --all-extras
	@echo ">>> Done! Activate with 'source .venv/bin/activate'"

.PHONY: test
test:
	uv run pytest inif/tests/

.PHONY: format
format:
	uv run ruff format .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: typecheck
typecheck:
	uv run ty check inif/

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + ; find . -name "*.pyc" -delete ; find . -name "*.pyo" -delete ; find . -type d -name ".pytest_cache" -exec rm -rf {} +

.PHONY: schema
schema:
	uv run python -c "from inif.schema import write_schema; write_schema('schemas/inif.schema.json')"
