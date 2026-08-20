.PHONY: install lint test data train serve clean

install:
	uv venv --python 3.11 .venv
	uv pip install -e ".[serve,train,dev]"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

test:
	.venv/bin/pytest -m "not network"

test-all:
	.venv/bin/pytest

data:
	.venv/bin/python -m flight_delay.data.build --months 2025-01 2025-02

train:
	.venv/bin/python -m flight_delay.train --backend local

serve:
	.venv/bin/uvicorn flight_delay.serve:app --reload --port 8000

clean:
	rm -rf data/interim data/processed mlruns models
