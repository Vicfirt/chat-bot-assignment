.PHONY: install lock test ingest up up-obs down eval loadtest

install:
	pip install -r requirements.lock

lock:
	uv pip compile requirements.txt -o requirements.lock

test:
	LLM_MODE=dummy python -m pytest -q

ingest:
	docker compose run --rm ingest

up:
	docker compose up --build

up-obs:
	docker compose --profile observability up --build

down:
	docker compose --profile observability --profile ingest down -v

eval:
	python -m eval.run_eval

loadtest:
	python -m loadtest.run_load --api-url http://localhost:8000 --n 100 --concurrency 4 --warmup 5
