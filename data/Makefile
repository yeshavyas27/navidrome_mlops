# Navidrome MLOps Data Pipeline
# Usage: make <target>
# Prerequisites: source ~/.chi_auth.sh

PYTHON = python3
PIPELINE = pipeline

.PHONY: all ingest validate build-dataset start-api generate clean help

all: ingest validate build-dataset

ingest:
	@echo "=== Running ingest pipeline ==="
	$(PYTHON) $(PIPELINE)/ingest.py
	$(PYTHON) $(PIPELINE)/parse_30music.py

validate:
	@echo "=== Running validation pipeline ==="
	$(PYTHON) $(PIPELINE)/validate.py

build-dataset:
	@echo "=== Building versioned dataset ==="
	$(PYTHON) /tmp/fix_triplets.py

start-api:
	@echo "=== Starting feedback API on port 8000 ==="
	uvicorn pipeline.feedback_api:app --host 0.0.0.0 --port 8000

generate:
	@echo "=== Running data generator ==="
	$(PYTHON) $(PIPELINE)/data_generator.py \
		--endpoint http://localhost:8000 \
		--users 100 \
		--events 500 \
		--songs 1000 \
		--delay 0.1 \
		--verbose

clean:
	@echo "=== Cleaning temp files ==="
	rm -f /tmp/*.parquet /tmp/*.csv /tmp/*.bin /tmp/*.zip

help:
	@echo "Available targets:"
	@echo "  make ingest         - Download FMA + 30Music, upload to Swift"
	@echo "  make validate       - Validate all datasets, reject bad rows"
	@echo "  make build-dataset  - Build versioned train/eval triplets"
	@echo "  make start-api      - Start feedback API on port 8000"
	@echo "  make generate       - Run data generator against API"
	@echo "  make all            - Run ingest + validate + build-dataset"
