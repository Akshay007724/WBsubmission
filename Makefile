# Common tasks. Every target works without an API key except `run-live`.
PY ?= .venv/bin/python
.PHONY: install lint test run run-live eda replay submission audit clean wheelhouse docker

install:            ## create .venv and install the package with dev tools
	python3 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"

lint:               ## ruff lint + format check + mypy
	$(PY) -m ruff check acquirer_engine tests
	$(PY) -m ruff format --check acquirer_engine tests
	$(PY) -m mypy

test:               ## tests with coverage (fails under 85%)
	$(PY) -m pytest --cov

run:                ## full pipeline, no key (deterministic template text)
	$(PY) -m acquirer_engine --provider offline

replay:             ## reproduce sample_output/ exactly from .cache/llm, no key
	$(PY) -m acquirer_engine --provider replay

run-live:           ## real run (needs ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)
	$(PY) -m acquirer_engine

eda:                ## stage 0 data profile only
	$(PY) -m acquirer_engine --eda-only

audit:              ## known-vulnerability scan of installed dependencies
	$(PY) -m pip_audit

submission:         ## build submission.zip (secret-scanned)
	PYTHON=$(PY) bash scripts/make_submission.sh

wheelhouse:         ## pre-fetch Linux wheels for the Docker build (helps on slow networks)
	$(PY) -m pip download -q --dest wheelhouse --only-binary=:all: --python-version 3.12 --implementation cp \
	  --platform manylinux_2_28_$$(uname -m | sed 's/arm64/aarch64/') \
	  --platform manylinux_2_17_$$(uname -m | sed 's/arm64/aarch64/') -r requirements.txt

docker:             ## build the image and reproduce sample_output/ inside it
	docker build -t acquirer-engine .
	mkdir -p out/docker && docker run --rm -v "$$PWD/out/docker:/app/out" acquirer-engine
	cmp out/docker/healthcare-services-200m-regional-private/report.json sample_output/report.json && echo "IDENTICAL"

clean:
	rm -rf out/ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov submission.zip
