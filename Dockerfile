# Slim image that reproduces the committed sample with no API key:
#   docker build -t acquirer-engine . && docker run --rm -v "$PWD/out:/app/out" acquirer-engine
# Any other CLI flags can be passed after the image name (e.g. --provider offline --sector Dental).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first (cached layer), then the package. Wheels pre-fetched into wheelhouse/ (make wheelhouse)
# make the install fully offline (--no-index), so a slow or flaky network inside the Docker VM cannot
# break the build; with an empty wheelhouse it installs from PyPI. The directory always exists (wheelhouse/.gitkeep), so COPY never fails.
COPY requirements.txt ./
COPY wheelhouse/ /wheelhouse/
RUN if ls /wheelhouse/*.whl >/dev/null 2>&1; \
      then pip install --no-index --find-links /wheelhouse -r requirements.txt; \
      else pip install --retries 10 --timeout 120 -r requirements.txt; fi
COPY pyproject.toml README.md ./
COPY acquirer_engine ./acquirer_engine
RUN pip install --no-deps .

# Data, sample targets and the validated-answer cache that replay serves.
COPY ma_transactions_500.csv ./
COPY targets ./targets
COPY .cache/llm ./.cache/llm

RUN useradd --create-home --uid 10001 engine && mkdir -p /app/out && chown -R engine /app
USER engine

ENTRYPOINT ["acquirer-engine"]
CMD ["--provider", "replay", "--out-dir", "/app/out"]
