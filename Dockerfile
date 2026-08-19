# Multi-stage build. The runtime image carries no build toolchain, no training
# dependencies (mlflow's server extras, pyarrow) beyond what inference needs, and no
# source data — it is what gets pulled on every cold start, so its size is latency.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir --target=/deps ".[serve]" && \
    find /deps -name "__pycache__" -type d -prune -exec rm -rf {} + && \
    find /deps -name "tests" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

# Run unprivileged: a compromised inference process should not own the filesystem.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

ENV PYTHONPATH=/deps \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/candidate

COPY --from=builder /deps /deps
COPY src/flight_delay ./flight_delay

# The model artifact is baked in rather than fetched at boot. Container Apps scales
# this service to zero, so a registry round-trip would be paid on every cold start and
# would add a network failure mode to the startup path.
COPY models/candidate ./models/candidate
COPY models/candidate_metrics.json ./models/candidate_metrics.json

USER appuser
EXPOSE 8000

# Container Apps injects PORT; default to 8000 for local runs.
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "python -m uvicorn flight_delay.serve:app --host 0.0.0.0 --port ${PORT}"]
