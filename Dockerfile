FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY plugins ./plugins
ARG UV_SYNC_ARGS=""
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project --no-dev $UV_SYNC_ARGS
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev $UV_SYNC_ARGS
COPY datasets.yaml ./
COPY examples ./examples
ENV CKVIZ_DATASETS=/app/datasets.yaml CKVIZ_DATA=/app/data
RUN useradd --create-home --uid 1000 ckviz && mkdir data pipelines && chown ckviz data pipelines
USER ckviz
EXPOSE 8000
CMD ["ckviz"]

FROM base AS dev
USER root
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked
COPY tests ./tests
RUN chown ckviz .
USER ckviz
CMD ["pytest", "-q"]
