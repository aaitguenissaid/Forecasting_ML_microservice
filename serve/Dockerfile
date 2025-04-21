FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY serve/src /app/src
COPY serve/uv.lock /app/uv.lock
COPY serve/pyproject.toml /app/pyproject.toml
COPY config/ /app/config/

RUN uv sync --frozen --no-cache

EXPOSE 8000

ENV PYTHONPATH=/app:/app/config:/app/models:/app/src

CMD ["/app/.venv/bin/uvicorn", "src.app:app", "--port", "8000", "--host", "0.0.0.0"]