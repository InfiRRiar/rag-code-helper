FROM python:3.13-slim


COPY pyproject.toml uv.lock* README.md* ./
RUN pip install -U pip uv && \ 
    uv sync --frozen --no-dev --group embs

COPY src/embs src/embs
COPY src/secrets.py src/secrets.py
COPY /.env /.env

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.embs.app:app", "--host", "0.0.0.0", "--port", "8080"]