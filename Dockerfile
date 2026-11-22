# ==============================================================================
# CottonGuard AI — Production Container Specification
# Multi-stage container for FastAPI ASGI serving and React 18 SPA static assets
# ==============================================================================

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend ./backend
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY alembic ./alembic
COPY alembic.ini .
COPY pyproject.toml .
COPY README.md .
COPY .env.example .

# Create artifacts directory structure
RUN mkdir -p artifacts/checkpoints artifacts/predictions artifacts/reports artifacts/manifests

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch FastAPI ASGI server
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
