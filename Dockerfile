# ==============================================================================
# CottonGuard AI — Production Container Specification
# Multi-stage container for FastAPI ASGI serving and React 18 SPA static assets
# ==============================================================================

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_HOME=/app/.cache/torch \
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
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torchvision.models as models; models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); models.resnet34(weights=models.ResNet34_Weights.DEFAULT)"

# Copy application source code and configurations
COPY backend ./backend
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY alembic ./alembic
COPY alembic.ini .
COPY pyproject.toml .
COPY README.md .
COPY .env.example .

# Copy model registry metadata and dataset manifests
COPY artifacts/model_registry.json ./artifacts/model_registry.json
COPY artifacts/manifests ./artifacts/manifests

# Create artifacts directory structure and ensure write permissions for non-root container users (e.g. Hugging Face UID 1000)
RUN mkdir -p artifacts/checkpoints/verification \
             artifacts/checkpoints/classification \
             artifacts/checkpoints/segmentation \
             artifacts/checkpoints/detection \
             artifacts/predictions \
             artifacts/reports \
             artifacts/logs \
             artifacts/explainability \
    && touch /app/cotton_leaf_ai.db \
    && chmod -R 777 /app

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Launch FastAPI ASGI server with dynamic port support (Hugging Face Spaces 7860, Railway/Render $PORT, default 8000)
CMD ["sh", "-c", "python scripts/download_checkpoints.py || true; exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
