# ---------------------------------------------
# Production Dockerfile for Render (Dash App)
# Minimal + does NOT modify your project files.
# Gunicorn is installed separately so you didn't need to add it to requirements.txt.
# ---------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PORT=8050

WORKDIR /app

# System deps (add more if a lib needs compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn==21.2.0

# Copy application source
COPY . .

# (Optional) create an unprivileged user
RUN useradd -m dashuser && chown -R dashuser:dashuser /app
USER dashuser

EXPOSE 8050

# Health check (simple) – Render just pings the port, but this helps locally
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:${PORT}/ || exit 1

# Gunicorn can point directly to Dash's underlying Flask server: app:server
# Render injects $PORT; fallback to 8050 locally
CMD ["bash", "-c", "gunicorn app:server --workers=2 --threads=2 --timeout=120 -b 0.0.0.0:${PORT:-8050}"]
