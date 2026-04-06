# ── Stage 1: Builder — install Python dependencies ─────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtualenv so we can copy cleanly to final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source
COPY . .

# Data directory — will be mounted as a volume in docker-compose
RUN mkdir -p data/chroma_db data/processed data/raw

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Default: expose Streamlit and FastAPI ports
EXPOSE 8501 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
