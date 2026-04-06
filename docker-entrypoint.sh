#!/usr/bin/env bash
# docker-entrypoint.sh — smart startup for the Finance Bot container.
#
# Behaviour:
#   1. If data/bm25_index.pkl does NOT exist → run ingestion first
#   2. Then start the requested service (default: streamlit)
#
# Environment variable SERVICE controls what to start:
#   SERVICE=streamlit   → start Streamlit frontend on :8501  (default)
#   SERVICE=api         → start FastAPI backend on :8000
#   SERVICE=ingest      → run ingestion only, then exit
#   SERVICE=both        → start API first, then Streamlit (foreground)

set -euo pipefail

BM25_INDEX="data/bm25_index.pkl"

# ── Auto-ingest if no index exists ──────────────────────────────────────────
if [ ! -f "$BM25_INDEX" ]; then
    echo "============================================================"
    echo " No BM25 index found — running Apple 10-K ingestion first..."
    echo "============================================================"
    python scripts/ingest_apple_html.py --dataset-dir ./Apple_Dataset
fi

# ── Start the requested service ─────────────────────────────────────────────
SERVICE="${SERVICE:-streamlit}"

case "$SERVICE" in
    streamlit)
        echo "Starting Streamlit on port 8501..."
        exec streamlit run app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    api)
        echo "Starting FastAPI on port 8000..."
        exec uvicorn api.main:app \
            --host 0.0.0.0 \
            --port 8000
        ;;
    ingest)
        echo "Running ingestion only..."
        python scripts/ingest_apple_html.py --dataset-dir ./Apple_Dataset
        echo "Ingestion complete."
        ;;
    both)
        echo "Starting FastAPI on port 8000 (background)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        echo "Starting Streamlit on port 8501..."
        exec streamlit run app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    *)
        echo "Unknown SERVICE='$SERVICE'. Use: streamlit | api | ingest | both"
        exit 1
        ;;
esac
