#!/usr/bin/env bash
# PAIMANA-AI :: full build + serve
set -e
cd "$(dirname "$0")"
echo "[1/4] Building CUF-schema dataset..."   && python3 data/generate_dataset.py
echo "[2/4] Training and evaluating models..." && python3 ml/train.py
echo "[3/4] Exporting static snapshots..."     && python3 ml/export_static.py
echo "[4/4] Starting server on http://localhost:8000 (Ctrl+C to stop)"
exec python3 -m uvicorn backend.app:app --port 8000
