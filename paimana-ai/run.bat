@echo off
REM PAIMANA-AI :: full build + serve
cd /d %~dp0
echo [1/4] Building CUF-schema dataset...
python data\generate_dataset.py || exit /b 1
echo [2/4] Training and evaluating models...
python ml\train.py || exit /b 1
echo [3/4] Exporting static snapshots...
python ml\export_static.py || exit /b 1
echo [4/4] Starting server on http://localhost:8000
python -m uvicorn backend.app:app --port 8000
