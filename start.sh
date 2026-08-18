#!/usr/bin/env bash
# Launch the FastAPI inference API (internal :8000), wait until it is ready,
# then launch the Streamlit dashboard on the public HF port (:7860).
set -euo pipefail

# 1) Start the API in the background.
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# 2) Wait (up to ~30s) for the API to finish loading model artifacts.
#    No curl in the slim image, so probe /health with Python's stdlib.
python - <<'PY'
import time, urllib.request
for _ in range(30):
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as r:
            if r.status == 200:
                print("API is healthy."); break
    except Exception:
        pass
    time.sleep(1)
else:
    print("WARNING: API did not report healthy in time; dashboard will retry on use.")
PY

# 3) Hand the foreground to Streamlit (port/address come from env in the Dockerfile).
#    If the API dies, take the container down so HF restarts it cleanly.
exec streamlit run src/ui/dashboard.py
