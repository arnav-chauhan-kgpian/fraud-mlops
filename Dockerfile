# ─────────────────────────────────────────────────────────────────────────────
# FraudShield AI — Hugging Face Spaces (Docker SDK)
# Runs BOTH services in one container:
#   • FastAPI inference API   → internal http://localhost:8000
#   • Streamlit dashboard     → public  http://0.0.0.0:7860  (HF exposes 7860)
# The dashboard calls the API over localhost inside the container.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Hugging Face Spaces runs the container as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies first (better layer caching).
COPY --chown=user:user requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-deploy.txt

# Copy the application (see .dockerignore for what's excluded, e.g. the 150MB CSV).
COPY --chown=user:user . .

# Make `from src...` imports resolve, and configure Streamlit for HF's iframe.
ENV PYTHONPATH=$HOME/app \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860
CMD ["bash", "start.sh"]
