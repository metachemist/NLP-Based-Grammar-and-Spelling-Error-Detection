# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# NLP-Based Grammar and Spelling Error Detection -- Streamlit app
# Layers 1-3 only (spaCy + pyspellchecker). Layer 4 (T5) is intentionally
# not installed here; the app disables it cleanly when transformers/torch
# are absent.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc files in the image
# - PYTHONUNBUFFERED: logs stream straight to the container output
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Then the application code.
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

# Streamlit serves here. Hosts that inject their own $PORT (Render, Railway,
# Fly.io, Cloud Run) override the CMD below or set STREAMLIT_SERVER_PORT.
EXPOSE 8501

# Fail the container's healthcheck if Streamlit stops responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
