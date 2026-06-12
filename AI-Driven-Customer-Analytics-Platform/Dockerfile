# ─────────────────────────────────────────────────────────────────
#  Dockerfile — AI-Driven Customer Analytics Platform
#  Base: Python 3.11-slim  |  Deployment: Docker / AWS / Azure / GCP
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create app directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────
COPY . .

# ── Create output directory ───────────────────────────────────────
RUN mkdir -p outputs

# ── Expose Streamlit port ─────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Streamlit config ──────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Entry point ───────────────────────────────────────────────────
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
