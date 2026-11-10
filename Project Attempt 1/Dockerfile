# ==============================================================================
# Dockerfile - US & Taiwan Semiconductor Supply-Chain Cointegration & ML Strategy
# Stock Markets Analytics Zoomcamp 2026 - Project Attempt 1
# ==============================================================================
FROM python:3.12-slim

LABEL maintainer="Stock Markets Analytics Zoomcamp 2026"
LABEL description="End-to-End Semiconductor Supply-Chain Cointegration & ML Trading Pipeline"

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system build dependencies and cron for scheduled execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project source code, scripts, and notebooks
COPY . /app/

# Ensure output and data directories exist
RUN mkdir -p /app/data /app/outputs/figures /app/outputs/reports /app/outputs/models \
    && chmod +x /app/scripts/run_cron.sh

# Default entrypoint executes the end-to-end pipeline in download/incremental regime
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--mode", "download"]

