# Build an image for the luckyIA Flask app.
# Build arg INSTALL_TORCH: set to "true" to attempt to install torch from the PyTorch CPU wheels.
# Usage examples:
#  docker build --build-arg INSTALL_TORCH=false -t luckyia:latest .
#  docker build --build-arg INSTALL_TORCH=true -t luckyia:latest .

FROM python:3.11-slim

ARG INSTALL_TORCH=false
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

WORKDIR /app

# system deps for common packages (pandas, numpy, sklearn). Keep small and safe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc git curl ca-certificates libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install. If INSTALL_TORCH is false we remove the torch line before install.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ "$INSTALL_TORCH" = "false" ]; then \
      sed '/^torch==/Id' requirements.txt > /app/requirements-no-torch.txt && \
      pip install --no-cache-dir -r /app/requirements-no-torch.txt; \
    else \
      # Try to install torch from official CPU index first, allow pip to resolve others
      pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu || true && \
      pip install --no-cache-dir -r /app/requirements.txt; \
    fi

# Installer gunicorn pour l'exécution en production
RUN python -m pip install --no-cache-dir gunicorn==20.1.0 || true

# Copy the project
COPY . /app

# Create a non-root user for running the app
RUN useradd -m -s /bin/bash appuser || true
RUN chown -R appuser:appuser /app
USER appuser

ENV FLASK_APP=web_app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7000

EXPOSE 7000

# Default command: run the Flask app. Use environment variable to override if needed.
CMD ["python", "web_app/app.py"]
