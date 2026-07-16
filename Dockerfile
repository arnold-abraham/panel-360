FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PANEL360_CONFIG_PATH=/app/config/config.ini

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY config ./config
COPY scripts ./scripts

# Baseline sample data; override with a bind mount for real deployments.
COPY files ./files

VOLUME ["/app/output"]

CMD ["python", "scripts/train_autoencoder.py"]
