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

# Directory structure only — no sample data is bundled (see files/README.md).
# Bind-mount your own data via docker-compose, or add .bin files locally before building.
COPY files ./files

VOLUME ["/app/output"]

CMD ["python", "scripts/train_autoencoder.py"]
