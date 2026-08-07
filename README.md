# PANEL360 – Predictive Maintenance via Thermal Imaging

## Problem

Electrical cabinets develop hotspots before they fail. Manual thermal inspections are
expensive and infrequent, so early warning signs often get missed between visits.

This project learns what "normal" thermal behavior looks like from historical panel
images, forecasts the next 24 hours, and automatically flags patterns that deviate from
the norm — turning periodic manual checks into continuous, automated monitoring.

## How it works

```
Thermal Images
      ↓
Preprocessing              (scaling, gap-aware segmentation, sliding windows)
      ↓
Convolutional Autoencoder  (compresses each 32x32 image into a latent vector)
      ↓
Latent Representation
      ↓
LSTM Seq2Seq               (learns temporal patterns, forecasts future latent vectors)
      ↓
Forecasted Latent Vectors
      ↓
Decoder                    (reconstructs latent vectors back into heatmaps)
      ↓
Forecasted Heatmaps
      ↓
Isolation Forest           (flags forecasted/observed vectors as outliers)
      ↓
Green / Yellow / Red Alarm
```

- **Autoencoder** — a CNN encoder/decoder trained to reconstruct each thermal image
  through a 64-dimensional bottleneck, so the bottleneck (the "latent vector") becomes a
  compact summary of the panel's thermal state at that hour.
- **LSTM Seq2Seq** — takes a sequence of past latent vectors (e.g. the last 7 days) and
  forecasts the next 24 hourly latent vectors.
- **Decoder** — reuses the autoencoder's decoder half to turn forecasted (and observed)
  latent vectors back into full heatmaps, so predictions stay visual and interpretable.
- **Isolation Forest** — two detectors (a looser "yellow" and a stricter "red") fitted on
  training-period latent vectors classify each hour — observed or forecasted — as
  Green/Yellow/Red, so anomalies surface without hand-set temperature thresholds.

## Project layout

```
config/config.ini          # default paths + forecast settings (overridable via PANEL360_* env vars)
src/panel360/
  data.py                  # binary panel loading, scaling, filename->datetime parsing
  augmentation.py          # thermal image rotation/flip augmentation
  timeseries.py            # gap-aware segmentation, resampling, sliding windows, blanking patterns
  anomaly.py                # Isolation Forest alarm classification
  models/
    autoencoder.py          # convolutional autoencoder (CustomAutoencoder, build_autoencoder)
    forecaster.py            # LSTM Seq2Seq forecaster
  pipelines/
    train_autoencoder.py    # trains the autoencoder, extracts latent_vectors.csv
    forecast.py               # forecasts latent vectors, reconstructs heatmaps, flags anomalies
    evaluate.py                # reports reconstruction MSE of a trained autoencoder on held-out data
scripts/                    # thin CLI entry points around the pipelines above
tests/                      # pytest unit tests for the pure-logic modules above
files/                      # sample binary panel data (32x32 float32 snapshots)
```

## Setup (local)

Requires Python 3.10 or 3.11 (TensorFlow 2.15 does not support 3.12+).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the pipelines

```bash
python scripts/train_autoencoder.py     # or: panel360-train
python scripts/run_forecast.py          # or: panel360-forecast
python scripts/evaluate_autoencoder.py  # or: panel360-evaluate
```

Paths and the forecast input window come from [config/config.ini](config/config.ini) and can be
overridden per-field with environment variables without touching the file:
`PANEL360_BINARY_FOLDER`, `PANEL360_ACTUAL_NEXT24_BINARY_FOLDER`, `PANEL360_TEST_BINARY_FOLDER`,
`PANEL360_OUTPUT_FOLDER`, `PANEL360_INPUT_DAYS` (and `PANEL360_CONFIG_PATH` to point at a different
ini file entirely).

### Outputs (written to `output_folder`)
- `scaler.joblib`, `convolutional_autoencoder.keras`, `latent_vectors.csv`
- `forecasted_latent_vectors.csv`, `forecasted_heatmaps/`, `seq2seq_forecast.keras`
- `anomaly_forecast_results.csv`, `max_temperatures_with_actual_plot.png`

## Tests

```bash
pytest
```

The suite covers data loading/scaling, augmentation, time-series segmentation/windowing,
anomaly classification, and config loading/validation without needing TensorFlow. Model-shape
tests in `tests/test_models.py` are skipped automatically if TensorFlow isn't installed
(e.g. on Python versions it doesn't support).

## Docker

Build once, then run each stage as a one-off job (they share the same image):

```bash
docker compose build
docker compose run --rm train
docker compose run --rm forecast
docker compose run --rm evaluate
```

`./files` and `./output` are bind-mounted into the container, so real data can be dropped into
`files/` on the host and results inspected in `output/` without rebuilding the image. Baseline
sample data is baked into the image for a quick smoke test if you don't mount over it.
