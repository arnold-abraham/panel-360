"""Forecast latent vectors 24h ahead, reconstruct heatmaps, and flag anomalies.

Usage: ``python -m panel360.pipelines.forecast``
"""

import logging
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from panel360.anomaly import classify_alarms, fit_anomaly_detectors
from panel360.config import ensure_output_dir, load_config, require_dir
from panel360.data import load_binary_data
from panel360.models.autoencoder import CustomAutoencoder
from panel360.models.forecaster import build_seq2seq_model
from panel360.timeseries import apply_pattern, create_sliding_windows_from_segments, split_into_segments_with_gaps

logger = logging.getLogger(__name__)

SEED = 42
FORECAST_HORIZON = 24
BLANKING_PATTERNS = ("all_6_days", "one_day", "two_days", "none")
HEATMAP_TEMP_RANGE = (10, 50)


def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def augment_with_blanking_patterns(X, y, patterns=BLANKING_PATTERNS):
    """Expand each 7-day training window into one copy per blanking pattern."""
    X_augmented, y_augmented = [], []
    for i in range(X.shape[0]):
        for pattern in patterns:
            X_augmented.append(apply_pattern(X[i], pattern))
            y_augmented.append(y[i])
    return np.array(X_augmented), np.array(y_augmented)


def time_series_split(X, y, train_frac=0.8, val_frac=0.1):
    """Chronological 80/10/10 split (no shuffling — order matters for time series)."""
    total = X.shape[0]
    train_size = int(train_frac * total)
    val_size = int(val_frac * total)
    return {
        "train": (X[:train_size], y[:train_size]),
        "val": (X[train_size:train_size + val_size], y[train_size:train_size + val_size]),
        "test": (X[train_size + val_size:], y[train_size + val_size:]),
        "train_size": train_size,
        "val_size": val_size,
    }


def save_heatmaps(predicted_temperatures, output_folder, vmin=HEATMAP_TEMP_RANGE[0], vmax=HEATMAP_TEMP_RANGE[1]):
    """Save one heatmap PNG per forecasted hour."""
    import matplotlib.pyplot as plt

    os.makedirs(output_folder, exist_ok=True)
    for i in range(predicted_temperatures.shape[0]):
        plt.figure(figsize=(6, 6))
        panel_data = predicted_temperatures[i, :, :, 0]
        plt.imshow(panel_data, cmap="seismic", interpolation="nearest", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(label="Temperature (°C)")
        cbar.set_ticks(np.linspace(vmin, vmax, num=11))
        plt.title(f"Predicted Temperature for Hour {i + 1}")
        plt.savefig(os.path.join(output_folder, f"heatmap_hour_{i + 1}.png"))
        plt.close()


def inverse_scale_images(images, image_scaler):
    """Invert MinMax scaling on a batch of reconstructed images."""
    flat = images.reshape(-1, 1)
    original_scale = image_scaler.inverse_transform(flat)
    return original_scale.reshape(images.shape)


def max_temperatures(images):
    """Per-hour max temperature over a batch of ``(n, 32, 32, 1)`` images."""
    n = images.shape[0]
    return images.reshape(n, -1).max(axis=1)


def run(config):
    set_seeds()
    require_dir(config.actual_next24_binary_folder, "actual_next24_binary_folder")
    output_folder = ensure_output_dir(config.output_folder)
    input_days = config.input_days

    latent_vectors_path = os.path.join(output_folder, "latent_vectors.csv")
    latent_vectors_df = pd.read_csv(latent_vectors_path, index_col=0, parse_dates=True)
    latent_vectors_df.sort_index(inplace=True)
    logger.info("Latent vectors: %s", latent_vectors_df.shape)

    segments = split_into_segments_with_gaps(latent_vectors_df, max_small_gap_hours=3)
    X, y = create_sliding_windows_from_segments(segments, input_days=input_days, forecast_horizon=FORECAST_HORIZON)
    logger.info("Windowed X/y shapes: %s %s", X.shape, y.shape)

    if input_days == 7:
        X, y = augment_with_blanking_patterns(X, y)
        logger.info("X/y shapes after blanking augmentation: %s %s", X.shape, y.shape)

    split = time_series_split(X, y)
    X_train, y_train = split["train"]
    X_val, y_val = split["val"]
    X_test, y_test = split["test"]
    logger.info("Train %s Val %s Test %s", X_train.shape, X_val.shape, X_test.shape)

    sequence_length = input_days * 24
    num_features = X.shape[-1]

    model = build_seq2seq_model(sequence_length, num_features, forecast_horizon=FORECAST_HORIZON)
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=10, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler],
    )

    test_loss = model.evaluate(X_test, y_test)
    logger.info("Test loss: %s", test_loss)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.legend()
    plt.title("Seq2Seq Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "seq2seq_loss_curves.png"))
    plt.close()

    model.save(os.path.join(output_folder, "seq2seq_forecast.keras"))

    # Forecast the next 24 hours from the most recent window.
    last_sequence = X[-1].reshape(1, sequence_length, num_features)
    forecast = model.predict(last_sequence).squeeze()

    last_timestamp = latent_vectors_df.index[-1]
    forecast_timestamps = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(FORECAST_HORIZON)]
    forecasted_latent_vectors_df = pd.DataFrame(forecast, index=forecast_timestamps, columns=latent_vectors_df.columns)
    forecasted_latent_vectors_df.to_csv(os.path.join(output_folder, "forecasted_latent_vectors.csv"))

    autoencoder_path = os.path.join(output_folder, "convolutional_autoencoder.keras")
    if not os.path.exists(autoencoder_path):
        logger.warning("No trained autoencoder found at %s; skipping image reconstruction.", autoencoder_path)
        return

    autoencoder = load_model(autoencoder_path, custom_objects={"CustomAutoencoder": CustomAutoencoder}, compile=False)
    decoder = autoencoder.decoder()
    image_scaler = joblib.load(os.path.join(output_folder, "scaler.joblib"))

    reconstructed_images = decoder.predict(forecast)
    reconstructed_images_original_scale = inverse_scale_images(reconstructed_images, image_scaler)
    save_heatmaps(reconstructed_images_original_scale, os.path.join(output_folder, "forecasted_heatmaps"))

    # Anomaly detection: fit only on the training-period latent vectors.
    train_end_index = split["train_size"] + sequence_length
    latent_vectors_train_df = latent_vectors_df.iloc[:train_end_index]
    clf_yellow, clf_red = fit_anomaly_detectors(latent_vectors_train_df)

    last_24_hours_df = latent_vectors_df.tail(FORECAST_HORIZON)
    historical_alarms = classify_alarms(clf_yellow, clf_red, last_24_hours_df)
    forecast_alarms = classify_alarms(clf_yellow, clf_red, forecasted_latent_vectors_df)

    reconstructed_last_24 = decoder.predict(last_24_hours_df.values)
    reconstructed_last_24_original_scale = inverse_scale_images(reconstructed_last_24, image_scaler)

    last_24_max_temps = max_temperatures(reconstructed_last_24_original_scale)
    forecast_24_max_temps = max_temperatures(reconstructed_images_original_scale)

    historical_results = last_24_hours_df.copy()
    historical_results["alarm"] = historical_alarms["alarm"]
    historical_results["max_temperature"] = last_24_max_temps
    historical_results["forecasted"] = "no"

    forecast_results = forecasted_latent_vectors_df.copy()
    forecast_results["alarm"] = forecast_alarms["alarm"]
    forecast_results["max_temperature"] = forecast_24_max_temps
    forecast_results["forecasted"] = "yes"

    combined_results = pd.concat([historical_results, forecast_results])
    combined_results["datetime"] = combined_results.index.strftime("%d/%m/%Y %H:%M")
    combined_results = combined_results[["datetime", "alarm", "max_temperature", "forecasted"]]
    combined_results.to_csv(os.path.join(output_folder, "anomaly_forecast_results.csv"), index=False)

    if config.influx_enabled:
        try:
            from panel360.storage.influx import write_anomaly_results
            write_anomaly_results(config, combined_results)
        except Exception:
            logger.warning("InfluxDB write failed; continuing (CSV already saved).", exc_info=True)

    # Compare against ground truth, if available.
    actual_next24_data, _ = load_binary_data(config.actual_next24_binary_folder)
    actual_24_max_temps = max_temperatures(actual_next24_data)

    x_axis = np.arange(1, 49)
    combined_max_temps = np.concatenate([last_24_max_temps, forecast_24_max_temps])

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, combined_max_temps, marker="o", linestyle="-", label="Predicted Next 24 Hours", color="b")
    plt.axvline(x=24.5, color="r", linestyle="--", label="Forecast Start")
    plt.plot(x_axis[:24], last_24_max_temps, marker="o", linestyle="-", label="Last 24 Hours", color="g")
    plt.plot(x_axis[24:], actual_24_max_temps, marker="o", linestyle="-", label="Actual Next 24 Hours", color="orange")
    plt.title("Max Temperatures for Last 24, Predicted Next 24, and Actual Next 24 Hours", fontsize=16)
    plt.xlabel("Time Step (1-48)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.xticks(ticks=np.arange(1, 49, step=2))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "max_temperatures_with_actual_plot.png"))
    plt.close()

    logger.info("Forecast pipeline outputs saved in: %s", output_folder)


def main():
    logging.basicConfig(level=logging.INFO)
    run(load_config())


if __name__ == "__main__":
    main()
