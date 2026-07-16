"""Train the convolutional autoencoder and extract latent vectors.

Usage: ``python -m panel360.pipelines.train_autoencoder``
"""

import logging
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from panel360.augmentation import augment_dataset
from panel360.config import ensure_output_dir, load_config, require_dir
from panel360.data import load_binary_data, parse_datetime, scale_data
from panel360.models.autoencoder import build_autoencoder

logger = logging.getLogger(__name__)

SEED = 42
MAX_TEMPERATURE_C = 90
MIN_TEMPERATURE_C = -30


def set_seeds(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def plot_loss(history, folder_path, title="Loss Curves"):
    """Plot training/validation loss curves and save them to ``folder_path``."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(folder_path, "loss_curves.png"))
    plt.close()


def save_reconstruction_preview(autoencoder, data, folder_path, n=5, rng=np.random):
    """Save a preview plot of original vs. reconstructed validation images."""
    import matplotlib.pyplot as plt

    n = min(n, data.shape[0])
    indices = rng.randint(0, data.shape[0], size=n)
    sample_images = data[indices]
    reconstructed = autoencoder.predict(sample_images)

    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(32, 32), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(32, 32), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "reconstructed_images.png"))
    plt.close()


def filter_by_temperature_range(data, filenames, max_temp=MAX_TEMPERATURE_C, min_temp=MIN_TEMPERATURE_C):
    """Drop panels whose unscaled max/min temperature falls outside a plausible range."""
    filtered_data, filtered_filenames = [], []
    for panel, filename in zip(data, filenames):
        if panel.max() < max_temp and panel.min() > min_temp:
            filtered_data.append(panel)
            filtered_filenames.append(filename)
    if not filtered_data:
        raise ValueError("No data met the min/max temperature criteria.")
    return np.array(filtered_data), np.array(filtered_filenames)


def build_latent_vectors_df(autoencoder, scaled_data, filenames):
    """Encode ``scaled_data`` and index the resulting latent vectors by hour, deduplicated."""
    latent_vectors = autoencoder.encoder().predict(scaled_data)
    df = pd.DataFrame(latent_vectors, index=filenames)
    df["datetime"] = [parse_datetime(fn) for fn in df.index]
    df.set_index("datetime", inplace=True)
    df.index = df.index.floor("h")
    df.sort_index(inplace=True)
    df.index.name = None
    return df[~df.index.duplicated(keep="first")]


def run(config):
    set_seeds()
    require_dir(config.binary_folder, "binary_folder")
    output_folder = ensure_output_dir(config.output_folder)

    data, filenames = load_binary_data(config.binary_folder)
    filtered_data, filtered_filenames = filter_by_temperature_range(data, filenames)

    scaled_data, scaler = scale_data(filtered_data)
    joblib.dump(scaler, os.path.join(output_folder, "scaler.joblib"))

    X_train, X_val = train_test_split(scaled_data, test_size=0.2, random_state=SEED)

    augmented_images = augment_dataset(X_train, num_augmentations=1)
    X_train_augmented = np.concatenate((X_train, augmented_images), axis=0)
    X_train_augmented = X_train_augmented[np.random.permutation(X_train_augmented.shape[0])]

    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
    autoencoder.encoder().summary()
    autoencoder.decoder().summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = autoencoder.fit(
        X_train_augmented, X_train_augmented,
        epochs=30,
        batch_size=8,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
    )

    autoencoder.save(os.path.join(output_folder, "convolutional_autoencoder.keras"))
    plot_loss(history, output_folder, title="Autoencoder Loss Curves")
    save_reconstruction_preview(autoencoder, X_val, output_folder)

    latent_vectors_df = build_latent_vectors_df(autoencoder, scaled_data, filtered_filenames)
    latent_vectors_path = os.path.join(output_folder, "latent_vectors.csv")
    latent_vectors_df.to_csv(latent_vectors_path)

    logger.info("All outputs saved in the folder: %s", output_folder)


def main():
    logging.basicConfig(level=logging.INFO)
    run(load_config())


if __name__ == "__main__":
    main()
