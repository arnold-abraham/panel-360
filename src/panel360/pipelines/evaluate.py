"""Evaluate a trained autoencoder's reconstruction quality on held-out data.

This is a manual evaluation/reporting script, not an automated test suite —
see ``tests/`` for that. Usage: ``python -m panel360.pipelines.evaluate``
"""

import logging
import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from panel360.config import ensure_output_dir, load_config, require_dir
from panel360.data import load_binary_data, scale_data
from panel360.models.autoencoder import CustomAutoencoder

logger = logging.getLogger(__name__)


def per_image_mse(originals, reconstructed):
    return np.mean((originals - reconstructed) ** 2, axis=(1, 2, 3))


def save_reconstruction_preview(autoencoder, data, file_names, output_folder, n=5):
    import matplotlib.pyplot as plt

    n = min(n, data.shape[0])
    indices = np.random.choice(data.shape[0], size=n, replace=False)
    sample_images = data[indices]
    sample_file_names = [file_names[i] for i in indices]
    reconstructed = autoencoder.predict(sample_images)
    mse_per_image = per_image_mse(sample_images, reconstructed)

    plt.figure(figsize=(15, 6))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(32, 32), cmap="gray")
        plt.title(f"File: {sample_file_names[i]}", fontsize=8)
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(32, 32), cmap="gray")
        plt.title(f"Reconstructed\nMSE: {mse_per_image[i]:.6f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "evaluation_reconstruction.png"))
    plt.close()


def evaluate_model(autoencoder, test_data, file_names, scaler, output_folder):
    """Report per-image and overall reconstruction MSE on held-out test data."""
    scaled_test_data, _ = scale_data(test_data, scaler=scaler)
    reconstructed = autoencoder.predict(scaled_test_data)
    mse_per_image = per_image_mse(scaled_test_data, reconstructed)
    for i, mse in enumerate(mse_per_image):
        logger.info("Image %d (%s): Reconstruction MSE = %.6f", i + 1, file_names[i], mse)
    overall_mse = np.mean(mse_per_image)
    logger.info("Overall Reconstruction MSE on test data: %s", overall_mse)
    save_reconstruction_preview(autoencoder, scaled_test_data, file_names, output_folder)
    return overall_mse


def run(config):
    require_dir(config.test_binary_folder, "test_binary_folder")
    output_folder = ensure_output_dir(config.output_folder)

    test_data, file_names = load_binary_data(config.test_binary_folder)
    scaler = joblib.load(os.path.join(output_folder, "scaler.joblib"))
    autoencoder_path = os.path.join(output_folder, "convolutional_autoencoder.keras")
    autoencoder = load_model(autoencoder_path, compile=False, custom_objects={"CustomAutoencoder": CustomAutoencoder})
    evaluate_model(autoencoder, test_data, file_names, scaler, output_folder)


def main():
    logging.basicConfig(level=logging.INFO)
    run(load_config())


if __name__ == "__main__":
    main()
