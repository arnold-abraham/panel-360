import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_utils import load_binary_data, scale_data
from autoencoder import CustomAutoencoder
from sklearn.metrics import mean_squared_error
import configparser
import joblib

def display_reconstruction(autoencoder, data, file_names, n=5):
    """
    Visualize original and reconstructed images with Reconstruction MSE and file names displayed.
    """
    indices = np.random.choice(data.shape[0], size=n, replace=False)
    sample_images = data[indices]
    sample_file_names = [file_names[i] for i in indices]
    reconstructed = autoencoder.predict(sample_images)
    mse_per_image = np.mean((sample_images - reconstructed) ** 2, axis=(1, 2, 3))

    plt.figure(figsize=(15, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(32, 32), cmap='gray')
        plt.title(f'File: {sample_file_names[i]}', fontsize=8)
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(32, 32), cmap='gray')
        plt.title(f'Reconstructed\nMSE: {mse_per_image[i]:.6f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model(autoencoder, test_data, file_names, scaler):
    """
    Test the autoencoder on new data and evaluate reconstruction loss for each image.
    """
    scaled_test_data, _ = scale_data(test_data, scaler=scaler)
    reconstructed = autoencoder.predict(scaled_test_data)
    mse_per_image = np.mean((scaled_test_data - reconstructed) ** 2, axis=(1, 2, 3))
    for i, mse in enumerate(mse_per_image):
        print(f"Image {i + 1}: Reconstruction MSE = {mse:.6f}")
    overall_mse = np.mean(mse_per_image)
    print(f"Overall Reconstruction MSE on test data: {overall_mse}")
    display_reconstruction(autoencoder, scaled_test_data, file_names)

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_binary_folder = config['PATHS']['test_binary_folder']
    output_folder = config['PATHS']['output_folder']
    test_data, file_names = load_binary_data(test_binary_folder)
    image_scaler_path = os.path.join(output_folder, 'scaler.joblib')
    scaler = joblib.load(image_scaler_path)
    autoencoder_path = os.path.join(output_folder, "convolutional_autoencoder.keras")
    autoencoder = load_model(
        autoencoder_path, compile=False, custom_objects={"CustomAutoencoder": CustomAutoencoder}
    )
    evaluate_model(autoencoder, test_data, file_names, scaler)

if __name__ == "__main__":
    main()