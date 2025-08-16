import os
from data_utils import load_binary_data, scale_data
from autoencoder import build_autoencoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import configparser
import joblib
import cv2
import pandas as pd
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def parse_datetime(filename):
    """
    Parse a filename (e.g., prefix_YYYYMMDD_HHMMSS.bin) into a datetime object.
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')  
    date_str = parts[1]          
    time_str = parts[2].replace('.bin', '') 
    datetime_str = date_str + time_str       
    datetime_obj = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S')
    return datetime_obj

def plot_loss(history, folder_path, title="Loss Curves"):
    """
    Plot training and validation loss curves and save to folder.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(folder_path, "loss_curves.png"))
    plt.show()

def display_reconstruction(autoencoder, data, folder_path, n=5):
    """
    Visualize original and reconstructed images and save the plot to folder.
    """
    indices = np.random.randint(0, data.shape[0], size=n)
    sample_images = data[indices]
    reconstructed = autoencoder.predict(sample_images)

    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original Image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(32, 32), cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Reconstructed Image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(32, 32), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "reconstructed_images.png"))
    plt.show()

def augment_thermal_image(thermal_image):
    probability = 0.1
    image = thermal_image.squeeze()  # Remove the singleton dimension
    if np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
       
    if np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.flip(image, 0)
    elif np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.flip(image, 1)
    elif np.random.choice([0, 1], p=[1-probability, probability]): 
        image = cv2.flip(image, -1)
    image = image.reshape((32, 32, 1))
    return image

def augment_dataset(X_train, num_augmentations=1):
    augmented_images = []
    for _ in range(num_augmentations):
        for img in X_train:
            aug_img = augment_thermal_image(img)
            augmented_images.append(aug_img)
    augmented_images = np.array(augmented_images)
    return augmented_images

config = configparser.ConfigParser()
config.read('config.ini')
binary_folder = config['PATHS']['binary_folder']
output_folder = config['PATHS']['output_folder']

# Load and preprocess data
data_sequences, filenames = load_binary_data(binary_folder)

filtered_data = []
filtered_filenames = []

for i, thermal_data in enumerate(data_sequences):
    # Check if max < 90 and min > -30 (based on the unscaled values)
    if thermal_data.max() < 90 and thermal_data.min() > -30:
        filtered_data.append(thermal_data)
        filtered_filenames.append(filenames[i])
    else:
        pass

filtered_data = np.array(filtered_data)
filtered_filenames = np.array(filtered_filenames)

# If all data is filtered out, you may want to handle that case:
if len(filtered_data) == 0:
    raise ValueError("No data met the min/max temperature criteria. Exiting.")

# Now scale only the valid data
scaled_data, scaler = scale_data(filtered_data)

# Save the scaler using joblib
scaler_path = os.path.join(output_folder, 'scaler.joblib')
joblib.dump(scaler, scaler_path)

# Split data into training and validation sets
X_train, X_val = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Apply data augmentation to X_train
num_augmentations = 1  
augmented_images = augment_dataset(X_train, num_augmentations=num_augmentations)

# Combine original and augmented data
X_train_augmented = np.concatenate((X_train, augmented_images), axis=0)

# Shuffle the augmented dataset
shuffle_indices = np.random.permutation(X_train_augmented.shape[0])
X_train_augmented = X_train_augmented[shuffle_indices]

# Build and train autoencoder
autoencoder = build_autoencoder()
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss="mse")
autoencoder.encoder().summary()
autoencoder.decoder().summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = autoencoder.fit(
    X_train_augmented, X_train_augmented,
    epochs=30,
    batch_size=8,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping]
)

model_path = os.path.join(output_folder, "convolutional_autoencoder.keras")
autoencoder.save(model_path)

plot_loss(history, folder_path=output_folder, title="Autoencoder Loss Curves")

display_reconstruction(autoencoder, X_val, folder_path=output_folder)

latent_vectors = autoencoder.encoder().predict(scaled_data)
latent_vectors_df = pd.DataFrame(latent_vectors, index=filtered_filenames)

datetimes = [parse_datetime(fn) for fn in latent_vectors_df.index]
latent_vectors_df['datetime'] = datetimes
latent_vectors_df.set_index('datetime', inplace=True)
latent_vectors_df.index = latent_vectors_df.index.floor('h')
latent_vectors_df.sort_index(inplace=True)
latent_vectors_df.index.name = None
latent_vectors_df = latent_vectors_df[~latent_vectors_df.index.duplicated(keep='first')]

latent_vectors_path = os.path.join(output_folder, "latent_vectors.csv")
latent_vectors_df.to_csv(latent_vectors_path)

print(f"All outputs saved in the folder: {output_folder}")
