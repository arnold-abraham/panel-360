import os
import struct
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_binary_data(binary_folder):
    """
    Loads and decodes all binary files in the specified folder into numpy arrays.
    Returns the data sequences and corresponding filenames.
    """
    data_sequences = []
    filenames = []
    filepaths = [os.path.join(binary_folder, fname)
                 for fname in sorted(os.listdir(binary_folder))
                 if fname.endswith('.bin')]
    for filepath in filepaths:
        with open(filepath, "rb") as file:
            binary_data = file.read()
        unpacked_data = struct.unpack("<1024f", binary_data)
        panel_data = np.array(unpacked_data).reshape((32, 32, 1))
        data_sequences.append(panel_data)
        filenames.append(os.path.basename(filepath))
    return np.array(data_sequences), filenames

def scale_data(data, scaler=None, save_path=None):
    """
    Scales data using MinMaxScaler and optionally saves the scaler.
    """
    data_flat = data.flatten().reshape(-1, 1) 
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data_flat = scaler.fit_transform(data_flat)
    else:
        scaled_data_flat = scaler.transform(data_flat)

    scaled_data = scaled_data_flat.reshape(data.shape) 
    return scaled_data, scaler