"""Loading and scaling of 32x32 thermal panel binary snapshots."""

import os
import struct

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PANEL_SHAPE = (32, 32, 1)
PANEL_FLOAT_COUNT = PANEL_SHAPE[0] * PANEL_SHAPE[1]


def load_binary_data(binary_folder):
    """Load every ``*.bin`` file in ``binary_folder`` into a stacked array.

    Each file must contain exactly ``32 * 32`` little-endian 32-bit floats
    (a single thermal panel snapshot). Returns ``(data, filenames)`` where
    ``data`` has shape ``(n, 32, 32, 1)`` and ``filenames`` are basenames
    sorted the same way as ``data``.
    """
    filepaths = sorted(
        os.path.join(binary_folder, fname)
        for fname in os.listdir(binary_folder)
        if fname.endswith(".bin")
    )

    data_sequences = []
    filenames = []
    for filepath in filepaths:
        with open(filepath, "rb") as file:
            binary_data = file.read()
        unpacked_data = struct.unpack(f"<{PANEL_FLOAT_COUNT}f", binary_data)
        panel_data = np.array(unpacked_data, dtype=np.float32).reshape(PANEL_SHAPE)
        data_sequences.append(panel_data)
        filenames.append(os.path.basename(filepath))

    if not data_sequences:
        return np.empty((0, *PANEL_SHAPE), dtype=np.float32), []
    return np.array(data_sequences), filenames


def scale_data(data, scaler=None):
    """Scale panel data to [0, 1] with a shared ``MinMaxScaler``.

    If ``scaler`` is ``None`` a new one is fit on ``data``; otherwise the
    given scaler is reused (for held-out/test data). Returns
    ``(scaled_data, scaler)``.
    """
    data_flat = data.flatten().reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_flat = scaler.fit_transform(data_flat)
    else:
        scaled_data_flat = scaler.transform(data_flat)

    scaled_data = scaled_data_flat.reshape(data.shape)
    return scaled_data, scaler


def parse_datetime(filename):
    """Parse a filename like ``prefix_YYYYMMDD_HHMMSS.bin`` into a ``Timestamp``."""
    basename = os.path.basename(filename)
    _, date_str, time_str = basename.split("_")
    time_str = time_str.replace(".bin", "")
    return pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S")
