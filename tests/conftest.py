import struct

import numpy as np
import pytest


def write_bin_file(path, panel):
    """Write a (32, 32) or (32, 32, 1) float array as a little-endian binary panel file."""
    flat = np.asarray(panel, dtype=np.float32).reshape(-1)
    assert flat.size == 1024
    with open(path, "wb") as f:
        f.write(struct.pack("<1024f", *flat.tolist()))


@pytest.fixture
def binary_folder(tmp_path):
    """A folder with two known synthetic panel .bin files."""
    folder = tmp_path / "binary_folder"
    folder.mkdir()
    panel_a = np.full((32, 32, 1), 20.0, dtype=np.float32)
    panel_b = np.full((32, 32, 1), 40.0, dtype=np.float32)
    write_bin_file(folder / "TTS01_20241013_003835.bin", panel_a)
    write_bin_file(folder / "TTS01_20241013_013835.bin", panel_b)
    return folder, [panel_a, panel_b]
