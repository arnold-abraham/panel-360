import numpy as np
import pandas as pd

from panel360.data import load_binary_data, parse_datetime, scale_data


def test_load_binary_data_roundtrip(binary_folder):
    folder, expected_panels = binary_folder
    data, filenames = load_binary_data(str(folder))

    assert data.shape == (2, 32, 32, 1)
    assert filenames == ["TTS01_20241013_003835.bin", "TTS01_20241013_013835.bin"]
    np.testing.assert_allclose(data[0], expected_panels[0])
    np.testing.assert_allclose(data[1], expected_panels[1])


def test_load_binary_data_empty_folder(tmp_path):
    data, filenames = load_binary_data(str(tmp_path))
    assert data.shape == (0, 32, 32, 1)
    assert filenames == []


def test_scale_data_fits_to_unit_range(binary_folder):
    folder, _ = binary_folder
    data, _ = load_binary_data(str(folder))

    scaled, scaler = scale_data(data)

    assert scaled.shape == data.shape
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    # Min and max panels (20 and 40) should map to 0 and 1 respectively.
    assert np.isclose(scaled.min(), 0.0)
    assert np.isclose(scaled.max(), 1.0)


def test_scale_data_reuses_existing_scaler(binary_folder):
    folder, _ = binary_folder
    data, _ = load_binary_data(str(folder))
    _, scaler = scale_data(data)

    new_panel = np.full((1, 32, 32, 1), 30.0, dtype=np.float32)
    scaled_new, reused_scaler = scale_data(new_panel, scaler=scaler)

    assert reused_scaler is scaler
    # 30 is the midpoint between the 20/40 fit range, so it should scale to 0.5.
    assert np.isclose(scaled_new.mean(), 0.5)


def test_parse_datetime():
    result = parse_datetime("TTS01_20241013_003835.bin")
    assert result == pd.Timestamp("2024-10-13 00:38:35")
