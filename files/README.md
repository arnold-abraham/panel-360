# Data directory

Real thermal captures used during development are not included in this repository
(source data from R&D test-center work, kept out for confidentiality reasons).

To run the pipelines, drop your own binary panel snapshots into the matching
subfolder (paths are configurable in [config/config.ini](../config/config.ini)):

- `binary_folder/` — training data
- `actual_next24_binary/` — ground-truth data for the 24h forecast window
- `test_binary_folder/` — held-out data used by `evaluate_autoencoder.py`

## Expected file format

Each `.bin` file is a raw 32×32 `float32` array (4096 bytes, row-major, one thermal
snapshot per file), named `<prefix>_<YYYYMMDD>_<HHMMSS>.bin` so the capture
timestamp can be parsed from the filename (see `panel360.data`).
