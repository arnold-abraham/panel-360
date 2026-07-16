import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def test_build_autoencoder_shapes_and_roundtrip():
    from panel360.models.autoencoder import build_autoencoder

    autoencoder = build_autoencoder(latent_dim=8)
    batch = np.random.rand(2, 32, 32, 1).astype("float32")

    output = autoencoder(batch)
    latent = autoencoder.encoder()(batch)

    assert output.shape == (2, 32, 32, 1)
    assert latent.shape == (2, 8)


def test_build_seq2seq_model_shapes():
    from panel360.models.forecaster import build_seq2seq_model

    model = build_seq2seq_model(sequence_length=168, num_features=8, forecast_horizon=24)
    batch = np.random.rand(2, 168, 8).astype("float32")

    output = model(batch)

    assert output.shape == (2, 24, 8)
