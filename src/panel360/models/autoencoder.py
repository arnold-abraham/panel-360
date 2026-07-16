"""Convolutional autoencoder for 32x32 thermal panel images."""

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape
from tensorflow.keras.models import Model

LATENT_DIM = 64


class CustomAutoencoder(Model):
    """Combines an encoder and decoder sub-model into a single trainable model."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_model = encoder
        self.decoder_model = decoder

    def call(self, inputs):
        latent = self.encoder_model(inputs)
        return self.decoder_model(latent)

    def encoder(self):
        return self.encoder_model

    def decoder(self):
        return self.decoder_model

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder_model.get_config(),
            "decoder": self.decoder_model.get_config(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = Model.from_config(config["encoder"])
        decoder = Model.from_config(config["decoder"])
        return cls(encoder, decoder)


def build_autoencoder(latent_dim=LATENT_DIM):
    """Build a CNN autoencoder for 32x32x1 images with a ``latent_dim`` bottleneck."""
    # ENCODER
    input_img = Input(shape=(32, 32, 1))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, activation="tanh")(x)
    encoder = Model(inputs=input_img, outputs=latent, name="encoder_64")

    # DECODER
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(8 * 8 * 64, activation="tanh")(decoder_input)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same")(x)
    decoded_output = Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)
    decoder = Model(inputs=decoder_input, outputs=decoded_output, name="decoder")

    return CustomAutoencoder(encoder, decoder)
