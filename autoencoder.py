from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D, Dropout)
 
class CustomAutoencoder(Model):
    def __init__(self, encoder, decoder):
        """
        Custom Autoencoder class that combines encoder and decoder models.
        """
        super(CustomAutoencoder, self).__init__()
        self.encoder_model = encoder
        self.decoder_model = decoder
 
    def call(self, inputs):
        latent = self.encoder_model(inputs)
        reconstructed = self.decoder_model(latent)
        return reconstructed
 
    def encoder(self):
        """
        Access the encoder part of the model.
        """
        return self.encoder_model
 
    def decoder(self):
        """
        Access the decoder part of the model.
        """
        return self.decoder_model
 
    def get_config(self):
        """
        Return the configuration of the custom model for serialization.
        """
        config = super(CustomAutoencoder, self).get_config()
        config.update({
            "encoder": self.encoder_model.get_config(),
            "decoder": self.decoder_model.get_config()
        })
        return config
 
    @classmethod
    def from_config(cls, config):
        """
        Recreate the model from its configuration.
        """
        encoder = Model.from_config(config["encoder"])
        decoder = Model.from_config(config["decoder"])
        return cls(encoder, decoder)
 
def build_autoencoder():
    """
    Builds a CNN autoencoder with a 64-dimensional latent representation.
    """
    latent_dim=64
    #     ENCODER
    input_img = Input(shape=(32, 32, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x) 

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  

    x = Flatten()(x)                
    latent = Dense(latent_dim, activation='tanh')(x)
    encoder = Model(inputs=input_img, outputs=latent, name="encoder_64")

    #     DECODER
    decoder_input = Input(shape=(latent_dim,)) 
    x = Dense(8 * 8 * 64, activation='tanh')(decoder_input)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)  
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)  
    decoded_output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x) 
    decoder = Model(inputs=decoder_input, outputs=decoded_output, name="decoder")
 
    autoencoder = CustomAutoencoder(encoder, decoder)
    return autoencoder
