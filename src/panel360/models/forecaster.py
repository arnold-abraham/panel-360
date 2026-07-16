"""LSTM Seq2Seq model for forecasting latent-vector sequences."""

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_seq2seq_model(sequence_length, num_features, forecast_horizon=24, lstm_units=96, dropout=0.2, learning_rate=0.001):
    """Build and compile an LSTM encoder-decoder for multi-step forecasting."""
    encoder_inputs = Input(shape=(sequence_length, num_features), name="encoder_inputs")
    encoder_lstm = LSTM(lstm_units, activation="tanh", return_sequences=False, name="encoder_lstm")(encoder_inputs)
    encoder_lstm = Dropout(dropout, name="encoder_dropout")(encoder_lstm)

    decoder_inputs = RepeatVector(forecast_horizon, name="decoder_repeat_vector")(encoder_lstm)
    decoder_lstm = LSTM(lstm_units, activation="tanh", return_sequences=True, name="decoder_lstm")(decoder_inputs)
    decoder_lstm = Dropout(dropout, name="decoder_dropout2")(decoder_lstm)
    decoder_outputs = TimeDistributed(Dense(num_features, activation="linear"), name="decoder_dense")(decoder_lstm)

    model = Model(encoder_inputs, decoder_outputs, name="Seq2Seq_Model")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model
