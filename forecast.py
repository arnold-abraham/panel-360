import os
import random
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.ensemble import IsolationForest

def split_into_segments_with_gaps(df, max_small_gap_hours=3):
    """
    Splits a time-indexed DataFrame into continuous segments.
    Segments are defined by consecutive rows whose timestamps differ by <= max_small_gap_hours.

    If no gap larger than max_small_gap_hours exists, the entire dataset is treated as one segment.
    """
    df = df.sort_index()  
    segments = []
    start_idx = 0
    df_index = df.index
    found_large_gap = False  

    for i in range(1, len(df_index)):
        time_diff_hours = (df_index[i] - df_index[i - 1]).total_seconds() / 3600.0
        if time_diff_hours > max_small_gap_hours:
            found_large_gap = True  
            segment = df.iloc[start_idx:i].copy()
            segment = process_and_fill_segment(segment, max_small_gap_hours)
            print(f"Segment {len(segments) + 1} from {df_index[start_idx]} to {df_index[i-1]}: {len(segment)} rows")
            if len(segment) > 0:
                segments.append(segment)
            start_idx = i  
    if not found_large_gap:
        print("No large gaps found. Treating entire dataset as a single segment.")
        segment = process_and_fill_segment(df.copy(), max_small_gap_hours)
        if len(segment) > 0:
            segments.append(segment)
    else:
        if start_idx < len(df):
            segment = process_and_fill_segment(df.iloc[start_idx:].copy(), max_small_gap_hours)
            print(f"Final segment {len(segments) + 1} from {df_index[start_idx]} to {df_index[-1]}: {len(segment)} rows")
            if len(segment) > 0:
                segments.append(segment)

    print(f"Total segments created: {len(segments)}")
    return segments


def process_and_fill_segment(segment, max_gap_hours=3):
    """
    Resamples a segment to hourly frequency and interpolates missing values
    using linear interpolation up to max_gap_hours.
    """
    segment = segment.resample('h').asfreq()
    segment = segment.interpolate(method='linear', limit=max_gap_hours, limit_direction='forward')
    segment.dropna(how='any', inplace=True)
    return segment


def create_sliding_windows_from_segments(segments, input_days=7, forecast_horizon=24):
    """
    Creates sliding windows for each continuous segment,
    then concatenates all windows into one big X, y.
    """
    sequence_length = input_days * 24
    all_X, all_y = [], []
    
    for seg in segments:
        if len(seg) < (sequence_length + forecast_horizon):
            continue
        data = seg.values  # shape: (num_hours, num_features)
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X_seq = data[i : i + sequence_length]
            y_seq = data[i + sequence_length : i + sequence_length + forecast_horizon]
            all_X.append(X_seq)
            all_y.append(y_seq)

    if not all_X:
        return np.empty((0,0,0)), np.empty((0,0,0))
    
    X = np.array(all_X)  # shape: (num_windows, sequence_length, num_features)
    y = np.array(all_y)  # shape: (num_windows, forecast_horizon, num_features)
    return X, y

def apply_pattern(x_seq, pattern):
    """
    Applies a 'blanking' pattern to the first 6 days of a 7-day sequence.
    x_seq shape: (168, num_features).
    """
    x_modified = x_seq.copy()
    if pattern == 'all_6_days':
        x_modified[:144, :] = 0
    elif pattern == 'one_day':
        day_to_blank = random.randint(0, 5)
        start = day_to_blank * 24
        end = (day_to_blank + 1) * 24
        x_modified[start:end, :] = 0
    elif pattern == 'two_days':
        days_to_blank = random.sample(range(6), 2)
        for d in days_to_blank:
            start = d * 24
            end = (d + 1) * 24
            x_modified[start:end, :] = 0
    return x_modified


def load_binary_data(folder):
    """
    Function that loads *.bin files in a folder 
    and returns them as a stacked NumPy array.
    """
    import glob
    all_files = glob.glob(os.path.join(folder, "*.bin"))
    all_files.sort()
    
    data_list = []
    for f in all_files:
        arr = np.fromfile(f, dtype=np.float32) 
        arr = arr.reshape(32, 32, 1)
        data_list.append(arr)
    data_array = np.stack(data_list, axis=0)
    return data_array, all_files

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    config = configparser.ConfigParser()
    config.read('config.ini')
    output_folder = config['PATHS']['output_folder']
    input_days = int(config['FORECAST']['input_days'])
    actual_next24_binary_folder = config['PATHS']['actual_next24_binary_folder']
    latent_vectors_path = os.path.join(output_folder, "latent_vectors.csv")

    latent_vectors_df = pd.read_csv(latent_vectors_path, index_col=0,parse_dates=True)
    latent_vectors_df.sort_index(inplace=True)

    print("Latent Vectors DF:", latent_vectors_df.shape)
    print(latent_vectors_df.head())

    segments = split_into_segments_with_gaps(latent_vectors_df, max_small_gap_hours=3)
    X, y = create_sliding_windows_from_segments(
        segments,
        input_days=input_days,
        forecast_horizon=24
    )
    print("X shape after gap handling and windowing:", X.shape)
    print("y shape after gap handling and windowing:", y.shape)

    if input_days == 7:
        patterns = ['all_6_days', 'one_day', 'two_days', 'none']
        X_augmented = []
        y_augmented = []
        for i in range(X.shape[0]):
            X_seq_original = X[i]
            y_seq = y[i]
            for pat in patterns:
                X_seq_modified = apply_pattern(X_seq_original, pat)
                X_augmented.append(X_seq_modified)
                y_augmented.append(y_seq)
        X = np.array(X_augmented)
        y = np.array(y_augmented)
        print("X shape after applying blanking patterns:", X.shape)
        print("y shape after applying blanking patterns:", y.shape)

    # TIME-SERIES SPLIT (80/10/10)
    total_sequences = X.shape[0]
    train_size = int(0.8 * total_sequences)
    val_size = int(0.1 * total_sequences)
    test_size = total_sequences - train_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    sequence_length = input_days * 24
    num_features = X.shape[-1]
    forecast_horizon = 24

    # Build LSTM Seq2Seq Model
    encoder_inputs = Input(shape=(sequence_length, num_features), name="encoder_inputs")
    encoder_lstm = LSTM(96, activation='tanh', return_sequences=False, name="encoder_lstm")(encoder_inputs)
    encoder_lstm = Dropout(0.2, name="encoder_dropout")(encoder_lstm)

    decoder_inputs = RepeatVector(forecast_horizon, name="decoder_repeat_vector")(encoder_lstm)
    decoder_lstm = LSTM(96, activation='tanh', return_sequences=True, name="decoder_lstm")(decoder_inputs)
    decoder_lstm = Dropout(0.2, name="decoder_dropout2")(decoder_lstm)
    decoder_outputs = TimeDistributed(Dense(num_features, activation='linear'), name="decoder_dense")(decoder_lstm)

    model = Model(encoder_inputs, decoder_outputs, name="Seq2Seq_Model")
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate on test set
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.legend()
    plt.title('Seq2Seq Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_loss_path = os.path.join(output_folder, "seq2seq_loss_curves.png")
    plt.savefig(plot_loss_path)
    plt.show()
    print(f"Loss curves saved to: {plot_loss_path}")

    # Save Model
    model_path = os.path.join(output_folder, "seq2seq_forecast.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # FORECASTING NEXT 24 HOURS
    # Here, we'll take the last window in X, and forecast
    last_sequence = X[-1].reshape(1, sequence_length, num_features)
    forecast = model.predict(last_sequence)  # shape: (1, 24, num_features)
    forecast = forecast.squeeze()            # shape: (24, num_features)

    # Reindex forecast with timestamps: let's assume the last segment ends at:
    last_timestamp = latent_vectors_df.index[-1]
    forecast_timestamps = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]

    forecasted_latent_vectors_df = pd.DataFrame(forecast, index=forecast_timestamps, 
                                                columns=latent_vectors_df.columns)
    forecasted_latent_vectors_path = os.path.join(output_folder, "forecasted_latent_vectors.csv")
    forecasted_latent_vectors_df.to_csv(forecasted_latent_vectors_path)
    print(f"Forecasted latent vectors saved to {forecasted_latent_vectors_path}")

    # --- LOAD PRE-TRAINED AUTOENCODER & RECONSTRUCT IMAGES ---
    autoencoder_path = os.path.join(output_folder, "convolutional_autoencoder.keras")
    if os.path.exists(autoencoder_path):
        from autoencoder import CustomAutoencoder
        autoencoder = load_model(autoencoder_path, custom_objects={'CustomAutoencoder': CustomAutoencoder}, compile=False)
        decoder = autoencoder.decoder()  

        image_scaler_path = os.path.join(output_folder, 'scaler.joblib')
        image_scaler = joblib.load(image_scaler_path)
        reconstructed_images = decoder.predict(forecast)  
        # Inverse scaling
        reconstructed_images_flat = reconstructed_images.reshape(-1, 1)
        reconstructed_images_original_scale = image_scaler.inverse_transform(reconstructed_images_flat)
        reconstructed_images_original_scale = reconstructed_images_original_scale.reshape(reconstructed_images.shape)

        def save_heatmaps(predicted_temperatures, output_folder):
            vmin = 10
            vmax = 50
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for i in range(predicted_temperatures.shape[0]):
                plt.figure(figsize=(6, 6))
                panel_data = predicted_temperatures[i, :, :, 0]
                plt.imshow(panel_data, cmap='seismic', interpolation='nearest', vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(label='Temperature (°C)')
                cbar.set_ticks(np.linspace(vmin, vmax, num=11)) 
                plt.title(f'Predicted Temperature for Hour {i+1}')
                plt.savefig(os.path.join(output_folder, f'heatmap_hour_{i+1}.png'))
                plt.close()

        heatmap_dir = os.path.join(output_folder, "forecasted_heatmaps")
        save_heatmaps(reconstructed_images_original_scale, heatmap_dir)
        print(f"Heatmaps saved to {heatmap_dir}")

    # --- ISOLATION FOREST: TRAIN ONLY ON TRAIN LATENT VECTORS ---
    train_end_index = train_size + sequence_length
    latent_vectors_train_df = latent_vectors_df.iloc[:train_end_index]  

    clf_yellow = IsolationForest(random_state=42, contamination=0.1)
    clf_red = IsolationForest(random_state=42, contamination=0.01)

    clf_yellow.fit(latent_vectors_train_df.values)
    clf_red.fit(latent_vectors_train_df.values)

    def anomaly_detection(pretrained_clf_yellow, pretrained_clf_red, seq_df):
        seq_df = seq_df.copy()
        preds_yellow = pretrained_clf_yellow.predict(seq_df.values)
        preds_red = pretrained_clf_red.predict(seq_df.values)

        seq_df['alarm'] = 'Green'
        seq_df.loc[preds_yellow == -1, 'alarm'] = 'Yellow'
        seq_df.loc[preds_red == -1, 'alarm'] = 'Red'
        return seq_df

    last_24_hours_df = latent_vectors_df.tail(24)
    last_24_latent_vectors = last_24_hours_df.values

    historical_alarms = anomaly_detection(clf_yellow, clf_red, last_24_hours_df)
    forecast_alarms = anomaly_detection(clf_yellow, clf_red, forecasted_latent_vectors_df)

    reconstructed_last_24 = decoder.predict(last_24_latent_vectors)
    reconstructed_last_24_flat = reconstructed_last_24.reshape(-1, 1)
    reconstructed_last_24_original_scale = image_scaler.inverse_transform(reconstructed_last_24_flat)
    reconstructed_last_24_original_scale = reconstructed_last_24_original_scale.reshape(reconstructed_last_24.shape)

    last_24_max_temps = reconstructed_last_24_original_scale.reshape(24, 32*32).max(axis=1)
    forecast_24_max_temps = reconstructed_images_original_scale.reshape(24, 32*32).max(axis=1)

    historical_results = last_24_hours_df.copy()
    historical_results['alarm'] = historical_alarms['alarm']
    historical_results['max_temperature'] = last_24_max_temps
    historical_results['forecasted'] = 'no'

    forecast_results = forecasted_latent_vectors_df.copy()
    forecast_results['alarm'] = forecast_alarms['alarm']
    forecast_results['max_temperature'] = forecast_24_max_temps
    forecast_results['forecasted'] = 'yes'

    combined_results = pd.concat([historical_results, forecast_results])
    combined_results['datetime'] = combined_results.index.strftime('%d/%m/%Y %H:%M')
    combined_results = combined_results[['datetime', 'alarm', 'max_temperature', 'forecasted']]

    final_results_path = os.path.join(output_folder, "anomaly_forecast_results.csv")
    combined_results.to_csv(final_results_path, index=False)
    print(f"Final combined anomaly and forecast results saved to {final_results_path}")

    # Load Actual Next 24 Hours and Compute Max Temperatures
    actual_next24_data, actual_next24_filenames = load_binary_data(actual_next24_binary_folder)
    actual_24_max_temps = actual_next24_data.reshape(24, 32*32).max(axis=1)

    # Plot Last 24 Hours, Predicted Next 24 Hours, and Actual Next 24 Hours Max Temperatures
    x_axis = np.arange(1, 49)  # 1 to 48
    max_temperatures = np.concatenate([last_24_max_temps, forecast_24_max_temps])

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, max_temperatures, marker='o', linestyle='-', label='Predicted Next 24 Hours', color='b')
    plt.axvline(x=24.5, color='r', linestyle='--', label='Forecast Start')

    # Plot the historical last 24 hours max temperatures (1-24)
    plt.plot(x_axis[:24], last_24_max_temps, marker='o', linestyle='-', label='Last 24 Hours', color='g')

    # Plot the actual next 24 hours max temperatures (25-48)
    plt.plot(x_axis[24:], actual_24_max_temps, marker='o', linestyle='-', label='Actual Next 24 Hours', color='orange')

    plt.title("Max Temperatures for Last 24, Predicted Next 24, and Actual Next 24 Hours", fontsize=16)
    plt.xlabel("Time Step (1-48)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.xticks(ticks=np.arange(1, 49, step=2))
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_folder, "max_temperatures_with_actual_plot.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    main()
