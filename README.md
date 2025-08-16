# PANEL 360

## Getting Started

To set up and run the project, follow the steps below:

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn joblib configparser opencv-python
```

### File Structure
- `config.ini`: Configuration file with paths.
- `main.py`: Prepares data, trains an autoencoder, and extracts latent vectors.
- `forecast.py`: Forecasts latent vectors, reconstructs images, and detects anomalies.
- `data_utils.py`: Utility functions for loading and processing binary data.
- `autoencoder.py`: Defines the convolutional autoencoder model.

### Running the Scripts

#### 1. Train Autoencoder
Run the following command to process data and train the autoencoder:
```
python main.py
```

#### 2. Forecast and Detect Anomalies
Run the forecasting and anomaly detection script:
```
python forecast.py
```

### Outputs
- `latent_vectors.csv`: Extracted latent vectors.
- `forecasted_latent_vectors.csv`: Forecasted latent representations.
- `forecasted_heatmaps/`: Reconstructed forecasted images.
- `anomaly_forecast_results.csv`: Anomaly detection results.
- `max_temperatures_with_actual_plot.png`: Temperature trend plot.