"""
RNN-based forecasting module for financial time series.

Provides the RNNForecaster class supporting:
- LSTM and GRU models
- Sequential feature preparation
- Iterative forecasting
- Training, evaluation (RMSE, MAE), and visualization
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from preprocessing import TimeSeriesDataset


class RNNForecaster:
    """
    RNN-based forecaster for financial time series using LSTM or GRU.

    Supports iterative forecasting, evaluation, and visualization.
    """
    def __init__(self, model_type='lstm', input_size=1, hidden_size=50,
                 num_layers=2, window_size=20, batch_size=32, device=None):
        """
        Args:
            model_type (str): 'lstm' or 'gru'.
            input_size (int): Number of input features (1 for price only).
            hidden_size (int): Number of hidden units in each RNN layer.
            num_layers (int): Number of RNN layers.
            window_size (int): Number of past steps to use for prediction.
            batch_size (int): Batch size for training.
            device (torch.device): Device to run the model on (CPU/GPU).
        """
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = MinMaxScaler()
        self.trained = False

    def _build_model(self):
        """Internal: build the PyTorch RNN model."""
        if self.model_type == 'lstm':
            self.model = nn.LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 batch_first=True)
        elif self.model_type == 'gru':
            self.model = nn.GRU(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        self.fc = nn.Linear(self.hidden_size, 1)
        self.model.to(self.device)
        self.fc.to(self.device)

    def prepare_data(self, train_series: pd.Series, test_series: pd.Series):
        """
        Normalize and create PyTorch Datasets for training and testing.

        Args:
            train_series (pd.Series): Training time series.
            test_series (pd.Series): Test time series.
        """
        # Fit scaler on train, transform both train and test
        train_scaled = self.scaler.fit_transform(train_series.values.reshape(-1,1))
        test_scaled = self.scaler.transform(test_series.values.reshape(-1,1))

        # Create datasets
        self.train_dataset = TimeSeriesDataset(train_scaled, self.window_size)
        combined = np.concatenate([train_scaled[-self.window_size:], test_scaled])
        self.test_dataset = TimeSeriesDataset(combined, self.window_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def train(self, epochs=50, lr=0.001):
        """
        Train the RNN model on the prepared training data.

        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate for Adam optimizer.
        """
        self._build_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=lr)
        self.model.train()
        self.fc.train()

        for epoch in range(epochs):
            losses = []
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for X_batch, y_batch in loop:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                rnn_out, _ = self.model(X_batch)
                y_pred = self.fc(rnn_out[:, -1, :])

                loss = criterion(y_pred.view(-1), y_batch.view(-1))                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                loop.set_postfix(loss=np.mean(losses))
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {np.mean(losses):.6f}")

        self.trained = True

    def predict(self):
        """
        Iterative forecast on the test set using trained model.

        Returns:
            np.ndarray: Predicted values in original scale.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        self.model.eval()
        self.fc.eval()
        predictions = []

        with torch.no_grad():
            for X_batch, _ in self.test_loader:
                X_batch = X_batch.to(self.device)
                rnn_out, _ = self.model(X_batch)
                y_pred = self.fc(rnn_out[:, -1, :])
                predictions.append(y_pred.cpu().item())

        # Inverse scaling
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Compute RMSE and MAE of the predictions.

        Args:
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: {'rmse': value, 'mae': value}
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {'rmse': rmse, 'mae': mae, 'mape': mape}

    def plot_forecast(self, y_true: pd.Series, y_pred: np.ndarray, title="RNN Forecast vs Actual"):
        """
        Plot actual vs predicted series.

        Args:
            y_true (pd.Series): True values.
            y_pred (np.ndarray): Predicted values.
            title (str): Plot title.
        """
        plt.figure(figsize=(12,6))
        plt.plot(y_true.index, y_true.values, label="Actual", color="black")
        plt.plot(y_true.index, y_pred, label=f"{self.model_type.upper()} Forecast", color="red")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
