"""
Object-oriented preprocessing pipeline for time series forecasting.

This module provides:
- The Preprocessor class for loading, cleaning, and feature engineering of financial data.
- The TimeSeriesDataset class for preparing sequential data for RNN-based models.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class Preprocessor:
    """
    A preprocessing class for financial time series forecasting.
    """

    def __init__(self, path: str):
        """
        Initialize the preprocessor with a data path.

        Args:
            path (str): Path to the CSV file containing the stock data.
        """
        self.path = path
        self.data = None
        self.train = None
        self.test = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and clean stock price data.

        Returns:
            pd.DataFrame: Cleaned DataFrame indexed by Date.
        """
        df = pd.read_csv(self.path)
        df["Volume"] = df["Volume"].str.replace(",", "", regex=False).astype(int)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        self.data = df
        return df

    def add_features(self) -> pd.DataFrame:
        """
        Add engineered features to the stock data (moving averages, volatility, Bollinger Bands).

        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before adding features. Call load_data().")

        df = self.data.copy()

        df["Daily_Return"] = df["Adj Close"].pct_change()
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

        df["SMA_30"] = df["Adj Close"].rolling(30).mean()
        df["SMA_100"] = df["Adj Close"].rolling(100).mean()
        df["EMA_30"] = df["Adj Close"].ewm(span=30, adjust=False).mean()
        df["EMA_100"] = df["Adj Close"].ewm(span=100, adjust=False).mean()

        df["Volatility_30d"] = df["Daily_Return"].rolling(30).std()

        win = 20
        df["BB_Mid"] = df["Adj Close"].rolling(win).mean()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["Adj Close"].rolling(win).std()
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["Adj Close"].rolling(win).std()

        self.data = df
        return df

    def split(self, test_size: float = 0.2):
        """
        Split the time series chronologically into train and test sets.

        Args:
            test_size (float): Fraction of data for testing (default: 0.2).

        Returns:
            (pd.DataFrame, pd.DataFrame): Train and test DataFrames.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before splitting.")

        split_idx = int(len(self.data) * (1 - test_size))
        self.train, self.test = self.data.iloc[:split_idx], self.data.iloc[split_idx:]
        return self.train, self.test

    def prepare_forecast_input(self, target_col: str = "Adj Close") -> pd.DataFrame:
        """
        Prepare the data for time series forecasting frameworks (Prophet, ARIMA, etc.).

        Args:
            target_col (str): Column name of the target variable (default: "Adj Close").

        Returns:
            pd.DataFrame: DataFrame with columns ['ds', 'y'] for Prophet.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before preparing forecast input.")

        return self.data[[target_col]].reset_index().rename(columns={"Date": "ds", target_col: "y"})

    def plot(self, cols=None):
        """
        Plot selected columns from the time series.

        Args:
            cols (list[str], optional): Columns to plot. Defaults to ['Adj Close'].
        """
        if self.data is None:
            raise ValueError("Data must be loaded before plotting.")

        if cols is None:
            cols = ["Adj Close"]

        plt.figure(figsize=(12, 6))
        for c in cols:
            plt.plot(self.data.index, self.data[c], label=c)
        plt.legend()
        plt.title("Time Series Overview")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.

    Converts a 1D array of values into sequences of fixed length
    suitable for RNN training.
    """
    def __init__(self, series: np.ndarray, window_size: int):
        """
        Args:
            series (np.ndarray): 1D array of normalized values.
            window_size (int): Number of past steps to use for prediction.
        """
        self.X, self.y = [], []
        for i in range(window_size, len(series)):
            self.X.append(series[i-window_size:i])
            self.y.append(series[i])
        
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]