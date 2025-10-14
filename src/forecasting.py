"""
Object-oriented forecasting module for financial time series.

Provides a Forecaster class that supports:
- Baseline forecasting (naïve / persistence)
- ARIMA / SARIMA models
- Prophet forecasting
- Optional log-transform for non-stationary series
- Model evaluation with MAPE, RMSE
- Visualization of forecasts vs actual data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


class Forecaster:
    """
    A forecasting class supporting multiple time series models.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_col: str = "Adj Close", train_start: str = None, log_transform: bool = False):
        """
        Initialize the forecaster with train and test data.

        Args:
            train (pd.DataFrame): Training time series (chronologically sorted).
            test (pd.DataFrame): Test time series (chronologically sorted).
            target_col (str): Name of the target variable column.
        """
        if train_start:
            self.train = train[train.index >= train_start].copy()
        else:
            self.train = train.copy()
    
        self.test = test
        self.target_col = target_col
        self.log_transform = log_transform

        # Internal transformed series
        if self.log_transform:
            self.train["_y"] = np.log1p(self.train[self.target_col])
            self.test["_y"] = np.log1p(self.test[self.target_col])
        else:
            self.train["_y"] = self.train[self.target_col]
            self.test["_y"] = self.test[self.target_col]

        self.model = None
        self.predictions = None

    def _inverse_transform(self, y_pred: pd.Series) -> pd.Series:
        """Inverse log transform if enabled."""
        return np.expm1(y_pred) if self.log_transform else y_pred


    def forecast_naive(self) -> pd.Series:
        """
        Naïve forecast: predicts that the next value equals the last observed value.

        Returns:
            pd.Series: Forecasted values for the test period.
        """
        last_value = self.train["_y"].iloc[-1]
        forecast  = pd.Series([last_value] * len(self.test), index=self.test.index)
        self.predictions = self._inverse_transform(forecast)
        return self.predictions

    def forecast_arima(self, order=(1, 1, 1)) -> pd.Series:
        """
        Fit and forecast using an ARIMA model.

        Args:
            order (tuple): ARIMA order (p, d, q).

        Returns:
            pd.Series: Forecasted values for the test period.
        """
        y_train = self.train["_y"]

        self.model = ARIMA(y_train, order=order).fit()
        forecast = self.model.forecast(steps=len(self.test))
        forecast.index = self.test.index
        self.predictions = self._inverse_transform(forecast)
        return self.predictions

    def forecast_prophet(self) -> pd.Series:
        """
        Fit and forecast using a Prophet model.

        Returns:
            pd.Series: Forecasted values for the test period.
        """
        prophet_train = self.train.reset_index().rename(columns={"Date": "ds", "_y": "y"})

        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_train)

        future = pd.DataFrame(self.test.index).reset_index(drop=True)
        future.columns = ["ds"]

        forecast = model.predict(future)
        self.model = model
        preds = pd.Series(forecast["yhat"].values, index=self.test.index)
        self.predictions = self._inverse_transform(preds)
        return self.predictions

    def evaluate(self, metric="mape") -> float:
        """
        Evaluate model forecasts against actual test data.

        Args:
            metric (str): Evaluation metric ('mape', 'rmse', or 'mae').

        Returns:
            float: Evaluation score.
        """
        if self.predictions is None:
            raise ValueError("No predictions found. Run a forecast method first.")

        y_true = self.test[self.target_col].values
        y_pred = self.predictions.values

        if metric == "mape":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif metric == "rmse":
            return sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric. Choose from 'mape', 'rmse', or 'mae'.")

    def plot_forecast(self, title="Forecast vs Actual"):
        """Plot forecasted values vs actual test values."""
        if self.predictions is None:
            raise ValueError("No predictions found. Run a forecast method first.")

        plt.figure(figsize=(12, 6))
        plt.plot(self.train.index, self.train[self.target_col], label="Train", color="blue")
        plt.plot(self.test.index, self.test[self.target_col], label="Test (Actual)", color="black")
        plt.plot(self.predictions.index, self.predictions, label="Forecast", color="red")
        plt.title(title + (" (log-transform)" if self.log_transform else ""))
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.show()

