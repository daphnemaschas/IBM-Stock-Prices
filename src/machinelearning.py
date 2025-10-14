"""
Machine Learning forecasting module for financial time series.

Provides a MLForecaster class supporting:
- XGBoost and LightGBM models
- Feature engineering for time series
- Train/test split handling
- Evaluation with RMSE, MAE, MAPE
- Forecast visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import xgboost as xgb
import lightgbm as lgb

class MLForecaster:
    """
    ML-based forecasting class for financial time series using XGBoost or LightGBM.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_col: str = "Adj Close", model_type: str = "xgb", n_lags: int = 5):
        """
        Args:
            train (pd.DataFrame): Training time series (chronologically sorted).
            test (pd.DataFrame): Test time series (chronologically sorted).
            target_col (str): Name of target variable.
            model_type (str): 'xgb' or 'lgb'.
            n_lags (int): Number of lag features to create.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.target_col = target_col
        self.model_type = model_type
        self.n_lags = n_lags
        self.model = None
        self.predictions = None

        self._create_features()

    def _create_features(self):
        """Create lag and rolling features for train and test sets."""
        # Train features
        for lag in range(1, self.n_lags + 1):
            self.train[f"lag_{lag}"] = self.train[self.target_col].shift(lag)
        self.train["rolling_mean_5"] = self.train[self.target_col].shift(1).rolling(5).mean()
        self.train["rolling_std_5"] = self.train[self.target_col].shift(1).rolling(5).std()
        self.train.dropna(inplace=True)
        
        # Test features
        self.test = self.test.copy()
        for lag in range(1, self.n_lags + 1):
            self.test[f"lag_{lag}"] = pd.concat([self.train[self.target_col], self.test[self.target_col]]).shift(lag).iloc[len(self.train):]
        rolling_series = pd.concat([self.train[self.target_col], self.test[self.target_col]]).shift(1)
        self.test["rolling_mean_5"] = rolling_series.rolling(5).mean().iloc[len(self.train):]
        self.test["rolling_std_5"] = rolling_series.rolling(5).std().iloc[len(self.train):]
        self.test.dropna(inplace=True)

    def fit(self):
        """Fit the ML model on the training set."""
        X_train = self.train.drop(columns=[self.target_col])
        y_train = self.train[self.target_col]

        if self.model_type == "xgb":
            self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500)
        elif self.model_type == "lgb":
            self.model = lgb.LGBMRegressor(n_estimators=500)
        else:
            raise ValueError("model_type must be 'xgb' or 'lgb'")

        self.model.fit(X_train, y_train)

    def predict(self) -> pd.Series:
        """Make predictions on the test set."""
        X_test = self.test.drop(columns=[self.target_col])
        self.predictions = pd.Series(self.model.predict(X_test), index=self.test.index)
        return self.predictions

    def evaluate(self, metric="rmse") -> float:
        """Evaluate predictions using RMSE, MAE, or MAPE."""
        if self.predictions is None:
            raise ValueError("Run predict() first.")

        y_true = self.test[self.target_col].values
        y_pred = self.predictions.values

        if metric == "rmse":
            return sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "mape":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            raise ValueError("metric must be 'rmse', 'mae', or 'mape'")

    def plot_forecast(self, title="ML Forecast vs Actual"):
        """Plot predicted vs actual series."""
        if self.predictions is None:
            raise ValueError("Run predict() first.")

        plt.figure(figsize=(12,6))
        plt.plot(self.train.index, self.train[self.target_col], label="Train", color="blue")
        plt.plot(self.test.index, self.test[self.target_col], label="Test (Actual)", color="black")
        plt.plot(self.predictions.index, self.predictions, label="Forecast", color="red")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.show()
