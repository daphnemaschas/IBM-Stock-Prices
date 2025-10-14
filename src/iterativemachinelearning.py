"""
Iterative Machine Learning forecasting module for financial time series.

Provides MLForecasterIterative class supporting:
- XGBoost and LightGBM models
- Sequential feature engineering for time series
- Iterative forecasting on the test set
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

class MLForecasterIterative:
    """
    ML-based iterative forecasting class for financial time series using XGBoost or LightGBM.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target_col: str = "Adj Close",
                 model_type: str = "xgb", n_lags: int = 5, rolling_window: int = 5):
        """
        Args:
            train (pd.DataFrame): Training time series (chronologically sorted).
            test (pd.DataFrame): Test time series (chronologically sorted).
            target_col (str): Name of target variable.
            model_type (str): 'xgb' or 'lgb'.
            n_lags (int): Number of lag features to create.
            rolling_window (int): Window size for rolling statistics.
        """
        self.train = train.copy()
        self.test = test.copy()
        self.target_col = target_col
        self.model_type = model_type
        self.n_lags = n_lags
        self.rolling_window = rolling_window
        self.model = None
        self.predictions = pd.Series(index=self.test.index, dtype=float)

    def _create_features(self, series: pd.Series) -> pd.DataFrame:
        """Create lag and rolling features for a given series."""
        df = pd.DataFrame({self.target_col: series})
        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)
        df[f"rolling_mean_{self.rolling_window}"] = df[self.target_col].shift(1).rolling(self.rolling_window).mean()
        df[f"rolling_std_{self.rolling_window}"] = df[self.target_col].shift(1).rolling(self.rolling_window).std()
        df.dropna(inplace=True)
        return df

    def fit_model(self, X: pd.DataFrame, y: pd.Series):
        """Instantiate and fit the model."""
        if self.model_type == "xgb":
            self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500)
        elif self.model_type == "lgb":
            self.model = lgb.LGBMRegressor(n_estimators=500)
        else:
            raise ValueError("model_type must be 'xgb' or 'lgb'")
        self.model.fit(X, y)

    def fit(self):
        """Fit the ML model on the training set using generated features."""
        train_features = self._create_features(self.train[self.target_col])
        X_train = train_features.drop(columns=[self.target_col])
        y_train = train_features[self.target_col]
        self.fit_model(X_train, y_train)

    def predict_iterative(self) -> pd.Series:
        """
        Make iterative predictions on the test set.
        Each prediction uses only past true values (train + already predicted test points).
        """
        history = self.train[self.target_col].copy()
        for idx in self.test.index:
            # Combine history with current test point placeholder
            temp_series = pd.concat([history, pd.Series([np.nan], index=[idx])])
            features = self._create_features(temp_series).iloc[[-1]]  # last row only
            X_test = features.drop(columns=[self.target_col])
            y_pred = self.model.predict(X_test)[0]
            self.predictions.at[idx] = y_pred
            history.at[idx] = y_pred  # add predicted value to history for next step
        return self.predictions

    def evaluate(self, metric="rmse") -> float:
        """Evaluate predictions using RMSE, MAE, or MAPE."""
        if self.predictions.isna().any():
            raise ValueError("Run predict_iterative() first.")
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

    def plot_forecast(self, title="Iterative ML Forecast vs Actual"):
        """Plot predicted vs actual series."""
        if self.predictions.isna().any():
            raise ValueError("Run predict_iterative() first.")
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