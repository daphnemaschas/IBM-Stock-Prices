## Context
IBM (International Business Machines Corporation) is one of the oldest and most influential technology companies in the world. Tracking IBM’s stock performance over time provides valuable insights into the evolution of the tech industry, the impact of macroeconomic events, and the effects of dividends and stock splits on shareholder returns.

Financial researchers and analysts often rely on adjusted prices to measure true returns, because raw close prices do not account for dividends and corporate actions. This dataset provides IBM’s historical stock data from 1980 to 2025, with an Adj Close column computed according to CRSP-style standards, making it suitable for academic, financial, and machine learning research.

The dataset and element of context have been found on Kaggle.

## Content
The dataset contains daily stock market data for IBM obtained from Yahoo Finance. Unlike raw downloads, this dataset includes a carefully constructed Adjusted Close column, calculated using dividends and stock splits.

## The dataset contains:

Date range: January 1980 – July 2025

Daily stock prices (Open, High, Low, Close)

Trading volume

Dividends and stock splits

Adjusted Close (manually computed using cumulative adjustment factors)

## Variables
Date → Trading day (YYYY-MM-DD).

Open → Stock price at market open.

High → Highest price during the trading day.

Low → Lowest price during the trading day.

Close → Stock price at market close.

Volume → Number of shares traded.

Dividends → Cash dividends paid on that date (if any).

Stock Splits → Ratio of stock splits on that date (e.g., 2.0 = 2-for-1 split).

Adj Close → Closing price adjusted for dividends and splits (CRSP-style adjustment).

## Acknowledgment
Data was sourced from Yahoo Finance

Adjustments were implemented following standards used by Center for Research in Security Prices (CRSP).

## Subjects tackled

1. Time Series Forecasting

- Classical models: ARIMA, SARIMA, Prophet

- Deep learning: LSTM, GRU, 1D CNN, Transformer-based models

2. Anomaly Detection

- Identifying periods of major financial crashes (2000, 2008, 2020, etc.)

- Outlier detection methods for abnormal market behavior

- Use of unsupervised deep learning (autoencoders, isolation forests)

3. Feature Engineering and Financial Indicators

- Creating derived features such as moving averages, RSI, MACD, and volatility

- Computing daily and cumulative returns

- Evaluating whether engineered features improve model performance

4. Visualization and Dashboard

- Interactive plots showing long-term historical stock trends

- Comparison of raw vs adjusted stock prices

- Streamlit dashboard for real-time forecasting and anomaly exploration