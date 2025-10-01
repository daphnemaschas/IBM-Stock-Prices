## Context
IBM (International Business Machines Corporation) is one of the oldest and most influential technology companies in the world. Tracking IBM’s stock performance over time provides valuable insights into the evolution of the tech industry, the impact of macroeconomic events, and the effects of dividends and stock splits on shareholder returns.

Financial researchers and analysts often rely on adjusted prices to measure true returns, because raw close prices do not account for dividends and corporate actions. This dataset provides IBM’s historical stock data from 1980 to 2025, with an Adj Close column computed according to CRSP-style standards, making it suitable for academic, financial, and machine learning research.

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

## Inspiration
This dataset can be useful for:

Financial analysts exploring long-term returns.

Data scientists building stock price prediction models.

Students practicing time-series forecasting.

Researchers performing event studies (e.g., effect of dividends, stock splits, or economic crises).