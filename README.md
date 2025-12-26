

# Macro-Driven Dynamic Asset Allocation Engine

A data-driven investment strategy backtester that combines **macroeconomic regime detection** with **technical momentum signals** to dynamically rebalance a multi-asset portfolio.

This project implements a **weekly rebalancing** strategy using a "History-Based" Markowitz optimization approach. Instead of using recent historical volatility, it identifies similar macroeconomic periods in the past to evaluate actual future risk and returns.

You can also find a png comparing the performance of the SP500 and the results I had with the following configuration : 

-target vol = 15%

-long only

-max weight = 25% (there is only 11 ETF)

-shrinkage = 0 (it increases return)

It gives a sharp ratio of 1.01 with an annual return of 11.50%, a risk of 11.43%, and a maximum drawdown of 13.23%.


## Strategy Methodology

The core logic relies on a three-step process executed every Friday:

### 1. Macro Regime Detection (Monthly)

The engine analyzes key macroeconomic indicators (GDP, CPI, Unemployment, Yield Curve, VIX, etc.) to classify the current economic environment into regimes:

* **Growth (Bullish)**
* **Stagflation / Recession (Bearish)**
* **Stable / Neutral**

### 2. Similarity Search (K-Nearest Neighbors)

Rather than optimizing based on the last 30 days of data, the algorithm looks back at history (starting from 2000) to find the **top  periods** that statistically resemble the current macroeconomic conditions.

* *Technique:* Z-score normalization + Euclidean distance on feature vectors.
* *Input:* Uses the covariance matrix and expected returns from these specific historical "neighbor" periods to feed the optimizer.

### 3. Tactical Momentum Overlay (Weekly)

To mitigate the lag inherent in monthly macro data, a **Daily Momentum Signal** (S&P 500 3-Month Momentum, smoothed) is injected weekly.

* **Signal:** 10-day moving average of the 3-month ROC.
* **Action:** Adjusts the target volatility and risk appetite. If momentum breaks down, the portfolio shifts defensively even if macro data hasn't updated yet.

### 4. Portfolio Optimization

* **Model:** Mean-Variance Optimization (Markowitz).
* **Constraints:** Long-only, Max weight per asset = 35% (to enforce conviction while maintaining diversification).
* **Shrinkage:** Set to 0 (Full reliance on the specific historical correlation structure detected).

## Repository Structure

* `backtest.py`: **Main Engine**. Handles data loading, weekly resampling, signal generation, optimization, and performance reporting.
* `generate_sp500_momentum.py`: **Data Prep**. Fetches daily S&P 500 data via `yfinance`, calculates the smoothed momentum signal, and exports the CSV.
* `macro_indicators_v2.csv`: Dataset containing historical macroeconomic indicators (CPI, GDP, Fed Funds, etc.).
* `asset_prices_daily_11ETF.csv`: Historical price data for the investable universe (SPY, QQQ, GLD, TLT, VNQ, etc.).
* `sp500_daily_mom.csv`: Generated file containing the smoothed daily momentum signal.

## Tech Stack

* **Python 3.10+**
* **Pandas & NumPy:** For vectorization and time-series manipulation.
* **Yfinance:** For fetching real-time market data.
* **Matplotlib:** For visualizing the Equity Curve and Drawdowns.
* **SciPy / NumPy.linalg:** For matrix operations and solving the quadratic optimization problem.

## How to Run

1. **Install dependencies:**
```bash
pip install pandas numpy yfinance matplotlib

```


2. **Generate the Momentum Signal:**
```bash
python generate_sp500_momentum.py

```


3. **Run the Backtest:**
```bash
python backtest.py

```

