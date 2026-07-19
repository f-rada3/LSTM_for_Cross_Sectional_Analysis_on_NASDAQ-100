# Cross Sectional Analysis of Returns on the NASDAQ-100

## Overview
This project is realized as a deep learning project for university and applies machine learning techniques to predict the directional movement of stock returns in the NASDAQ-100 index using Long Short-Term Memory (LSTM) networks, GRU networks, and Random Forests. Inspired by the methodology of Fischer & Krauss (Deep learning with long short-term memory networks for financial market predictions), the goal is to evaluate the predictive power of LSTM networks in a period of increased market efficiency (post-2010).

The project focuses on cross-sectional analysis, where the task is to predict whether the return of a stock on a given day will be above or below the daily cross-sectional median of all tickers in the dataset. The results are evaluated using a long-short trading strategy, which simulates portfolio returns based on the model's predictions.
___
## Project Objectives
1. **Task:**
Perform binary classification to predict whether a stock's return on the following day will be above or below the daily median return of all stocks.

2. **Dataset:**
The dataset includes daily adjusted closing prices for all stocks that were part of the NASDAQ-100 index between 2008 and 2026. Survivorship bias is avoided by including stocks that left or joined the index during this period.

3. **Evaluation:**
The models are evaluated using both classification metrics (e.g., accuracy, AUC) and financial metrics derived from the long-short strategy:

- Mean Return (%)

- Standard Deviation of Returns (%)

- Sharpe Ratio (Annualized)

- Trade Accuracy
___

## Methodology

### 1. Data Preprocessing

- **Returns Calculation:**
Daily percentage returns are calculated as: $ r_t = \frac{Adj\_Close(t) - Adj\_Close(t-1)}{Adj\_Close(t-1)}$

- **Cross-Sectional Median:**
For each day, the median return across all tickers is calculated. Stocks are labeled as:
    (i) 1 if their return is greater than or equal to the median.
    (ii) 0 otherwise.

- **Sequence Construction:**
The LSTM and GRU models are trained on sequences of 240 consecutive daily returns (approximately one year of trading data).

### 2. Study Periods
The dataset is divided into 15 rolling study periods, each spanning 1,000 days:
- Training Period: 750 days.
- Trading Period: 250 days (out-of-sample evaluation).

The rolling window advances by 250 days for each new study period.

### 3. Models
- **LSTM Baseline:**
A standard LSTM network with 25 hidden units, dropout regularization, and binary cross-entropy loss.

- **LSTM with Focal Loss:**
An LSTM network using Focal Loss to handle class imbalance and focus on difficult examples.

- **GRU Network:**
A Gated Recurrent Unit (GRU) network with a similar architecture to the LSTM.

- **Random Forest:**
A memory-free baseline model trained on flattened feature vectors.

#### 4. Trading Strategy
- **Long-Short Portfolio:**
Long: Top k stocks with the highest predicted probability of outperforming the median.
    (i) Short: Bottom k stocks with the lowest predicted probability.
    (ii) Portfolio return:  $\text{Return} = \frac{\text{Mean Long Return} - \text{Mean Short Return}}{2}$

- **Portfolio Size:**
A fixed portfolio size of k=10 is used.
___
## Results
The models are evaluated on both classification performance and financial performance. Key metrics include:

1. Validation Metrics:

    - Accuracy: Percentage of correct predictions.
    - AUC: Area under the ROC curve.
2. Trading Metrics:

    - Mean Return (%): Average daily return of the long-short portfolio.
    - Standard Deviation (%): Volatility of daily returns.
    - Sharpe Ratio: Risk-adjusted return.
    - Trade Accuracy: Percentage of correct predictions during the trading period.
___
### Key Findings
1. **LSTM Performance:**
The LSTM models outperform the Random Forest baseline in terms of Sharpe ratio and mean return, demonstrating the ability of recurrent networks to capture temporal dependencies in financial data.

2. **Focal Loss:**
The LSTM with Focal Loss achieves better performance in imbalanced scenarios, focusing on difficult-to-classify examples.

3. **Market Efficiency:**
The predictive power of the models varies across study periods, reflecting changes in market efficiency and volatility regimes.

4. **Volatility and Tail Risk:**
Rolling statistics (e.g., skewness, kurtosis) highlight the non-stationarity of financial markets, with extreme events (e.g., 2020 COVID-19 crisis) significantly impacting model performance.
___
```.txt
├── NASDAQ-100.ipynb          # Main Jupyter Notebook with the full analysis
├── data/
│   ├── NASDAQ100_between_2008_2026.csv  # List of tickers in the NASDAQ-100
│   ├── daily_prices.csv                 # Adjusted closing prices for all tickers
├── models/
│   ├── lstm_baseline.h5                 # Trained LSTM baseline model
│   ├── lstm_focal.h5                    # Trained LSTM with Focal Loss
│   ├── random_forest.pkl                # Trained Random Forest model
├── results/
│   ├── metrics_summary.csv             # Summary of evaluation metrics
│   ├── equity_curves.png               # Cumulative return plots
├── README.md                           # Project description
```
___

## Future Work
1. Dynamic Universe:
Incorporate historical changes in the NASDAQ-100 index composition to improve the realism of the backtest.

2. Feature Engineering:
Explore additional features such as rolling skewness, kurtosis, and macroeconomic indicators.

3. Alternative Models:
Test other architectures (e.g., Transformers) and ensemble methods.

4. Transaction Costs:
Include transaction costs to evaluate the real-world feasibility of the strategy.

5. Cross Validation Over Different Time Periods
