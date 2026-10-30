# 🇨🇦 TSX Stock Price Direction Predictor
*An Honest, Leakage-Free Portfolio Project in Financial Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app.py)

---

## 📌 1. The Research Question
> **Can simple machine learning models (Logistic Regression, Random Forest, XGBoost) systematically beat naive baselines when predicting the next-day direction (Up/Down) of major Toronto Stock Exchange (TSX) equities?**

In introductory machine learning tutorials, an accuracy score of **54%** is often celebrated as proof that an algorithm has "beaten a 50% random coin flip." In financial markets, this assumption is fatally flawed:
1. **Equities Have an Upward Drift**: Over a multi-year horizon, major stocks rise more often than they fall. Across our 10-year dataset, TSX stocks closed positive on **52.5% to 56.2% of days**.
2. **The True Hurdle**: A naive trader who does nothing except predict "Up" every single morning automatically achieves **56% to 59% accuracy** on trending stocks like `RY.TO` and `TD.TO`.
3. **The Core Thesis**: Any machine learning model that achieves 54% accuracy on `RY.TO` is actually **underperforming** a zero-intelligence majority guesser. This project tests whether ML models can produce genuine statistical alpha after accounting for market drift, walk-forward validation, and transaction friction.

---

## 📊 2. Dataset & Data Engineering

### Asset Universe
* **Financials**: `RY.TO` (Royal Bank of Canada), `TD.TO` (Toronto-Dominion Bank)
* **Technology**: `SHOP.TO` (Shopify Inc.)
* **Energy & Infrastructure**: `ENB.TO` (Enbridge Inc.), `CNQ.TO` (Canadian Natural Resources)
* **Market Benchmark**: `^GSPTSE` (S&P/TSX Composite Index)

### Data Ingestion & Sanitization ([`src/data.py`](src/data.py))
* **10 Years of Daily OHLCV**: 2,510 trading days downloaded via Yahoo Finance (October 2016 to October 2026).
* **Adjusted vs. Unadjusted Prices**: `auto_adjust=False` was enforced to preserve both raw `Close` and `Adj Close`. Corporate actions (such as Shopify's 10-for-1 split in June 2022) would create catastrophic false -90% drops if unadjusted.
* **Timezone Stripping**: Timestamps were stripped to timezone-naive `YYYY-MM-DD` dates to prevent timezone-offset mismatches when merging equities with index bars.
* **Calendar Alignment**: Verified 100% calendar synchrony across all 6 instruments (zero missing cross-asset dates).

---

## 🛠️ 3. Methodology & Zero Look-Ahead Hygiene

### Feature Engineering ([`src/features.py`](src/features.py))
Every feature computed for trading day $t$ uses exclusively information observable **at or before market close on day $t$**:

| Feature Category | Indicators | Look-Ahead Protection & Stationarity Rationale |
| :--- | :--- | :--- |
| **Lagged Returns** | `return_1d`, `return_2d`, `return_5d`, `return_lag1`, `return_lag2` | Uses $(P_t - P_{t-k}) / P_{t-k}$. Backward-looking only. |
| **Moving Average Ratios** | `sma_ratio_10`, `sma_ratio_50`, `sma_10_50_ratio` | Expressed as $(P_t / \text{SMA}_k) - 1.0$. Raw dollar moving averages are non-stationary; price-to-SMA ratios are stationary percentage premiums/discounts. |
| **Momentum** | `rsi_14`, `momentum_5d`, `momentum_10d` | Wilder’s 14-day smoothed RSI bounded $\in [0, 100]$ and price rates of change. |
| **Rolling Volatility** | `volatility_10d`, `volatility_20d` | Annualized rolling standard deviation of daily returns: $\sigma \times \sqrt{252}$. |
| **Volume Dynamics** | `volume_change_1d`, `volume_change_5d`, `volume_ratio_20d` | Relative volume compared to its 20-day moving average. Zero-volume sessions safely handled. |
| **Macro Market Context** | `market_return_1d`, `market_return_5d`, `market_volatility_20d`, `excess_return_1d` | Lagged returns and rolling volatility of the `^GSPTSE` benchmark index. |

* **Target Definition**: $y_t = \mathbf{1}(P_{t+1} > P_t) \in \{0, 1\}$. Tomorrow's return is shifted back onto row $t$ using `shift(-1)` to align features with supervision labels.
* **NaN Dropping Rule**: 49 warm-up rows (required for the 50-day moving average) and 1 terminal row (where tomorrow is unknown) were cleanly dropped, leaving **2,460 rows** with zero NaNs.
* **Empirical Perturbation Test**: Altering day $t+1$ price by $+50\%$ produced exactly **$0.000000$** change across all day $t$ features.

---

### Validation & Modeling Strategy ([`src/models.py`](src/models.py))

```text
Full Dataset (2,460 Trading Days)
├── Training & Tuning Window (First 80% = 1,968 days: 2016-12-23 to 2024-10-28)
│   ├── Fold 1: Train 328 days  -> Validate 328 days
│   ├── Fold 2: Train 656 days  -> Validate 328 days
│   ├── Fold 3: Train 984 days  -> Validate 328 days
│   ├── Fold 4: Train 1,312 days -> Validate 328 days
│   └── Fold 5: Train 1,640 days -> Validate 328 days
└── UNTOUCHED Holdout Test Set (Recent 20% = 492 days: 2024-10-29 to 2026-10-15)
    └── Evaluated ONCE. Never touched during tuning.
```

1. **No Shuffling**: Financial time series must never be randomly split.
2. **Leakage-Free Preprocessing**: `StandardScaler` is wrapped in an `sklearn.pipeline.Pipeline` and fit strictly on training folds.
3. **Regularized Models**:
   * **Logistic Regression**: $L_2$ regularization ($C \in [0.01, 10.0]$).
   * **Random Forest**: Shallow ensembles (`max_depth`: 3, 5; `n_estimators`: 50, 100).
   * **XGBoost**: Shallow gradient boosted trees (`max_depth`: 2, 3; `learning_rate`: 0.03, 0.1).
4. **Naive Baselines**:
   * **Always Up Baseline**: Predicts 1 every day (Buy-and-Hold / Majority Class proxy).
   * **Tomorrow = Today Baseline**: Predicts tomorrow's direction will match today's direction (1-Day Momentum).

---

## 📈 4. Key Results

### Classification Metrics (492-Day Holdout Test Set)

Evaluated on unseen test data from October 2024 to October 2026:

| Ticker | Model | Test Accuracy | **Always Up (Hurdle)** | Tomorrow=Today | **Beat Always Up?** | Precision | Recall | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RY.TO`** | **Always Up** | **56.50%** | **56.50%** | 51.02% | — | 56.50% | 100.0% | 0.500 | 0.4350 |
| | Logistic Regression | 52.44% | 56.50% | 51.02% | ❌ NO | 55.07% | 85.97% | 0.489 | 0.2485 |
| | Random Forest | 52.85% | 56.50% | 51.02% | ❌ NO | 55.30% | 86.33% | 0.502 | 0.2478 |
| | XGBoost | 53.86% | 56.50% | 51.02% | ❌ NO | 56.12% | 84.17% | 0.524 | 0.2469 |
| **`TD.TO`** | **Always Up** | **59.35%** | **59.35%** | 48.78% | — | 59.35% | 100.0% | 0.500 | 0.4065 |
| | Logistic Regression | 52.64% | 59.35% | 48.78% | ❌ NO | 57.78% | 75.00% | 0.500 | 0.2475 |
| | Random Forest | 53.46% | 59.35% | 48.78% | ❌ NO | 58.36% | 75.34% | 0.480 | 0.2482 |
| | XGBoost | 55.08% | 59.35% | 48.78% | ❌ NO | 60.17% | 71.92% | 0.510 | 0.2477 |
| **`SHOP.TO`**| Always Up | 51.63% | 51.63% | 50.41% | — | 51.63% | 100.0% | 0.500 | 0.4837 |
| | **Logistic Regression**| **52.03%**| 51.63% | 50.41% | **✅ YES** | 52.13% | 86.61% | 0.515 | 0.2498 |
| | Random Forest | 50.61% | 51.63% | 50.41% | ❌ NO | 51.48% | 75.20% | 0.521 | 0.2504 |
| | XGBoost | 50.61% | 51.63% | 50.41% | ❌ NO | 51.58% | 70.87% | 0.530 | 0.2503 |
| **`ENB.TO`** | Always Up | 55.28% | 55.28% | 54.88% | — | 55.28% | 100.0% | 0.500 | 0.4472 |
| | Logistic Regression | 55.08% | 55.28% | 54.88% | ❌ NO | 55.76% | 90.81% | 0.483 | 0.2496 |
| | **Random Forest** | **55.89%** | 55.28% | 54.88% | **✅ YES** | 55.81% | 97.06% | 0.497 | 0.2480 |
| | XGBoost | 50.00% | 55.28% | 54.88% | ❌ NO | 53.61% | 70.96% | 0.493 | 0.2531 |
| **`CNQ.TO`** | Always Up | 54.07% | 54.07% | 51.42% | — | 54.07% | 100.0% | 0.500 | 0.4593 |
| | **Logistic Regression**| **55.08%**| 54.07% | 51.42% | **✅ YES** | 56.52% | 73.31% | 0.551 | 0.2483 |
| | Random Forest | 52.24% | 54.07% | 51.42% | ❌ NO | 53.79% | 82.71% | 0.516 | 0.2486 |
| | XGBoost | 53.86% | 54.07% | 51.42% | ❌ NO | 54.86% | 82.71% | 0.535 | 0.2477 |
| **`^GSPTSE`**| Always Up | 59.96% | 59.96% | 51.42% | — | 59.96% | 100.0% | 0.500 | 0.4004 |
| | **Logistic Regression**| **60.57%**| 59.96% | 51.42% | **✅ YES** | 60.54% | 98.31% | 0.516 | 0.2419 |
| | Random Forest | 58.13% | 59.96% | 51.42% | ❌ NO | 60.28% | 88.47% | 0.504 | 0.2433 |
| | XGBoost | 59.76% | 59.96% | 51.42% | ❌ NO | 60.57% | 94.24% | 0.514 | 0.2425 |

---

### Economic Backtest with 0.1% Transaction Costs ([`src/backtest.py`](src/backtest.py))

* **Strategy Rule**: Long when model predicts Up ($\hat{y}_t = 1$), Hold Cash ($0\%$ return) when model predicts Down ($\hat{y}_t = 0$).
* **Friction Fee**: 0.10% (10 bps) deducted per position flip.
* **Starting Capital**: $10,000.

| Ticker | Model | Strategy Return | Buy & Hold Return | Excess Alpha | Strategy Sharpe | BnH Sharpe | Strategy Max DD | BnH Max DD | Total Trades | Fee Drag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RY.TO`** | XGBoost | +36.33% | +71.70% | -35.37% | 1.05 | **1.87** | **-10.16%** | -14.03% | 131 | 13.10% |
| **`TD.TO`** | XGBoost | +75.52% | +137.16% | -61.65% | 2.12 | **3.16** | -10.23% | -9.14% | 133 | 13.30% |
| **`SHOP.TO`**| **Random Forest** | **+130.87%**| +115.25% | **+15.62%** | **0.95** | 0.76 | **-34.30%** | -47.66% | 108 | 10.80% |
| | **XGBoost** | **+132.73%**| +115.25% | **+17.48%** | **0.98** | 0.76 | **-31.37%** | -47.66% | 116 | 11.60% |
| **`ENB.TO`** | **Random Forest** | **+41.68%** | +34.81% | **+6.87%** | **1.07** | 0.87 | -17.60% | -17.30% | 25 | 2.50% |
| **`CNQ.TO`** | **Logistic Reg.**| **+77.28%** | +75.80% | **+1.47%** | **1.24** | 1.05 | **-22.43%** | -24.46% | 133 | 13.30% |
| **`^GSPTSE`**| **Logistic Reg.**| **+52.34%** | +47.46% | **+4.88%** | **1.62** | 1.45 | **-11.70%** | -12.79% | 27 | 2.70% |

---

### Key Visualizations

1. **ROC Curves ([`reports/figures/roc_curves_all.png`](reports/figures/roc_curves_all.png))**:
   Shows probability discrimination across all 6 assets clustering near AUC $\approx 0.50 - 0.55$.
2. **Out-of-Sample Equity Curves ([`reports/figures/equity_curves_all.png`](reports/figures/equity_curves_all.png))**:
   Illustrates capital growth trajectories with transaction costs vs. passive Buy & Hold.
3. **Drawdown Comparison Charts ([`reports/figures/SHOP.TO_backtest_detail.png`](reports/figures/SHOP.TO_backtest_detail.png))**:
   Demonstrates how switching to cash on Shopify during tech pullbacks cut the maximum drawdown by **16.3 percentage points**.

---

## ⚠️ 5. Critical Limitations & Domain Reality Check

### 1. Overfitting Risk & Noise Dominance
* Daily stock returns have a signal-to-noise ratio where predictable components rarely exceed $R^2 < 1\text{--}2\%$.
* In our regime evaluation ([`notebooks/02_feature_importance_and_regimes.ipynb`](notebooks/02_feature_importance_and_regimes.ipynb)), tree models achieved **62% to 65% accuracy** on training data, but compressed back to **50% to 55%** on the holdout test set.
* *Mitigation*: Strictly shallow trees (`max_depth`: 2–5) and walk-forward cross-validation were necessary to prevent full memorization.

### 2. Market Efficiency & Frictional Drag
* Under the Efficient Market Hypothesis (EMH), public historical OHLCV indicators are swiftly priced in by institutional algorithmic trading and market makers.
* Strategies executing 80 to 133 trades burned **8.0% to 13.3% of portfolio equity** solely in transaction fees at 0.10% per trade. In trending bull markets, whipsaw turnover destroyed paper profits.

### 3. Survivorship Bias
* Our universe (`RY.TO`, `TD.TO`, `SHOP.TO`, `ENB.TO`, `CNQ.TO`) was chosen in **2026** based on their current status as TSX mega-caps.
* Canadian companies that went bankrupt, merged under distress, or were delisted over the last 10 years are absent. This survivor conditioning artificially inflates both the 10-year Buy-and-Hold return and the naive upward drift.

---

## 🚀 6. Installation & How to Run

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/tsx-predictor.git
cd tsx-predictor

# Create and activate Python 3.11 virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
# 1. Download 10 years of raw OHLCV CSVs
python -m src.data

# 2. Engineer features with zero look-ahead bias
python -m src.features

# 3. Train models with TimeSeriesSplit walk-forward CV
python -m src.models

# 4. Generate classification evaluation & ROC charts
python -m src.evaluate

# 5. Run economic backtest & equity curves
python -m src.backtest
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with the stock selector, view live confusion matrices, and test custom transaction cost assumptions.

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```
Executes 39 automated unit tests verifying data integrity, zero look-ahead bias, model inference, friction mathematics, and web app functionality.

### 5. Explore Jupyter Notebooks
```bash
jupyter notebook notebooks/
```
* **`01_eda.ipynb`**: Price history, returns distribution, kurtosis, rolling volatility, and correlation heatmap.
* **`02_feature_importance_and_regimes.ipynb`**: Gini/Gain feature importance across models, 2020 crash & 2022 rate hike regime analysis, and in-depth limitations study.

---

## 📂 Project Structure
```text
tsx-predictor/
├── data/
│   ├── raw/                  # 10 years of OHLCV CSVs downloaded via yfinance
│   ├── processed/            # Engineered feature matrices with labels & target returns
│   └── models/               # Serialized .joblib models (LR, RF, XGBoost)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis & baseline establishment
│   └── 02_feature_importance_and_regimes.ipynb # Feature ranking & macro regimes
├── reports/
│   └── figures/              # Exported ROC curves, confusion matrices & equity charts
├── src/
│   ├── __init__.py
│   ├── data.py               # Ingestion, validation & timezone cleaning
│   ├── features.py           # Technical indicators & zero look-ahead target
│   ├── baselines.py          # Always-Up & Tomorrow=Today naive estimators
│   ├── models.py             # TimeSeriesSplit walk-forward training
│   ├── evaluate.py           # Classification metrics & ROC visualizer
│   └── backtest.py           # Long/Cash backtesting engine with friction
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # 39 end-to-end unit tests (pytest)
├── app.py                    # Multi-asset interactive Streamlit dashboard
├── requirements.txt          # Reproducible project dependencies
└── README.md                 # Complete project documentation & empirical findings
```

---

## 👨‍💻 Author & Portfolio Note
*Built for educational and portfolio demonstration purposes. Models are designed to showcase rigorous time-series evaluation hygiene, statistical baselining, and transaction cost modeling.*
