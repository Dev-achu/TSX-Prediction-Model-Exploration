"""Baseline models and time-series benchmark evaluation for TSX Stock Predictor.

Why are baselines essential in financial machine learning?
---------------------------------------------------------
In standard machine learning (e.g. image classification or spam detection),
a model with 55% accuracy is often deemed a failure, while 95% is great.
In financial direction prediction:
1. Markets have a natural upward drift (equities rise over time). Therefore,
   simply guessing "Up" every single day often achieves 52% to 59% accuracy!
2. A machine learning model that achieves 54% accuracy might seem like an edge
   over a 50% random coin flip, but if "Always Up" achieved 58%, the ML model
   actually destroyed value.
3. Therefore, before touching XGBoost or Neural Networks, we MUST establish
   clear naive benchmarks on the exact same chronological test split.

Baselines Implemented:
---------------------
1. AlwaysUpBaseline:
   - Always predicts direction = 1 ("Up").
   - Represents the naive Buy-and-Hold / Majority Class strategy.
2. TomorrowEqualsTodayBaseline:
   - Predicts tomorrow's direction will equal today's direction.
   - Represents the 1-day sign momentum (trend persistence) strategy.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from src.data import ALL_DEFAULT_TICKERS, DEFAULT_RAW_DATA_DIR, load_raw_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Preparation & Look-Ahead-Free Target Definition
# ---------------------------------------------------------------------------

def prepare_directional_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares daily returns, today's direction, and next-day target direction.

    Timing and Zero Look-Ahead Bias Rules:
    -------------------------------------
    For any trading day t:
    - Return_t = (Adj_Close_t - Adj_Close_{t-1}) / Adj_Close_{t-1}
      This is today's return, available strictly at or after market close on day t.
    - Today_Direction_t = 1 if Return_t > 0 else 0
      This is known at close of day t.
    - Target_t = 1 if Return_{t+1} > 0 else 0
      This is tomorrow's price direction (the future event we want to predict).

    Notice:
    - Day t=0 has no prior day, so Return_0 is NaN and is dropped.
    - The final row in the dataset has no future day (t+1 does not exist yet),
      so Target_{last} is NaN and is dropped.

    Args:
        df: Raw OHLCV DataFrame with DatetimeIndex and 'Adj Close'.

    Returns:
        pd.DataFrame containing 'Adj Close', 'Return', 'Today_Direction',
        and 'Target', indexed by Date.
    """
    clean_df = df.copy()

    # Ensure chronological order (oldest to newest)
    clean_df = clean_df.sort_index(ascending=True)

    # 1. Today's return: percentage change from yesterday's close to today's close
    clean_df["Return"] = clean_df["Adj Close"].pct_change()

    # 2. Today's direction: 1 if today closed higher than yesterday, 0 otherwise
    clean_df["Today_Direction"] = (clean_df["Return"] > 0).astype(int)

    # 3. Target: Tomorrow's direction.
    # shift(-1) shifts the future return back into today's row so it serves as the label.
    clean_df["Target"] = (clean_df["Return"].shift(-1) > 0).astype(float)

    # Drop the first row (NaN return) and last row (unknown tomorrow target)
    clean_df = clean_df.dropna(subset=["Return", "Target"])
    clean_df["Target"] = clean_df["Target"].astype(int)

    return clean_df


def time_based_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time series data chronologically into train and test sets.

    CRITICAL RULE:
    --------------
    NEVER shuffle financial time series data! Random splitting leaks future
    information into the training set (look-ahead bias) and ruins evaluation.
    Here:
    - Train set = First (1 - test_ratio) portion (e.g. first 80%)
    - Test set  = Most recent test_ratio portion (e.g. final 20%)

    Args:
        df: Prepared DataFrame with DatetimeIndex.
        test_ratio: Fraction of most recent data to reserve for testing (default 0.20).

    Returns:
        Tuple of (train_df, test_df).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    n_samples = len(df)
    split_idx = int(n_samples * (1.0 - test_ratio))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Verify chronological integrity: train max date must strictly precede test min date
    assert train_df.index.max() < test_df.index.min(), (
        f"Data leakage detected! Train max date ({train_df.index.max()}) "
        f"is not strictly before test min date ({test_df.index.min()})"
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# Scikit-Learn Compatible Baseline Estimators
# ---------------------------------------------------------------------------

class AlwaysUpBaseline(BaseEstimator, ClassifierMixin):
    """Naive Baseline 1: Always predicts 'Up' (class 1).

    Concept:
    - Equities have a positive expected return over long horizons.
    - This baseline models a passive 'Buy and Hold' direction guesser.
    - Represents the Majority Class accuracy hurdle.
    """

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> AlwaysUpBaseline:
        """No parameters to estimate; returns self."""
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts 1 for all samples."""
        n_samples = len(X)
        return np.ones(n_samples, dtype=int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns 100% probability for class 1."""
        n_samples = len(X)
        proba = np.zeros((n_samples, 2), dtype=float)
        proba[:, 1] = 1.0
        return proba


class TomorrowEqualsTodayBaseline(BaseEstimator, ClassifierMixin):
    """Naive Baseline 2: Predicts tomorrow's direction equals today's direction.

    Concept:
    - Represents 1-day sign momentum / trend persistence.
    - If today was an Up day (Return > 0), predict tomorrow will be Up.
    - If today was a Down day (Return <= 0), predict tomorrow will be Down.
    - Tests the Random Walk hypothesis (whether consecutive days exhibit momentum or reversal).
    """

    def __init__(self, direction_col: str = "Today_Direction") -> None:
        self.direction_col = direction_col
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
    ) -> TomorrowEqualsTodayBaseline:
        """No parameters to estimate; returns self."""
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts today's known direction as tomorrow's predicted direction."""
        if isinstance(X, pd.DataFrame):
            if self.direction_col in X.columns:
                return X[self.direction_col].values.astype(int)
            elif "Return" in X.columns:
                return (X["Return"] > 0).astype(int).values
            elif "return_1d" in X.columns:
                return (X["return_1d"] > 0).astype(int).values
            else:
                # If direction column isn't found, assume first column is today's return or direction
                first_col = X.iloc[:, 0].values
                return (first_col > 0).astype(int)
        elif isinstance(X, np.ndarray):
            # Assume first feature column is today's return/direction
            first_col = X[:, 0] if X.ndim > 1 else X
            return (first_col > 0).astype(int)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns deterministic probabilities based on today's direction."""
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2), dtype=float)
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_baselines_for_ticker(
    ticker: str,
    test_ratio: float = 0.20,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Dict[str, any]:
    """Loads raw data for a ticker, splits chronologically, and evaluates both baselines.

    Args:
        ticker: Symbol name (e.g. 'RY.TO' or '^GSPTSE').
        test_ratio: Fraction of most recent data for test set (default 0.20).
        raw_dir: Path to directory containing raw CSVs.

    Returns:
        Dictionary containing ticker evaluation metrics on the test set.
    """
    raw_df = load_raw_data(ticker=ticker, raw_dir=raw_dir)
    prepared_df = prepare_directional_dataset(raw_df)

    train_df, test_df = time_based_train_test_split(prepared_df, test_ratio=test_ratio)

    y_train = train_df["Target"]
    y_test = test_df["Target"]

    # Model 1: Always Up
    always_up = AlwaysUpBaseline()
    always_up_pred_train = always_up.predict(train_df)
    always_up_pred_test = always_up.predict(test_df)
    always_up_train_acc = accuracy_score(y_train, always_up_pred_train)
    always_up_test_acc = accuracy_score(y_test, always_up_pred_test)

    # Model 2: Tomorrow Equals Today
    momentum = TomorrowEqualsTodayBaseline()
    mom_pred_train = momentum.predict(train_df)
    mom_pred_test = momentum.predict(test_df)
    mom_train_acc = accuracy_score(y_train, mom_pred_train)
    mom_test_acc = accuracy_score(y_test, mom_pred_test)

    test_up_count = int(y_test.sum())
    test_down_count = int(len(y_test) - test_up_count)

    return {
        "Ticker": ticker,
        "Total Days": len(prepared_df),
        "Train Days": len(train_df),
        "Test Days": len(test_df),
        "Test Start": test_df.index.min().strftime("%Y-%m-%d"),
        "Test End": test_df.index.max().strftime("%Y-%m-%d"),
        "Test Up Days": test_up_count,
        "Test Down Days": test_down_count,
        "Baseline 1: Always Up (Test Acc)": always_up_test_acc,
        "Baseline 2: Today Direction (Test Acc)": mom_test_acc,
        "Baseline 1 (Train Acc)": always_up_train_acc,
        "Baseline 2 (Train Acc)": mom_train_acc,
    }


def evaluate_baselines_all(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    test_ratio: float = 0.20,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> pd.DataFrame:
    """Evaluates both naive baselines across all specified tickers.

    Args:
        tickers: List of ticker strings.
        test_ratio: Fraction of most recent data for test set (default 0.20).
        raw_dir: Path to directory containing raw CSVs.

    Returns:
        Formatted pandas DataFrame summarizing baseline performance.
    """
    records = []
    for ticker in tickers:
        metrics = evaluate_baselines_for_ticker(
            ticker=ticker,
            test_ratio=test_ratio,
            raw_dir=raw_dir,
        )
        records.append(metrics)

    df_results = pd.DataFrame(records)
    return df_results


def format_results_table(df_results: pd.DataFrame) -> pd.DataFrame:
    """Formats baseline evaluation DataFrame into a presentation-ready table."""
    display_df = df_results.copy()
    display_df["Always Up (Test Acc)"] = (
        display_df["Baseline 1: Always Up (Test Acc)"] * 100
    ).map("{:.2f}%".format)
    display_df["Tomorrow=Today (Test Acc)"] = (
        display_df["Baseline 2: Today Direction (Test Acc)"] * 100
    ).map("{:.2f}%".format)
    display_df["Always Up (Train Acc)"] = (
        display_df["Baseline 1 (Train Acc)"] * 100
    ).map("{:.2f}%".format)
    display_df["Tomorrow=Today (Train Acc)"] = (
        display_df["Baseline 2 (Train Acc)"] * 100
    ).map("{:.2f}%".format)

    columns_to_show = [
        "Ticker",
        "Test Days",
        "Test Start",
        "Test End",
        "Always Up (Test Acc)",
        "Tomorrow=Today (Test Acc)",
        "Always Up (Train Acc)",
        "Tomorrow=Today (Train Acc)",
    ]
    return display_df[columns_to_show]


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TSX NAIVE BASELINES EVALUATION (CHRONOLOGICAL 80/20 SPLIT)")
    print("=" * 80)
    print("Testing the most recent 20% of data (approx. past 2 years of trading).")
    print("Baselines tested:")
    print("  1. Always Up (Majority Class / Buy-and-Hold)")
    print("  2. Tomorrow Equals Today (1-Day Momentum / Persistence)")
    print("-" * 80)

    raw_results = evaluate_baselines_all()
    formatted = format_results_table(raw_results)
    print(formatted.to_string(index=False))
    print("=" * 80 + "\n")
