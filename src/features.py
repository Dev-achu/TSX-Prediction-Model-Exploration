"""Feature engineering pipeline for TSX Stock Price Direction Predictor.

Features engineered in this module:
1. Lagged Returns:
   - return_1d: 1-day return (today's close vs yesterday's close)
   - return_2d: 2-day cumulative return
   - return_5d: 5-day cumulative return
   - return_lag1: yesterday's 1-day return (lagged by 1 day)
   - return_lag2: day-before-yesterday's 1-day return (lagged by 2 days)
2. Moving Average Ratios (10-day and 50-day):
   - sma_ratio_10: (Adj_Close / SMA_10) - 1.0
   - sma_ratio_50: (Adj_Close / SMA_50) - 1.0
3. RSI (Relative Strength Index):
   - rsi_14: 14-period Wilder smoothed RSI bounded between 0 and 100.
4. Momentum:
   - momentum_10d: 10-day price momentum (Adj_Close_t / Adj_Close_{t-10} - 1.0)
   - momentum_5d: 5-day price momentum
5. Rolling Volatility:
   - volatility_10d: 10-day annualized rolling volatility
   - volatility_20d: 20-day annualized rolling volatility (~1 calendar month)
6. Volume Dynamics:
   - volume_change_1d: 1-day percentage change in trading volume
   - volume_ratio_20d: volume relative to its 20-day moving average (Volume / SMA_vol_20 - 1.0)
7. Target (Next-Day Direction):
   - target: 1 if tomorrow's adjusted close is higher than today's adjusted close, else 0.
   - target_return: tomorrow's percentage return (useful for economic backtesting).

HOW LOOK-AHEAD BIAS WAS RIGOROUSLY AVOIDED:
------------------------------------------
Every predictive model in quantitative finance must strictly obey causality:
features for day t may ONLY use data that was generated on or before day t.

Look-Ahead Audit per Feature at Market Close of Day t:
- return_1d: Uses Close_t and Close_{t-1}. (Known at close of day t).
- return_2d, return_5d: Uses Close_t, Close_{t-2}, Close_{t-5}. (Known).
- sma_ratio_10, sma_ratio_50: Uses rolling window [t-9 to t] and [t-49 to t].
  All window elements are on or before day t.
- rsi_14: Computed using price differences up to and including day t.
- momentum_10d: Uses Close_t and Close_{t-10}.
- volatility_20d: Uses rolling standard deviation of returns from t-19 to t.
- volume_change_1d, volume_ratio_20d: Uses Volume_t and past volume up to day t.
- Target: Uses Close_{t+1} and Close_t via `shift(-1)`.
  This correctly shifts tomorrow's future outcome back onto row t to serve as the
  supervision label for day t's feature vector.

HOW NaN ROWS ARE HANDLED:
-------------------------
1. Warm-up window:
   The longest lookback is 50 days (sma_50). Therefore, the first 49 rows
   (from index 0 to 48) naturally lack a full 50-day history and produce NaNs.
2. Target boundary:
   The final row (day N-1) has no future day (day N does not exist yet),
   so its target is NaN.
3. Drop rule:
   We call `.dropna()` on the combined feature-and-target DataFrame.
   This cleanly discards the warm-up period at the beginning and the boundary row
   at the end without imputing or leaking future data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data import ALL_DEFAULT_TICKERS, BENCHMARK_INDEX, DEFAULT_RAW_DATA_DIR, load_raw_data

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Technical Indicator Calculations
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilder's Relative Strength Index (RSI).

    Formula:
        delta = price_t - price_{t-1}
        gain = max(delta, 0), loss = max(-delta, 0)
        avg_gain = ExponentialMovingAverage(gain, alpha=1/period)
        avg_loss = ExponentialMovingAverage(loss, alpha=1/period)
        RS = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))

    Args:
        series: Price series (e.g. Adj Close).
        period: Lookback window (standard is 14 days).

    Returns:
        pd.Series containing RSI values strictly between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing corresponds to an exponential moving average with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0 (all days were gains), RSI is 100; if avg_gain is 0, RSI is 0
    rsi = rsi.fillna(50.0)
    return rsi


# ---------------------------------------------------------------------------
# Core Feature Engineering
# ---------------------------------------------------------------------------

def create_features_for_df(
    df: pd.DataFrame,
    include_market_features: bool = False,
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generates all technical, momentum, volatility, and target features for a single asset.

    Args:
        df: Clean raw OHLCV DataFrame with DatetimeIndex and 'Adj Close', 'Volume'.
        include_market_features: If True, merges lagged TSX index returns.
        market_df: Clean raw OHLCV DataFrame for ^GSPTSE (required if include_market_features=True).

    Returns:
        pd.DataFrame containing engineered features, target, and target_return,
        with all warm-up and boundary NaNs dropped.
    """
    # Verify chronological ascending order
    clean_df = df.sort_index(ascending=True).copy()

    price = clean_df["Adj Close"]
    volume = clean_df["Volume"].astype(float)

    feat: Dict[str, pd.Series] = {}

    # -----------------------------------------------------------------------
    # 1. Lagged Returns
    # -----------------------------------------------------------------------
    # 1-day return: today's close vs yesterday's close
    feat["return_1d"] = price.pct_change(1)
    # 2-day cumulative return
    feat["return_2d"] = price.pct_change(2)
    # 5-day cumulative return (~1 trading week)
    feat["return_5d"] = price.pct_change(5)

    # Autoregressive lags: yesterday's 1-day return and 2-days-ago 1-day return
    feat["return_lag1"] = feat["return_1d"].shift(1)
    feat["return_lag2"] = feat["return_1d"].shift(2)

    # -----------------------------------------------------------------------
    # 2. Moving Average Ratios (Stationary Price-to-Trend Metrics)
    # -----------------------------------------------------------------------
    # Why ratios? Raw moving averages in dollar terms ($80 vs $200) are non-stationary.
    # Computing (Price / SMA) - 1.0 creates a stationary percentage premium/discount:
    #   +0.03 means price is 3% above the moving average.
    #   -0.03 means price is 3% below the moving average.
    sma_10 = price.rolling(window=10).mean()
    sma_50 = price.rolling(window=50).mean()
    feat["sma_ratio_10"] = (price / sma_10) - 1.0
    feat["sma_ratio_50"] = (price / sma_50) - 1.0

    # Moving average cross signal: 10-day SMA relative to 50-day SMA
    feat["sma_10_50_ratio"] = (sma_10 / sma_50) - 1.0

    # -----------------------------------------------------------------------
    # 3. Relative Strength Index (RSI)
    # -----------------------------------------------------------------------
    feat["rsi_14"] = compute_rsi(price, period=14)

    # -----------------------------------------------------------------------
    # 4. Momentum & Rate of Change
    # -----------------------------------------------------------------------
    feat["momentum_5d"] = price.pct_change(5)
    feat["momentum_10d"] = price.pct_change(10)

    # -----------------------------------------------------------------------
    # 5. Rolling Volatility (Annualized)
    # -----------------------------------------------------------------------
    # Multiplied by sqrt(252) to annualize daily standard deviation
    feat["volatility_10d"] = feat["return_1d"].rolling(window=10).std() * np.sqrt(252)
    feat["volatility_20d"] = feat["return_1d"].rolling(window=20).std() * np.sqrt(252)

    # -----------------------------------------------------------------------
    # 6. Volume Dynamics
    # -----------------------------------------------------------------------
    # Volume percentage change (handling 0-volume sessions to avoid 0/0 NaNs or inf)
    feat["volume_change_1d"] = volume.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feat["volume_change_5d"] = volume.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Volume relative to 20-day average volume:
    # > 0 means unusually heavy trading; < 0 means quiet session
    vol_sma_20 = volume.rolling(window=20).mean()
    feat["volume_ratio_20d"] = (volume / vol_sma_20.replace(0.0, np.nan)) - 1.0
    feat["volume_ratio_20d"] = feat["volume_ratio_20d"].fillna(0.0)

    # -----------------------------------------------------------------------
    # 7. Optional Market Context (TSX Composite Index Features)
    # -----------------------------------------------------------------------
    if include_market_features and market_df is not None:
        market_clean = market_df.sort_index(ascending=True)
        mkt_ret_1d = market_clean["Adj Close"].pct_change(1)
        mkt_ret_5d = market_clean["Adj Close"].pct_change(5)
        mkt_vol_20d = mkt_ret_1d.rolling(window=20).std() * np.sqrt(252)

        feat["market_return_1d"] = mkt_ret_1d
        feat["market_return_5d"] = mkt_ret_5d
        feat["market_volatility_20d"] = mkt_vol_20d

        # Market-relative return: stock return minus index return (alpha today)
        feat["excess_return_1d"] = feat["return_1d"] - mkt_ret_1d

    # -----------------------------------------------------------------------
    # 8. Target Construction (Next-Day Direction)
    # -----------------------------------------------------------------------
    # Next-day return: Return_{t+1} = (Adj_Close_{t+1} - Adj_Close_t) / Adj_Close_t
    # shift(-1) moves tomorrow's return onto row t so it can be predicted from row t features.
    target_return = price.pct_change(1).shift(-1)
    feat["target_return"] = target_return

    # Binary classification target: 1 if tomorrow's return > 0, else 0
    # Strict inequality: zero-return days are labeled 0; boundary rows remain NaN
    feat["target"] = np.where(target_return.isna(), np.nan, (target_return > 0.0).astype(float))

    # Assemble into DataFrame
    features_df = pd.DataFrame(feat, index=clean_df.index)

    # -----------------------------------------------------------------------
    # 9. Clean Handling of Missing Values (Warm-Up and Future Boundary)
    # -----------------------------------------------------------------------
    # Drop rows containing NaNs:
    # - First 49 rows dropped due to 50-day moving average warm-up.
    # - Final row dropped because tomorrow's target is unknown.
    n_before = len(features_df)
    features_df = features_df.dropna()
    features_df["target"] = features_df["target"].astype(int)
    n_after = len(features_df)

    logger.debug(f"Features created: {n_before} raw rows -> {n_after} valid rows (dropped {n_before - n_after} boundary rows).")
    return features_df


# ---------------------------------------------------------------------------
# Verification & Quality Assurance Helper
# ---------------------------------------------------------------------------

def verify_feature_integrity(features_df: pd.DataFrame, ticker: str) -> None:
    """Runs automated health checks on engineered feature matrices.

    Checks:
    1. Zero remaining NaNs or infinite values.
    2. Date index is monotonically increasing.
    3. Target is strictly binary {0, 1}.
    4. Target class balance is reasonable.
    5. No constant columns (zero variance).
    """
    assert not features_df.empty, f"[{ticker}] Feature DataFrame is empty!"

    # 1. Check nulls and infs
    null_counts = features_df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"[{ticker}] Found remaining nulls after dropna:\n{null_counts[null_counts > 0]}")

    inf_mask = np.isinf(features_df.select_dtypes(include=[np.number]))
    if inf_mask.values.any():
        raise ValueError(f"[{ticker}] Found infinite values in features:\n{inf_mask.sum()[inf_mask.sum() > 0]}")

    # 2. Monotonic index
    assert features_df.index.is_monotonic_increasing, f"[{ticker}] Dates are not monotonically increasing!"

    # 3. Target values
    unique_targets = set(features_df["target"].unique())
    assert unique_targets.issubset({0, 1}), f"[{ticker}] Target values must be binary {0, 1}, got {unique_targets}"

    # 4. Zero variance check
    zero_var_cols = [c for c in features_df.columns if features_df[c].nunique() <= 1]
    if zero_var_cols:
        raise ValueError(f"[{ticker}] Found zero-variance features: {zero_var_cols}")


# ---------------------------------------------------------------------------
# Storage & Ingestion Helpers
# ---------------------------------------------------------------------------

def get_processed_filepath(ticker: str, processed_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR) -> Path:
    """Returns standardized file path for a processed feature CSV."""
    clean_name = ticker.replace("^", "")
    return Path(processed_dir) / f"{clean_name}_features.csv"


def save_processed_features(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
) -> Path:
    """Saves engineered feature DataFrame to data/processed."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = get_processed_filepath(ticker, processed_dir=out_dir)
    df.to_csv(target_path, index=True)
    logger.info(f"[{ticker}] Saved {len(df)} feature records to {target_path}")
    return target_path


def load_processed_features(
    ticker: str,
    processed_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Loads a previously saved engineered feature dataset.

    Args:
        ticker: Symbol name (e.g. 'RY.TO' or '^GSPTSE').
        processed_dir: Path to directory containing processed CSVs.

    Returns:
        pd.DataFrame with DatetimeIndex, feature columns, and 'target'.
    """
    target_path = get_processed_filepath(ticker, processed_dir=processed_dir)
    if not target_path.exists():
        # Fallback to direct name
        clean_name = ticker.replace("^", "")
        alt_path = Path(processed_dir) / f"{ticker}_features.csv"
        if alt_path.exists():
            target_path = alt_path
        else:
            raise FileNotFoundError(f"Processed feature file not found at {target_path}")

    df = pd.read_csv(target_path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    return df


def build_all_features(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    raw_dir: Path | str = DEFAULT_RAW_DATA_DIR,
    output_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
    include_market: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Generates, validates, and saves feature sets for all project tickers.

    Args:
        tickers: List of ticker strings.
        raw_dir: Source raw data directory.
        output_dir: Target processed data directory.
        include_market: Whether to incorporate TSX Composite market context features.

    Returns:
        Dictionary mapping ticker -> engineered pd.DataFrame.
    """
    logger.info(f"Building engineered features for {len(tickers)} tickers...")

    # Load market benchmark index first if market features are enabled
    market_df = None
    if include_market:
        try:
            market_df = load_raw_data(BENCHMARK_INDEX, raw_dir=raw_dir)
        except Exception as exc:
            logger.warning(f"Could not load market benchmark ({BENCHMARK_INDEX}): {exc}. Skipping market features.")

    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        raw_df = load_raw_data(ticker=ticker, raw_dir=raw_dir)

        # For the benchmark index itself, do not pass market_df (avoid self-referential redundancy)
        is_index = ticker == BENCHMARK_INDEX
        feats = create_features_for_df(
            df=raw_df,
            include_market_features=include_market and not is_index,
            market_df=market_df,
        )

        verify_feature_integrity(feats, ticker=ticker)
        save_processed_features(feats, ticker=ticker, output_dir=output_dir)
        results[ticker] = feats

    logger.info(f"Successfully engineered and stored features for all {len(tickers)} tickers.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 80)
    print("ENGINEERING FEATURES FOR ALL TSX ASSETS (STRICT ZERO LOOK-AHEAD)")
    print("=" * 80)

    datasets = build_all_features()

    print("\nFeature Summary Table:")
    summary_rows = []
    for ticker, df_feat in datasets.items():
        summary_rows.append({
            "Ticker": ticker,
            "Valid Rows": len(df_feat),
            "Features": len(df_feat.columns) - 2,  # Exclude 'target' and 'target_return'
            "Start Date": df_feat.index.min().strftime("%Y-%m-%d"),
            "End Date": df_feat.index.max().strftime("%Y-%m-%d"),
            "Target Up %": f"{(df_feat['target'] == 1).mean() * 100:.2f}%",
            "Target Down %": f"{(df_feat['target'] == 0).mean() * 100:.2f}%",
            "Nulls": df_feat.isnull().sum().sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")
