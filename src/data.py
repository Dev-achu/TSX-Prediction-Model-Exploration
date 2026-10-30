"""Data ingestion module for TSX Stock Price Direction Predictor.

This module is responsible for:
1. Downloading 10 years of daily OHLCV (Open, High, Low, Close, Volume) data
   for selected Toronto Stock Exchange (TSX) equities and the S&P/TSX Composite index (^GSPTSE).
2. Sanitizing and validating the raw data (handling missing values, timezones, and sorting).
3. Storing each ticker's clean raw data as a CSV in data/raw/.
4. Providing reusable loading utility functions for downstream feature engineering and models.

Why these design decisions?
- auto_adjust=False: We preserve both 'Close' and 'Adj Close'. Stock splits (e.g. Shopify in 2022)
  and dividends distort historical unadjusted prices. We need Adjusted Close for accurate return
  calculations, while raw Open/High/Low/Close reflect historical exchange quotes.
- Timezone stripping: yfinance returns timezone-aware timestamps (America/Toronto or UTC). Stripping
  to timezone-naive dates (YYYY-MM-DD) avoids timezone-mismatch bugs when joining stocks with the index.
- Chronological sorting: Time series data must NEVER be shuffled or inverted. We enforce ascending order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# Configure clean logging format for execution feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default asset universe requested for the project
DEFAULT_STOCKS: List[str] = ["RY.TO", "TD.TO", "SHOP.TO", "ENB.TO", "CNQ.TO"]
BENCHMARK_INDEX: str = "^GSPTSE"  # S&P/TSX Composite Index
ALL_DEFAULT_TICKERS: List[str] = DEFAULT_STOCKS + [BENCHMARK_INDEX]

# Project directory paths
DEFAULT_RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def ticker_to_filename(ticker: str) -> str:
    """Converts a ticker symbol to a safe filename.

    Example:
        '^GSPTSE' -> 'GSPTSE.csv'
        'RY.TO'   -> 'RY.TO.csv'

    Why? The leading '^' character in market indices can cause escaping issues
    in Unix shells and command-line scripts.
    """
    clean_name = ticker.replace("^", "")
    return f"{clean_name}.csv"


def download_ticker_data(
    ticker: str,
    period: str = "10y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Downloads daily historical market data for a single ticker via yfinance.

    Args:
        ticker: The Yahoo Finance ticker symbol (e.g. 'RY.TO' or '^GSPTSE').
        period: Valid period strings like '10y', '5y', 'max'. Default is '10y'.
        start: Optional start date 'YYYY-MM-DD'. If provided with end, period is ignored.
        end: Optional end date 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame containing historical market data indexed by Date.

    Raises:
        ValueError: If yfinance returns an empty DataFrame or download fails.
    """
    logger.info(f"Downloading data for ticker '{ticker}' (period={period})...")
    try:
        yf_ticker = yf.Ticker(ticker)
        if start and end:
            df = yf_ticker.history(start=start, end=end, auto_adjust=False)
        else:
            df = yf_ticker.history(period=period, auto_adjust=False)
    except Exception as exc:
        logger.error(f"Network or API exception downloading {ticker}: {exc}")
        raise

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Please verify the symbol.")

    return df


def clean_raw_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validates, cleans, and standardizes raw OHLCV DataFrame.

    Steps performed:
    1. Removes any multi-index columns if present.
    2. Strips timezone offsets so all dates are standardized to YYYY-MM-DD.
    3. Sorts index chronologically (ascending).
    4. Drops duplicate dates if any exist.
    5. Checks and handles missing values (forward fill for prices, zero for volume).
    6. Validates presence of essential financial columns.

    Args:
        df: Raw DataFrame directly from yfinance.
        ticker: Symbol name (for contextual logging).

    Returns:
        Cleaned, chronologically sorted DataFrame with DatetimeIndex named 'Date'.
    """
    df_clean = df.copy()

    # Flatten MultiIndex columns if yfinance returned them
    if isinstance(df_clean.columns, pd.MultiIndex):
        # Keep top-level price metric name (e.g. ('Close', 'RY.TO') -> 'Close')
        df_clean.columns = [col[0] for col in df_clean.columns]

    # Standardize index name
    df_clean.index.name = "Date"

    # Convert index to timezone-naive datetime (avoids tz alignment errors)
    if hasattr(df_clean.index, "tz") and df_clean.index.tz is not None:
        df_clean.index = df_clean.index.tz_localize(None)
    df_clean.index = pd.to_datetime(df_clean.index).normalize()

    # Sort strictly chronologically: Oldest -> Newest
    df_clean = df_clean.sort_index(ascending=True)

    # Check for duplicate dates
    duplicate_count = df_clean.index.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"[{ticker}] Found {duplicate_count} duplicate dates. Keeping first occurrence.")
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

    # Essential column check
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df_clean.columns]
    if missing_cols:
        raise KeyError(f"[{ticker}] Missing required market columns: {missing_cols}")

    # Inspect missing values
    null_counts = df_clean[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"[{ticker}] Missing values detected:\n{null_counts[null_counts > 0]}")
        # Financial time series handling:
        # Prices are forward-filled (today's missing price assumes previous day's value)
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df_clean.columns]
        df_clean[price_cols] = df_clean[price_cols].ffill().bfill()
        # Volume is filled with 0 if missing
        if "Volume" in df_clean.columns:
            df_clean["Volume"] = df_clean["Volume"].fillna(0)

    # Verify no invalid non-positive prices
    for p_col in ["Open", "High", "Low", "Close"]:
        invalid_mask = df_clean[p_col] <= 0
        if invalid_mask.any():
            logger.warning(f"[{ticker}] Detected {invalid_mask.sum()} non-positive values in {p_col}.")

    return df_clean


def save_raw_data(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """Saves cleaned raw DataFrame to CSV in the target directory.

    Args:
        df: Cleaned raw market DataFrame.
        ticker: Symbol name (used to build file name).
        output_dir: Target directory path.

    Returns:
        Path to the saved CSV file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = ticker_to_filename(ticker)
    target_file = out_path / filename

    df.to_csv(target_file, index=True)
    logger.info(f"[{ticker}] Saved {len(df)} records ({df.index.min().date()} to {df.index.max().date()}) to {target_file}")
    return target_file


def load_raw_data(
    ticker: str,
    raw_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> pd.DataFrame:
    """Loads a previously saved raw CSV file for a given ticker.

    Supports both '^GSPTSE' and 'GSPTSE' or 'RY.TO' syntax.

    Args:
        ticker: The stock or index symbol.
        raw_dir: Path to directory holding raw CSVs.

    Returns:
        pd.DataFrame with parsed DatetimeIndex and standard columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist in raw_dir.
    """
    raw_path = Path(raw_dir)
    clean_name = ticker.replace("^", "")

    # Try exact match, sanitized match, or with caret
    candidates = [
        raw_path / f"{clean_name}.csv",
        raw_path / f"{ticker}.csv",
        raw_path / f"^{clean_name}.csv",
    ]

    for cand in candidates:
        if cand.exists():
            df = pd.read_csv(cand, index_col="Date", parse_dates=True)
            # Ensure index is datetime sorted ascending
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            return df

    raise FileNotFoundError(
        f"Could not locate raw CSV for '{ticker}' in {raw_path}. Looked for: {[str(c) for c in candidates]}"
    )


def download_all(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    period: str = "10y",
    output_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> Dict[str, pd.DataFrame]:
    """Batch downloads, cleans, and saves data for all requested tickers.

    Args:
        tickers: List of ticker strings to download.
        period: Period string (default '10y').
        output_dir: Destination folder for CSV files.

    Returns:
        Dictionary mapping ticker -> cleaned pd.DataFrame.
    """
    logger.info(f"Starting batch download for {len(tickers)} symbols into {output_dir}...")
    results: Dict[str, pd.DataFrame] = {}
    failed_tickers: List[str] = []

    for ticker in tickers:
        try:
            raw_df = download_ticker_data(ticker=ticker, period=period)
            clean_df = clean_raw_data(df=raw_df, ticker=ticker)
            save_raw_data(df=clean_df, ticker=ticker, output_dir=output_dir)
            results[ticker] = clean_df
        except Exception as exc:
            logger.error(f"Failed to process ticker '{ticker}': {exc}")
            failed_tickers.append(ticker)

    if failed_tickers:
        logger.warning(f"Batch completed with {len(failed_tickers)} failures: {failed_tickers}")
    else:
        logger.info(f"Batch successfully completed for all {len(tickers)} symbols.")

    return results


if __name__ == "__main__":
    # When executed directly, download all 6 project tickers for 10 years
    download_all()
