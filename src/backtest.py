"""Backtesting engine for TSX Stock Price Direction Predictor.

Strategy Specification:
-----------------------
- Signal:
  At the close of trading day t, the model produces prediction y_pred_t in {0, 1}.
  - If y_pred_t == 1: Enter / maintain Long position (pos_t = 1.0) for day t+1.
  - If y_pred_t == 0: Exit to Cash (pos_t = 0.0) for day t+1, earning 0% return.
- Execution & Timing:
  - Position pos_t captures the price change from day t close to day t+1 close (Return_{t+1}).
  - Zero look-ahead: The decision for day t uses ONLY day t features.
- Transaction Costs:
  - 0.1% (10 basis points = 0.0010) per trade.
  - Incurred whenever the position changes: turnover_t = |pos_t - pos_{t-1}|.
  - Transaction cost: cost_t = turnover_t * cost_per_trade.
  - Net Strategy Return: R_net_{t+1} = pos_t * Return_{t+1} - cost_t.
- Benchmark:
  - Buy-and-Hold: pos_t = 1.0 for all days, paying a single 0.1% entry fee on day 1.

Metrics Computed:
-----------------
1. Cumulative Return (%): Total wealth growth over the test period.
2. Annualized Return (CAGR %): Normalized compound annual growth rate.
3. Annualized Volatility (%): Risk of the strategy's equity trajectory.
4. Sharpe Ratio: (CAGR - RiskFreeRate) / Annualized Volatility.
5. Maximum Drawdown (Max DD %): Worst peak-to-trough drop in equity.
6. Calmar Ratio: CAGR / |Max DD| (return generated per unit of downside crash risk).
7. Turnover & Cost Drag (%): Total frictional losses to trading fees.
8. Win Rate (%): Fraction of invested days that yielded positive returns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import ALL_DEFAULT_TICKERS
from src.features import DEFAULT_PROCESSED_DATA_DIR, load_processed_features
from src.models import DEFAULT_MODELS_DIR, prepare_train_test_data

logger = logging.getLogger(__name__)

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


# ---------------------------------------------------------------------------
# Core Quantitative Backtest Engine
# ---------------------------------------------------------------------------

def calculate_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """Computes peak-to-date running maximum and percentage drawdown series."""
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return drawdown


def run_strategy_backtest(
    dates: pd.DatetimeIndex,
    returns_next: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    ticker: str,
    model_name: str,
    cost_per_trade: float = 0.0010,  # 0.1% per trade
    risk_free_rate: float = 0.02,     # 2.0% annual risk-free rate
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """Simulates a Long/Cash trading strategy with realistic transaction fees.

    Args:
        dates: DatetimeIndex for the backtest window.
        returns_next: Realized asset returns for day t+1 (target_return).
        predictions: Model binary predictions at close of day t (0 or 1).
        ticker: Asset symbol.
        model_name: Name of the strategy or model.
        cost_per_trade: Friction fee rate per trade (default 0.0010 = 0.1%).
        risk_free_rate: Annualized risk-free benchmark rate (default 0.02 = 2%).
        initial_capital: Starting portfolio value (default $10,000).

    Returns:
        Dictionary of performance metrics, equity series, and trade logs.
    """
    n_days = len(returns_next)
    y_pred = np.asarray(predictions, dtype=float)
    ret_arr = np.asarray(returns_next, dtype=float)

    # 1. Position array: 1.0 (Long) or 0.0 (Cash)
    pos = y_pred.copy()

    # 2. Turnover calculation
    # Previous position (day t-1): initial position before day 0 is cash (0.0)
    pos_prev = np.roll(pos, 1)
    pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    trade_costs = turnover * cost_per_trade

    # 3. Strategy Net Daily Returns
    strategy_returns = (pos * ret_arr) - trade_costs

    # 4. Buy and Hold Net Daily Returns (Entry fee paid on day 0)
    bnh_costs = np.zeros(n_days)
    bnh_costs[0] = cost_per_trade
    bnh_returns = ret_arr - bnh_costs

    # 5. Cumulative Equity Curves
    strat_equity = initial_capital * np.cumprod(1.0 + strategy_returns)
    bnh_equity = initial_capital * np.cumprod(1.0 + bnh_returns)

    strat_eq_series = pd.Series(strat_equity, index=dates, name=f"{model_name} Equity")
    bnh_eq_series = pd.Series(bnh_equity, index=dates, name="Buy & Hold Equity")

    # 6. Cumulative & Annualized Metrics
    strat_total_return = (strat_equity[-1] / initial_capital) - 1.0
    bnh_total_return = (bnh_equity[-1] / initial_capital) - 1.0

    years = n_days / 252.0
    strat_cagr = (strat_equity[-1] / initial_capital) ** (1.0 / years) - 1.0 if strat_equity[-1] > 0 else -1.0
    bnh_cagr = (bnh_equity[-1] / initial_capital) ** (1.0 / years) - 1.0 if bnh_equity[-1] > 0 else -1.0

    strat_vol = np.std(strategy_returns) * np.sqrt(252)
    bnh_vol = np.std(bnh_returns) * np.sqrt(252)

    strat_sharpe = (strat_cagr - risk_free_rate) / strat_vol if strat_vol > 0 else 0.0
    bnh_sharpe = (bnh_cagr - risk_free_rate) / bnh_vol if bnh_vol > 0 else 0.0

    # 7. Drawdown Analysis
    strat_dd_series = calculate_drawdown_series(strat_eq_series)
    bnh_dd_series = calculate_drawdown_series(bnh_eq_series)

    strat_max_dd = strat_dd_series.min()
    bnh_max_dd = bnh_dd_series.min()

    strat_calmar = strat_cagr / abs(strat_max_dd) if abs(strat_max_dd) > 0 else 0.0
    bnh_calmar = bnh_cagr / abs(bnh_max_dd) if abs(bnh_max_dd) > 0 else 0.0

    # 8. Operational & Trading Statistics
    total_trades = int(np.sum(turnover))
    total_cost_paid = float(np.sum(trade_costs))
    exposure_pct = float(np.mean(pos))
    invested_days = np.sum(pos > 0)
    win_days = np.sum((pos > 0) & (ret_arr > 0))
    win_rate = (win_days / invested_days) if invested_days > 0 else 0.0

    return {
        "ticker": ticker,
        "model_name": model_name,
        "test_days": n_days,
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        # Strategy Metrics
        "strat_total_return": strat_total_return,
        "strat_cagr": strat_cagr,
        "strat_volatility": strat_vol,
        "strat_sharpe": strat_sharpe,
        "strat_max_dd": strat_max_dd,
        "strat_calmar": strat_calmar,
        # Buy & Hold Metrics
        "bnh_total_return": bnh_total_return,
        "bnh_cagr": bnh_cagr,
        "bnh_volatility": bnh_vol,
        "bnh_sharpe": bnh_sharpe,
        "bnh_max_dd": bnh_max_dd,
        "bnh_calmar": bnh_calmar,
        # Operational Stats
        "total_trades": total_trades,
        "total_cost_pct": total_cost_paid,
        "market_exposure": exposure_pct,
        "win_rate": win_rate,
        # Time Series
        "strategy_equity_curve": strat_eq_series,
        "bnh_equity_curve": bnh_eq_series,
        "strategy_drawdown": strat_dd_series,
        "bnh_drawdown": bnh_dd_series,
    }


# ---------------------------------------------------------------------------
# Multi-Stock Backtest Runner
# ---------------------------------------------------------------------------

def backtest_models_for_ticker(
    ticker: str,
    test_ratio: float = 0.20,
    cost_per_trade: float = 0.0010,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> List[Dict[str, Any]]:
    """Runs backtests for Logistic Regression, Random Forest, and XGBoost on a stock."""
    _, X_test, _, y_test, _, returns_test = prepare_train_test_data(ticker, test_ratio=test_ratio)
    clean_ticker = ticker.replace("^", "")
    models_path = Path(models_dir)

    results = []
    model_configs = [
        ("Logistic Regression", f"{clean_ticker}_logistic_regression.joblib"),
        ("Random Forest", f"{clean_ticker}_random_forest.joblib"),
        ("XGBoost", f"{clean_ticker}_xgboost.joblib"),
    ]

    for model_name, filename in model_configs:
        file_path = models_path / filename
        if not file_path.exists():
            continue

        model = joblib.load(file_path)
        preds = model.predict(X_test)

        bt_res = run_strategy_backtest(
            dates=X_test.index,
            returns_next=returns_test.values,
            predictions=preds,
            ticker=ticker,
            model_name=model_name,
            cost_per_trade=cost_per_trade,
        )
        results.append(bt_res)

    return results


def backtest_all_stocks(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    test_ratio: float = 0.20,
    cost_per_trade: float = 0.0010,
    output_csv: Optional[Path | str] = DEFAULT_PROCESSED_DATA_DIR / "backtest_results.csv",
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """Runs backtest across all tickers and returns a tidy summary DataFrame."""
    summary_rows = []
    detailed_map = {}

    for ticker in tickers:
        stock_results = backtest_models_for_ticker(ticker, test_ratio=test_ratio, cost_per_trade=cost_per_trade)
        detailed_map[ticker] = stock_results

        for r in stock_results:
            summary_rows.append({
                "Ticker": r["ticker"],
                "Model": r["model_name"],
                "Strat Return (%)": r["strat_total_return"] * 100,
                "BnH Return (%)": r["bnh_total_return"] * 100,
                "Outperformance (%)": (r["strat_total_return"] - r["bnh_total_return"]) * 100,
                "Strat Sharpe": r["strat_sharpe"],
                "BnH Sharpe": r["bnh_sharpe"],
                "Strat Max DD (%)": r["strat_max_dd"] * 100,
                "BnH Max DD (%)": r["bnh_max_dd"] * 100,
                "Trades": r["total_trades"],
                "Cost Drag (%)": r["total_cost_pct"] * 100,
                "Exposure (%)": r["market_exposure"] * 100,
                "Win Rate (%)": r["win_rate"] * 100,
            })

    summary_df = pd.DataFrame(summary_rows)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        logger.info(f"Saved backtest summary to {out_path}")

    return summary_df, detailed_map


def format_backtest_table(df: pd.DataFrame) -> pd.DataFrame:
    """Formats backtest summary DataFrame into a presentation table."""
    disp = df.copy()
    disp["Strat Return"] = disp["Strat Return (%)"].map("{:+.2f}%".format)
    disp["BnH Return"] = disp["BnH Return (%)"].map("{:+.2f}%".format)
    disp["Alpha (Excess)"] = disp["Outperformance (%)"].map("{:+.2f}%".format)
    disp["Strat Sharpe"] = disp["Strat Sharpe"].map("{:.2f}".format)
    disp["BnH Sharpe"] = disp["BnH Sharpe"].map("{:.2f}".format)
    disp["Strat Max DD"] = disp["Strat Max DD (%)"].map("{:.2f}%".format)
    disp["BnH Max DD"] = disp["BnH Max DD (%)"].map("{:.2f}%".format)
    disp["Friction Drag"] = disp["Cost Drag (%)"].map("{:.2f}%".format)
    disp["Exposure"] = disp["Exposure (%)"].map("{:.1f}%".format)

    cols = [
        "Ticker",
        "Model",
        "Strat Return",
        "BnH Return",
        "Alpha (Excess)",
        "Strat Sharpe",
        "BnH Sharpe",
        "Strat Max DD",
        "BnH Max DD",
        "Trades",
        "Friction Drag",
    ]
    return disp[cols]


# ---------------------------------------------------------------------------
# Visualization Routines (Equity Curves & Drawdowns)
# ---------------------------------------------------------------------------

def plot_equity_curves_all(
    detailed_map: Dict[str, List[Dict[str, Any]]],
    output_path: Path | str = DEFAULT_FIGURES_DIR / "equity_curves_all.png",
) -> Path:
    """Plots a 2x3 panel of equity curves (Strategy vs. Buy & Hold vs. Cash) across all assets."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tickers = list(detailed_map.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = {
        "Logistic Regression": "navy",
        "Random Forest": "forestgreen",
        "XGBoost": "crimson",
    }

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        results = detailed_map[ticker]
        if not results:
            continue

        # Plot Buy & Hold
        bnh_eq = results[0]["bnh_equity_curve"]
        bnh_ret = results[0]["bnh_total_return"] * 100
        ax.plot(bnh_eq.index, bnh_eq, color="black", linestyle="--", linewidth=2.0, label=f"Buy & Hold ({bnh_ret:+.1f}%)")

        # Plot Model Strategies
        for res in results:
            m_name = res["model_name"]
            strat_eq = res["strategy_equity_curve"]
            strat_ret = res["strat_total_return"] * 100
            c = colors.get(m_name, "purple")
            ax.plot(strat_eq.index, strat_eq, color=c, linewidth=1.7, label=f"{m_name} ({strat_ret:+.1f}%)")

        # Initial capital horizontal reference
        ax.axhline(10000.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{ticker} Out-of-Sample Equity Curves ($10k Initial)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Strategy vs. Buy-and-Hold Equity Curves (492-Day Holdout Test with 0.1% Friction)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved equity curves plot to {out_file}")
    return out_file


def plot_stock_detailed_backtest(
    stock_results: List[Dict[str, Any]],
    ticker: str,
    output_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> Path:
    """Generates a detailed 2-panel chart (Equity Curve & Underwater Drawdown) for a single asset."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_ticker = ticker.replace("^", "")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    colors = {
        "Logistic Regression": "navy",
        "Random Forest": "forestgreen",
        "XGBoost": "crimson",
    }

    # Top Panel: Equity
    bnh_eq = stock_results[0]["bnh_equity_curve"]
    bnh_ret = stock_results[0]["bnh_total_return"] * 100
    ax1.plot(bnh_eq.index, bnh_eq, color="black", linestyle="--", linewidth=2.0, label=f"Buy & Hold ({bnh_ret:+.1f}%)")

    for res in stock_results:
        m_name = res["model_name"]
        strat_eq = res["strategy_equity_curve"]
        strat_ret = res["strat_total_return"] * 100
        c = colors.get(m_name, "purple")
        ax1.plot(strat_eq.index, strat_eq, color=c, linewidth=1.8, label=f"{m_name} ({strat_ret:+.1f}%)")

    ax1.set_title(f"{ticker}: Backtest Equity Progression & Drawdowns ($10,000 Starting Capital)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom Panel: Underwater Drawdowns
    bnh_dd = stock_results[0]["bnh_drawdown"] * 100
    ax2.plot(bnh_dd.index, bnh_dd, color="black", linestyle="--", linewidth=1.5, label="Buy & Hold DD")

    for res in stock_results:
        m_name = res["model_name"]
        strat_dd = res["strategy_drawdown"] * 100
        c = colors.get(m_name, "purple")
        ax2.plot(strat_dd.index, strat_dd, color=c, linewidth=1.5, label=f"{m_name} DD")

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    out_file = out_dir / f"{clean_ticker}_backtest_detail.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 90)
    print("TSX STRATEGY BACKTEST: LONG / CASH WITH 0.1% TRANSACTION COSTS")
    print("=" * 90)
    print("Assumptions:")
    print("  - Universe: 5 Equities + TSX Composite Index")
    print("  - Period: 492 Holdout Trading Days (2024 to 2026)")
    print("  - Transaction Fee: 0.10% (10 bps) per turnover event")
    print("  - Risk-Free Rate: 2.0% annual")
    print("-" * 90)

    summary_df, detailed_map = backtest_all_stocks()
    table = format_backtest_table(summary_df)
    print(table.to_string(index=False))
    print("=" * 90 + "\n")

    # Generate charts
    eq_plot = plot_equity_curves_all(detailed_map)
    print(f"Equity curves plot saved to: {eq_plot}")

    for ticker, res_list in detailed_map.items():
        dt_plot = plot_stock_detailed_backtest(res_list, ticker)
        print(f"Detailed backtest plot for {ticker} saved to: {dt_plot}")
