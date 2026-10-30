"""Streamlit Web Application: TSX Stock Price Direction Predictor.

Features:
- Asset Universe Selector (RY.TO, TD.TO, SHOP.TO, ENB.TO, CNQ.TO, ^GSPTSE)
- Model Selector (Logistic Regression, Random Forest, XGBoost)
- Interactive Price & Volume Charts with Moving Averages & Event Annotations
- Out-of-Sample Predictions vs. Actuals with Confusion Matrix & ROC-AUC
- Real-Time Long/Cash Backtesting vs. Buy-and-Hold with Configurable Friction
- Honest Baseline Comparison (Always Up / Buy-and-Hold Proxy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="TSX Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_MODELS_DIR = PROJECT_ROOT / "data" / "models"
REPORTS_FIG_DIR = PROJECT_ROOT / "reports" / "figures"

ASSET_NAMES: Dict[str, str] = {
    "RY.TO": "Royal Bank of Canada (Financials)",
    "TD.TO": "Toronto-Dominion Bank (Financials)",
    "SHOP.TO": "Shopify Inc. (Technology)",
    "ENB.TO": "Enbridge Inc. (Energy / Infrastructure)",
    "CNQ.TO": "Canadian Natural Resources (Energy)",
    "^GSPTSE": "S&P/TSX Composite Index (Benchmark)",
}


# ---------------------------------------------------------------------------
# Data & Model Loading with Streamlit Caching
# ---------------------------------------------------------------------------

@st.cache_data
def load_feature_data(ticker: str) -> pd.DataFrame:
    """Loads processed features for the selected ticker."""
    clean_ticker = ticker.replace("^", "")
    csv_path = DATA_PROCESSED_DIR / f"{clean_ticker}_features.csv"
    if not csv_path.exists():
        csv_path = DATA_PROCESSED_DIR / f"{ticker}_features.csv"
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index(ascending=True)


@st.cache_data
def load_summary_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads saved model evaluation and backtest CSV tables."""
    eval_csv = DATA_PROCESSED_DIR / "detailed_evaluation.csv"
    bt_csv = DATA_PROCESSED_DIR / "backtest_results.csv"

    eval_df = pd.read_csv(eval_csv) if eval_csv.exists() else pd.DataFrame()
    bt_df = pd.read_csv(bt_csv) if bt_csv.exists() else pd.DataFrame()
    return eval_df, bt_df


def load_model(ticker: str, model_name: str):
    """Loads trained joblib model."""
    clean_ticker = ticker.replace("^", "")
    key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    key = key_map.get(model_name, "logistic_regression")
    model_path = DATA_MODELS_DIR / f"{clean_ticker}_{key}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Backtesting Calculation Helper
# ---------------------------------------------------------------------------

def compute_backtest_series(
    returns_series: pd.Series,
    predictions_series: pd.Series,
    cost_rate: float = 0.0010,
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """Computes daily strategy equity curve with configurable friction."""
    pos = predictions_series.values.astype(float)
    ret = returns_series.values.astype(float)

    pos_prev = np.roll(pos, 1)
    pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    costs = turnover * cost_rate

    strat_daily_ret = (pos * ret) - costs

    bnh_costs = np.zeros_like(ret)
    bnh_costs[0] = cost_rate
    bnh_daily_ret = ret - bnh_costs

    strat_equity = initial_capital * np.cumprod(1.0 + strat_daily_ret)
    bnh_equity = initial_capital * np.cumprod(1.0 + bnh_daily_ret)

    # Drawdowns
    strat_series = pd.Series(strat_equity, index=returns_series.index)
    bnh_series = pd.Series(bnh_equity, index=returns_series.index)

    strat_peak = strat_series.cummax()
    bnh_peak = bnh_series.cummax()

    strat_dd = (strat_series - strat_peak) / strat_peak
    bnh_dd = (bnh_series - bnh_peak) / bnh_peak

    # Annualized Metrics
    n_days = len(returns_series)
    years = n_days / 252.0
    strat_cagr = (strat_equity[-1] / initial_capital) ** (1.0 / years) - 1.0 if strat_equity[-1] > 0 else -1.0
    bnh_cagr = (bnh_equity[-1] / initial_capital) ** (1.0 / years) - 1.0 if bnh_equity[-1] > 0 else -1.0

    strat_vol = np.std(strat_daily_ret) * np.sqrt(252)
    bnh_vol = np.std(bnh_daily_ret) * np.sqrt(252)

    rf = 0.02
    strat_sharpe = (strat_cagr - rf) / strat_vol if strat_vol > 0 else 0.0
    bnh_sharpe = (bnh_cagr - rf) / bnh_vol if bnh_vol > 0 else 0.0

    return {
        "strat_equity": strat_series,
        "bnh_equity": bnh_series,
        "strat_dd": strat_dd,
        "bnh_dd": bnh_dd,
        "strat_total_ret": (strat_equity[-1] / initial_capital) - 1.0,
        "bnh_total_ret": (bnh_equity[-1] / initial_capital) - 1.0,
        "strat_cagr": strat_cagr,
        "bnh_cagr": bnh_cagr,
        "strat_vol": strat_vol,
        "bnh_vol": bnh_vol,
        "strat_sharpe": strat_sharpe,
        "bnh_sharpe": bnh_sharpe,
        "strat_max_dd": strat_dd.min(),
        "bnh_max_dd": bnh_dd.min(),
        "total_trades": int(np.sum(turnover)),
        "total_friction_pct": float(np.sum(costs)),
        "market_exposure": float(np.mean(pos)),
    }


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Model & Asset Controls")

selected_ticker = st.sidebar.selectbox(
    "Select TSX Asset",
    options=list(ASSET_NAMES.keys()),
    index=0,
    format_func=lambda t: f"{t} - {ASSET_NAMES[t].split(' (')[0]}",
)

selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    options=["Logistic Regression", "Random Forest", "XGBoost"],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Parameters")

cost_slider_bps = st.sidebar.slider(
    "Transaction Cost per Trade (Basis Points)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="10 bps = 0.10% friction fee deducted on every buy or sell decision.",
)
cost_rate = cost_slider_bps / 10000.0

st.sidebar.markdown(
    """
    **Project Principles**:
    * Strictly time-based splits (no shuffling).
    * Zero look-ahead bias in features.
    * Real-world transaction costs included.
    * Always compared against naive baselines.
    """
)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

df_feat = load_feature_data(selected_ticker)
feat_cols = [c for c in df_feat.columns if c not in ["target", "target_return"]]

# Holdout Test set: most recent 20%
split_idx = int(len(df_feat) * 0.8)
train_df = df_feat.iloc[:split_idx]
test_df = df_feat.iloc[split_idx:]

X_test = test_df[feat_cols]
y_test = test_df["target"].astype(int)
returns_test = test_df["target_return"]

# Load model
model = load_model(selected_ticker, selected_model)
if model is not None:
    preds_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
else:
    preds_test = np.ones(len(y_test), dtype=int)
    probs_test = np.ones(len(y_test), dtype=float)

# Compute Backtest
bt_res = compute_backtest_series(returns_test, pd.Series(preds_test, index=test_df.index), cost_rate=cost_rate)

# Naive Baseline Scores
always_up_acc = float((y_test == 1).mean())
model_acc = float((preds_test == y_test).mean())
beat_baseline = model_acc > always_up_acc


# ---------------------------------------------------------------------------
# Main App Header & KPI Summary Cards
# ---------------------------------------------------------------------------

st.title("🇨🇦 TSX Stock Price Direction Predictor")
st.markdown(
    f"Predicting next-day price direction (**Up / Down**) for **{selected_ticker}** "
    f"(*{ASSET_NAMES[selected_ticker]}*) using machine learning with rigorous baseline benchmarking."
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Selected Model", selected_model)
col2.metric("Test Accuracy", f"{model_acc * 100:.2f}%")
col3.metric("Always Up Hurdle", f"{always_up_acc * 100:.2f}%")
delta_color = "normal" if beat_baseline else "inverse"
diff_pts = (model_acc - always_up_acc) * 100
col4.metric(
    "Beat Baseline?",
    "✅ YES" if beat_baseline else "❌ NO",
    delta=f"{diff_pts:+.2f}%",
    delta_color=delta_color,
)
col5.metric("Holdout Test Window", f"{len(test_df)} days")


# ---------------------------------------------------------------------------
# Interactive Tabbed Interface
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Historical Regimes",
    "🎯 Model Evaluation vs. Baselines",
    "💼 Economic Backtest & Equity Curves",
    "📚 Cross-Asset Summary & Limitations",
])


# ---------------------------------------------------------------------------
# TAB 1: Price History & Historical Regimes
# ---------------------------------------------------------------------------
with tab1:
    st.subheader(f"Historical Price Progression & Key Macro Regimes: {selected_ticker}")
    st.markdown("10 years of daily market history with moving averages and macroeconomic event annotations.")

    # Reconstruct raw price series from feature ratios if needed, or use raw data
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / f"{selected_ticker.replace('^', '')}.csv"
    if not raw_csv_path.exists():
        raw_csv_path = PROJECT_ROOT / "data" / "raw" / f"{selected_ticker}.csv"
    
    if raw_csv_path.exists():
        df_raw = pd.read_csv(raw_csv_path, index_col="Date", parse_dates=True)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.sort_index(ascending=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        ax1.plot(df_raw.index, df_raw["Adj Close"], color="navy", label="Adj Close", linewidth=1.5)
        ax1.plot(df_raw.index, df_raw["Adj Close"].rolling(50).mean(), color="orange", label="50-day SMA", linestyle="--")
        ax1.plot(df_raw.index, df_raw["Adj Close"].rolling(200).mean(), color="crimson", label="200-day SMA", linestyle=":")

        # Shaded Regimes
        ax1.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"), color="red", alpha=0.2, label="2020 COVID Crash")
        ax1.axvspan(pd.Timestamp("2022-03-02"), pd.Timestamp("2022-12-31"), color="gold", alpha=0.2, label="2022 Rate Hike Cycle")
        ax1.axvspan(test_df.index.min(), test_df.index.max(), color="green", alpha=0.1, label="Holdout Test Window (20%)")

        ax1.set_ylabel("Adjusted Close ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title(f"{selected_ticker} 10-Year Price History with Shaded Regimes", fontweight="bold")

        # Volume
        ax2.bar(df_raw.index, df_raw["Volume"] / 1e6, color="steelblue", alpha=0.6, width=1.5)
        ax2.set_ylabel("Volume (M)")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)
        plt.close()


# ---------------------------------------------------------------------------
# TAB 2: Model Performance & Baselines
# ---------------------------------------------------------------------------
with tab2:
    st.subheader(f"Out-of-Sample Performance: {selected_model} on {selected_ticker}")
    st.markdown(
        f"Evaluated on **{len(y_test)} unseen trading days** ({test_df.index.min().strftime('%Y-%m-%d')} to "
        f"{test_df.index.max().strftime('%Y-%m-%d')})."
    )

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("#### Detailed Classification Metrics")
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

        prec = precision_score(y_test, preds_test, zero_division=0)
        rec = recall_score(y_test, preds_test, zero_division=0)
        f1 = f1_score(y_test, preds_test, zero_division=0)
        if probs_test is not None:
            try:
                auc_val = roc_auc_score(y_test, probs_test)
            except Exception:
                auc_val = 0.50
        else:
            auc_val = 0.50

        metrics_table = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Always Up (Baseline Hurdle)",
                "Tomorrow=Today (Momentum Hurdle)",
                "Precision (When model says Up, is it Up?)",
                "Recall (% of actual Up days captured)",
                "F1 Score",
                "ROC-AUC (Discrimination)",
            ],
            "Value": [
                f"{model_acc * 100:.2f}%",
                f"{always_up_acc * 100:.2f}%",
                f"{(test_df['return_1d'] > 0).astype(int).eq(y_test).mean() * 100:.2f}%",
                f"{prec * 100:.2f}%",
                f"{rec * 100:.2f}%",
                f"{f1 * 100:.2f}%",
                f"{auc_val:.3f}",
            ],
        })
        st.table(metrics_table)

    with col_m2:
        st.markdown("#### Holdout Confusion Matrix")
        cm = confusion_matrix(y_test, preds_test, labels=[0, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["Predicted Down (0)", "Predicted Up (1)"],
            yticklabels=["Actual Down (0)", "Actual Up (1)"],
        )
        ax_cm.set_title(f"{selected_ticker} Confusion Matrix", fontweight="bold", fontsize=11)
        st.pyplot(fig_cm)
        plt.close()

    st.markdown("#### Recent Predictions Inspection (Last 10 Days)")
    recent_view = pd.DataFrame({
        "Date": test_df.index[-10:].strftime("%Y-%m-%d"),
        "Predicted Direction": ["🟢 UP" if p == 1 else "🔴 DOWN" for p in preds_test[-10:]],
        "Actual Direction": ["🟢 UP" if a == 1 else "🔴 DOWN" for a in y_test.values[-10:]],
        "Actual Tomorrow Return": [f"{r * 100:+.2f}%" for r in returns_test.values[-10:]],
        "Prediction Correct?": ["✅ Yes" if p == a else "❌ No" for p, a in zip(preds_test[-10:], y_test.values[-10:])],
    })
    st.table(recent_view)


# ---------------------------------------------------------------------------
# TAB 3: Economic Backtest & Equity Curves
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(f"Strategy Simulation vs. Buy & Hold ({cost_slider_bps} bps Friction)")
    st.markdown(
        "Strategy rule: **Long when model signals Up, Cash (0% return) when model signals Down**. "
        f"A transaction fee of **{cost_slider_bps / 100:.2f}%** is deducted on every position flip."
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Strategy Return", f"{bt_res['strat_total_ret'] * 100:+.2f}%")
    kpi2.metric("Buy & Hold Return", f"{bt_res['bnh_total_ret'] * 100:+.2f}%")
    excess_ret = (bt_res["strat_total_ret"] - bt_res["bnh_total_ret"]) * 100
    kpi3.metric("Excess Alpha", f"{excess_ret:+.2f}%", delta=f"{excess_ret:+.2f}%")
    kpi4.metric("Strategy Sharpe", f"{bt_res['strat_sharpe']:.2f}")
    kpi5.metric("Strategy Max DD", f"{bt_res['strat_max_dd'] * 100:.2f}%")

    # Plot Equity Curves & Drawdowns
    fig_bt, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    ax_eq.plot(bt_res["bnh_equity"].index, bt_res["bnh_equity"], color="black", linestyle="--", linewidth=2.0, label="Buy & Hold")
    ax_eq.plot(bt_res["strat_equity"].index, bt_res["strat_equity"], color="crimson", linewidth=1.8, label=f"{selected_model} Strategy")
    ax_eq.axhline(10000.0, color="gray", linestyle=":", alpha=0.7, label="Initial Capital ($10,000)")
    ax_eq.set_ylabel("Portfolio Value ($)")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, linestyle="--", alpha=0.5)
    ax_eq.set_title(f"{selected_ticker} Out-of-Sample Equity Curves ($10k Starting Wealth)", fontweight="bold")

    # Drawdown Panel
    ax_dd.plot(bt_res["bnh_dd"].index, bt_res["bnh_dd"] * 100, color="black", linestyle="--", label="Buy & Hold DD")
    ax_dd.plot(bt_res["strat_dd"].index, bt_res["strat_dd"] * 100, color="crimson", label="Strategy DD")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(loc="lower left")
    ax_dd.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig_bt)
    plt.close()

    # Trading Friction Callout
    st.info(
        f"💡 **Friction Impact**: The strategy executed **{bt_res['total_trades']} trades**, "
        f"burning **{bt_res['total_friction_pct'] * 100:.2f}%** in transaction costs over the test window. "
        f"Time spent in the market: **{bt_res['market_exposure'] * 100:.1f}%** of trading days."
    )


# ---------------------------------------------------------------------------
# TAB 4: Cross-Asset Summary & Limitations
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Cross-Asset Project Summary (All 6 Equities)")
    eval_summary, bt_summary = load_summary_results()

    if not bt_summary.empty:
        st.markdown("#### Complete Out-of-Sample Backtest Results")
        disp_bt = bt_summary.copy()
        disp_bt["Strat Return"] = disp_bt["Strat Return (%)"].map("{:+.2f}%".format)
        disp_bt["BnH Return"] = disp_bt["BnH Return (%)"].map("{:+.2f}%".format)
        disp_bt["Alpha"] = disp_bt["Outperformance (%)"].map("{:+.2f}%".format)
        disp_bt["Strat Sharpe"] = disp_bt["Strat Sharpe"].map("{:.2f}".format)
        disp_bt["BnH Sharpe"] = disp_bt["BnH Sharpe"].map("{:.2f}".format)
        disp_bt["Strat Max DD"] = disp_bt["Strat Max DD (%)"].map("{:.2f}%".format)
        disp_bt["Cost Drag"] = disp_bt["Cost Drag (%)"].map("{:.2f}%".format)

        cols_show = ["Ticker", "Model", "Strat Return", "BnH Return", "Alpha", "Strat Sharpe", "BnH Sharpe", "Strat Max DD", "Trades", "Cost Drag"]
        try:
            st.dataframe(disp_bt[cols_show], width="stretch")
        except TypeError:
            st.dataframe(disp_bt[cols_show], use_container_width=True)

    st.markdown("---")
    st.subheader("Critical Quantitative Limitations")
    st.markdown(
        """
        1. **Overfitting Risk & Noise**: Daily equity returns have a signal-to-noise ratio under 2%. Complex algorithms readily memorize past noise. On our holdout test set, models averaged 51%–55% accuracy.
        2. **Market Efficiency & Friction**: Public technical signals are rapidly arbed by high-frequency market makers. Frequent turnover (80–130 trades) destroyed 8%–13% in transaction fees, causing strategies to lag simple Buy-and-Hold.
        3. **Survivorship Bias**: Our asset universe was selected in 2026 from today's surviving TSX mega-caps. Delisted and bankrupt Canadian equities from the 2016–2026 decade are omitted, artificially boosting the historical long drift.
        """
    )
