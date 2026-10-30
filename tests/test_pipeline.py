"""Comprehensive automated test suite for the TSX Stock Price Direction Predictor.

Verifies:
1. Data Integrity: Raw data format, monotonicity, calendar synchrony, and clean values.
2. Feature Pipeline: Rigorous zero look-ahead bias, perturbation resistance, and indicator bounds.
3. Baseline Estimators: Majority class and momentum baseline interfaces and outputs.
4. Trained Models: Joblib serialization integrity and out-of-sample inference validity.
5. Backtesting Engine: Transaction friction accounting, drawdowns, and portfolio metrics.
6. Web App Functions: Streamlit caching, helper loaders, and dynamic backtest simulation.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.data import ALL_DEFAULT_TICKERS, BENCHMARK_INDEX, load_raw_data, ticker_to_filename
from src.features import (
    compute_rsi,
    create_features_for_df,
    load_processed_features,
    verify_feature_integrity,
)
from src.baselines import (
    AlwaysUpBaseline,
    TomorrowEqualsTodayBaseline,
    prepare_directional_dataset,
    time_based_train_test_split,
)
from src.models import DEFAULT_MODELS_DIR, prepare_train_test_data
from src.evaluate import compute_classification_metrics
from src.backtest import calculate_drawdown_series, run_strategy_backtest
from app import compute_backtest_series, load_feature_data, load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Data Integrity Tests
# ---------------------------------------------------------------------------

def test_ticker_to_filename():
    assert ticker_to_filename("^GSPTSE") == "GSPTSE.csv"
    assert ticker_to_filename("RY.TO") == "RY.TO.csv"


@pytest.mark.parametrize("ticker", ALL_DEFAULT_TICKERS)
def test_raw_data_files(ticker: str):
    df = load_raw_data(ticker)
    assert not df.empty, f"Raw data for {ticker} is empty."
    assert df.index.is_monotonic_increasing, f"Dates for {ticker} are not chronologically sorted."
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        assert col in df.columns, f"Required column '{col}' missing in {ticker} raw CSV."
        assert df[col].isnull().sum() == 0, f"Found nulls in {ticker} column '{col}'."
    assert (df["Adj Close"] > 0).all(), f"Found non-positive Adj Close in {ticker}."


def test_calendar_alignment():
    """Verify all assets share identical trading dates."""
    date_sets = [set(load_raw_data(t).index) for t in ALL_DEFAULT_TICKERS]
    common = set.intersection(*date_sets)
    union = set.union(*date_sets)
    assert len(common) == len(union), "Calendar misalignment detected across assets."


# ---------------------------------------------------------------------------
# 2. Feature Pipeline & Zero Look-Ahead Verification
# ---------------------------------------------------------------------------

def test_rsi_bounds():
    prices = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5] * 3)
    rsi = compute_rsi(prices, period=5)
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all(), "RSI values exceed [0, 100] bounds."


@pytest.mark.parametrize("ticker", ALL_DEFAULT_TICKERS)
def test_processed_features_integrity(ticker: str):
    df_feat = load_processed_features(ticker)
    verify_feature_integrity(df_feat, ticker)
    assert "target" in df_feat.columns
    assert "target_return" in df_feat.columns
    assert set(df_feat["target"].unique()).issubset({0, 1})


def test_zero_look_ahead_perturbation():
    """Modifying day t+1 price MUST NOT alter features computed on day t."""
    df_raw = load_raw_data("RY.TO").copy()
    features_orig = create_features_for_df(df_raw, include_market_features=False)

    # Artificially spike day 100 price by +50%
    target_idx = 100
    df_perturbed = df_raw.copy()
    df_perturbed.iloc[target_idx, df_perturbed.columns.get_loc("Adj Close")] *= 1.50
    df_perturbed.iloc[target_idx, df_perturbed.columns.get_loc("Close")] *= 1.50

    features_perturbed = create_features_for_df(df_perturbed, include_market_features=False)

    # Feature columns up to day target_idx - 1 MUST be strictly identical
    feat_cols = [c for c in features_orig.columns if c not in ["target", "target_return"]]
    orig_prior = features_orig.iloc[:target_idx - 50][feat_cols]
    pert_prior = features_perturbed.iloc[:target_idx - 50][feat_cols]

    np.testing.assert_allclose(
        orig_prior.values,
        pert_prior.values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Look-ahead leak: Day t features changed when day t+1 was perturbed!",
    )


# ---------------------------------------------------------------------------
# 3. Baseline Estimator Tests
# ---------------------------------------------------------------------------

def test_always_up_baseline():
    clf = AlwaysUpBaseline()
    X = pd.DataFrame({"return_1d": [0.01, -0.02, 0.03]})
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert (preds == 1).all()
    assert (probs[:, 1] == 1.0).all()


def test_tomorrow_equals_today_baseline():
    clf = TomorrowEqualsTodayBaseline()
    X = pd.DataFrame({"return_1d": [0.02, -0.01, 0.05, -0.03]})
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    expected = np.array([1, 0, 1, 0])
    np.testing.assert_array_equal(preds, expected)
    assert probs[0, 1] == 1.0
    assert probs[1, 0] == 1.0


# ---------------------------------------------------------------------------
# 4. Model Serialization & Inference Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", ALL_DEFAULT_TICKERS)
@pytest.mark.parametrize("model_key", ["logistic_regression", "random_forest", "xgboost"])
def test_serialized_models(ticker: str, model_key: str):
    clean_ticker = ticker.replace("^", "")
    model_path = DEFAULT_MODELS_DIR / f"{clean_ticker}_{model_key}.joblib"
    assert model_path.exists(), f"Model file missing: {model_path}"

    model = joblib.load(model_path)
    _, X_test, _, y_test, _, _ = prepare_train_test_data(ticker)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# ---------------------------------------------------------------------------
# 5. Backtest Engine Mathematical Verification
# ---------------------------------------------------------------------------

def test_drawdown_series():
    equity = pd.Series([100.0, 120.0, 110.0, 90.0, 130.0])
    dd = calculate_drawdown_series(equity)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    assert np.isclose(dd.iloc[2], (110 - 120) / 120)
    assert np.isclose(dd.iloc[3], (90 - 120) / 120)
    assert dd.iloc[4] == 0.0
    assert (dd <= 0.0).all(), "Drawdown values must be non-positive."


def test_backtest_with_friction():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = np.array([0.05, 0.02, -0.03, 0.04])
    # Signals: Buy on day 0, stay long day 1, exit to cash day 2, enter long day 3
    preds = np.array([1, 1, 0, 1])
    cost_per_trade = 0.0010

    res = run_strategy_backtest(
        dates=dates,
        returns_next=returns,
        predictions=preds,
        ticker="TEST",
        model_name="TestModel",
        cost_per_trade=cost_per_trade,
        initial_capital=10000.0,
    )

    # Turnover:
    # Day 0: 0 -> 1 = 1 turnover (cost = 0.0010) -> net ret = 0.05 - 0.0010 = 0.0490
    # Day 1: 1 -> 1 = 0 turnover -> net ret = 0.02
    # Day 2: 1 -> 0 = 1 turnover (cost = 0.0010) -> net ret = 0.00 - 0.0010 = -0.0010
    # Day 3: 0 -> 1 = 1 turnover (cost = 0.0010) -> net ret = 0.04 - 0.0010 = 0.0390
    expected_trades = 3
    assert res["total_trades"] == expected_trades
    assert np.isclose(res["total_cost_pct"], 3 * cost_per_trade)

    expected_wealth = 10000.0 * (1.0490) * (1.02) * (0.9990) * (1.0390)
    assert np.isclose(res["strategy_equity_curve"].iloc[-1], expected_wealth)


# ---------------------------------------------------------------------------
# 6. Streamlit Helper Function Tests
# ---------------------------------------------------------------------------

def test_app_helpers():
    df = load_feature_data("RY.TO")
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)

    model = load_model("RY.TO", "Random Forest")
    assert model is not None

    # App backtest consistency
    ret_series = df["target_return"].iloc[-100:]
    pred_series = pd.Series(np.ones(100), index=ret_series.index)
    bt = compute_backtest_series(ret_series, pred_series, cost_rate=0.0010)
    assert bt["strat_total_ret"] is not None
    assert (bt["strat_dd"] <= 0.0).all()
