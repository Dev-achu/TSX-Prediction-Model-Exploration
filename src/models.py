"""Model training and walk-forward validation pipeline for TSX Predictor.

Models Implemented:
1. Logistic Regression:
   - Linear baseline with L2 penalty.
   - Scaled using StandardScaler inside a Pipeline to prevent train-to-test leakage.
2. Random Forest Classifier:
   - Bagging ensemble of shallow decision trees.
   - Handles non-linear feature interactions and fat-tailed return distributions.
3. XGBoost Classifier (Extreme Gradient Boosting):
   - Regularized gradient boosted tree ensemble.
   - State-of-the-art tabular benchmark.

Time-Series Cross-Validation Principles:
----------------------------------------
1. Strict Chronological Split (Train 80% / Holdout Test 20%):
   - The final 20% of data (most recent ~2 years) is strictly held out.
   - It is NEVER touched during hyperparameter tuning or model selection.
2. Walk-Forward Validation (TimeSeriesSplit, 5 Folds):
   - Within the 80% training window, we use an expanding walk-forward split.
   - Fold k trains strictly on data up to time t_k, and validates on (t_k, t_k + window].
   - Guarantees zero future-to-past information leakage.
3. Baseline Benchmarks:
   - Every model's test performance is compared side-by-side with naive baselines
     (Always Up and Tomorrow = Today).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.baselines import AlwaysUpBaseline, TomorrowEqualsTodayBaseline
from src.data import ALL_DEFAULT_TICKERS
from src.features import DEFAULT_PROCESSED_DATA_DIR, load_processed_features

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Data Preparation for Modeling
# ---------------------------------------------------------------------------

def prepare_train_test_data(
    ticker: str,
    test_ratio: float = 0.20,
    processed_dir: Path | str = DEFAULT_PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Loads processed features for a ticker and performs a strict time-based split.

    Args:
        ticker: Symbol name (e.g. 'RY.TO' or '^GSPTSE').
        test_ratio: Fraction of most recent data to reserve for testing (default 0.20).
        processed_dir: Path to directory containing processed feature CSVs.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, returns_train, returns_test).
    """
    df = load_processed_features(ticker, processed_dir=processed_dir)

    # Separate feature matrix from target and future return
    feature_cols = [c for c in df.columns if c not in ["target", "target_return"]]
    X = df[feature_cols].copy()
    y = df["target"].astype(int)
    target_returns = df["target_return"]

    # Strict chronological split (never shuffled!)
    split_idx = int(len(df) * (1.0 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    returns_train, returns_test = target_returns.iloc[:split_idx], target_returns.iloc[split_idx:]

    # Integrity assertion
    assert X_train.index.max() < X_test.index.min(), "Data leakage! Train and test overlap in time."

    return X_train, X_test, y_train, y_test, returns_train, returns_test


# ---------------------------------------------------------------------------
# Model Definitions & Hyperparameter Grids
# ---------------------------------------------------------------------------

def get_model_definitions() -> Dict[str, Dict[str, Any]]:
    """Defines candidate models, pipelines, and minimal hyperparameter search spaces.

    Design Choices:
    - Logistic Regression uses a Pipeline with StandardScaler so features with
      different scales (e.g. RSI 0-100 vs returns 0.01) don't distort weights.
      StandardScaler is fit strictly on training folds inside GridSearchCV.
    - Tree models (RandomForest, XGBoost) use shallow depths (2-5) to strongly
      regularize against financial market noise and prevent overfitting.
    """
    return {
        "Logistic Regression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "min_samples_split": [10, 20],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8],
            },
        },
    }


# ---------------------------------------------------------------------------
# Walk-Forward Training & Evaluation Routine
# ---------------------------------------------------------------------------

def train_and_evaluate_model(
    model_name: str,
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Tunes and evaluates a single model using TimeSeriesSplit walk-forward validation.

    Walk-Forward Steps:
    1. Grid search with TimeSeriesSplit(n_splits=5) over X_train and y_train only.
       The holdout test set (X_test, y_test) is NEVER touched during this stage.
    2. Best model is refitted on the entire training set (80%).
    3. Final evaluation is scored on the unseen holdout test set (20%).

    Args:
        model_name: Display name of the model.
        estimator: Scikit-learn estimator or Pipeline.
        param_grid: Hyperparameter grid.
        X_train, y_train: Training split.
        X_test, y_test: Holdout test split.
        n_splits: Number of walk-forward folds (default 5).

    Returns:
        Dictionary containing best hyperparameters, CV scores, and test metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,  # Refit best estimator on full 80% train set
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Evaluate on untouched holdout test set
    test_preds = best_estimator.predict(X_test)
    test_proba = best_estimator.predict_proba(X_test)[:, 1]

    # Metrics
    test_accuracy = accuracy_score(y_test, test_preds)
    test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
    test_roc_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_brier = brier_score_loss(y_test, test_proba)

    return {
        "model_name": model_name,
        "best_estimator": best_estimator,
        "best_params": best_params,
        "cv_accuracy": best_cv_score,
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_acc,
        "test_roc_auc": test_roc_auc,
        "test_f1": test_f1,
        "test_brier_score": test_brier,
        "test_predictions": test_preds,
        "test_probabilities": test_proba,
    }


def train_models_for_ticker(
    ticker: str,
    test_ratio: float = 0.20,
    n_splits: int = 5,
    save_models: bool = True,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Trains all models for a specific ticker and computes baseline benchmarks.

    Args:
        ticker: Symbol name (e.g. 'RY.TO').
        test_ratio: Fraction for test set.
        n_splits: Number of CV folds.
        save_models: Whether to serialize trained models with joblib.
        models_dir: Directory to store model files.

    Returns:
        Tuple of (list of model results, dictionary of baseline test accuracies).
    """
    logger.info(f"[{ticker}] Preparing time-series splits (test_ratio={test_ratio})...")
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_data(ticker, test_ratio=test_ratio)

    # 1. Compute Naive Baselines on the exact same test set
    always_up = AlwaysUpBaseline()
    always_up_test_acc = accuracy_score(y_test, always_up.predict(X_test))

    mom_baseline = TomorrowEqualsTodayBaseline()
    mom_test_acc = accuracy_score(y_test, mom_baseline.predict(X_test))

    baselines = {
        "Always Up": always_up_test_acc,
        "Tomorrow=Today": mom_test_acc,
    }

    # 2. Train and tune ML models
    model_defs = get_model_definitions()
    results: List[Dict[str, Any]] = []

    models_path = Path(models_dir)
    if save_models:
        models_path.mkdir(parents=True, exist_ok=True)

    for name, config in model_defs.items():
        logger.info(f"[{ticker}] Tuning {name} with {n_splits}-fold walk-forward CV...")
        res = train_and_evaluate_model(
            model_name=name,
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_splits=n_splits,
        )
        res["ticker"] = ticker
        res["always_up_test_acc"] = always_up_test_acc
        res["momentum_test_acc"] = mom_test_acc
        res["test_samples"] = len(y_test)

        # Check whether model beat the majority baseline
        res["beat_always_up"] = res["test_accuracy"] > always_up_test_acc
        res["beat_momentum"] = res["test_accuracy"] > mom_test_acc

        if save_models:
            clean_ticker = ticker.replace("^", "")
            clean_name = name.lower().replace(" ", "_")
            filepath = models_path / f"{clean_ticker}_{clean_name}.joblib"
            joblib.dump(res["best_estimator"], filepath)
            res["model_file"] = str(filepath)

        results.append(res)

    return results, baselines


def train_all_tickers(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    test_ratio: float = 0.20,
    n_splits: int = 5,
    output_csv: Path | str = DEFAULT_PROCESSED_DATA_DIR / "model_results.csv",
) -> pd.DataFrame:
    """Trains all models across all tickers and compiles a comprehensive results table.

    Args:
        tickers: List of ticker symbols.
        test_ratio: Test split ratio.
        n_splits: Walk-forward folds.
        output_csv: Path to save the summary results CSV.

    Returns:
        pd.DataFrame summarizing performance across all tickers and models.
    """
    all_rows = []

    for ticker in tickers:
        ticker_results, _ = train_models_for_ticker(
            ticker=ticker,
            test_ratio=test_ratio,
            n_splits=n_splits,
        )

        for r in ticker_results:
            all_rows.append({
                "Ticker": r["ticker"],
                "Model": r["model_name"],
                "CV Accuracy": r["cv_accuracy"],
                "Test Accuracy": r["test_accuracy"],
                "Always Up (Baseline)": r["always_up_test_acc"],
                "Today Dir (Baseline)": r["momentum_test_acc"],
                "Beat Baseline?": "YES" if r["beat_always_up"] else "NO",
                "Test ROC-AUC": r["test_roc_auc"],
                "Test F1": r["test_f1"],
                "Test Brier Score": r["test_brier_score"],
                "Best Params": str(r["best_params"]),
            })

    results_df = pd.DataFrame(all_rows)

    # Save to CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved evaluation results across all models and stocks to {out_path}")

    return results_df


def format_model_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Formats model results into a presentation table."""
    disp = df.copy()
    disp["CV Accuracy"] = (disp["CV Accuracy"] * 100).map("{:.2f}%".format)
    disp["Test Accuracy"] = (disp["Test Accuracy"] * 100).map("{:.2f}%".format)
    disp["Always Up (Baseline)"] = (disp["Always Up (Baseline)"] * 100).map("{:.2f}%".format)
    disp["Today Dir (Baseline)"] = (disp["Today Dir (Baseline)"] * 100).map("{:.2f}%".format)
    disp["Test ROC-AUC"] = disp["Test ROC-AUC"].map("{:.3f}".format)
    disp["Test Brier"] = disp["Test Brier Score"].map("{:.4f}".format)

    cols = [
        "Ticker",
        "Model",
        "CV Accuracy",
        "Test Accuracy",
        "Always Up (Baseline)",
        "Today Dir (Baseline)",
        "Beat Baseline?",
        "Test ROC-AUC",
        "Test Brier",
    ]
    return disp[cols]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 90)
    print("TSX MODEL TRAINING: WALK-FORWARD VALIDATION & ZERO TEST LEAKAGE")
    print("=" * 90)
    print("Walk-forward setup:")
    print("  - Train/Test Split: First 80% (train/tuning) vs. Last 20% (untouched holdout test)")
    print("  - Tuning Method: TimeSeriesSplit(5 folds) strictly on training set")
    print("  - Models: Logistic Regression, Random Forest, XGBoost")
    print("-" * 90)

    summary_df = train_all_tickers()
    table = format_model_results_table(summary_df)
    print("\n" + table.to_string(index=False))
    print("=" * 90 + "\n")
