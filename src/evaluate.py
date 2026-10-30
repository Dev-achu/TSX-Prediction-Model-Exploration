"""Comprehensive evaluation module for TSX stock price direction models.

Metrics Explained for Beginners:
--------------------------------
1. Accuracy:
   - (True Positives + True Negatives) / Total Samples
   - Percentage of days where the model correctly predicted whether the stock went Up or Down.
   - CAVEAT: In an upward trending market where 57% of days are Up, a naive model that always
     guesses Up gets 57% accuracy without learning anything! Always benchmark against baselines.
2. Precision (Positive Predictive Value):
   - True Positives / (True Positives + False Positives)
   - When the model signals 'Up', how often is the stock actually Up?
   - Crucial for trading: high precision minimizes buying on days that turn out to be negative.
3. Recall (Sensitivity / True Positive Rate):
   - True Positives / (True Positives + False Negatives)
   - Out of all the actual Up days in the market, what fraction did the model capture?
4. F1-Score:
   - Harmonic mean of Precision and Recall: 2 * (Prec * Rec) / (Prec + Rec).
   - Balances false alarms against missed opportunities.
5. Balanced Accuracy:
   - Average of recall obtained on each class: 0.5 * (True Positive Rate + True Negative Rate).
   - If an Always-Up model predicts 100% Up, its recall on Down days is 0%, giving a balanced
     accuracy of exactly 50.0%! This exposes naive majority guessing immediately.
6. ROC-AUC (Area Under the Receiver Operating Characteristic):
   - Measures the model's ability to rank probabilities correctly across all possible thresholds.
   - 0.50 = random coin flip, 1.00 = perfect separation.
7. Brier Score:
   - Mean squared difference between predicted probability and actual binary outcome.
   - 0.25 represents an uninformative 50/50 probability. Lower is better.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.baselines import AlwaysUpBaseline, TomorrowEqualsTodayBaseline
from src.data import ALL_DEFAULT_TICKERS
from src.features import DEFAULT_PROCESSED_DATA_DIR, load_processed_features
from src.models import DEFAULT_MODELS_DIR, prepare_train_test_data

logger = logging.getLogger(__name__)

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


# ---------------------------------------------------------------------------
# Metric Computation Core
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, float]:
    """Calculates all key directional classification metrics.

    Args:
        y_true: Ground truth binary target (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability for class 1 (optional).

    Returns:
        Dictionary of metric scores.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_arr, y_prob_arr)
        except Exception:
            metrics["roc_auc"] = 0.50
        metrics["brier_score"] = brier_score_loss(y_true_arr, y_prob_arr)
    else:
        metrics["roc_auc"] = np.nan
        metrics["brier_score"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Multi-Model Evaluation per Asset
# ---------------------------------------------------------------------------

def evaluate_models_for_ticker(
    ticker: str,
    test_ratio: float = 0.20,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> pd.DataFrame:
    """Evaluates all trained models and naive baselines for a specific ticker on holdout test data.

    Args:
        ticker: Symbol name (e.g. 'RY.TO').
        test_ratio: Holdout test split ratio.
        models_dir: Directory where trained .joblib models are stored.

    Returns:
        pd.DataFrame containing side-by-side performance metrics.
    """
    _, X_test, _, y_test, _, _ = prepare_train_test_data(ticker, test_ratio=test_ratio)
    clean_ticker = ticker.replace("^", "")
    models_path = Path(models_dir)

    rows = []

    # 1. Baseline: Always Up (Majority Class)
    always_up = AlwaysUpBaseline()
    pred_up = always_up.predict(X_test)
    prob_up = always_up.predict_proba(X_test)[:, 1]
    m_up = compute_classification_metrics(y_test, pred_up, prob_up)
    m_up.update({"ticker": ticker, "model": "Always Up (Baseline)", "model_type": "Baseline"})
    rows.append(m_up)

    # 2. Baseline: Tomorrow = Today (1-Day Momentum)
    mom = TomorrowEqualsTodayBaseline()
    pred_mom = mom.predict(X_test)
    prob_mom = mom.predict_proba(X_test)[:, 1]
    m_mom = compute_classification_metrics(y_test, pred_mom, prob_mom)
    m_mom.update({"ticker": ticker, "model": "Tomorrow=Today (Baseline)", "model_type": "Baseline"})
    rows.append(m_mom)

    # 3. Machine Learning Models
    model_names = [
        ("Logistic Regression", f"{clean_ticker}_logistic_regression.joblib"),
        ("Random Forest", f"{clean_ticker}_random_forest.joblib"),
        ("XGBoost", f"{clean_ticker}_xgboost.joblib"),
    ]

    for display_name, filename in model_names:
        file_path = models_path / filename
        if not file_path.exists():
            logger.warning(f"Model file not found: {file_path}. Skipping.")
            continue

        model = joblib.load(file_path)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_classification_metrics(y_test, pred, prob)
        metrics.update({"ticker": ticker, "model": display_name, "model_type": "ML Model"})
        rows.append(metrics)

    df_eval = pd.DataFrame(rows)
    return df_eval


def evaluate_all_stocks(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    test_ratio: float = 0.20,
    output_csv: Optional[Path | str] = DEFAULT_PROCESSED_DATA_DIR / "detailed_evaluation.csv",
) -> pd.DataFrame:
    """Evaluates all stocks and compiles a comprehensive evaluation DataFrame."""
    all_dfs = []
    for ticker in tickers:
        df_stock = evaluate_models_for_ticker(ticker, test_ratio=test_ratio)
        all_dfs.append(df_stock)

    combined = pd.concat(all_dfs, ignore_index=True)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info(f"Detailed evaluation metrics saved to {out_path}")

    return combined


def format_evaluation_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Formats metrics DataFrame into a clean display table."""
    disp = eval_df.copy()
    disp["Accuracy"] = (disp["accuracy"] * 100).map("{:.2f}%".format)
    disp["Precision"] = (disp["precision"] * 100).map("{:.2f}%".format)
    disp["Recall"] = (disp["recall"] * 100).map("{:.2f}%".format)
    disp["F1 Score"] = (disp["f1_score"] * 100).map("{:.2f}%".format)
    disp["Bal. Acc"] = (disp["balanced_accuracy"] * 100).map("{:.2f}%".format)
    disp["ROC-AUC"] = disp["roc_auc"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    disp["Brier"] = disp["brier_score"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    cols = [
        "ticker",
        "model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Bal. Acc",
        "ROC-AUC",
        "Brier",
    ]
    return disp[cols].rename(columns={"ticker": "Ticker", "model": "Model"})


# ---------------------------------------------------------------------------
# Visualization Routines (ROC Curves & Confusion Matrices)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    ticker: str,
    test_ratio: float = 0.20,
    output_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> Path:
    """Plots confusion matrices for Logistic Regression, Random Forest, and XGBoost on a stock."""
    _, X_test, _, y_test, _, _ = prepare_train_test_data(ticker, test_ratio=test_ratio)
    clean_ticker = ticker.replace("^", "")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_info = [
        ("Logistic Regression", f"{clean_ticker}_logistic_regression.joblib"),
        ("Random Forest", f"{clean_ticker}_random_forest.joblib"),
        ("XGBoost", f"{clean_ticker}_xgboost.joblib"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (name, filename) in enumerate(models_info):
        file_path = DEFAULT_MODELS_DIR / filename
        if not file_path.exists():
            continue
        model = joblib.load(file_path)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Pred Down (0)", "Pred Up (1)"],
            yticklabels=["Actual Down (0)", "Actual Up (1)"],
        )
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        axes[idx].set_title(f"{name}\nAcc: {acc:.1f}% | Prec: {prec:.1f}% | Rec: {rec:.1f}%", fontsize=11, fontweight="bold")

    plt.suptitle(f"{ticker}: Holdout Test Set Confusion Matrices (492 Days)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = out_dir / f"{clean_ticker}_confusion_matrices.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    return out_file


def plot_roc_curves_all(
    tickers: List[str] = ALL_DEFAULT_TICKERS,
    test_ratio: float = 0.20,
    output_path: Path | str = DEFAULT_FIGURES_DIR / "roc_curves_all.png",
) -> Path:
    """Plots a multi-panel grid of ROC curves for all assets."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    model_configs = [
        ("Logistic Regression", "logistic_regression", "navy"),
        ("Random Forest", "random_forest", "forestgreen"),
        ("XGBoost", "xgboost", "crimson"),
    ]

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        _, X_test, _, y_test, _, _ = prepare_train_test_data(ticker, test_ratio=test_ratio)
        clean_ticker = ticker.replace("^", "")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")

        for model_name, key, color in model_configs:
            file_path = DEFAULT_MODELS_DIR / f"{clean_ticker}_{key}.joblib"
            if not file_path.exists():
                continue
            model = joblib.load(file_path)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{model_name} (AUC = {auc_score:.2f})")

        ax.set_title(f"{ticker} ROC Curve", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Receiver Operating Characteristic (ROC) Curves Across TSX Assets", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves plot to {out_file}")
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 90)
    print("TSX MODEL CLASSIFICATION EVALUATION (ACCURACY, PRECISION, RECALL, F1, ROC-AUC)")
    print("=" * 90)

    eval_results = evaluate_all_stocks()
    table = format_evaluation_table(eval_results)
    print(table.to_string(index=False))
    print("=" * 90 + "\n")

    # Generate visual artifacts
    roc_plot = plot_roc_curves_all()
    print(f"ROC Curves saved to: {roc_plot}")
    for ticker in ALL_DEFAULT_TICKERS:
        cm_plot = plot_confusion_matrices(ticker)
        print(f"Confusion matrices for {ticker} saved to: {cm_plot}")
