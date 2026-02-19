"""
classifier.py
-------------
Trains, evaluates, and saves a no-show prediction model.
Reads from data/processed/model_ready.csv.
Saves trained model to outputs/results/model.pkl.

Usage:
    python src/models/classifier.py
    python src/models/classifier.py --data_path data/processed/model_ready.csv
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "scheduling_interval",
    "appointment_hour",
    "appointment_day_of_week",
    "appointment_month",
    "is_morning",
    "is_monday",
    "is_friday",
    "age",
    "sex_encoded",
    "insurance_encoded",
    "patient_prior_noshows",
    "patient_prior_noshows_rate",
    "patient_prior_appointments",
    "daily_slot_utilization",
    "rolling_7d_noshows_rate",
]

TARGET_COL = "no_show"

# Temporal split boundaries
TRAIN_END   = "2022-12-31"
VAL_END     = "2023-12-31"
# Test = 2024 (everything after VAL_END)

# XGBoost handles class imbalance via scale_pos_weight
# Value = attended / did-not-attend, computed from EDA: 13.01
SCALE_POS_WEIGHT = 13.01


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_and_split(data_path: str):
    """
    Load model_ready.csv and split temporally into train / val / test.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, test_df)
    """
    log.info(f"Loading model-ready data from: {data_path}")
    df = pd.read_csv(
        data_path,
        dtype={"appointment_id": str},
        low_memory=False,
    )
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])

    # Drop rows with any null in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    after = len(df)
    if before - after > 0:
        log.warning(f"  Dropped {before - after:,} rows with nulls in features/target")

    train = df[df["appointment_date"] <= TRAIN_END]
    val   = df[(df["appointment_date"] > TRAIN_END) & (df["appointment_date"] <= VAL_END)]
    test  = df[df["appointment_date"] > VAL_END]

    log.info(f"  Train : {len(train):,} rows ({train['appointment_date'].min().date()} → {train['appointment_date'].max().date()})")
    log.info(f"  Val   : {len(val):,}   rows ({val['appointment_date'].min().date()} → {val['appointment_date'].max().date()})")
    log.info(f"  Test  : {len(test):,}  rows ({test['appointment_date'].min().date()} → {test['appointment_date'].max().date()})")

    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        rate = split_df[TARGET_COL].mean() * 100
        log.info(f"  {split_name} no-show rate: {rate:.1f}%")

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]

    return X_train, y_train, X_val, y_val, X_test, y_test, test


# ── Models ────────────────────────────────────────────────────────────────────

def build_baseline(X_train, y_train):
    """
    Logistic regression baseline with standard scaling.
    Gives us a floor to beat with XGBoost.
    """
    log.info("Training baseline (logistic regression)...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ))
    ])
    pipe.fit(X_train, y_train)
    log.info("  Baseline trained. ✓")
    return pipe


def build_xgboost(X_train, y_train):
    """
    XGBoost classifier tuned for imbalanced classification.
    scale_pos_weight compensates for the 13:1 class ratio.
    """
    log.info("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr",        # PR-AUC suits imbalanced data better than log-loss
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        verbose=False,
    )
    log.info(f"  XGBoost trained. Best iteration: {model.best_iteration}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X, y, split_name: str, model_name: str) -> dict:
    """
    Evaluate a model on a split. Returns a metrics dict.
    Imports kept local to avoid top-level sklearn dependency sprawl.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
    )

    probs = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, probs)
    pr_auc  = average_precision_score(y, probs)

    # Find threshold that maximises F1 for the positive class
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    preds = (probs >= best_thresh).astype(int)

    log.info(f"\n── {model_name} on {split_name} ───────────────────────────────")
    log.info(f"  ROC-AUC  : {roc_auc:.4f}")
    log.info(f"  PR-AUC   : {pr_auc:.4f}")
    log.info(f"  Best threshold (max F1): {best_thresh:.3f}")
    log.info(f"\n{classification_report(y, preds, target_names=['attended', 'no_show'])}")

    cm = confusion_matrix(y, preds)
    log.info(f"  Confusion matrix:\n{cm}")

    return {
        "model": model_name,
        "split": split_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_thresh,
    }


def evaluate_calibration(model, X, y, model_name: str, out_dir: str) -> None:
    """
    Plot calibration curve — are predicted probabilities meaningful?
    A 70% predicted no-show should actually no-show ~70% of the time.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    probs = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")

    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve — {model_name}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"calibration_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    log.info(f"  Calibration curve saved to: {path}")


def plot_feature_importance(model, out_dir: str) -> None:
    """Plot XGBoost feature importances."""
    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=FEATURE_COLS).sort_values(ascending=True)

    plt.figure(figsize=(9, 6))
    feat_imp.plot(kind="barh", color="steelblue")
    plt.title("XGBoost Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feature_importance_xgboost.png")
    plt.savefig(path)
    plt.close()
    log.info(f"  Feature importance plot saved to: {path}")


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, metrics: dict, out_dir: str) -> str:
    """Save the trained model and metadata to disk."""
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics, "feature_cols": FEATURE_COLS}, f)

    log.info(f"Model saved to: {model_path}")
    return model_path


def save_predictions(model, test_df: pd.DataFrame, best_threshold: float, out_dir: str) -> str:
    """
    Save test set predictions with probabilities and risk tiers.
    This is the output consumed by the optimization layer.
    """
    X_test = test_df[FEATURE_COLS]
    probs  = model.predict_proba(X_test)[:, 1]

    out = test_df[["appointment_id", "appointment_date", TARGET_COL]].copy()
    out["no_show_prob"]  = probs
    out["predicted_label"] = (probs >= best_threshold).astype(int)
    out["risk_tier"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test_predictions.csv")
    out.to_csv(path, index=False)
    log.info(f"Test predictions saved to: {path}")

    log.info("\n  Risk tier breakdown (test set):")
    tier_counts = out["risk_tier"].value_counts()
    for tier, count in tier_counts.items():
        log.info(f"    {tier:<8} {count:>6,}  ({count/len(out)*100:.1f}%)")

    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def run(data_path:  str = "data/processed/model_ready.csv",
        out_dir:    str = "outputs/results",
        fig_dir:    str = "outputs/figures") -> dict:
    """
    Full training pipeline.
    Returns metrics dict for the best model (XGBoost).
    """
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = load_and_split(data_path)

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline = build_baseline(X_train, y_train)
    evaluate(baseline, X_val, y_val, "Validation", "Logistic Regression")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    # Pass val set for early stopping
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    log.info(f"XGBoost best iteration: {xgb.best_iteration}")

    val_metrics  = evaluate(xgb, X_val,  y_val,  "Validation", "XGBoost")
    test_metrics = evaluate(xgb, X_test, y_test, "Test",       "XGBoost")

    evaluate_calibration(xgb, X_test, y_test, "XGBoost", fig_dir)
    plot_feature_importance(xgb, fig_dir)

    save_model(xgb, test_metrics, out_dir)
    save_predictions(xgb, test_df, test_metrics["best_threshold"], out_dir)

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train no-show prediction model.")
    parser.add_argument("--data_path", default="data/processed/model_ready.csv")
    parser.add_argument("--out_dir",   default="outputs/results")
    parser.add_argument("--fig_dir",   default="outputs/figures")
    args = parser.parse_args()

    metrics = run(data_path=args.data_path, out_dir=args.out_dir, fig_dir=args.fig_dir)
    log.info(f"\nFinal test ROC-AUC : {metrics['roc_auc']:.4f}")
    log.info(f"Final test PR-AUC  : {metrics['pr_auc']:.4f}")
    sys.exit(0)