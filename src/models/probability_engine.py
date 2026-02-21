"""
probability_engine.py
---------------------
Hybrid no-show probability model.

Context:
    The ML classifier (classifier.py) achieved ROC-AUC ≈ 0.50 on held-out data,
    indicating the available features lack sufficient predictive signal for a
    learned model to outperform chance. This is a legitimate finding — no-show
    behaviour in this dataset is distributed roughly uniformly across all feature
    slices (age, scheduling interval, time of day, insurance provider).

    This module implements a statistically grounded fallback:
    - Patients with sufficient history → use their personal historical no-show rate
    - Patients with limited history   → blend personal rate with population base rate
    - First-time patients             → use population base rate

    This approach is transparent, explainable, and defensible. It also mirrors
    how clinics reason about risk in practice.

Usage:
    python src/models/probability_engine.py
    python src/models/probability_engine.py --features_path data/processed/features.csv
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Population base rate — computed from full model-eligible dataset (EDA)
BASE_NO_SHOW_RATE = 0.074

# Minimum prior appointments before trusting a patient's personal rate fully
HISTORY_THRESHOLD_FULL  = 3   # 3+ appointments → full personal rate
HISTORY_THRESHOLD_BLEND = 1   # 1-2 appointments → blended rate

# Blend weight for limited-history patients
# 0.5 = equal weight between personal rate and base rate
BLEND_WEIGHT = 0.5

# Risk tier boundaries (used for reminder prioritisation downstream)
RISK_TIERS = {
    "LOW":    (0.00, 0.10),
    "MEDIUM": (0.10, 0.20),
    "HIGH":   (0.20, 1.00),
}


# ── Core probability function ─────────────────────────────────────────────────

def compute_no_show_probability(
    prior_appointments: float,
    prior_noshows: float,
    base_rate: float = BASE_NO_SHOW_RATE,
    history_threshold_full: int = HISTORY_THRESHOLD_FULL,
    blend_weight: float = BLEND_WEIGHT,
) -> float:
    """
    Compute no-show probability for a single appointment.

    Args:
        prior_appointments : Number of prior appointments for this patient
                             at the time of this appointment (leakage-safe).
        prior_noshows      : Number of prior no-shows for this patient.
        base_rate          : Population-level no-show rate (default 7.4%).
        history_threshold_full : Min prior appointments to use personal rate fully.
        blend_weight       : Weight on personal rate for limited-history blend.

    Returns:
        Estimated no-show probability (float between 0 and 1).

    Logic:
        - 0 prior appointments  → base_rate
        - 1-2 prior appointments → blend_weight * personal_rate + (1-blend_weight) * base_rate
        - 3+ prior appointments → personal_rate
    """
    prior_appointments = int(prior_appointments)
    prior_noshows      = int(prior_noshows)

    if prior_appointments == 0:
        return base_rate

    personal_rate = prior_noshows / prior_appointments

    if prior_appointments >= history_threshold_full:
        return personal_rate
    else:
        # Limited history — blend toward base rate
        return blend_weight * personal_rate + (1 - blend_weight) * base_rate


def assign_risk_tier(prob: float) -> str:
    """Assign a risk tier label based on no-show probability."""
    for tier, (low, high) in RISK_TIERS.items():
        if low <= prob < high:
            return tier
    return "HIGH"  # catch prob == 1.0


# ── Vectorised application ────────────────────────────────────────────────────

def compute_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the hybrid probability model to a DataFrame.

    Expects columns: patient_prior_appointments, patient_prior_noshows
    Adds columns:    no_show_prob, risk_tier, probability_source

    probability_source documents which branch of the logic was used —
    useful for transparency and debugging.
    """
    log.info("Computing no-show probabilities...")

    required = ["patient_prior_appointments", "patient_prior_noshows"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. Run features.py first.")

    df = df.copy()

    # Vectorised probability computation
    prior_appts   = df["patient_prior_appointments"].fillna(0).astype(int)
    prior_noshows = df["patient_prior_noshows"].fillna(0).astype(int)
    personal_rate = np.where(prior_appts > 0, prior_noshows / prior_appts, BASE_NO_SHOW_RATE)

    df["no_show_prob"] = np.where(
        prior_appts == 0,
        BASE_NO_SHOW_RATE,
        np.where(
            prior_appts >= HISTORY_THRESHOLD_FULL,
            personal_rate,
            BLEND_WEIGHT * personal_rate + (1 - BLEND_WEIGHT) * BASE_NO_SHOW_RATE,
        )
    )

    # Source label for transparency
    df["probability_source"] = np.where(
        prior_appts == 0,
        "base_rate",
        np.where(
            prior_appts >= HISTORY_THRESHOLD_FULL,
            "personal_rate",
            "blended",
        )
    )

    # Risk tier
    df["risk_tier"] = df["no_show_prob"].apply(assign_risk_tier)

    # Summary
    source_counts = df["probability_source"].value_counts()
    tier_counts   = df["risk_tier"].value_counts()

    log.info("  Probability source breakdown:")
    for source, count in source_counts.items():
        log.info(f"    {source:<15} {count:>7,}  ({count/len(df)*100:.1f}%)")

    log.info("  Risk tier breakdown:")
    for tier, count in tier_counts.items():
        log.info(f"    {tier:<8} {count:>7,}  ({count/len(df)*100:.1f}%)")

    log.info(f"  Mean predicted no-show probability: {df['no_show_prob'].mean():.3f}")
    log.info(f"  (Population base rate: {BASE_NO_SHOW_RATE:.3f})")

    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame) -> dict:
    """
    Evaluate the hybrid engine against ground truth where available.
    Requires a 'no_show' column (0/1).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    if "no_show" not in df.columns:
        log.warning("No ground truth 'no_show' column found — skipping evaluation.")
        return {}

    clean = df[["no_show", "no_show_prob"]].dropna()
    y     = clean["no_show"]
    probs = clean["no_show_prob"]

    roc_auc  = roc_auc_score(y, probs)
    pr_auc   = average_precision_score(y, probs)
    brier    = brier_score_loss(y, probs)

    log.info("\n── Hybrid Engine Evaluation ─────────────────────────────────")
    log.info(f"  ROC-AUC     : {roc_auc:.4f}")
    log.info(f"  PR-AUC      : {pr_auc:.4f}")
    log.info(f"  Brier Score : {brier:.4f}  (lower = better, naive = {BASE_NO_SHOW_RATE*(1-BASE_NO_SHOW_RATE):.4f})")
    log.info("─────────────────────────────────────────────────────────────")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier}


def plot_calibration(df: pd.DataFrame, out_dir: str) -> None:
    """
    Calibration plot — are our probabilities meaningful?
    Bins appointments by predicted probability and checks actual no-show rate.
    """
    import matplotlib.pyplot as plt

    if "no_show" not in df.columns:
        return

    df = df[["no_show", "no_show_prob"]].dropna().copy()
    df["prob_bin"] = pd.cut(df["no_show_prob"], bins=10)

    calibration = df.groupby("prob_bin", observed=True).agg(
        mean_pred=("no_show_prob", "mean"),
        actual_rate=("no_show", "mean"),
        count=("no_show", "count"),
    ).reset_index()

    plt.figure(figsize=(7, 5))
    plt.plot(calibration["mean_pred"], calibration["actual_rate"], "s-",
             label="Hybrid Engine", color="steelblue")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual No-Show Rate")
    plt.title("Calibration Curve — Hybrid Probability Engine")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calibration_hybrid_engine.png")
    plt.savefig(path)
    plt.close()
    log.info(f"  Calibration plot saved to: {path}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, out_dir: str) -> str:
    """Save the scored DataFrame to outputs/results/probabilities.csv."""
    os.makedirs(out_dir, exist_ok=True)

    keep_cols = [
        "appointment_id", "appointment_date", "patient_id",
        "no_show_prob", "risk_tier", "probability_source",
        "patient_prior_appointments", "patient_prior_noshows",
    ]
    # Add no_show ground truth if present
    if "no_show" in df.columns:
        keep_cols.append("no_show")

    out = df[[c for c in keep_cols if c in df.columns]]
    path = os.path.join(out_dir, "probabilities.csv")
    out.to_csv(path, index=False)
    log.info(f"Probabilities saved to: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def run(features_path: str = "data/processed/features.csv",
        out_dir:       str = "outputs/results",
        fig_dir:       str = "outputs/figures") -> pd.DataFrame:
    """
    Full probability engine pipeline.
    Returns scored DataFrame (used by optimization layer).
    """
    log.info(f"Loading features from: {features_path}")
    df = pd.read_csv(
        features_path,
        dtype={"appointment_id": str, "patient_id": str, "slot_id": str},
        low_memory=False,
    )
    log.info(f"  Loaded {len(df):,} rows")

    df = compute_probabilities(df)
    metrics = evaluate(df)
    plot_calibration(df, fig_dir)
    save(df, out_dir)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid no-show probability engine.")
    parser.add_argument("--features_path", default="data/processed/features.csv")
    parser.add_argument("--out_dir",       default="outputs/results")
    parser.add_argument("--fig_dir",       default="outputs/figures")
    args = parser.parse_args()

    run(features_path=args.features_path, out_dir=args.out_dir, fig_dir=args.fig_dir)
    sys.exit(0)