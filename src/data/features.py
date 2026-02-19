"""
features.py
-----------
Builds the feature matrix from master.csv for no-show prediction.
Writes the result to data/processed/features.csv.

Usage:
    python src/data/features.py
    python src/data/features.py --master_path data/processed/master.csv --out_dir data/processed
"""

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Statuses we include in the feature matrix (ground truth only)
MODEL_STATUSES = {"attended", "did not attend"}

# Target: 1 = no-show, 0 = attended
TARGET_COL = "no_show"

# Final feature columns (in order) — update this as features are added/removed
FEATURE_COLS = [
    # Appointment-level
    "scheduling_interval",
    "appointment_hour",
    "appointment_day_of_week",
    "appointment_month",
    "is_morning",
    "is_monday",
    "is_friday",
    # Patient-level
    "age",
    "sex_encoded",
    "insurance_encoded",
    "patient_prior_noshows",
    "patient_prior_noshows_rate",
    "patient_prior_appointments",
    # Schedule-level
    "daily_slot_utilization",
    "rolling_7d_noshows_rate",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time_to_hour(series: pd.Series) -> pd.Series:
    """Extract hour (0-23) from a time or string column. NaT/NaN-safe."""
    parsed = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    return parsed.dt.hour


def _date_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a date-object column to datetime64 for .dt accessor support."""
    return pd.to_datetime(series)


# ── Feature builders ──────────────────────────────────────────────────────────

def build_appointment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appointment-level time and scheduling features.
    All derived from columns already present in master.csv.
    """
    log.info("Building appointment-level features...")

    appt_dt = _date_to_datetime(df["appointment_date"])

    df["appointment_hour"]        = _parse_time_to_hour(df["appointment_time"])
    df["appointment_day_of_week"] = appt_dt.dt.dayofweek   # 0=Mon, 6=Sun
    df["appointment_month"]       = appt_dt.dt.month
    df["is_morning"]              = (df["appointment_hour"] < 12).astype(int)
    df["is_monday"]               = (df["appointment_day_of_week"] == 0).astype(int)
    df["is_friday"]               = (df["appointment_day_of_week"] == 4).astype(int)

    # scheduling_interval is already in master — just validate it's numeric
    df["scheduling_interval"] = pd.to_numeric(df["scheduling_interval"], errors="coerce")

    return df


def build_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level features including leakage-safe historical no-show rates.

    Historical features are computed using only appointments BEFORE the current
    one for each patient (sorted by appointment_date). The current row is never
    included in its own history.
    """
    log.info("Building patient-level features...")

    # ── Encode categoricals ───────────────────────────────────────────────────

    df["sex_encoded"] = df["sex"].map({"Male": 0, "Female": 1}).fillna(-1).astype(int)

    # Frequency-encode insurance: rank providers by count, rarer = higher int
    insurance_counts = df["insurance"].value_counts()
    df["insurance_encoded"] = df["insurance"].map(insurance_counts).fillna(0).astype(int)

    # ── Rolling historical no-show features ──────────────────────────────────
    # Sort by patient + date so expanding windows are chronological
    df = df.sort_values(["patient_id", "appointment_date"]).reset_index(drop=True)

    # Binary no-show flag for the history computation
    df["_noshowed"] = (df["status"] == "did not attend").astype(int)

    # Expanding sum/count per patient, shifted by 1 so current row is excluded
    grp = df.groupby("patient_id")["_noshowed"]

    df["patient_prior_noshows"]      = grp.transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    df["patient_prior_appointments"] = grp.transform(lambda x: x.shift(1).expanding().count()).fillna(0)

    df["patient_prior_noshows_rate"] = np.where(
        df["patient_prior_appointments"] > 0,
        df["patient_prior_noshows"] / df["patient_prior_appointments"],
        0.0,   # no history → default to 0
    )

    df.drop(columns=["_noshowed"], inplace=True)

    return df


def build_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slot and schedule-level features capturing how busy a given day is
    and recent trends in no-show rates.
    """
    log.info("Building schedule-level features...")

    # ── Daily slot utilization ────────────────────────────────────────────────
    daily_booked = df.groupby("appointment_date")["slot_id"].count().rename("daily_booked")
    daily_total  = df.groupby("appointment_date")["slot_id"].nunique().rename("daily_total")
    daily_util   = (daily_booked / daily_total).rename("daily_slot_utilization")

    df = df.merge(daily_util.reset_index(), on="appointment_date", how="left")

    # ── 7-day rolling no-show rate ────────────────────────────────────────────
    # Compute daily no-show rate on attended + did not attend rows only,
    # then roll over 7 days and shift by 1 so today's data isn't included.
    daily_noshows = (
        df[df["status"].isin(MODEL_STATUSES)]
        .groupby("appointment_date")
        .apply(lambda x: (x["status"] == "did not attend").mean(), include_groups=False)
        .rename("_daily_noshows_rate")
        .reset_index()
    )
    daily_noshows["appointment_date"] = pd.to_datetime(daily_noshows["appointment_date"])
    daily_noshows = daily_noshows.sort_values("appointment_date").set_index("appointment_date")

    rolling_7d = (
        daily_noshows["_daily_noshows_rate"]
        .rolling("7D", min_periods=1)
        .mean()
        .shift(1)
        .rename("rolling_7d_noshows_rate")
        .reset_index()
    )

    df["_appt_date_dt"] = pd.to_datetime(df["appointment_date"])
    df = df.merge(rolling_7d, left_on="_appt_date_dt", right_on="appointment_date",
                  how="left", suffixes=("", "_r"))
    df.drop(columns=["_appt_date_dt", "appointment_date_r"], inplace=True, errors="ignore")
    df["rolling_7d_noshows_rate"] = df["rolling_7d_noshows_rate"].fillna(0.0)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the binary target column and filter to model-eligible rows only.
    Excludes cancelled, unknown, and scheduled statuses.
    """
    log.info("Building target variable...")

    df = df[df["status"].isin(MODEL_STATUSES)].copy()
    df[TARGET_COL] = (df["status"] == "did not attend").astype(int)

    no_show_rate = df[TARGET_COL].mean() * 100
    log.info(f"  Model rows (attended + did not attend): {len(df):,}")
    log.info(f"  No-show rate in model set: {no_show_rate:.1f}%")

    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> None:
    log.info("Validating feature matrix...")

    missing = df[FEATURE_COLS].isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        log.warning("  Missing values detected:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            log.warning(f"    {col:<35} {count:>6,}  ({pct:.1f}%)")
    else:
        log.info("  No missing values in feature columns. ✓")

    log.info(f"  Feature matrix shape: {df[FEATURE_COLS].shape[0]:,} rows × {len(FEATURE_COLS)} features")


# ── Summary ───────────────────────────────────────────────────────────────────

def summary(df: pd.DataFrame) -> None:
    log.info("\n── Feature Summary ──────────────────────────────────────")
    log.info(f"  Rows            : {len(df):,}")
    log.info(f"  Features        : {len(FEATURE_COLS)}")
    log.info(f"  Target (no_show): {df[TARGET_COL].sum():,} positive ({df[TARGET_COL].mean()*100:.1f}%)")
    log.info(f"  Date range      : {df['appointment_date'].min()} → {df['appointment_date'].max()}")
    log.info("\n  Feature stats:")
    log.info(df[FEATURE_COLS].describe().to_string())
    log.info("─────────────────────────────────────────────────────────\n")


# ── Writer ────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Save full feature matrix (all cols, for reference)
    full_path = os.path.join(out_dir, "features.csv")
    df.to_csv(full_path, index=False)
    log.info(f"Full feature table saved to: {full_path}")

    # Save model-ready matrix (features + target + appointment_id + appointment_date only)
    model_cols = ["appointment_id", "appointment_date", TARGET_COL] + FEATURE_COLS
    model_path = os.path.join(out_dir, "model_ready.csv")
    df[model_cols].to_csv(model_path, index=False)
    log.info(f"Model-ready table saved to:  {model_path}")

    return full_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run(master_path: str = "data/processed/master.csv",
        out_dir:     str = "data/processed") -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns the feature DataFrame (useful when called as a module).
    """
    log.info(f"Loading master table from: {master_path}")
    df = pd.read_csv(
        master_path,
        dtype={"appointment_id": str, "slot_id": str, "patient_id": str},
        low_memory=False,
    )
    log.info(f"  Loaded {len(df):,} rows")

    df = build_appointment_features(df)
    df = build_patient_features(df)
    df = build_schedule_features(df)
    df = build_target(df)

    validate_features(df)
    summary(df)
    save(df, out_dir)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix from master.csv.")
    parser.add_argument("--master_path", default="data/processed/master.csv")
    parser.add_argument("--out_dir",     default="data/processed")
    args = parser.parse_args()

    run(master_path=args.master_path, out_dir=args.out_dir)
    sys.exit(0)