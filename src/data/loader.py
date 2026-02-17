import argparse
import logging
import os
import sys

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# ID columns that must stay zero-padded strings (e.g. "00001", not 1)
ID_COLS = ["patient_id", "appointment_id", "slot_id"]

# Expected statuses in appointments.status
VALID_STATUSES = {"attended", "did not attend", "cancelled"}

# Columns that must be non-null for attended appointments
ATTENDED_REQUIRED = ["check_in_time", "start_time", "end_time", "appointment_duration", "waiting_time"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_patients(path: str) -> pd.DataFrame:
    """
    Load patients.csv.

    Columns: patient_id, name, sex, dob, insurance
    """
    log.info(f"Loading patients from: {path}")

    df = pd.read_csv(
        path,
        dtype={"patient_id": str},  # preserve zero-padding
    )

    # Zero-pad patient_id to 5 characters to match appointments table
    df["patient_id"] = df["patient_id"].str.zfill(5)

    # Parse date of birth
    df["dob"] = pd.to_datetime(df["dob"], format="%Y-%m-%d").dt.date

    # Normalise sex casing
    df["sex"] = df["sex"].str.strip().str.title()

    log.info(f"  Patients loaded: {len(df):,} rows")
    _check_duplicates(df, "patient_id", "patients")

    return df


def load_slots(path: str) -> pd.DataFrame:
    """
    Load slots.csv.

    Columns: slot_id, appointment_date, appointment_time, is_available
    """
    log.info(f"Loading slots from: {path}")

    df = pd.read_csv(
        path,
        dtype={"slot_id": str},
    )

    df["slot_id"] = df["slot_id"].str.zfill(7)

    # Parse date and time
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], format="%Y-%m-%d").dt.date
    df["appointment_time"] = pd.to_datetime(df["appointment_time"], format="%H:%M:%S").dt.time

    # Parse is_available — comes in as string "True"/"False"
    df["is_available"] = df["is_available"].map({"True": True, "False": False}).astype(bool)

    log.info(f"  Slots loaded: {len(df):,} rows")
    _check_duplicates(df, "slot_id", "slots")

    return df


def load_appointments(path: str) -> pd.DataFrame:
    """
    Load appointments.csv.

    Columns: appointment_id, slot_id, scheduling_date, appointment_date,
             appointment_time, scheduling_interval, status, check_in_time,
             appointment_duration, start_time, end_time, waiting_time,
             patient_id, sex, age, age_group
    """
    log.info(f"Loading appointments from: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "appointment_id": str,
            "slot_id":        str,
            "patient_id":     str,
        },
    )

    # Zero-pad IDs
    df["appointment_id"] = df["appointment_id"].str.zfill(7)
    df["slot_id"]        = df["slot_id"].str.zfill(7)
    df["patient_id"]     = df["patient_id"].str.zfill(5)

    # Parse dates
    df["scheduling_date"]   = pd.to_datetime(df["scheduling_date"],   format="%Y-%m-%d").dt.date
    df["appointment_date"]  = pd.to_datetime(df["appointment_date"],  format="%Y-%m-%d").dt.date

    # Parse time columns — these are nullable (empty for no-shows/cancellations)
    for col in ["appointment_time", "check_in_time", "start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce").dt.time

    # Numeric columns
    df["scheduling_interval"]   = pd.to_numeric(df["scheduling_interval"],   errors="coerce")
    df["appointment_duration"]  = pd.to_numeric(df["appointment_duration"],  errors="coerce")
    df["waiting_time"]          = pd.to_numeric(df["waiting_time"],          errors="coerce")
    df["age"]                   = pd.to_numeric(df["age"],                   errors="coerce").astype("Int64")

    # Normalise status and sex casing
    df["status"] = df["status"].str.strip().str.lower()
    df["sex"]    = df["sex"].str.strip().str.title()

    log.info(f"  Appointments loaded: {len(df):,} rows")
    _check_duplicates(df, "appointment_id", "appointments")

    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _check_duplicates(df: pd.DataFrame, key_col: str, table_name: str) -> None:
    dupes = df[key_col].duplicated().sum()
    if dupes:
        log.warning(f"  [{table_name}] {dupes:,} duplicate {key_col} values found.")
    else:
        log.info(f"  [{table_name}] No duplicate {key_col} values. ✓")


def validate(appointments: pd.DataFrame, slots: pd.DataFrame, patients: pd.DataFrame) -> None:
    """
    Run referential integrity and business-logic checks.
    Logs warnings rather than raising so the pipeline can continue.
    """
    log.info("Running validation checks...")

    # 1. Referential integrity: slot_id
    orphan_slots = ~appointments["slot_id"].isin(slots["slot_id"])
    if orphan_slots.sum():
        log.warning(f"  {orphan_slots.sum():,} appointments reference a slot_id not in slots.csv")
    else:
        log.info("  All slot_id references valid. ✓")

    # 2. Referential integrity: patient_id
    orphan_patients = ~appointments["patient_id"].isin(patients["patient_id"])
    if orphan_patients.sum():
        log.warning(f"  {orphan_patients.sum():,} appointments reference a patient_id not in patients.csv")
    else:
        log.info("  All patient_id references valid. ✓")

    # 3. Valid status values
    bad_status = ~appointments["status"].isin(VALID_STATUSES)
    if bad_status.sum():
        log.warning(f"  {bad_status.sum():,} appointments have unexpected status values: "
                    f"{appointments.loc[bad_status, 'status'].unique()}")
    else:
        log.info("  All status values valid. ✓")

    # 4. Attended appointments should have timing data
    attended = appointments["status"] == "attended"
    for col in ATTENDED_REQUIRED:
        missing = attended & appointments[col].isna()
        if missing.sum():
            log.warning(f"  {missing.sum():,} attended appointments missing '{col}'")
    if attended.sum():
        log.info("  Attended appointment timing checks complete. ✓")

    # 5. appointment_date consistency between appointments and slots
    merged_check = appointments[["slot_id", "appointment_date"]].merge(
        slots[["slot_id", "appointment_date"]].rename(columns={"appointment_date": "slot_date"}),
        on="slot_id",
        how="left",
    )
    date_mismatch = merged_check["appointment_date"] != merged_check["slot_date"]
    if date_mismatch.sum():
        log.warning(f"  {date_mismatch.sum():,} appointments have appointment_date "
                    f"inconsistent with their slot's date")
    else:
        log.info("  appointment_date consistent with slot dates. ✓")


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge(appointments: pd.DataFrame, slots: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the three tables into a single master DataFrame.

    Join strategy:
        appointments
            LEFT JOIN slots    ON slot_id       (adds is_available; date/time already in appointments)
            LEFT JOIN patients ON patient_id    (adds name, dob, insurance)

    Left joins preserve all appointments even if a slot or patient record
    is missing (orphans flagged in validation above).
    """
    log.info("Merging tables...")

    # Drop redundant date/time cols from slots before joining (already in appointments)
    slots_slim = slots[["slot_id", "is_available"]].copy()

    master = (
        appointments
        .merge(slots_slim,  on="slot_id",   how="left")
        .merge(
            patients[["patient_id", "name", "dob", "insurance"]],
            on="patient_id",
            how="left",
            suffixes=("", "_patient"),
        )
    )

    # Drop the duplicate sex column from appointments (patients has the authoritative one)
    # appointments also carries sex/age directly — we keep those and note they may differ
    # from patient records for older entries; keep both for now, flag in EDA.

    log.info(f"  Master table shape: {master.shape[0]:,} rows × {master.shape[1]} columns")

    return master


# ── Writer ────────────────────────────────────────────────────────────────────

def save(master: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "master.csv")
    master.to_csv(out_path, index=False)
    log.info(f"Master table saved to: {out_path}")
    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def summary(master: pd.DataFrame) -> None:
    log.info("\n── Dataset Summary ──────────────────────────────────────")
    log.info(f"  Total appointments : {len(master):,}")
    log.info(f"  Unique patients    : {master['patient_id'].nunique():,}")
    log.info(f"  Unique slots       : {master['slot_id'].nunique():,}")
    log.info(f"  Date range         : {master['appointment_date'].min()} → {master['appointment_date'].max()}")
    log.info(f"\n  Status breakdown:")
    counts = master["status"].value_counts()
    for status, count in counts.items():
        pct = count / len(master) * 100
        log.info(f"    {status:<20} {count:>7,}  ({pct:.1f}%)")
    log.info("─────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> pd.DataFrame:
    """
    Full load → validate → merge → save pipeline.
    Returns the master DataFrame (useful when called as a module).
    """
    patients     = load_patients(os.path.join(raw_dir, "patients.csv"))
    slots        = load_slots(os.path.join(raw_dir, "slots.csv"))
    appointments = load_appointments(os.path.join(raw_dir, "appointments.csv"))

    validate(appointments, slots, patients)

    master = merge(appointments, slots, patients)

    summary(master)
    save(master, out_dir)

    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and merge appointment data.")
    parser.add_argument("--raw_dir", default="data/raw",       help="Directory containing raw CSVs")
    parser.add_argument("--out_dir", default="data/processed", help="Directory to write master.csv")
    args = parser.parse_args()

    master = run(raw_dir=args.raw_dir, out_dir=args.out_dir)
    sys.exit(0)

