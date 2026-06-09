"""
run_pipeline.py
---------------
End-to-end pipeline for the Hospital Appointment Optimizer.

Usage:
    python scripts/run_pipeline.py --date 2024-11-15
    python scripts/run_pipeline.py --date 2024-11-15 --skip_data_prep
    python scripts/run_pipeline.py --date 2024-11-15 --skip_data_prep --call_capacity 30
"""
import argparse
import logging
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd

from src.data.loader                import run as run_loader
from src.data.features              import run as run_features
from src.models.probability_engine  import run as run_probability_engine
from src.optimization.formulation   import (
    build_reminder_model,
    build_waitlist_model,
    compute_schedule_health,
    reminder_baseline,
)
from src.optimization.scheduler     import (
    solve_reminder_model,
    solve_waitlist_model,
    save_results,
)
from src.data.loader                import load_slots

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def stage(name: str):
    log.info(f"\n{'─'*55}")
    log.info(f"  STAGE: {name}")
    log.info(f"{'─'*55}")


def run_optimization(
    date:          str,
    probs_path:    str,
    master_path:   str,
    call_capacity: int,
    out_dir:       str,
):
    probs  = pd.read_csv(probs_path,  dtype={"appointment_id": str, "patient_id": str})
    master = pd.read_csv(master_path, dtype={"appointment_id": str, "patient_id": str,
                                              "slot_id": str}, low_memory=False)

    probs["appointment_date"]  = pd.to_datetime(probs["appointment_date"])
    master["appointment_date"] = pd.to_datetime(master["appointment_date"])

    target_dt  = pd.to_datetime(date)
    day_probs  = probs[probs["appointment_date"] == target_dt].copy()
    day_master = master[master["appointment_date"] == target_dt].copy()

    if day_probs.empty:
        log.error(f"No appointments found for {date}.")
        sys.exit(1)

    log.info(f"Appointments on {date}: {len(day_probs)}")

    slots = load_slots("data/raw/slots.csv")
    slots["appointment_date"] = pd.to_datetime(slots["appointment_date"])
    total_slots = len(slots[slots["appointment_date"] == target_dt])

    health = compute_schedule_health(day_probs, total_slots)

    reminder_model  = build_reminder_model(day_probs, call_capacity=call_capacity)
    reminder_greedy = reminder_baseline(reminder_model)
    reminder_lp     = solve_reminder_model(reminder_model)

    log.info(f"\n── Reminder Call List (top {call_capacity}) ────────────────")
    log.info(reminder_lp.to_string(index=False))
    log.info(f"\nGreedy vs LP expected recovery: "
             f"{reminder_greedy['expected_recovery'].sum():.4f} vs "
             f"{reminder_lp['expected_recovery'].sum():.4f}")

    high_risk  = day_probs[day_probs["risk_tier"] == "HIGH"].copy()
    high_risk  = high_risk.merge(
        day_master[["appointment_id", "appointment_time", "slot_id"]],
        on="appointment_id", how="left"
    )
    open_slots = high_risk[["slot_id", "appointment_time"]].copy()
    open_slots["appointment_date"] = date

    future = probs[
        (probs["appointment_date"] > target_dt) &
        (probs["appointment_date"] <= target_dt + pd.Timedelta(days=30)) &
        (probs["risk_tier"] == "MEDIUM")
    ].head(50).copy()

    future = future.merge(
        master[["appointment_id", "appointment_date"]].rename(
            columns={"appointment_date": "current_appointment_date"}
        ),
        on="appointment_id", how="left"
    )
    future["patient_prior_noshows_rate"] = future.get(
        "patient_prior_noshows_rate",
        pd.Series([0.074] * len(future))
    )

    waitlist_matches = pd.DataFrame()
    if not open_slots.empty and not future.empty:
        waitlist_model   = build_waitlist_model(open_slots, future, date)
        waitlist_matches = solve_waitlist_model(waitlist_model)

        if not waitlist_matches.empty:
            log.info(f"\n── Waitlist Matches ─────────────────────────────────")
            log.info(waitlist_matches.to_string(index=False))
    else:
        log.info("No open slots or waitlisted patients for this date.")

    save_results(reminder_lp, waitlist_matches, health, date, out_dir)

    log.info(f"\n{'='*55}")
    log.info(f"  SCHEDULE HEALTH SUMMARY — {date}")
    log.info(f"{'='*55}")
    log.info(f"  Raw utilization     : {health['raw_utilization']:.1%}")
    log.info(f"  Expected utilization: {health['expected_util']:.1%}")
    log.info(f"  High-risk appts     : {health['risk_HIGH']}")
    log.info(f"  Reminder calls      : {len(reminder_lp)}")
    log.info(f"  Waitlist matches    : {len(waitlist_matches)}")
    log.info(f"{'='*55}\n")


def run_pipeline(
    date:           str,
    raw_dir:        str  = "data/raw",
    processed_dir:  str  = "data/processed",
    results_dir:    str  = "outputs/results",
    figures_dir:    str  = "outputs/figures",
    call_capacity:  int  = 20,
    skip_data_prep: bool = False,
):
    pipeline_start = time.time()

    log.info(f"\n{'='*55}")
    log.info(f"  HOSPITAL APPOINTMENT OPTIMIZER")
    log.info(f"  Target date   : {date}")
    log.info(f"  Call capacity : {call_capacity}")
    log.info(f"  Skip data prep: {skip_data_prep}")
    log.info(f"{'='*55}")

    master_path   = os.path.join(processed_dir, "master.csv")
    features_path = os.path.join(processed_dir, "features.csv")
    probs_path    = os.path.join(results_dir,   "probabilities.csv")

    if not skip_data_prep:
        stage("1 / 4 — Load & Merge Raw Data")
        t0 = time.time()
        run_loader(raw_dir=raw_dir, out_dir=processed_dir)
        log.info(f"  Completed in {time.time() - t0:.1f}s")

        stage("2 / 4 — Feature Engineering")
        t0 = time.time()
        run_features(master_path=master_path, out_dir=processed_dir)
        log.info(f"  Completed in {time.time() - t0:.1f}s")

        stage("3 / 4 — No-Show Probability Engine")
        t0 = time.time()
        run_probability_engine(features_path=features_path, out_dir=results_dir, fig_dir=figures_dir)
        log.info(f"  Completed in {time.time() - t0:.1f}s")
    else:
        for path, name in [(master_path, "master.csv"),
                           (features_path, "features.csv"),
                           (probs_path, "probabilities.csv")]:
            if not os.path.exists(path):
                log.error(f"{name} not found at {path}. Run without --skip_data_prep first.")
                sys.exit(1)
        log.info("Stages 1–3 skipped — using existing processed files.")

    stage("4 / 4 — Schedule Optimization")
    t0 = time.time()
    run_optimization(
        date=date,
        probs_path=probs_path,
        master_path=master_path,
        call_capacity=call_capacity,
        out_dir=results_dir,
    )
    log.info(f"  Completed in {time.time() - t0:.1f}s")

    elapsed  = time.time() - pipeline_start
    date_str = date.replace("-", "")

    log.info(f"\n{'='*55}")
    log.info(f"  PIPELINE COMPLETE — {elapsed:.1f}s total")
    log.info(f"{'='*55}")
    log.info(f"  Outputs written to: {results_dir}/")
    log.info(f"    reminder_calls_{date_str}.csv")
    log.info(f"    waitlist_matches_{date_str}.csv")
    log.info(f"    schedule_health_{date_str}.csv")
    log.info(f"    probabilities.csv")
    log.info(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Hospital Appointment Optimizer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date",           required=True)
    parser.add_argument("--raw_dir",        default="data/raw")
    parser.add_argument("--processed_dir",  default="data/processed")
    parser.add_argument("--results_dir",    default="outputs/results")
    parser.add_argument("--figures_dir",    default="outputs/figures")
    parser.add_argument("--call_capacity",  default=20, type=int)
    parser.add_argument("--skip_data_prep", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        date=args.date,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        call_capacity=args.call_capacity,
        skip_data_prep=args.skip_data_prep,
    )
    sys.exit(0)
