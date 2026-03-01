"""
run_pipeline.py
---------------
End-to-end pipeline runner for the Hospital Appointment Optimizer.

Chains all stages in order:
    1. Load & merge raw CSVs          (loader.py)
    2. Feature engineering            (features.py)
    3. Compute no-show probabilities  (probability_engine.py)
    4. Run optimization for a date    (scheduler.py)
       a. Schedule health score
       b. Reminder allocation (Gurobi knapsack)
       c. Waitlist matching   (Gurobi assignment IP)

Usage:
    # Full pipeline from raw data for a target date
    python scripts/run_pipeline.py --date 2024-11-15

    # Skip data prep if master.csv / features.csv already exist
    python scripts/run_pipeline.py --date 2024-11-15 --skip_data_prep

    # Custom call capacity
    python scripts/run_pipeline.py --date 2024-11-15 --call_capacity 30
"""

import argparse
import logging
import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from src/ regardless of where script is called from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.loader              import run as run_loader
from src.data.features            import run as run_features
from src.models.probability_engine import run as run_probability_engine
from src.optimization.scheduler   import run_demo as run_optimization

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage(name: str):
    """Simple stage header printer."""
    log.info(f"\n{'─'*55}")
    log.info(f"  STAGE: {name}")
    log.info(f"{'─'*55}")


def run_pipeline(
    date:            str,
    raw_dir:         str  = "data/raw",
    processed_dir:   str  = "data/processed",
    results_dir:     str  = "outputs/results",
    figures_dir:     str  = "outputs/figures",
    call_capacity:   int  = 20,
    skip_data_prep:  bool = False,
) -> None:
    """
    Run the full pipeline for a given target date.

    Args:
        date           : Target date in YYYY-MM-DD format.
        raw_dir        : Directory containing raw CSVs.
        processed_dir  : Directory for processed data outputs.
        results_dir    : Directory for optimization outputs.
        figures_dir    : Directory for plots.
        call_capacity  : Max reminder calls for the target date.
        skip_data_prep : If True, skip loader + features + probabilities
                         and use existing processed files. Useful for
                         re-running optimization on a different date.
    """
    pipeline_start = time.time()

    log.info(f"\n{'='*55}")
    log.info(f"  HOSPITAL APPOINTMENT OPTIMIZER")
    log.info(f"  Target date   : {date}")
    log.info(f"  Call capacity : {call_capacity}")
    log.info(f"  Skip data prep: {skip_data_prep}")
    log.info(f"{'='*55}")

    # ── Stage 1: Load & merge ─────────────────────────────────────────────────
    master_path = os.path.join(processed_dir, "master.csv")

    if not skip_data_prep:
        stage("1 / 4 — Load & Merge Raw Data")
        t0 = time.time()
        run_loader(raw_dir=raw_dir, out_dir=processed_dir)
        log.info(f"  Completed in {time.time() - t0:.1f}s")
    else:
        if not os.path.exists(master_path):
            log.error(f"master.csv not found at {master_path}. "
                      "Run without --skip_data_prep first.")
            sys.exit(1)
        log.info("Stage 1 skipped — using existing master.csv")

    # ── Stage 2: Feature engineering ─────────────────────────────────────────
    features_path   = os.path.join(processed_dir, "features.csv")
    model_ready_path = os.path.join(processed_dir, "model_ready.csv")

    if not skip_data_prep:
        stage("2 / 4 — Feature Engineering")
        t0 = time.time()
        run_features(master_path=master_path, out_dir=processed_dir)
        log.info(f"  Completed in {time.time() - t0:.1f}s")
    else:
        if not os.path.exists(features_path):
            log.error(f"features.csv not found at {features_path}. "
                      "Run without --skip_data_prep first.")
            sys.exit(1)
        log.info("Stage 2 skipped — using existing features.csv")

    # ── Stage 3: Probability engine ───────────────────────────────────────────
    probs_path = os.path.join(results_dir, "probabilities.csv")

    if not skip_data_prep:
        stage("3 / 4 — No-Show Probability Engine")
        t0 = time.time()
        run_probability_engine(
            features_path=features_path,
            out_dir=results_dir,
            fig_dir=figures_dir,
        )
        log.info(f"  Completed in {time.time() - t0:.1f}s")
    else:
        if not os.path.exists(probs_path):
            log.error(f"probabilities.csv not found at {probs_path}. "
                      "Run without --skip_data_prep first.")
            sys.exit(1)
        log.info("Stage 3 skipped — using existing probabilities.csv")

    # ── Stage 4: Optimization ─────────────────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Hospital Appointment Optimizer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date", required=True,
        help="Target date for optimization (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--raw_dir", default="data/raw",
        help="Directory containing raw CSVs",
    )
    parser.add_argument(
        "--processed_dir", default="data/processed",
        help="Directory for processed data",
    )
    parser.add_argument(
        "--results_dir", default="outputs/results",
        help="Directory for optimization outputs",
    )
    parser.add_argument(
        "--figures_dir", default="outputs/figures",
        help="Directory for figures",
    )
    parser.add_argument(
        "--call_capacity", default=20, type=int,
        help="Max reminder calls for the target date",
    )
    parser.add_argument(
        "--skip_data_prep", action="store_true",
        help="Skip stages 1-3 and use existing processed files",
    )

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