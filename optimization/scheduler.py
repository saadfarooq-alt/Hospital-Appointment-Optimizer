"""
scheduler.py
------------
Gurobi solver layer for the two optimization models defined in formulation.py.

Consumes model dicts from formulation.py and returns clean DataFrames.
This module handles all Gurobi-specific code so the rest of the pipeline
stays solver-agnostic.

Usage:
    Called as a module by run_pipeline.py.
    Can also be run standalone for testing:
        python src/optimization/scheduler.py --date 2024-06-15
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Solver ────────────────────────────────────────────────────────────────────

def _get_solver():
    """
    Try to import Gurobi. Falls back to PuLP + CBC if unavailable.
    Returns a string indicating which solver is active.
    """
    try:
        import gurobipy  # noqa: F401
        log.info("Solver: Gurobi ✓")
        return "gurobi"
    except (ImportError, Exception):
        try:
            import pulp  # noqa: F401
            log.warning("Gurobi unavailable — falling back to PuLP (CBC solver).")
            return "pulp"
        except ImportError:
            raise RuntimeError(
                "No solver found. Install gurobipy or pulp:\n"
                "  pip install pulp --break-system-packages"
            )


# ── Reminder Allocation Solver ────────────────────────────────────────────────

def solve_reminder_model(model: dict, solver: str = None) -> pd.DataFrame:
    """
    Solve the reminder allocation knapsack.

    Args:
        model  : Dict from formulation.build_reminder_model()
        solver : 'gurobi' | 'pulp' | None (auto-detect)

    Returns:
        DataFrame with selected appointments, ranked by call priority.
        Columns: call_rank, appointment_id, patient_id,
                 no_show_prob, expected_recovery, method
    """
    if solver is None:
        solver = _get_solver()

    values   = model["values"]
    cap      = model["call_capacity"]
    n        = model["n_patients"]

    log.info(f"Solving reminder allocation: n={n}, capacity={cap}, solver={solver}")

    if solver == "gurobi":
        result = _solve_reminder_gurobi(model, values, cap, n)
    else:
        result = _solve_reminder_pulp(model, values, cap, n)

    log.info(f"  Selected {len(result)} appointments to call")
    log.info(f"  Total expected recovery: {result['expected_recovery'].sum():.3f}")

    return result


def _solve_reminder_gurobi(model, values, cap, n) -> pd.DataFrame:
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("reminder_allocation")
    m.setParam("OutputFlag", 0)  # suppress Gurobi console output

    # Decision variables: x[i] = 1 if we call patient i
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximise expected recoveries
    m.setObjective(
        gp.quicksum(values[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE,
    )

    # Capacity constraint
    m.addConstr(gp.quicksum(x[i] for i in range(n)) <= cap, "capacity")

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        log.warning(f"Gurobi did not find optimal solution (status={m.Status}). "
                    "Falling back to greedy.")
        from src.optimization.formulation import reminder_baseline
        return reminder_baseline(model)

    selected = [i for i in range(n) if x[i].X > 0.5]
    selected.sort(key=lambda i: values[i], reverse=True)

    return _build_reminder_result(model, selected, values, method="gurobi_lp")


def _solve_reminder_pulp(model, values, cap, n) -> pd.DataFrame:
    import pulp

    prob = pulp.LpProblem("reminder_allocation", pulp.LpMaximize)
    x    = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective
    prob += pulp.lpSum(values[i] * x[i] for i in range(n))

    # Capacity constraint
    prob += pulp.lpSum(x[i] for i in range(n)) <= cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        log.warning("PuLP did not find optimal solution. Falling back to greedy.")
        from src.optimization.formulation import reminder_baseline
        return reminder_baseline(model)

    selected = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    selected.sort(key=lambda i: values[i], reverse=True)

    return _build_reminder_result(model, selected, values, method="pulp_cbc")


def _build_reminder_result(model, selected_indices, values, method) -> pd.DataFrame:
    rows = []
    for rank, i in enumerate(selected_indices, start=1):
        rows.append({
            "call_rank":         rank,
            "appointment_id":    model["appointment_ids"][i],
            "patient_id":        model["patient_ids"][i],
            "no_show_prob":      round(model["no_show_probs"][i], 4),
            "expected_recovery": round(values[i], 4),
            "method":            method,
        })
    return pd.DataFrame(rows)


# ── Waitlist Matching Solver ──────────────────────────────────────────────────

def solve_waitlist_model(model: dict, solver: str = None) -> pd.DataFrame:
    """
    Solve the waitlist matching assignment IP.

    Args:
        model  : Dict from formulation.build_waitlist_model()
        solver : 'gurobi' | 'pulp' | None (auto-detect)

    Returns:
        DataFrame of patient-slot assignments.
        Columns: slot_id, patient_id, score, method,
                 appointment_time (from open_slots),
                 current_appointment_date (from waitlisted)
    """
    if solver is None:
        solver = _get_solver()

    scores     = model["scores"]
    n_patients = model["n_patients"]
    n_slots    = model["n_slots"]

    log.info(f"Solving waitlist matching: {n_patients} patients, "
             f"{n_slots} slots, solver={solver}")

    if n_patients == 0 or n_slots == 0:
        log.warning("Empty waitlist or no open slots — skipping.")
        return pd.DataFrame()

    if solver == "gurobi":
        result = _solve_waitlist_gurobi(model, scores, n_patients, n_slots)
    else:
        result = _solve_waitlist_pulp(model, scores, n_patients, n_slots)

    # Enrich with slot and patient metadata
    result = _enrich_waitlist_result(result, model)

    log.info(f"  Matched {len(result)} patient-slot pairs")
    if not result.empty:
        log.info(f"  Total assignment score: {result['score'].sum():.3f}")

    return result


def _solve_waitlist_gurobi(model, scores, n_patients, n_slots) -> pd.DataFrame:
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("waitlist_matching")
    m.setParam("OutputFlag", 0)

    # x[i, s] = 1 if patient i is assigned to slot s
    x = m.addVars(n_patients, n_slots, vtype=GRB.BINARY, name="x")

    # Objective: maximise total assignment score
    m.setObjective(
        gp.quicksum(scores[i, s] * x[i, s]
                    for i in range(n_patients)
                    for s in range(n_slots)),
        GRB.MAXIMIZE,
    )

    # Each patient assigned to at most 1 slot
    for i in range(n_patients):
        m.addConstr(
            gp.quicksum(x[i, s] for s in range(n_slots)) <= 1,
            f"patient_{i}_once",
        )

    # Each slot filled by at most 1 patient
    for s in range(n_slots):
        m.addConstr(
            gp.quicksum(x[i, s] for i in range(n_patients)) <= 1,
            f"slot_{s}_once",
        )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        log.warning(f"Gurobi waitlist did not find optimal (status={m.Status}). "
                    "Falling back to greedy.")
        from src.optimization.formulation import waitlist_baseline
        return waitlist_baseline(model)

    assignments = []
    for i in range(n_patients):
        for s in range(n_slots):
            if x[i, s].X > 0.5:
                assignments.append({
                    "slot_id":    model["slot_ids"][s],
                    "patient_id": model["patient_ids"][i],
                    "score":      round(scores[i, s], 4),
                    "method":     "gurobi_ip",
                })

    return pd.DataFrame(assignments)


def _solve_waitlist_pulp(model, scores, n_patients, n_slots) -> pd.DataFrame:
    import pulp

    prob = pulp.LpProblem("waitlist_matching", pulp.LpMaximize)

    x = [[pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
          for s in range(n_slots)]
         for i in range(n_patients)]

    # Objective
    prob += pulp.lpSum(
        scores[i, s] * x[i][s]
        for i in range(n_patients)
        for s in range(n_slots)
    )

    # Each patient assigned to at most 1 slot
    for i in range(n_patients):
        prob += pulp.lpSum(x[i][s] for s in range(n_slots)) <= 1

    # Each slot filled by at most 1 patient
    for s in range(n_slots):
        prob += pulp.lpSum(x[i][s] for i in range(n_patients)) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    assignments = []
    for i in range(n_patients):
        for s in range(n_slots):
            if pulp.value(x[i][s]) > 0.5:
                assignments.append({
                    "slot_id":    model["slot_ids"][s],
                    "patient_id": model["patient_ids"][i],
                    "score":      round(scores[i, s], 4),
                    "method":     "pulp_cbc",
                })

    return pd.DataFrame(assignments)


def _enrich_waitlist_result(result: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Join slot times and patient details back onto the assignment result."""
    if result.empty:
        return result

    slots_meta = model["open_slots"][["slot_id", "appointment_time"]].copy()
    wait_meta  = model["waitlisted"][
        ["patient_id", "current_appointment_date", "patient_prior_noshows_rate"]
    ].copy()

    result = result.merge(slots_meta, on="slot_id",   how="left")
    result = result.merge(wait_meta,  on="patient_id", how="left")

    return result


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(
    reminder_calls:    pd.DataFrame,
    waitlist_matches:  pd.DataFrame,
    health:            dict,
    target_date:       str,
    out_dir:           str,
) -> None:
    """Save all optimization outputs for a given date."""
    os.makedirs(out_dir, exist_ok=True)
    date_str = target_date.replace("-", "")

    # Reminder call list
    if not reminder_calls.empty:
        path = os.path.join(out_dir, f"reminder_calls_{date_str}.csv")
        reminder_calls.to_csv(path, index=False)
        log.info(f"Reminder calls saved to: {path}")

    # Waitlist matches
    if not waitlist_matches.empty:
        path = os.path.join(out_dir, f"waitlist_matches_{date_str}.csv")
        waitlist_matches.to_csv(path, index=False)
        log.info(f"Waitlist matches saved to: {path}")

    # Schedule health
    health_df = pd.DataFrame([{**health, "date": target_date}])
    path = os.path.join(out_dir, f"schedule_health_{date_str}.csv")
    health_df.to_csv(path, index=False)
    log.info(f"Schedule health saved to: {path}")


# ── Demo / standalone run ─────────────────────────────────────────────────────

def run_demo(
    date:          str,
    probs_path:    str = "outputs/results/probabilities.csv",
    master_path:   str = "data/processed/master.csv",
    call_capacity: int = 20,
    out_dir:       str = "outputs/results",
):
    """
    Run both optimization models for a single target date.
    Loads probabilities.csv and master.csv, filters to the target date,
    and produces reminder calls + waitlist matches + health score.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.optimization.formulation import (
        build_reminder_model,
        build_waitlist_model,
        compute_schedule_health,
        reminder_baseline,
    )

    log.info(f"\n{'='*55}")
    log.info(f"  Running optimization for date: {date}")
    log.info(f"{'='*55}")

    # ── Load data ─────────────────────────────────────────────────────────────
    probs = pd.read_csv(probs_path,  dtype={"appointment_id": str, "patient_id": str})
    master = pd.read_csv(master_path, dtype={"appointment_id": str, "patient_id": str,
                                              "slot_id": str}, low_memory=False)

    probs["appointment_date"]  = pd.to_datetime(probs["appointment_date"])
    master["appointment_date"] = pd.to_datetime(master["appointment_date"])

    target_dt = pd.to_datetime(date)

    # Filter to target date
    day_probs  = probs[probs["appointment_date"] == target_dt].copy()
    day_master = master[master["appointment_date"] == target_dt].copy()

    if day_probs.empty:
        log.error(f"No appointments found for {date}. Try a different date.")
        return

    log.info(f"Appointments on {date}: {len(day_probs)}")

    # ── Schedule health ───────────────────────────────────────────────────────
    from src.data.loader import load_slots
    slots = load_slots("data/raw/slots.csv")
    slots["appointment_date"] = pd.to_datetime(slots["appointment_date"])
    total_slots = len(slots[slots["appointment_date"] == target_dt])

    health = compute_schedule_health(day_probs, total_slots)

    # ── Reminder allocation ───────────────────────────────────────────────────
    reminder_model  = build_reminder_model(day_probs, call_capacity=call_capacity)
    reminder_greedy = reminder_baseline(reminder_model)
    reminder_lp     = solve_reminder_model(reminder_model)

    log.info(f"\n── Reminder Call List (top {call_capacity}) ────────────────")
    log.info(reminder_lp.to_string(index=False))
    log.info(f"\nGreedy vs LP expected recovery: "
             f"{reminder_greedy['expected_recovery'].sum():.4f} vs "
             f"{reminder_lp['expected_recovery'].sum():.4f}")

    # ── Waitlist matching ─────────────────────────────────────────────────────
    # Simulate: treat HIGH-risk appointments as potential cancellations
    # and MEDIUM-risk patients on other dates as waitlisted
    high_risk = day_probs[day_probs["risk_tier"] == "HIGH"].copy()
    high_risk = high_risk.merge(
        day_master[["appointment_id", "appointment_time", "slot_id"]],
        on="appointment_id", how="left"
    )

    # Simulated open slots from high-risk appointments
    open_slots = high_risk[["slot_id", "appointment_time"]].copy()
    open_slots["appointment_date"] = date

    # Simulated waitlist: medium-risk patients from the next 30 days
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
            log.info(f"\n── Waitlist Matches ────────────────────────────────")
            log.info(waitlist_matches.to_string(index=False))
    else:
        log.info("No open slots or waitlisted patients for this date.")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(reminder_lp, waitlist_matches, health, date, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"  SCHEDULE HEALTH SUMMARY — {date}")
    log.info(f"{'='*55}")
    log.info(f"  Raw utilization     : {health['raw_utilization']:.1%}")
    log.info(f"  Expected utilization: {health['expected_util']:.1%}")
    log.info(f"  High-risk appts     : {health['risk_HIGH']}")
    log.info(f"  Reminder calls      : {len(reminder_lp)}")
    log.info(f"  Waitlist matches    : {len(waitlist_matches)}")
    log.info(f"{'='*55}\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run schedule optimization for a target date.")
    parser.add_argument("--date",          required=True,  help="Target date YYYY-MM-DD")
    parser.add_argument("--probs_path",    default="outputs/results/probabilities.csv")
    parser.add_argument("--master_path",   default="data/processed/master.csv")
    parser.add_argument("--call_capacity", default=20, type=int)
    parser.add_argument("--out_dir",       default="outputs/results")
    args = parser.parse_args()

    run_demo(
        date=args.date,
        probs_path=args.probs_path,
        master_path=args.master_path,
        call_capacity=args.call_capacity,
        out_dir=args.out_dir,
    )
    sys.exit(0)