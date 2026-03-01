"""
scheduler.py
------------
Gurobi solver layer for the two optimization models defined in formulation.py.
Consumes model dicts from formulation.py and returns clean DataFrames.
"""

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _get_solver():
    try:
        import gurobipy
        log.info("Solver: Gurobi ✓")
        return "gurobi"
    except (ImportError, Exception):
        try:
            import pulp
            log.warning("Gurobi unavailable — falling back to PuLP (CBC solver).")
            return "pulp"
        except ImportError:
            raise RuntimeError(
                "No solver found. Install gurobipy or pulp:\n"
                "  pip install pulp --break-system-packages"
            )


def solve_reminder_model(model: dict, solver: str = None) -> pd.DataFrame:
    if solver is None:
        solver = _get_solver()

    values = model["values"]
    cap    = model["call_capacity"]
    n      = model["n_patients"]

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
    m.setParam("OutputFlag", 0)

    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(values[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE,
    )

    m.addConstr(gp.quicksum(x[i] for i in range(n)) <= cap, "capacity")
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        log.warning(f"Gurobi did not find optimal solution (status={m.Status}). Falling back to greedy.")
        from src.optimization.formulation import reminder_baseline
        return reminder_baseline(model)

    selected = [i for i in range(n) if x[i].X > 0.5]
    selected.sort(key=lambda i: values[i], reverse=True)

    return _build_reminder_result(model, selected, values, method="gurobi_lp")


def _solve_reminder_pulp(model, values, cap, n) -> pd.DataFrame:
    import pulp

    prob = pulp.LpProblem("reminder_allocation", pulp.LpMaximize)
    x    = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(values[i] * x[i] for i in range(n))
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


def solve_waitlist_model(model: dict, solver: str = None) -> pd.DataFrame:
    if solver is None:
        solver = _get_solver()

    scores     = model["scores"]
    n_patients = model["n_patients"]
    n_slots    = model["n_slots"]

    log.info(f"Solving waitlist matching: {n_patients} patients, {n_slots} slots, solver={solver}")

    if n_patients == 0 or n_slots == 0:
        log.warning("Empty waitlist or no open slots — skipping.")
        return pd.DataFrame()

    if solver == "gurobi":
        result = _solve_waitlist_gurobi(model, scores, n_patients, n_slots)
    else:
        result = _solve_waitlist_pulp(model, scores, n_patients, n_slots)

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

    x = m.addVars(n_patients, n_slots, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(scores[i, s] * x[i, s]
                    for i in range(n_patients)
                    for s in range(n_slots)),
        GRB.MAXIMIZE,
    )

    for i in range(n_patients):
        m.addConstr(gp.quicksum(x[i, s] for s in range(n_slots)) <= 1, f"patient_{i}_once")

    for s in range(n_slots):
        m.addConstr(gp.quicksum(x[i, s] for i in range(n_patients)) <= 1, f"slot_{s}_once")

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        log.warning(f"Gurobi waitlist did not find optimal (status={m.Status}). Falling back to greedy.")
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

    prob += pulp.lpSum(
        scores[i, s] * x[i][s]
        for i in range(n_patients)
        for s in range(n_slots)
    )

    for i in range(n_patients):
        prob += pulp.lpSum(x[i][s] for s in range(n_slots)) <= 1

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
    if result.empty:
        return result

    slots_meta = model["open_slots"][["slot_id", "appointment_time"]].copy()
    wait_meta  = model["waitlisted"][
        ["patient_id", "current_appointment_date", "patient_prior_noshows_rate"]
    ].copy()

    result = result.merge(slots_meta, on="slot_id",   how="left")
    result = result.merge(wait_meta,  on="patient_id", how="left")

    return result


def save_results(
    reminder_calls:   pd.DataFrame,
    waitlist_matches: pd.DataFrame,
    health:           dict,
    target_date:      str,
    out_dir:          str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    date_str = target_date.replace("-", "")

    if not reminder_calls.empty:
        path = os.path.join(out_dir, f"reminder_calls_{date_str}.csv")
        reminder_calls.to_csv(path, index=False)
        log.info(f"Reminder calls saved to: {path}")

    if not waitlist_matches.empty:
        path = os.path.join(out_dir, f"waitlist_matches_{date_str}.csv")
        waitlist_matches.to_csv(path, index=False)
        log.info(f"Waitlist matches saved to: {path}")

    health_df = pd.DataFrame([{**health, "date": target_date}])
    path = os.path.join(out_dir, f"schedule_health_{date_str}.csv")
    health_df.to_csv(path, index=False)
    log.info(f"Schedule health saved to: {path}")