"""
formulation.py
--------------
LP/IP mathematical formulations for clinical schedule optimization.

Two models:

1. Reminder Allocation (0-1 Knapsack)
   ─────────────────────────────────
   Given a day's appointments and a fixed staff call capacity N,
   decide which patients to call with a reminder to maximise the
   expected number of recovered appointments.

   Decision variable : x_i ∈ {0, 1}  — call patient i or not
   Objective         : maximise Σ (no_show_prob_i × recovery_rate) × x_i
   Constraint        : Σ x_i ≤ N  (call capacity)

   Intuition: prioritise high-risk patients first. A patient with a
   70% no-show probability is worth more than one at 8%, assuming
   a reminder call recovers a fixed proportion of at-risk patients.

2. Waitlist Matching (Assignment IP)
   ──────────────────────────────────
   When a slot opens (cancellation), assign the best waitlisted
   patient to fill it.

   Decision variable : x_{i,s} ∈ {0,1} — assign patient i to slot s
   Objective         : maximise Σ score(i,s) × x_{i,s}
     where score(i,s) = w1 × days_until_their_appt_normalised
                      + w2 × reliability_score (1 - personal_noshowrate)
   Constraints:
     - Each patient assigned to at most 1 slot
     - Each slot filled by at most 1 patient

   Intuition: prefer patients whose current appointment is far away
   (they benefit most from moving up) and who are reliable (low
   personal no-show rate).
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Fraction of at-risk patients recovered by a reminder call
# Conservative estimate — literature suggests 20-40% of no-shows
# can be recovered with a phone reminder
RECOVERY_RATE = 0.30

# Weights for waitlist matching score
# w1: how much we value moving a patient up vs w2: reliability
WAITLIST_W1 = 0.6   # days-until-appointment weight
WAITLIST_W2 = 0.4   # reliability weight

# Default daily call capacity if not specified
DEFAULT_CALL_CAPACITY = 20


# ── Reminder Allocation ───────────────────────────────────────────────────────

def build_reminder_model(
    appointments: pd.DataFrame,
    call_capacity: int = DEFAULT_CALL_CAPACITY,
    recovery_rate: float = RECOVERY_RATE,
) -> dict:
    """
    Build the reminder allocation knapsack LP.

    Args:
        appointments  : DataFrame with columns [appointment_id, patient_id,
                        no_show_prob, risk_tier]. One row per appointment
                        for the target date.
        call_capacity : Max number of reminder calls staff can make (N).
        recovery_rate : Fraction of at-risk patients recovered by a call.

    Returns:
        A model dict consumed by scheduler.solve_reminder_model():
        {
            "type"          : "reminder_knapsack",
            "n_patients"    : int,
            "call_capacity" : int,
            "values"        : np.ndarray   # expected recoveries per call
            "appointment_ids": list
            "patient_ids"   : list
            "no_show_probs" : np.ndarray
        }
    """
    required = ["appointment_id", "patient_id", "no_show_prob"]
    _check_columns(appointments, required, "build_reminder_model")

    df = appointments.copy().reset_index(drop=True)

    # Value of calling patient i = expected appointments recovered
    # = no_show_probability × recovery_rate
    # (calling a low-risk patient has near-zero expected value)
    values = (df["no_show_prob"] * recovery_rate).values

    n = len(df)
    log.info(f"Reminder model: {n} appointments, capacity={call_capacity}, "
             f"total expected recoverable={values.sum():.2f}")

    return {
        "type":            "reminder_knapsack",
        "n_patients":      n,
        "call_capacity":   call_capacity,
        "values":          values,
        "appointment_ids": df["appointment_id"].tolist(),
        "patient_ids":     df["patient_id"].tolist(),
        "no_show_probs":   df["no_show_prob"].values,
    }


def reminder_baseline(model: dict) -> pd.DataFrame:
    """
    Greedy baseline for reminder allocation (no Gurobi needed).
    Simply ranks by no_show_prob descending and takes the top N.
    Used to benchmark the LP solution.

    Returns a DataFrame with the selected appointments and call order.
    """
    probs = model["no_show_probs"]
    cap   = model["call_capacity"]

    ranked_idx = np.argsort(probs)[::-1][:cap]

    result = pd.DataFrame({
        "call_rank":      range(1, len(ranked_idx) + 1),
        "appointment_id": [model["appointment_ids"][i] for i in ranked_idx],
        "patient_id":     [model["patient_ids"][i]     for i in ranked_idx],
        "no_show_prob":   [probs[i]                    for i in ranked_idx],
        "expected_recovery": [probs[i] * RECOVERY_RATE for i in ranked_idx],
        "method":         "greedy_baseline",
    })

    log.info(f"Greedy baseline: {len(result)} calls, "
             f"total expected recovery = {result['expected_recovery'].sum():.3f}")

    return result


# ── Waitlist Matching ─────────────────────────────────────────────────────────

def build_waitlist_model(
    open_slots:       pd.DataFrame,
    waitlisted:       pd.DataFrame,
    target_date:      str,
    w1:               float = WAITLIST_W1,
    w2:               float = WAITLIST_W2,
) -> dict:
    """
    Build the waitlist matching assignment IP.

    Args:
        open_slots   : DataFrame of slots that have opened up.
                       Columns: [slot_id, appointment_date, appointment_time]
        waitlisted   : DataFrame of patients on the waitlist.
                       Columns: [patient_id, appointment_id,
                                 current_appointment_date,
                                 patient_prior_noshows_rate]
        target_date  : The date slots have opened on (str 'YYYY-MM-DD').
                       Used to compute days_until_current_appt.
        w1, w2       : Score weights (must sum to 1.0).

    Returns:
        A model dict consumed by scheduler.solve_waitlist_model():
        {
            "type"        : "waitlist_assignment",
            "n_slots"     : int,
            "n_patients"  : int,
            "scores"      : np.ndarray  shape (n_patients, n_slots)
            "slot_ids"    : list
            "patient_ids" : list
            "open_slots"  : DataFrame
            "waitlisted"  : DataFrame
        }
    """
    assert abs(w1 + w2 - 1.0) < 1e-6, "Weights w1 + w2 must equal 1.0"

    required_slots   = ["slot_id", "appointment_date", "appointment_time"]
    required_waitlist = ["patient_id", "current_appointment_date",
                         "patient_prior_noshows_rate"]
    _check_columns(open_slots, required_slots,    "build_waitlist_model (slots)")
    _check_columns(waitlisted, required_waitlist, "build_waitlist_model (waitlist)")

    slots_df = open_slots.copy().reset_index(drop=True)
    wait_df  = waitlisted.copy().reset_index(drop=True)

    target_dt = pd.to_datetime(target_date)

    # ── Score matrix ─────────────────────────────────────────────────────────
    # score(i, s) for patient i and slot s

    # Component 1: days until patient's current appointment (normalised 0-1)
    wait_df["current_appointment_date"] = pd.to_datetime(
        wait_df["current_appointment_date"]
    )
    days_until = (wait_df["current_appointment_date"] - target_dt).dt.days.clip(lower=0)

    # Normalise to [0, 1] — patient furthest away gets score 1.0
    max_days = days_until.max()
    days_score = (days_until / max_days) if max_days > 0 else days_until * 0

    # Component 2: reliability score = 1 - personal no-show rate
    reliability = 1.0 - wait_df["patient_prior_noshows_rate"].fillna(0.074)

    # Combined patient score (independent of slot — extend to matrix)
    patient_scores = (w1 * days_score + w2 * reliability).values

    n_patients = len(wait_df)
    n_slots    = len(slots_df)

    # scores[i, s] — currently slot-independent but structured for extension
    # (e.g. time-of-day preference could be added as a slot dimension)
    scores = np.tile(patient_scores.reshape(-1, 1), (1, n_slots))

    log.info(f"Waitlist model: {n_patients} patients, {n_slots} open slots")
    log.info(f"  Score range: {scores.min():.3f} – {scores.max():.3f}")

    return {
        "type":        "waitlist_assignment",
        "n_slots":     n_slots,
        "n_patients":  n_patients,
        "scores":      scores,
        "slot_ids":    slots_df["slot_id"].tolist(),
        "patient_ids": wait_df["patient_id"].tolist(),
        "open_slots":  slots_df,
        "waitlisted":  wait_df,
    }


def waitlist_baseline(model: dict) -> pd.DataFrame:
    """
    Greedy baseline for waitlist matching.
    For each open slot, assigns the highest-scoring unassigned patient.
    Used to benchmark the IP solution.
    """
    scores      = model["scores"]
    patient_ids = model["patient_ids"]
    slot_ids    = model["slot_ids"]
    n_patients  = model["n_patients"]
    n_slots     = model["n_slots"]

    assigned_patients = set()
    assignments = []

    for s in range(n_slots):
        best_score   = -1
        best_patient = None
        best_idx     = None

        for i in range(n_patients):
            if i not in assigned_patients and scores[i, s] > best_score:
                best_score   = scores[i, s]
                best_patient = patient_ids[i]
                best_idx     = i

        if best_patient is not None:
            assigned_patients.add(best_idx)
            assignments.append({
                "slot_id":    slot_ids[s],
                "patient_id": best_patient,
                "score":      best_score,
                "method":     "greedy_baseline",
            })

    result = pd.DataFrame(assignments)
    if not result.empty:
        log.info(f"Greedy waitlist baseline: {len(result)} assignments, "
                 f"total score = {result['score'].sum():.3f}")

    return result


# ── Schedule Health Score ─────────────────────────────────────────────────────

def compute_schedule_health(
    appointments: pd.DataFrame,
    total_slots: int,
) -> dict:
    """
    Compute a schedule health score for a given day.

    Combines:
    - Raw utilization   : booked / total_slots
    - Expected attended : Σ (1 - no_show_prob) / total_slots
    - Risk summary      : counts per tier

    Args:
        appointments : DataFrame with [no_show_prob, risk_tier] for the day.
        total_slots  : Total available slots for the day.

    Returns:
        Dict with health metrics.
    """
    n_booked          = len(appointments)
    expected_attended = (1 - appointments["no_show_prob"]).sum()
    raw_utilization   = n_booked / total_slots if total_slots > 0 else 0
    expected_util     = expected_attended / total_slots if total_slots > 0 else 0

    tier_counts = appointments["risk_tier"].value_counts().to_dict()

    health = {
        "total_slots":        total_slots,
        "booked_slots":       n_booked,
        "raw_utilization":    round(raw_utilization, 4),
        "expected_attended":  round(expected_attended, 1),
        "expected_util":      round(expected_util, 4),
        "risk_HIGH":          tier_counts.get("HIGH",   0),
        "risk_MEDIUM":        tier_counts.get("MEDIUM", 0),
        "risk_LOW":           tier_counts.get("LOW",    0),
        "mean_no_show_prob":  round(appointments["no_show_prob"].mean(), 4),
    }

    log.info("Schedule health:")
    log.info(f"  Raw utilization     : {raw_utilization:.1%}")
    log.info(f"  Expected utilization: {expected_util:.1%}")
    log.info(f"  Expected attended   : {expected_attended:.1f} / {total_slots} slots")
    log.info(f"  Risk breakdown      : HIGH={health['risk_HIGH']}, "
             f"MEDIUM={health['risk_MEDIUM']}, LOW={health['risk_LOW']}")

    return health


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}")
