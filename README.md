# Hospital Appointment Optimizer

> End-to-end clinical scheduling optimization system combining no-show probability estimation with LP-based reminder allocation and waitlist matching.

---

## Overview

This system processes 10 years of clinical appointment data across 111k appointments, 37k patients, and 104k scheduling slots. For any target date, it produces three operational outputs: a ranked reminder call list identifying which patients staff should contact before their appointment, a waitlist match recommendation filling cancelled slots with the best-fit patients, and a schedule health score summarising expected utilization for the day. The system avoids overbooking entirely — instead optimizing the use of existing capacity through targeted outreach and intelligent slot reallocation.

---

## Results

**No-show prediction:** An XGBoost classifier trained on a temporal split (2015–2022 train, 2023 val, 2024 test) achieved ROC-AUC of 0.50 on held-out data — effectively random. EDA confirmed that no-show behaviour is distributed uniformly across all available features (scheduling interval, age, time of day, insurance provider), with near-zero point-biserial correlations across the board. This is a legitimate finding rather than a modelling failure: the available features do not contain sufficient signal for a learned model to outperform chance on this population.

**Hybrid probability engine:** In place of the learned model, a statistically grounded hybrid engine was built. Patients with 3+ prior appointments use their personal historical no-show rate. Patients with 1–2 appointments blend their personal rate with the population base rate (7.4%) at equal weight. First-time patients receive the base rate. This approach is transparent, explainable, and matches how clinicians reason about risk in practice.

**Optimization (sample date: 2024-11-15):**
- 38 appointments across 40 available slots — 95.0% raw utilization
- 85.6% expected utilization after accounting for predicted no-shows
- 9 high-risk appointments identified for reminder calls (out of 20 call capacity)
- 9 open slots filled via waitlist matching, total assignment score 8.43
- Gurobi LP and greedy baseline produced identical expected recovery (1.093) — expected for a unit-weight knapsack, confirming the greedy heuristic is optimal for this problem structure

---

## Dataset

Three relational tables sourced from [Medical Appointment Scheduling System](https://www.kaggle.com/datasets/carogonzalezgaltier/medical-appointment-scheduling-system):

| Table | Rows | Description |
|---|---|---|
| `patients.csv` | 36,697 | Patient demographics and insurance |
| `appointments.csv` | 111,488 | Appointment lifecycle: booking, status, timing |
| `slots.csv` | 104,360 | 15-minute slot grid with availability |

**Key fields:**
- `scheduling_interval` — days between booking and appointment date
- `status` — attended / did not attend / cancelled
- `waiting_time` — minutes spent in waiting room (attended only)
- `check_in_time`, `start_time`, `end_time` — full visit timeline

---

## Methodology

### Phase 1 — Data Foundation

Three CSVs are loaded, type-cast, and merged into a single master table via left joins: appointments ← slots (on `slot_id`) ← patients (on `patient_id`). Left joins preserve all appointments even if a slot or patient record is missing, with orphans flagged in validation rather than silently dropped. ID columns are zero-padded to fixed widths before joining to prevent silent mismatches. Five referential integrity and business logic checks run on every load.

### Phase 2 — Feature Engineering

15 features are engineered across three groups. Appointment-level features include `scheduling_interval`, `appointment_hour`, `appointment_day_of_week`, `appointment_month`, `is_morning`, `is_monday`, and `is_friday`. Patient-level features include `age`, `sex_encoded`, `insurance_encoded` (frequency-encoded), `patient_prior_noshows`, `patient_prior_noshows_rate`, and `patient_prior_appointments`. Schedule-level features include `daily_slot_utilization` and `rolling_7d_noshows_rate`.

The historical patient features are computed using an expanding window sorted by `appointment_date`, shifted by one row per patient so the current appointment is never included in its own history. This prevents data leakage without requiring a manual date cutoff.

### Phase 3 — No-Show Classifier & Probability Engine

A logistic regression baseline and XGBoost classifier were trained on a temporal split (train: 2015–2022, val: 2023, test: 2024) with `scale_pos_weight=13.01` to address the 13:1 class imbalance. Both models achieved ROC-AUC ≈ 0.50 on held-out data. EDA showed near-zero point-biserial correlations between all features and the no-show target, explaining the result — the outcome is uniformly distributed across all feature slices in this dataset.

A hybrid probability engine was built as a statistically grounded alternative. It assigns no-show probabilities based on personal history depth: full personal rate for patients with 3+ prior appointments, a 50/50 blend with the 7.4% base rate for patients with 1–2 appointments, and the base rate for first-time patients. The classifier is retained in the repo as documentation of the full ML attempt.

### Phase 4 — Optimization

**Reminder Allocation (0-1 Knapsack LP)**

Given a day's appointments and a staff call capacity N, the model selects which patients to call with a reminder.

- Decision variable: $x_i \in \{0,1\}$ — call patient $i$ or not
- Objective: $\text{maximise} \sum_i (\text{no\_show\_prob}_i \times \text{recovery\_rate}) \cdot x_i$
- Constraint: $\sum_i x_i \leq N$

A recovery rate of 30% is assumed, consistent with the clinical literature on phone reminder effectiveness. A greedy baseline (rank by probability, take top N) is computed alongside the LP for benchmarking.

**Waitlist Matching (Assignment IP)**

When slots open up, the model assigns the best-fit waitlisted patients to fill them.

- Decision variable: $x_{i,s} \in \{0,1\}$ — assign patient $i$ to slot $s$
- Objective: $\text{maximise} \sum_{i,s} \text{score}(i,s) \cdot x_{i,s}$
- Score: $0.6 \times \text{days\_until\_current\_appt (normalised)} + 0.4 \times (1 - \text{personal\_noshowrate})$
- Constraints: each patient assigned to at most one slot; each slot filled by at most one patient

Both models are solved with Gurobi (academic license) with automatic fallback to PuLP/CBC if Gurobi is unavailable.

---

## Repo Structure

```
hospital-appointment-optimizer/
├── data/
│   ├── raw/                        # Original CSVs (not tracked in git)
│   └── processed/                  # master.csv, features.csv, model_ready.csv
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis, no-show distributions
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model.ipynb           # Classifier training and evaluation
│   └── 04_optimization.ipynb       # LP formulations and results
├── src/
│   ├── data/
│   │   ├── loader.py               # Load, validate, and merge the 3 CSVs
│   │   └── features.py             # Feature engineering pipeline
│   ├── models/
│   │   ├── classifier.py           # XGBoost no-show classifier (documented)
│   │   ├── probability_engine.py   # Hybrid probability model (used in pipeline)
│   │   └── evaluate.py             # Metrics, calibration, plots
│   └── optimization/
│       ├── formulation.py          # LP/IP math definitions and baselines
│       └── scheduler.py            # Gurobi/PuLP solver layer
├── scripts/
│   └── run_pipeline.py             # End-to-end runner for a given target date
├── outputs/
│   ├── figures/                    # EDA plots, calibration curves, feature importance
│   └── results/                    # Reminder calls, waitlist matches, health scores
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/saadfarooq-alt/Hospital-Appointment-Optimizer
cd Hospital-Appointment-Optimizer
pip install -r requirements.txt

# 2. Place raw CSVs in data/raw/
#    patients.csv, appointments.csv, slots.csv

# 3. Run the full pipeline for a target date
python scripts/run_pipeline.py --date 2024-11-15

# 4. Re-run on a different date without reprocessing data
python scripts/run_pipeline.py --date 2024-11-22 --skip_data_prep

# 5. Adjust call capacity
python scripts/run_pipeline.py --date 2024-11-15 --skip_data_prep --call_capacity 30
```

**Outputs saved to `outputs/results/`:**
- `reminder_calls_YYYYMMDD.csv` — ranked call list for the target date
- `waitlist_matches_YYYYMMDD.csv` — patient-slot assignments
- `schedule_health_YYYYMMDD.csv` — utilization and risk summary

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
gurobipy
matplotlib
seaborn
scipy
jupyter
```

> Gurobi requires a valid license. Free academic licenses are available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/). The pipeline falls back to PuLP/CBC automatically if Gurobi is unavailable.

---

## Additional Paths


---
## Author

**Sa'ad Farooq** — [LinkedIn](https://www.linkedin.com/in/sa-ad-farooq-057a5825b/) · [GitHub](https://github.com/saadfarooq-alt) · s4farooq@uwaterloo.ca
