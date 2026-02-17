# Hospital Appointment Optimizer

> End-to-end clinical scheduling optimization system combining no-show prediction with LP-based reminder allocation and waitlist matching.

---

## Overview

<!-- 
  TODO: Fill in after project is complete.
  2-3 sentences describing what the system does, what data it uses, and what problem it solves.
  Example: "This system ingests appointment, patient, and slot data from [X] clinic and produces
  daily no-show risk scores, reminder priority lists, and waitlist fill recommendations to
  maximize schedule utilization without overbooking."
-->

---

## Results

<!-- 
  TODO: Fill in after Phase 3 & 4 are complete.
  Key metrics to report:
  - No-show classifier: ROC-AUC, precision, recall, calibration score
  - Reminder LP: Expected appointments recovered per day vs. baseline
  - Waitlist matching: Average utilization improvement on holdout dates
  - Example: "Gradient boosting classifier achieved AUC of X on held-out dates.
    Reminder allocation recovered an estimated Y% of at-risk appointments.
    Schedule utilization improved from Z% to Z% on simulated days."
-->

---

## Dataset

Three relational tables sourced from [Medical Appointment Scheduling System](https://www.kaggle.com/datasets/carogonzalezgaltier/medical-appointment-scheduling-system):

| Table | Rows | Description |
|---|---|---|
| `patients.csv` | 36,698 | Patient demographics and insurance |
| `appointments.csv` | 111,489 | Appointment lifecycle: booking, status, timing |
| `slots.csv` | 104,361 | 15-minute slot grid with availability |

**Key fields:**
- `scheduling_interval` — days between booking and appointment date
- `status` — attended / did not attend / cancelled
- `waiting_time` — minutes spent in waiting room (attended only)
- `check_in_time`, `start_time`, `end_time` — full visit timeline

> Raw data not included in this repo. Download from Kaggle and place CSVs in `data/raw/`.

---

## Methodology

### Phase 1 — Data Foundation
<!-- TODO: Brief description of merging strategy and any data quality issues found. -->

### Phase 2 — Feature Engineering
<!-- 
  TODO: List features used. Expected:
  - Appointment-level: scheduling_interval, hour, day_of_week, month, is_morning
  - Patient-level: age, sex, insurance, historical_noshows_rate
  - Slot-level: daily_utilization_rate
-->

### Phase 3 — No-Show Classifier
<!-- 
  TODO: Describe model choices and train/test split strategy.
  Note: Temporal split used (train on earlier dates, test on later dates) to prevent leakage.
  Baseline: logistic regression. Main model: XGBoost / LightGBM.
  Evaluation: ROC-AUC, precision-recall, calibration curve.
-->

### Phase 4 — Optimization

#### Reminder Allocation (Knapsack LP)
<!-- 
  TODO: Describe formulation.
  Decision variable: x_i ∈ {0,1} — call patient i or not
  Objective: maximize Σ (no_show_prob_i × recovery_value_i) × x_i
  Constraint: Σ x_i ≤ N (daily call capacity)
-->

#### Waitlist Matching (Assignment IP)
<!-- 
  TODO: Describe formulation.
  Decision variable: x_{i,s} ∈ {0,1} — assign waitlisted patient i to open slot s
  Objective: maximize Σ score(i,s) × x_{i,s}
    where score = f(days until their current appointment, 1 - no_show_prob)
  Constraints: each patient assigned to ≤ 1 slot, each slot filled by ≤ 1 patient
-->

---

## Repo Structure

```
hospital-appointment-optimizer/
├── data/
│   ├── raw/                        # Original CSVs (not tracked in git)
│   └── processed/                  # Cleaned, merged, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory analysis, no-show distributions
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model.ipynb           # Classifier training and evaluation
│   └── 04_optimization.ipynb       # LP formulations and results
├── src/
│   ├── data/
│   │   ├── loader.py               # Load and merge the 3 CSVs
│   │   └── features.py             # Feature engineering pipeline
│   ├── models/
│   │   ├── classifier.py           # No-show prediction model
│   │   └── evaluate.py             # Metrics, calibration, plots
│   └── optimization/
│       ├── formulation.py          # LP/IP math definitions
│       └── scheduler.py            # Gurobi solver layer, returns DataFrames
├── scripts/
│   └── run_pipeline.py             # End-to-end runner for a given target date
├── outputs/
│   ├── figures/                    # EDA plots, model performance charts
│   └── results/                    # Optimized schedules, model artifacts
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/saadfarooq-alt/hospital-appointment-optimizer
cd hospital-appointment-optimizer
pip install -r requirements.txt

# 2. Place raw CSVs in data/raw/
#    patients.csv, appointments.csv, slots.csv

# 3. Run the full pipeline for a target date
python scripts/run_pipeline.py --date 2015-06-15
```

**Output:** Reminder priority list, waitlist match recommendations, and schedule health score saved to `outputs/results/`.

---

## Requirements

<!-- TODO: Fill in after dependencies are finalized. -->

```
pandas
numpy
scikit-learn
xgboost          # or lightgbm
gurobipy
matplotlib
seaborn
jupyter
```

> Gurobi requires a valid license. Free academic licenses available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Skills Demonstrated

<!-- TODO: Refine after project is complete. Keep this honest and specific. -->

- Relational data modeling and multi-table joins across 111k+ records
- Temporal train/test splitting to prevent leakage in time-series classification
- Probability calibration for ML outputs consumed by downstream optimization
- Linear and integer programming with Gurobi (knapsack + assignment formulations)
- End-to-end pipeline design: raw data → features → predictions → decisions

---

## Author

**Sa'ad Farooq** — [LinkedIn](https://www.linkedin.com/in/sa-ad-farooq-057a5825b/) · [GitHub](https://github.com/saadfarooq-alt) · s4farooq@uwaterloo.ca
