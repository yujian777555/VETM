# Phase 1.5C-R1 — Repair 3D Unit-HV Reuse and Run a Staged Healthy Matrix

## Planner decision

Phase 1.5C commit `40a915bc74c9c5ac200b86b818c047dc524c5b5c` contains valid engineering improvements, but the phase is not scientifically complete.

Phase 2 remains blocked.

This revision is narrowly scoped to:
1. repair the 3-objective unit-HV reuse bug;
2. correct runtime/cost accounting and metric saturation logic;
3. make the executor fast enough for staged experiments;
4. obtain the smallest decisive healthy matrix before spending on the full 20–30 task design;
5. run the complete simple predictability baseline suite.

Do not implement final VETM/Cross Attention/LLM.

---

## 1. Fix the 3-objective unit-HV reuse bug

Current bug:

`phase1_5c_pilot_reuse.py` falls back to `n_obj=2` because Phase 1.5B CSV has no `n_obj` column.

Therefore DTLZ2_m3 rows are divided by `1.1^2` rather than `1.1^3`.

### Required repair

Infer `n_obj` from trusted task metadata, preferably:
- the task config registry keyed by `task_id`, or
- an explicit mapping generated from config.

Do not infer dimensionality from a default.

Add tests that verify:
- m=2 -> divisor `1.1^2`
- m=3 -> divisor `1.1^3`

Regenerate:
- `results/phase1_5C_transfer_matrix.csv`
- `results/phase1_5C_condition_summary.csv`
- `results/phase1_5C_summary.json`

Record both old and corrected held-out MAE for audit purposes.

---

## 2. Fix metric saturation logic

Do not define saturation as low variation across seeds.

Replace:
`np.ptp(HV_across_seeds) < eps`

with a metric-health criterion tied to optimization quality/progress, for example:
- unit-HV near the maximum box occupancy, and/or
- negligible improvement over a pre-registered late-search window.

The rule must be fixed before running the matrix.

Keep:
- `zero_HV_ratio <= 0.10`
- finite HV/IGD
- valid normalization

A stable but non-saturated task must not be rejected merely because seeds agree.

---

## 3. Correct compute-cost accounting

Report three costs separately:

1. **Worst-case calibration ceiling**
   - all candidate budgets used for all tasks.

2. **Adaptive expected calibration cost**
   - calibration stops after the first healthy budget.

3. **Main matrix cost**
   - healthy_tasks × active_interventions × 10 paired seeds
   - plus one cached baseline per Task × Seed.

Do not double every matrix condition to 20 seeds.

Only candidate sign reversals are extended with **10 additional seeds**, so report extension cost separately.

---

## 4. Executor performance work

Before launching the staged matrix, benchmark runtime.

Required:
- benchmark one representative 2k, 5k, and 10k run;
- record wall-clock seconds and evaluations/sec;
- estimate staged/full matrix wall time.

Add safe parallel execution across independent Task × Seed × Intervention jobs using process-level parallelism where possible.

Requirements:
- deterministic seed assignment;
- no shared mutable RNG;
- identical results between serial and parallel execution for a small fixture;
- configurable worker count;
- crash/failure rows preserved.

Do not change optimization semantics merely to gain speed.

---

## 5. Adaptive budget calibration

Use candidate budgets:
- 2,000
- 5,000
- 10,000
- 20,000 evaluations

Use 5 baseline seeds.

For each task:
- start at 2k;
- stop at the first healthy non-saturated budget;
- only unresolved tasks proceed to the next budget.

Store actual executed runs, not placeholder rows.

Output:
- `results/phase1_5C_R1_budget_calibration.csv`
- `results/phase1_5C_R1_metric_health.json`

---

## 6. Staged matrix instead of all-at-once execution

### Stage A — decisive matrix

Run approximately **12 metric-healthy task configurations** with:
- all active deterministic interventions;
- 10 paired seeds;
- cached baseline once per Task × Seed.

Task set must include enough diversity to test:
- multiple ZDT problem identities;
- at least two configurations from DTLZ1/DTLZ2 if healthy;
- within-family dimension variation.

Do not cherry-pick tasks based on intervention outcomes. Selection is based only on metric health and pre-registered structural coverage.

### Stage A Go gate

Continue to Stage B only if:
1. at least one within-family confirmed sign reversal exists;
2. HV_unit and IGD agree qualitatively on the reversal;
3. at least one task-conditioned simple predictor improves over zero/global baselines on held-out task splits.

If these fail, stop and report NO-GO without spending on the full matrix.

### Stage B — expansion

If Stage A passes, expand to **20+ healthy task configurations**.

Candidate sign reversals are then extended from 10 to 20 paired seeds.

---

## 7. Structured predictor features

Prediction rows are aggregated Task × Intervention conditions.

### Intervention features

Actually use `Intervention.descriptor(...)`:
- category
- parameter identity
- absolute value
- signed change
- relative magnitude

Encode categorical fields explicitly.

### Task features

At minimum:
- n_var
- n_obj
- population size
- budget
- bound statistics

Optionally add a **separate pre-decision baseline probe** if implemented cleanly.

If dynamic features are added, clearly define the probe budget and keep them causally available before the transfer decision.

Do not use:
- benchmark name as a learned feature
- final target outcome
- transfer label
- future trajectory information

---

## 8. Complete simple baseline suite

Actually execute and report:

### Regression / utility
- zero-gain
- global intervention mean
- nearest-neighbor
- ridge/linear regression
- small MLP

Metrics:
- MAE
- Spearman

### Negative-transfer detection
- majority/random
- nearest-neighbor
- logistic regression
- small MLP classifier

Metrics:
- AUROC
- balanced accuracy / F1
- Brier score

Use aggregated conditions, not seed rows.

For ridge/logistic models, include a proper intercept or centered target treatment.

---

## 9. Held-out protocols

Stage A minimum:
- leave-one-problem/configuration-out where feasible;
- family-held-out if both families have sufficient healthy tasks;
- dimension-OOD when train/test coverage allows.

All train-derived global/intervention statistics must be computed only from training tasks.

Invalid tasks never enter train or test.

---

## 10. Required scientific outputs

Create:
- `results/phase1_5C_R1_budget_calibration.csv`
- `results/phase1_5C_R1_metric_health.json`
- `results/phase1_5C_R1_runtime_benchmark.json`
- `results/phase1_5C_R1_transfer_matrix.csv`
- `results/phase1_5C_R1_condition_summary.csv`
- `results/phase1_5C_R1_predictability.json`
- `results/phase1_5C_R1_summary.json`
- `docs/PHASE1_5C_R1_REPORT.md`

Report:
- corrected DTLZ2 unit-HV scaling;
- tests passed;
- runtime benchmark;
- actual calibration runs;
- selected task budgets;
- Stage A task/intervention/run counts;
- confirmed within-family/cross-family reversals;
- HV/IGD agreement;
- all simple predictor results;
- GO/NO-GO for Stage B and Phase 2.

## Executor constraints

- Do not implement final VETM.
- Do not alter health/tolerance thresholds after seeing outcomes.
- Do not include invalid tasks in prediction.
- Do not claim the old Phase 1.5C held-out MAE as final until the 3D scaling bug is repaired.
- Do not silently reduce tasks/interventions/seeds; use the explicit staged stop rule instead.
