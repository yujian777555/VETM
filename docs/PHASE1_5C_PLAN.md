# Phase 1.5C — Healthy Matrix Expansion and Predictability Test

## Planner decision

Phase 1.5B fixed major parts of the transfer-metric pipeline and produced the first credible context-dependent transfer signal, but Phase 2 remains blocked.

This phase has two goals:

1. build a sufficiently large **metric-healthy** Task × Intervention matrix;
2. test whether transfer utility is actually predictable under held-out task/family splits after removing invalid-task contamination.

Do **not** implement the final VETM architecture, Cross Attention, LLM, or agent layer in this phase.

---

## 1. Mandatory corrections before new experiments

### 1.1 Metric health threshold

The Planner health gate is:

`zero_HV_ratio > 0.10 => metric-invalid`

The current implementation uses >0.95. Fix this and add unit tests.

Metric-invalid tasks must be excluded from:
- sign-reversal evidence;
- predictor training;
- held-out evaluation;
- global intervention means.

They may remain in diagnostic reports.

### 1.2 Unit-normalized HV

The current objective coordinates are normalized, but HV itself is not unit-normalized across objective dimensions.

Define:

[
HV_{unit} = \frac{HV(F_{norm}, r)}{\prod_k r_k}
]

with the same pre-registered normalized reference value `r_k = 1.1`.

Use `HV_unit` as the primary transfer utility.

Keep:
- raw HV;
- coordinate-normalized HV;
- unit-normalized HV;
- IGD or IGD+.

The practical tolerance is applied to `delta_HV_unit`, not raw HV.

### 1.3 Held-out contamination fix

Recompute Phase 1.5B held-out analysis after excluding:
- ZDT4_n10_m2;
- ZDT6_n20_m2;
- DTLZ1_n7_m3;

and any other task failing the health gate.

Do not interpret the previous 4.3× worse-than-zero result as evidence that cross-family prediction is impossible.

---

## 2. Budget calibration for hard tasks

Before the full matrix, calibrate difficult tasks using **baseline-only** runs.

Candidate budgets:
- 2,000
- 5,000
- 10,000
- 20,000 evaluations

Use at least 5 seeds for the calibration sweep.

For each task configuration choose the **smallest** budget satisfying:

- zero_HV_ratio <= 0.10;
- all metrics finite;
- baseline HV is not completely saturated;
- IGD is finite.

Freeze the selected budget before intervention experiments.

Do not choose a budget based on which intervention looks best.

Store:
- `results/phase1_5C_budget_calibration.csv`
- `results/phase1_5C_metric_health.json`

If a task remains censored at 20k evaluations, mark it unsupported for the current benchmark instead of changing the reference point after seeing intervention outcomes.

---

## 3. Healthy task grid

Construct approximately **20–30 metric-healthy task configurations**.

Minimum target:

### ZDT
ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 with multiple dimensions such as:
- n_var=10
- n_var=20
- n_var=30

### DTLZ
DTLZ1 and DTLZ2 with:
- n_obj=2 and/or 3;
- more than one valid n_var where appropriate.

If DTLZ3/DTLZ4 are added, implement known-value tests first.

Do not add >3 objective tasks until the HV implementation is independently validated for that dimension or a trusted external implementation becomes available.

Task identity may remain metadata but must not be a direct learned predictor feature in the final held-out tests.

---

## 4. Intervention set

Keep at least the current 20 deterministic mutation/crossover/selection interventions, filtered for task-specific no-ops.

Do not silently reduce the intervention set.

For every active intervention store a structured descriptor:
- category;
- parameter name;
- absolute value;
- signed change from baseline;
- relative magnitude from baseline;
- applicability.

These descriptors will become predictor inputs.

---

## 5. Main paired transfer matrix

For every healthy Task × Intervention condition:

- baseline and intervention share the same seed;
- baseline initial population must be identical when the algorithm semantics permit it;
- equal function-evaluation budget;
- baseline is cached once per Task × Seed and reused across interventions;
- failures are preserved;
- 10 paired seeds for the full matrix.

For candidate sign reversals, re-run/extend to **20 paired seeds**.

Required outputs:
- `results/phase1_5C_transfer_matrix.csv`
- `results/phase1_5C_condition_summary.csv`
- `results/phase1_5C_summary.json`

For each condition report:
- mean/median delta_HV_unit;
- bootstrap 95% CI;
- effect size;
- mean delta_IGD;
- positive / neutral / negative state.

---

## 6. Sign-reversal confirmation

The scientific phenomenon is stronger than “some interventions help and some hurt.”

Measure whether the **same intervention** changes sign as task context changes.

Report separately:

1. within-family reversals;
2. cross-family reversals;
3. dimension/objective-count reversals;
4. agreement between HV_unit and IGD direction.

A reversal counts as confirmed only when both sides are non-neutral under the pre-registered CI+tolerance rule.

Target evidence before Phase 2:
- at least one confirmed within-family reversal;
- at least one confirmed cross-family reversal;
- secondary metric direction qualitatively agrees for the confirmed conditions.

If only family-level reversal exists, report that honestly; do not call it a general validity boundary yet.

---

## 7. Predictability dataset

Build one prediction row per aggregated Task × Intervention condition.

Do not train on individual seed rows as independent samples.

### Task features

Use only features available before the intervention decision, for example:
- n_var;
- n_obj;
- budget;
- population size;
- variable-bound statistics;
- baseline early-run HV/IGD;
- early HV/IGD slope;
- nondominated ratio;
- diversity/crowding summaries;
- stagnation/improvement-rate features.

Do not include:
- target final outcome;
- transfer label;
- benchmark/problem name as a direct predictor;
- future trajectory information.

### Intervention features

Use the structured descriptor from Section 4.

---

## 8. Simple predictability baselines

Before final VETM, compare:

- zero-gain predictor;
- global intervention mean;
- nearest-neighbor;
- ridge/linear regression;
- logistic regression for negative-transfer detection;
- small MLP.

A NumPy implementation is acceptable if scikit-learn remains unavailable.

Targets:
1. regression of `delta_HV_unit`;
2. negative-transfer detection;
3. ranking of intervention utility.

Report:
- MAE;
- Spearman correlation;
- AUROC;
- balanced accuracy / F1;
- Brier score where applicable.

---

## 9. Held-out protocols

Use condition-level splits with no target leakage.

Required where data volume permits:

1. leave-problem-out;
2. family-held-out ZDT ↔ DTLZ;
3. structural OOD by dimension and/or objective count.

Invalid tasks are never included in train or test.

Source experience/intervention statistics used by a baseline must be computed from the training side only.

---

## 10. Go / No-Go gate for Phase 2

### GO only if the following are supported

1. The healthy transfer matrix contains bootstrap-stable sign reversals.
2. At least one reversal exists within a family, not only between ZDT and DTLZ.
3. Confirmed reversal directions are qualitatively consistent with IGD/IGD+.
4. At least one simple task-conditioned model improves over trivial zero/global baselines on held-out tasks using pre-registered metrics.
5. Improvement is not caused by benchmark-name leakage or metric-invalid tasks.

Prefer bootstrap confidence intervals on the model-vs-baseline error difference rather than selecting a favorable test split.

### NO-GO / redesign

If task-conditioned prediction remains indistinguishable from trivial baselines after the healthy matrix expansion, do not build Cross Attention/VETM. Reframe toward operator-sensitivity/negative-transfer analysis or redesign the task representation.

---

## 11. Deliverables

Commit:
- metric-health threshold fix;
- unit-normalized HV implementation;
- corrected Phase 1.5B held-out analysis;
- budget calibration;
- expanded healthy task grid;
- structured intervention descriptors;
- Phase 1.5C transfer matrix;
- simple predictor baselines;
- tests;
- `docs/PHASE1_5C_REPORT.md`.

Completion report must include:
- commit hash;
- tests passed;
- healthy/invalid task counts;
- selected per-task budgets;
- condition/run counts;
- confirmed sign reversals;
- HV/IGD consistency;
- held-out baseline/model results;
- Planner GO/NO-GO recommendation.

## Executor constraints

- Do not implement final VETM/Cross Attention.
- Do not add LLM/agent components.
- Do not alter the tolerance or health thresholds after inspecting held-out outcomes.
- Do not include metric-invalid tasks in prediction analyses.
- Do not treat the old Phase 1.5B held-out result as valid evidence after the contamination issue is known.
