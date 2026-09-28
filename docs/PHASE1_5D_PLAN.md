# Phase 1.5D — Predictability Robustness with Early Dynamic Task Context

## Planner decision

Phase 1.5C-R1 established a real transfer-validity phenomenon:

- 20 metric-healthy task configurations;
- 390 aggregated Task × Intervention conditions;
- independent seed-block replication of candidate signs;
- multiple within-family and cross-family intervention sign reversals;
- qualitative HV/IGD agreement.

The remaining blocker is **predictability robustness**.

Phase 2 remains blocked until we show that task-conditioned transfer utility is predictably better than trivial baselines under an aggregate held-out criterion, not just on selected favorable splits.

Do not implement final VETM/Cross Attention/LLM in this phase.

---

## 1. Freeze the independent replication interpretation

Use seeds 0–9 only for candidate discovery.

Use seeds 10–19 as the **independent confirmation block**.

Primary replication reporting must use seeds 10–19 alone.

Current independently verified summary:

- candidate conditions: 46
- same-sign replication: 35 / 46 = 76.1%
- neutral on holdout: 11 / 46
- sign flips: 0 / 46

Independent seed-block reversals that remain visible:

### Within-family
- mutation_eta_01
- crossover_eta_01
- mutation_prob_02
- mutation_prob_03
- mutation_prob_04

### Cross-family
- crossover_eta_01
- mutation_prob_04

The combined 20-seed estimate may be used for final effect-size estimation after independent replication has been reported.

Do not call a condition independently replicated merely because the combined 20-seed CI is non-neutral.

---

## 2. Fix static predictor implementation

### 2.1 Standardize nearest-neighbor features

Nearest-neighbor distance must use training-set standardization.

For every split:
- fit mean/std on training rows only;
- transform train and test using training statistics;
- then fit/query kNN.

Add a regression test showing that rescaling one raw feature does not change neighbor identity after standardization.

### 2.2 Remove problem-name-derived features

Current bounds are hard-coded through a `problem == "ZDT4"` branch.

Replace with generic problem metadata:
- lower-bound mean/std/min/max;
- upper-bound mean/std/min/max;
- bound-width statistics.

Read these from `ProblemSpec`.

Do not use benchmark/problem name as a learned feature.

### 2.3 Structured intervention descriptors

Use `Intervention.descriptor(...)` directly.

Include:
- category one-hot;
- parameter identity one-hot;
- absolute value;
- signed change from baseline;
- relative magnitude.

---

## 3. Add pre-decision dynamic task context

Static descriptors are currently insufficient for robust family/OOD prediction.

Build a cheap baseline probe that is available **before** choosing an intervention.

### Probe seeds

Use dedicated probe seeds:
- 100
- 101
- 102
- 103
- 104

These seeds must be disjoint from transfer-evaluation seeds 0–19.

### Probe budget

For each task with calibrated evaluation budget `B`:

[
B_{probe} = \min(1000, \max(400, 0.2B))
]

Round to a valid multiple of population size.

Freeze this rule before running probes.

### Dynamic features

Extract only information observable during the probe:

- initial unit-HV;
- final probe unit-HV;
- unit-HV slope;
- unit-HV AUC / mean trajectory value;
- initial and final IGD;
- IGD slope;
- nondominated population ratio;
- nondominated-ratio slope;
- objective-spread / diversity summary;
- stagnation / improvement-rate summary.

All HV features must use the same calibrated unit-HV definition as the transfer matrix.

Do not use final full-budget target outcome.

Prefer task-level probe aggregates over the 5 dedicated probe seeds:
- mean;
- standard deviation where useful.

Store:
`results/phase1_5D_task_context.csv`

---

## 4. Predictor ablation groups

Every prediction row remains one aggregated Task × Intervention condition.

Compare exactly these feature groups:

1. **intervention-only**
2. **static task + intervention**
3. **dynamic task + intervention**
4. **static + dynamic task + intervention**
5. **context-shuffled negative control**

The context-shuffled control must preserve intervention identity while permuting task-context vectors among training tasks.

If static+dynamic does not outperform intervention-only and shuffled-context controls, there is no evidence that the predictor learned a genuine validity boundary.

---

## 5. Models

Keep the model suite intentionally simple:

### Regression
- zero gain
- global intervention mean
- standardized kNN
- ridge
- small MLP

### Negative-transfer detection
- majority
- random expectation / repeated-random baseline
- standardized kNN
- logistic regression
- small MLP classifier

Run the MLP with at least 5 training seeds and report mean/std rather than selecting one favorable seed.

---

## 6. Pre-register the primary evaluation criterion

Do not use “any model on any split beats baseline.”

### Primary split family

Use **leave-one-problem-out** across all supported problem identities.

Compute one MAE per held-out task configuration, then aggregate at the **task level**.

Primary regression endpoint:

[
\Delta MAE = MAE_{static+dynamic} - MAE_{zero}
]

Also compare against the global-intervention baseline.

Use a task-block bootstrap with at least 5000 resamples.

### GO requirement for predictive signal

The static+dynamic model must satisfy all:

1. macro task-level MAE is lower than zero-gain;
2. macro task-level MAE is lower than global-intervention mean;
3. 95% task-block bootstrap CI for `MAE_model - MAE_zero` has upper bound < 0;
4. Spearman correlation is positive and materially non-zero on the aggregated held-out predictions;
5. context-shuffled control is worse than the real task-context model.

If no simple model meets this, Phase 2 remains NO-GO.

Do not choose the model after examining test results. Select the model class using training/internal validation only, or report each model without post-hoc winner selection.

---

## 7. Secondary generalization tests

Report, but do not use them alone to pass the primary gate:

- ZDT → DTLZ family-held-out;
- DTLZ → ZDT family-held-out;
- objective-count 2 → 3;
- dimension OOD.

For each:
- MAE;
- Spearman;
- negative-transfer AUROC;
- balanced accuracy;
- Brier score.

If family/OOD still fails but the primary leave-problem aggregate passes, Phase 2 may proceed with claims explicitly limited to the supported generalization regime.

---

## 8. Negative-transfer detection

Because the paper is validity-aware, negative-transfer detection is a key secondary endpoint.

Primary label:
[
y_{neg} = 1[\Delta HV_{unit} < -0.01]
]

For each leave-one-problem split, report:
- AUROC;
- balanced accuracy;
- F1;
- Brier.

Aggregate by task, not condition count alone.

Also report class prevalence.

A high AUROC on one favorable problem is not sufficient.

---

## 9. Statistical controls

### Task-block bootstrap

Resample held-out task configurations, not individual intervention conditions.

### Permutation/context test

Repeat task-context permutation at least 200 times.

Report where the real model falls relative to the shuffled-context null distribution.

### Multiple splits

Do not report “best split” as the main evidence.

Main tables must contain:
- all splits;
- macro task-level aggregate;
- bootstrap CI.

---

## 10. Outputs

Create:

- `results/phase1_5D_task_context.csv`
- `results/phase1_5D_static_features.csv`
- `results/phase1_5D_predictability.json`
- `results/phase1_5D_bootstrap.json`
- `results/phase1_5D_permutation.json`
- `results/phase1_5D_summary.json`
- `docs/PHASE1_5D_REPORT.md`

The report must explicitly separate:

1. transfer-phenomenon evidence;
2. independent sign replication;
3. static-only predictability;
4. dynamic-context gain;
5. family/OOD limits;
6. Phase 2 GO/NO-GO.

---

## 11. Phase 2 gate

### GO

Proceed to Phase 2 only if:

- independent sign reversals remain confirmed;
- static+dynamic task context beats trivial/intervention-only baselines under the pre-registered aggregate task-level criterion;
- improvement survives task-block bootstrap;
- context shuffling materially degrades performance.

### NO-GO

If dynamic context still cannot beat trivial baselines robustly:

- do not build Cross Attention/VETM;
- reframe the work toward mechanistic operator sensitivity / negative-transfer structure;
- analyze why the validity boundary exists but is not predictable from available pre-decision context.

## Executor constraints

- No benchmark name as predictor feature.
- No future/full-budget target outcomes in task context.
- No use of transfer-evaluation seeds for probe context.
- No post-hoc split/model selection to pass the gate.
- No final VETM/Cross Attention/LLM in Phase 1.5D.
