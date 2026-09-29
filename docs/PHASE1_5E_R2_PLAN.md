# Phase 1.5E-R2 — Mean-Score Consistency Repair and Final Fork

## Planner decision

Phase 1.5E-R1 repaired the raw objective-scale leak, but the primary causal fingerprint still contains one mathematical inconsistency in `delta_empirical_mean`.

This phase is a **strict implementation repair only**.

It is the final predictor test.

Do not:
- add/change probe anchors;
- add/change task configurations;
- change probe/evaluation seeds;
- add/change feature families;
- change the primary model;
- tune ridge regularization;
- change held-out splits;
- change GO thresholds;
- add a deeper network, Transformer, Cross Attention, LLM, or agent logic.

After R2:
- PASS -> Planner may open Phase 2;
- FAIL -> predictor path is permanently closed and the project pivots to mechanistic operator-sensitivity / negative-transfer boundary analysis.

---

## 1. Fix the empirical mean-score definition

Current R1 bug:

```python
base_mean = np.mean(base_norm_front)
improvement = np.mean(np.sum(norm_front, axis=1)) - base_mean
```

The baseline and probe are not scored with the same statistic.

Use exactly:

```python
def empirical_mean_score(norm_front):
    return float(np.mean(np.sum(norm_front, axis=1)))

base_mean = empirical_mean_score(base_norm_front)
probe_mean = empirical_mean_score(norm_front)
delta_empirical_mean = probe_mean - base_mean
```

Equivalent implementations are acceptable only if they are algebraically identical.

Do not divide one side by `n_obj` without dividing the other side identically.

---

## 2. Add invariance tests before rerunning experiments

Add tests covering at least:

### Identical-front invariant

For an identical normalized baseline and probe front:

`delta_empirical_mean == 0`

for:
- m=2;
- m=3.

### Translation/scaling consistency

Given raw fronts transformed with the same initial-population center/IQR:

- compute normalized baseline/probe;
- verify `delta_empirical_mean` matches the explicit definition above.

### Regression test for the R1 bug

Construct a small m=3 fixture where:
- `mean(F)` differs from `mean(sum(F, axis=1))`;
- verify the old asymmetric formula would be non-zero for identical fronts;
- verify the repaired function returns zero.

Do not rerun the final gate until these tests pass.

---

## 3. Preserve empirical scaling

Keep exactly the R1 empirical normalization:

[
c_k = median(F^{init}_{:,k})
]

[
s_k = max(Q_{.75}-Q_{.25}, 10^{-12})
]

[
	ilde f_k = (f_k-c_k)/s_k
]

Use the same center/scale for:
- baseline one-step branch;
- every probe branch of the same Task × ProbeSeed.

Keep normalized:
- delta empirical mean;
- delta objective spread;
- empirical HV.

No raw objective values enter the primary fingerprint.

---

## 4. Preserve the causal probe protocol

Keep exactly:

- 20 metric-healthy tasks;
- seeds 200 and 201;
- byte-identical initial populations/objectives;
- one generation;
- the existing eight off-grid anchors;
- collision filtering exactly as R1;
- current oracle-free static features;
- current target intervention descriptors.

Do not rerun long-horizon transfer experiments. Reuse Phase 1.5C-R1 targets.

---

## 5. Regenerate per-seed and aggregate fingerprints

Regenerate:

- `results/phase1_5E_R2_fingerprint_by_seed.csv`
- `results/phase1_5E_R2_causal_fingerprint.csv`
- `results/phase1_5E_R2_scale_audit.json`
- `results/phase1_5E_R2_stability.json`

Also report, for every task:

- R1 max absolute response;
- R2 max absolute response;
- max absolute `delta_empirical_mean`;
- non-finite count.

Explicitly highlight:
- ZDT6_n10_m2;
- ZDT6_n20_m2;
- ZDT6_n30_m2;
- DTLZ1_n6_m2;
- DTLZ1_n7_m3.

---

## 6. Rerun the exact primary regression gate

Primary model remains:

**ridge regression on causal fingerprint + intervention descriptor**

Same ablations:

1. intervention-only
2. oracle-free passive + intervention
3. causal fingerprint + intervention
4. passive + causal + intervention
5. task-level shuffled causal fingerprint + intervention

No model/hyperparameter search.

Evaluation remains leave-one-problem-out.

Compute predictions for all held-out conditions and aggregate errors by the 20 task configurations.

Report:

- macro task MAE;
- macro task Spearman;
- per-problem results;
- per-task-configuration results.

---

## 7. Bootstrap

Use the same 20 task-configuration blocks.

5000 resamples.

Required comparisons:

- causal - zero;
- causal - intervention-only;
- causal - global-intervention mean.

Store:
`results/phase1_5E_R2_bootstrap.json`.

---

## 8. Correct permutation p-value

Use exactly 1000 whole-task fingerprint permutations.

Do not shuffle individual feature columns.

Use the pre-registered corrected empirical p-value:

[
p = \frac{1 + \#\{MAE_{perm} \le MAE_{real}\}}{1001}
]

Store:
- real MAE;
- shuffled mean;
- shuffled 5th/50th/95th percentiles;
- exceedance count;
- corrected p-value.

Write:
`results/phase1_5E_R2_permutation.json`.

---

## 9. Complete negative-transfer reporting

Primary negative-transfer classifier remains:

**logistic regression on causal fingerprint + intervention descriptor**

Secondary:
- standardized kNN;
- small MLP.

Label:

[
y_{neg}=1[\Delta HV_{unit}<-0.01]
]

For each held-out task configuration, report:

- prevalence;
- AUROC if both classes are present;
- balanced accuracy;
- F1;
- Brier.

For MLP:
- 5 training seeds;
- report mean and standard deviation for AUROC, balanced accuracy, F1, and Brier;
- no best-seed selection.

Aggregate these metrics at the task-configuration level.

Write:
`results/phase1_5E_R2_negative_transfer.json`.

This is secondary and cannot rescue a failed primary regression gate.

---

## 10. Preserve probe-cost accounting

Reuse the R1 probe protocol/cost.

Write:
`results/phase1_5E_R2_probe_cost.json`.

Report:
- min/median/max probe fraction;
- count > 0.25;
- count >= 0.50.

No efficiency claim is allowed.

---

## 11. Binding final GO / NO-GO gate

The pre-registered causal-ridge representation must satisfy **all**:

1. macro task MAE < zero-gain;
2. macro task MAE < intervention-only ridge;
3. macro task MAE < global intervention mean;
4. causal-zero bootstrap 95% CI upper bound < 0;
5. causal-intervention-only bootstrap 95% CI upper bound < 0;
6. macro task Spearman > 0.30;
7. real causal-context MAE < shuffled-context mean;
8. corrected permutation p < 0.05.

No partial pass.

### GO

Only then may Planner open Phase 2.

### NO-GO

If any required condition fails:

- close the predictor/VETM architecture path;
- do not try additional representations or model capacity;
- pivot the research to:
  - mechanistic operator sensitivity;
  - stable negative-transfer structure;
  - intervention-response phase diagrams;
  - validity-boundary characterization;
  - conditions under which successful source interventions reverse sign.

---

## 12. Required outputs

Commit:

- repaired mean-score function;
- new invariant/regression tests;
- `results/phase1_5E_R2_fingerprint_by_seed.csv`;
- `results/phase1_5E_R2_causal_fingerprint.csv`;
- `results/phase1_5E_R2_scale_audit.json`;
- `results/phase1_5E_R2_stability.json`;
- `results/phase1_5E_R2_predictability.json`;
- `results/phase1_5E_R2_bootstrap.json`;
- `results/phase1_5E_R2_permutation.json`;
- `results/phase1_5E_R2_negative_transfer.json`;
- `results/phase1_5E_R2_probe_cost.json`;
- `results/phase1_5E_R2_summary.json`;
- `docs/PHASE1_5E_R2_REPORT.md`.

Completion report must include:

- commit hash;
- test count;
- exact repaired formula;
- R1 vs R2 scale audit;
- fingerprint stability;
- macro MAE table;
- three bootstrap CIs;
- macro Spearman;
- corrected permutation p-value;
- negative-transfer aggregate metrics;
- probe cost;
- binding Phase 2 GO/NO-GO;
- if NO-GO, explicit statement that predictor rescue is closed and pivot begins.
