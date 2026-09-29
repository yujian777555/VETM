# Phase 1.5E-R1 — Empirical-Scale Repair and Final Predictability Gate

## Planner decision

Phase 1.5E was executed with the correct causal-probe concept, but the final NO-GO cannot yet be treated as the mandatory fork because two response features were left on raw objective scale.

This revision is a **protocol repair only**.

It is the final predictor test.

Do not:
- change the eight probe anchors;
- add a deeper model;
- tune the representation after seeing held-out results;
- change the GO thresholds;
- add LLM/agent/Cross Attention logic.

After this revision:
- PASS -> Phase 2 may open;
- FAIL -> mandatory pivot to mechanistic operator-sensitivity / negative-transfer boundary analysis.

---

## 1. Preserve all Phase 1.5E protocol choices

Keep exactly:

- tasks: existing 20 metric-healthy task configurations;
- transfer targets: existing Phase 1.5C-R1 condition means;
- probe seeds: 200 and 201;
- common initial population per Task × ProbeSeed;
- one generation;
- same eight off-grid anchors;
- leave-one-problem-out protocol;
- primary regression model: ridge;
- 20 task-configuration bootstrap blocks;
- 5000 bootstrap resamples;
- 1000 task-context permutations.

Do not alter these after inspecting R1 outcomes.

---

## 2. Fix empirical objective scaling

For every Task × ProbeSeed, use only the common initial population.

Let:

[
c_k = \operatorname{median}(F^{init}_{:,k})
]

and

[
s_k = \max(Q_{0.75}(F^{init}_{:,k})-Q_{0.25}(F^{init}_{:,k}), \epsilon)
]

with a fixed epsilon such as `1e-12`.

For every baseline/probe objective vector:

[
\tilde f_k = (f_k-c_k)/s_k
]

The same `c,s` must be reused for:
- baseline one-step branch;
- every probe anchor branch.

### Required corrected response features

Compute these in the empirically normalized space:

- `delta_empirical_mean`
- `delta_objective_spread`
- empirical-HV inputs

Specifically:

[
mean_score(F)=\frac{1}{|F|}\sum_{x\in F}\sum_k \tilde f_k(x)
]

and

[
spread(F)=mean_k(std(\tilde F_{:,k}))
]

Then:

- `delta_empirical_mean = mean_score(probe)-mean_score(baseline)`
- `delta_objective_spread = spread(probe)-spread(baseline)`

Do not retain raw-objective versions in the primary feature matrix.

Raw values may be stored only for diagnostics with clearly separate column names.

### Scale-audit output

For every task, record:
- maximum absolute normalized response feature;
- median absolute normalized response feature;
- count of non-finite features.

Create:
`results/phase1_5E_R1_scale_audit.json`

The audit must explicitly compare DTLZ1 to the remaining tasks.

---

## 3. Keep scale-invariant causal features

The following may remain as currently defined:

- nondominated fraction change;
- dominance win-rate difference;
- offspring-survival difference.

Crowding-distance features may be retained only if the implementation remains internally objective-range normalized, as in the existing crowding-distance calculation.

Empirical HV must continue using only the initial-population-derived normalization/reference box.

No known Pareto-front information is allowed.

---

## 4. Persist raw per-seed fingerprints

Create:

`results/phase1_5E_R1_fingerprint_by_seed.csv`

One row per:
- task configuration;
- probe seed.

Include every active anchor response feature before averaging.

Then create:

`results/phase1_5E_R1_causal_fingerprint.csv`

with task-level mean/std aggregates.

Do not discard collision/availability indicators.

---

## 5. Fingerprint stability

Using the two raw probe seeds, compute task-level fingerprint stability.

For each task:

1. use only response dimensions available for both seeds;
2. exclude constant availability flags from correlation;
3. compute:
   - Pearson correlation;
   - Spearman correlation;
   - cosine similarity;
   - normalized L2 distance.

Report:
- per-task metrics;
- median across tasks;
- IQR across tasks.

Create:
`results/phase1_5E_R1_stability.json`

This is descriptive, not a separate GO gate, but instability must be discussed.

---

## 6. Primary regression rerun

Use exactly the same target and pre-registered primary model:

**ridge regression with causal fingerprint + intervention descriptor**

Also report the same ablations:

1. intervention-only
2. oracle-free passive + intervention
3. causal fingerprint + intervention
4. passive + causal + intervention
5. shuffled causal fingerprint + intervention

No hyperparameter search.

Keep training-only standardization.

---

## 7. Task-configuration-level evaluation

For each leave-one-problem-out split:

- generate predictions for every held-out condition;
- compute error separately for every held-out task configuration.

There must be 20 task blocks total.

Primary macro metrics:

- MAE
- Spearman

Primary bootstrap comparisons:

1. causal - zero
2. causal - intervention-only
3. causal - global intervention mean

Use 5000 task-block bootstrap resamples.

---

## 8. Permutation test

Repeat the existing whole-task fingerprint permutation with 1000 permutations.

Requirements:

- shuffle complete causal fingerprints among training tasks;
- keep intervention descriptors attached to their original condition rows;
- never shuffle individual feature columns;
- evaluate the same leave-one-problem-out predictions.

Empirical p-value:

[
p=\frac{1+\#\{MAE_{perm}\le MAE_{real}\}}{1001}
]

Use the +1 correction.

---

## 9. Negative-transfer detection

The Phase 1.5E plan required this and it was not delivered.

Label:

[
y_{neg}=1[\Delta HV_{unit}<-0.01]
]

Primary classifier:

- logistic regression on causal fingerprint + intervention descriptor.

Secondary:
- standardized kNN;
- small MLP.

For MLP:
- 5 training seeds;
- report mean and standard deviation;
- no best-seed selection.

For every held-out task configuration report:

- prevalence;
- AUROC where both classes exist;
- balanced accuracy;
- F1;
- Brier.

Aggregate at the task level.

Create:
`results/phase1_5E_R1_negative_transfer.json`

This remains a secondary endpoint and cannot rescue a failed primary regression gate.

---

## 10. Probe cost

Reuse the existing cost accounting, but report summary statistics:

- min probe fraction;
- median probe fraction;
- max probe fraction;
- number of tasks with probe fraction > 0.25;
- number of tasks with probe fraction >= 0.50.

Do not make an efficiency claim if the probe cost is a large fraction of full-budget optimization.

---

## 11. Final GO / NO-GO gate

The causal-fingerprint ridge must satisfy **all**:

1. macro task-level MAE < zero-gain;
2. macro task-level MAE < intervention-only ridge;
3. macro task-level MAE < global intervention mean;
4. 95% task-block bootstrap CI for causal-zero has upper bound < 0;
5. 95% task-block bootstrap CI for causal-intervention-only has upper bound < 0;
6. macro task-level Spearman > 0.30;
7. real causal context beats shuffled-context mean;
8. corrected permutation p < 0.05.

No partial pass.

### If GO

Planner may open Phase 2:
- causal task representation;
- intervention encoder;
- uncertainty-aware utility model;
- risk-aware transfer routing.

### If NO-GO

The predictor path is closed.

Do not:
- try Transformer/Cross Attention;
- enlarge MLP;
- hand-select favorable problems;
- retune anchors;
- add oracle features.

Pivot the paper to:
- mechanistic operator sensitivity;
- negative-transfer structure;
- intervention-response phase diagrams;
- validity-boundary characterization.

---

## 12. Required outputs

Commit:

- corrected fingerprint generator;
- tests for normalized response features;
- tests for no-oracle calls;
- per-seed fingerprint output;
- `results/phase1_5E_R1_scale_audit.json`;
- `results/phase1_5E_R1_fingerprint_by_seed.csv`;
- `results/phase1_5E_R1_causal_fingerprint.csv`;
- `results/phase1_5E_R1_stability.json`;
- `results/phase1_5E_R1_predictability.json`;
- `results/phase1_5E_R1_bootstrap.json`;
- `results/phase1_5E_R1_permutation.json`;
- `results/phase1_5E_R1_negative_transfer.json`;
- `results/phase1_5E_R1_probe_cost.json`;
- `results/phase1_5E_R1_summary.json`;
- `docs/PHASE1_5E_R1_REPORT.md`.

Completion report must include:

- commit hash;
- tests passed;
- corrected scale audit;
- DTLZ1 feature-scale before/after comparison;
- fingerprint stability;
- macro MAE table;
- all three bootstrap CIs;
- macro Spearman;
- permutation p-value;
- negative-transfer metrics;
- probe-cost summary;
- final Phase 2 GO/NO-GO;
- if NO-GO, explicit confirmation that the project pivots and no more predictor rescue will be attempted.
