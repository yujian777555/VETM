# Phase 1.5E — Oracle-Free Causal Response Fingerprint

## Planner decision

Phase 1.5D established a conservative NO-GO for passive static + early-trajectory context.

The transfer-boundary phenomenon itself remains supported by the independently replicated sign reversals from Phase 1.5C-R1.

This phase is the **final representation rescue before a research pivot**.

The question is:

> Can a cheap, oracle-free causal response fingerprint of the target task predict the long-horizon utility of algorithm-design interventions better than trivial transfer baselines?

Do **not** implement final VETM, Cross Attention, LLM, or agent logic in this phase.

If this phase fails the pre-registered gate, stop the predictor-architecture path and reframe the paper around mechanistic transfer sensitivity / negative-transfer structure.

---

## 1. First repair the Phase 1.5D statistical audit

Before new probing, recompute the Phase 1.5D primary aggregate correctly.

For each leave-one-problem-out split:

1. predict every held-out condition;
2. compute MAE separately for each held-out **task configuration**;
3. collect one model-minus-baseline MAE difference per task configuration.

The bootstrap unit is the task configuration, not the problem identity.

Required:
- 20 task-configuration blocks;
- 5000 bootstrap resamples;
- real-model vs zero-gain;
- real-model vs intervention-only;
- real-model vs global-intervention mean.

Write:
- `results/phase1_5D_R1_taskblock_audit.json`

This audit is diagnostic only. Do not retroactively change the Phase 1.5D feature/model choices.

---

## 2. Oracle-free requirement

Predictor inputs in Phase 1.5E must not use:

- `reference_front(problem)`;
- true Pareto-front ideal/nadir;
- IGD / IGD+;
- benchmark name;
- final full-budget outcomes;
- transfer labels;
- any statistic requiring oracle knowledge of the optimum/front.

Known Pareto fronts and calibrated unit-HV remain allowed only for:
- long-horizon benchmark labels;
- scientific evaluation after prediction.

The task representation itself must be deployable on an unknown black-box multi-objective task.

---

## 3. Causal fingerprint concept

Passive trajectories did not robustly predict intervention utility.

Instead, create a **low-fidelity causal response fingerprint** by applying a small fixed panel of off-grid probe interventions for exactly one generation from a common initial population.

The probe asks:

> How does this task immediately respond to controlled changes in search behavior?

This is not the target long-horizon intervention result. It is a cheap diagnostic experiment.

---

## 4. Probe protocol

### Dedicated probe seeds

Use:
- 200
- 201

These are disjoint from:
- transfer evaluation seeds 0–19;
- passive-context probe seeds 100–104.

### Common initial population

For every Task × ProbeSeed:

1. generate one initial population of size 50;
2. evaluate it once;
3. reuse the exact same initial population for:
   - baseline one-step evolution;
   - every probe intervention.

The probe API must support injecting a fixed initial population.

Add a deterministic test proving baseline/probe branches begin from byte-identical initial decision vectors.

### Probe horizon

Exactly **one generation** after the common initial population.

Each branch may consume the same number of new function evaluations.

Record exact probe cost.

---

## 5. Off-grid probe panel

Probe actions must not be identical to target intervention registry values.

Use the following fixed panel:

1. mutation probability = 0.5 × baseline `1/n_var`
2. mutation probability = 2.0 × baseline `1/n_var`
3. polynomial mutation eta = 7
4. polynomial mutation eta = 30
5. crossover probability = 0.8
6. SBX eta = 12
7. SBX eta = 30
8. tournament size = 6

These are **probe-only anchors** and must not be added to the target intervention registry.

If a probe becomes numerically identical to baseline after clipping or validation, report it and exclude that probe dimension rather than silently keeping a no-op.

---

## 6. Oracle-free response features

All response features are computed relative to the baseline one-step branch from the same initial population and seed.

### Initial-population empirical scaling

For each objective, derive scaling only from the common initial population.

Allowed examples:
- median + IQR;
- min/max with epsilon floor.

Freeze the exact rule before producing final fingerprints.

Do not use the true Pareto front.

### Required response features per probe anchor

At minimum compute:

- change in nondominated fraction;
- dominance win-rate of probe population/front vs baseline one-step population/front;
- change in empirical objective spread;
- change in crowding-distance summary;
- change in empirically normalized objective aggregate;
- empirical-HV change using an initial-population-derived normalization/reference box;
- offspring survival-rate difference if available.

For empirical HV:
- derive normalization/reference exclusively from the initial population;
- use the exact same empirical box for baseline and all anchors of that Task × ProbeSeed;
- treat this only as a probe feature, not the scientific outcome metric.

Aggregate the two probe seeds with:
- mean;
- absolute seed difference or standard deviation.

Create:
- `results/phase1_5E_causal_fingerprint.csv`
- one row per task configuration.

---

## 7. Passive oracle-free task features

Build a small passive feature set from the common initial population only.

Examples:
- n_var;
- n_obj;
- evaluation budget;
- bound-width statistics;
- objective variance;
- objective Spearman-correlation summary;
- nondominated fraction;
- dominance ratio;
- decision-objective correlation summary;
- empirical spread.

No benchmark name or oracle front information.

Write:
- `results/phase1_5E_oracle_free_static.csv`

---

## 8. Prediction targets

Use the existing long-horizon Phase 1.5C-R1 aggregated Task × Intervention conditions.

Primary regression target:
- `mean delta_HV_unit`

Primary negative-transfer label:
[
y_{neg}=1[Delta HV_{unit}<-0.01]
]

Known-front HV remains acceptable here because it is the benchmark outcome label, not an input.

---

## 9. Feature ablations

Evaluate exactly:

1. **intervention-only**
2. **oracle-free passive + intervention**
3. **causal fingerprint + intervention**
4. **passive + causal fingerprint + intervention**
5. **shuffled causal fingerprint + intervention**

The primary representation is:
- **causal fingerprint + intervention**

Use passive+causal only as a secondary ablation.

---

## 10. Primary model

Use **ridge regression** as the pre-registered primary regression model.

Reason:
- the goal is to test the representation, not model capacity.

Do not choose a different primary model after seeing test results.

Secondary:
- standardized kNN;
- small MLP reported descriptively.

For negative-transfer detection:
- logistic regression is primary;
- standardized kNN and small MLP are secondary.

For MLPs:
- use 5 training seeds;
- report mean ± std;
- do not select the best seed.

---

## 11. Held-out protocol

Primary evaluation:
- leave-one-problem-out;
- no target-problem conditions in training.

Within each held-out problem:
- compute metrics separately for every task configuration.

Primary bootstrap unit:
- task configuration.

There should be approximately 20 task blocks, not 7 problem blocks.

Secondary:
- ZDT → DTLZ;
- DTLZ → ZDT;
- objective-count 2 → 3;
- dimension OOD.

---

## 12. Primary GO / NO-GO gate

For the pre-registered ridge using **causal fingerprint + intervention**:

### Required for GO

1. macro task-level MAE < zero-gain MAE;
2. macro task-level MAE < intervention-only ridge MAE;
3. 95% task-block bootstrap CI for
   `MAE_causal - MAE_zero`
   has upper bound < 0;
4. 95% task-block bootstrap CI for
   `MAE_causal - MAE_intervention_only`
   has upper bound < 0;
5. held-out Spearman is positive with macro value > 0.30;
6. causal fingerprint beats its shuffled-context control;
7. a 1000-permutation task-context test gives empirical `p < 0.05`.

No “best split wins” rule is allowed.

### Secondary negative-transfer evidence

Report macro task-level:
- AUROC;
- balanced accuracy;
- F1;
- Brier;
- prevalence.

This is supportive but does not replace the primary regression gate.

---

## 13. Probe-cost accounting

Report the actual cost of creating a fingerprint:

- initial-population evaluations;
- baseline one-step evaluations;
- 8 anchor one-step evaluations;
- number of probe seeds;
- total evaluations per task.

Report:
[
probe_cost / calibrated_full_budget
]

for every task.

Do not claim evaluation efficiency yet if the causal fingerprint consumes a large fraction of the target-task budget.

If Phase 1.5E passes, Phase 2 may optimize probe selection/cost using uncertainty or active validity learning.

---

## 14. Required controls

### Fingerprint stability
Report similarity/correlation between seed-200 and seed-201 fingerprints.

### Target-intervention leakage check
Probe anchors are off-grid and not in the target intervention registry.

### Oracle check
Add a test/search assertion that the fingerprint-generation code does not call:
- `reference_front`;
- `calibrate_task`;
- IGD functions.

### Shuffle control
Shuffle complete task-level causal fingerprints across training tasks, never individual feature columns.

---

## 15. Deliverables

Commit:

- fixed initial-population injection support;
- one-step probe implementation;
- tests;
- `results/phase1_5D_R1_taskblock_audit.json`;
- `results/phase1_5E_causal_fingerprint.csv`;
- `results/phase1_5E_oracle_free_static.csv`;
- `results/phase1_5E_predictability.json`;
- `results/phase1_5E_bootstrap.json`;
- `results/phase1_5E_permutation.json`;
- `results/phase1_5E_probe_cost.json`;
- `results/phase1_5E_summary.json`;
- `docs/PHASE1_5E_REPORT.md`.

The report must separate:
1. Phase 1.5D corrected task-block audit;
2. causal-fingerprint stability;
3. oracle-free feature construction;
4. primary regression gate;
5. negative-transfer detection;
6. probe cost;
7. GO/NO-GO.

---

## 16. Final fork after Phase 1.5E

### If GO

Proceed to Phase 2 with:
- causal task fingerprint;
- intervention encoder;
- uncertainty-aware transfer utility prediction;
- risk-aware routing.

### If NO-GO

Do not try a deeper predictor merely to rescue the result.

Reframe the project toward:
- mechanistic operator sensitivity;
- stable negative-transfer structure;
- intervention-response phase diagrams;
- validity-boundary characterization without claiming robust zero/few-shot prediction.

This fork is mandatory.

## Executor constraints

- No final VETM architecture.
- No Cross Attention.
- No LLM/agent layer.
- No oracle Pareto-front information in predictor features.
- No changing probe anchors after inspecting target outcomes.
- No post-hoc primary-model selection.
- No changing GO thresholds after seeing results.
