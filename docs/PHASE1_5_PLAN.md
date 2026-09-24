# Phase 1.5 — Establish a Scientifically Valid Transfer Matrix

## Planner decision

Phase 1 engineering foundation is accepted, but the scientific Go/No-Go gate is **not yet passed**.

Do **not** implement Cross Attention or the final VETM architecture in this phase.

The goal of Phase 1.5 is to determine whether algorithm-design experience has a stable, task-conditioned validity boundary under a rigorous multi-objective evaluation protocol.

## Why Phase 1.5 is required

The current Phase 1 prototype has several limitations that prevent a scientific claim:

1. The pre-leakage-fix MLP AUROC must not be treated as valid evidence.
2. Transfer labels currently use the final-population mean objective, which is not an acceptable primary multi-objective quality measure.
3. The current experience space is mostly a small set of NSGA-II parameter variations rather than a broad controlled intervention library.
4. Current runs use a very small budget and only three seeds.
5. The task space has little structural variation, so a model may learn benchmark identity rather than a transferable validity boundary.

Phase 1.5 must fix these issues before any complex model is introduced.

---

## 1. Primary scientific question

For a controlled algorithm-design intervention \(e_i\) and target task configuration \(T_j\), estimate the transfer effect

\[
\Delta HV_{ij}=HV(A\oplus e_i,T_j)-HV(A,T_j)
\]

under paired random seeds and equal evaluation budgets.

The key phenomenon to establish is:

> The same intervention is beneficial on some tasks and harmful on others, and this sign change is stable enough to be predicted from task/intervention context.

---

## 2. Evaluation metric

### Primary metric

Use **normalized Hypervolume (HV)** with a task-specific but pre-registered fixed reference point.

Requirements:

- Do not derive the reference point from the evaluated final population.
- The same task configuration must use the same reference point for baseline and intervention runs.
- Add numerical unit tests for known fronts.
- For 3+ objectives, use a validated standard implementation rather than an ad-hoc estimator.

### Secondary metric

Use **IGD or IGD+** when a trusted reference Pareto front is available.

Runtime may be recorded, but it is not the Phase 1.5 transfer label.

---

## 3. Task configuration grid

A task is not just a benchmark name.

Represent a task as a configuration containing at least:

- problem family
- problem name
- number of variables
- number of objectives
- population size
- evaluation budget

Construct approximately **30–50 task configurations**.

Minimum coverage:

- ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
- DTLZ1, DTLZ2
- WFG family if a validated implementation is available

Vary dimensions and, for DTLZ/WFG, objective count.

Do not encode benchmark name directly into the learned baseline features used for the final held-out generalization test. It may remain in metadata.

---

## 4. Controlled intervention library

Build at least **20–30 deterministic interventions**.

Each intervention must have:

- unique intervention_id
- category
- complete parameters
- deterministic semantics
- applicability constraints

Cover multiple categories where feasible:

### Mutation
- mutation probability increase/decrease
- polynomial mutation eta increase/decrease
- scheduled/adaptive mutation

### Crossover
- SBX probability increase/decrease
- SBX eta increase/decrease
- scheduled crossover

### Selection
- tournament pressure variants
- altered parent-selection pressure

### Diversity
- stronger/weaker crowding pressure
- diversity preservation variant
- random immigrants

### Restart
- stagnation-triggered restart
- partial restart

### Archive / elitism
- external archive on/off or size variants, if implemented rigorously

### Population / refinement
- population-size adaptation
- bounded local refinement / top-k refinement, if budget accounting is exact

Avoid LLM-generated interventions in Phase 1.5.

---

## 5. Paired experimental protocol

For every eligible Task × Intervention condition:

- run baseline and intervention under the **same seed**
- use identical evaluation budget
- use identical initial population when technically possible
- record all failures
- do not silently retry failed runs

Start with **10 paired seeds** per condition.

For conditions near decision boundaries or used in final analysis, increase to **20 paired seeds** when computationally feasible.

Record:

- baseline HV
- intervention HV
- delta_HV
- baseline IGD/IGD+
- intervention IGD/IGD+
- runtime
- function evaluations
- failure status
- seed
- config hash
- code commit

---

## 6. Aggregated transfer label

Do not label a single noisy run as valid simply because delta > 0.

Aggregate by Task × Intervention.

For each condition report:

- mean paired delta_HV
- median paired delta_HV
- standard deviation
- 95% bootstrap confidence interval
- effect size

Create three descriptive states:

- positive
- neutral
- negative

The exact practical-effect tolerance must be fixed before inspecting the full matrix and stored in config.

A suggested initial rule is:

- positive: CI supports gain beyond a pre-registered practical tolerance
- negative: CI supports degradation beyond the same tolerance
- neutral: otherwise

Do not change the tolerance after seeing the held-out results.

---

## 7. Required analyses

Generate a real **Task × Intervention Transfer Matrix**.

Required analyses:

1. Percentage of interventions that show both positive and negative transfer across tasks.
2. Per-intervention sign entropy / heterogeneity.
3. Negative-transfer rate of naive global reuse.
4. Distribution of effect sizes.
5. Task-feature association with transfer effect.
6. Family-held-out generalization.

Produce heatmaps/tables from aggregated paired results, not single-run labels.

---

## 8. Predictability baselines

Only after the transfer matrix is built, train simple models to test whether the phenomenon is learnable.

Required baselines:

- random / majority
- global-best historical intervention
- nearest-neighbor
- logistic regression
- tree baseline such as XGBoost/RandomForest if dependency policy permits
- small MLP

Targets:

- sign classification
- delta_HV regression

Report at minimum:

- AUROC for positive-vs-nonpositive or negative-transfer detection
- F1 / balanced accuracy
- Spearman correlation for delta_HV ranking
- MAE for delta_HV
- Brier score / calibration where probability outputs exist

---

## 9. Data splitting

Do not rely on a single random split.

Required evaluation:

1. Task-configuration-held-out
2. Family-held-out / leave-family-out where possible
3. Structural OOD, e.g. lower objective counts for train and higher objective counts for test

The source experience/intervention bank must respect the same split boundary. No target-family experience may leak into a family-held-out source bank.

---

## 10. Go / No-Go gate

### GO to Phase 2 only if all are supported

1. A non-trivial subset of interventions exhibits stable positive/negative sign reversals across tasks.
2. Those reversals persist under paired multi-seed evaluation and are not explained by single-seed noise.
3. At least one simple model predicts transfer sign or transfer utility above trivial baselines on held-out tasks/families.
4. Results remain qualitatively consistent when using HV and at least one secondary Pareto-quality metric.

### NO-GO / redesign

If interventions are almost universally good/bad, sign reversals disappear with more seeds, or held-out prediction collapses to chance, do not build VETM yet. Revisit task diversity, intervention design, or the research hypothesis.

---

## 11. Required deliverables

Commit:

- implementation changes
- intervention registry
- task-grid configs
- metric/reference-front code
- tests
- raw aggregated statistics or reproducible generation scripts
- `results/phase1_5_transfer_matrix.csv`
- `results/phase1_5_summary.json`
- `docs/PHASE1_5_REPORT.md`

The report must explicitly state the Planner Go/No-Go evidence without claiming final VETM effectiveness.

## Executor constraints

- Do not change the research hypothesis.
- Do not implement Cross Attention/VETM.
- Do not add an LLM or agent layer.
- Do not tune the protocol using held-out test outcomes.
- Preserve raw results and failed runs.
- If compute cost is unexpectedly large, stop after a pilot and report estimated full-run cost instead of silently shrinking the protocol.
