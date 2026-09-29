# Phase 2M — Mechanistic Transfer Boundary Mapping

## Planner decision

The predictive VETM path is closed after the binding Phase 1.5E-R2 NO-GO.

Do not continue trying to predict intervention utility with a larger model.

The project pivots to the strongest replicated scientific result already present in the repository:

> The same evolutionary-search intervention can produce stable positive or negative long-horizon effects depending on task context.

Phase 2M asks:

> **When and through what search dynamics do evolutionary interventions reverse sign across multi-objective optimization tasks?**

The goal is not zero-shot prediction. The goal is a mechanistic, reproducible map of intervention-response regimes and negative-transfer boundaries.

---

## 1. Freeze the predictor branch

Treat the following as final predictor evidence:

- Phase 1.5C-R1: stable sign-reversal phenomenon established;
- Phase 1.5D: passive context prediction NO-GO;
- Phase 1.5E/R1/R2: oracle-free causal fingerprint prediction NO-GO;
- no further predictor architecture rescue.

Create a short archival note:

`docs/PREDICTOR_PATH_CLOSED.md`

It must record:
- final commit;
- final primary gate values;
- why the predictor path was closed;
- what evidence remains valid;
- that future mechanistic experiments are a pivot, not post-hoc predictor tuning.

Also fix the two archival reproducibility issues without rerunning the predictor study:
1. make the committed permutation formula use the +1 correction;
2. make the negative-transfer reporting code capable of emitting balanced accuracy and MLP AUROC mean/std.

These are code/report hygiene only. Do not reinterpret the final predictor decision.

---

## 2. Primary mechanistic hypotheses

Do not assume benchmark identity itself is the mechanism.

Test whether sign reversals are associated with measurable changes in search dynamics.

### H-M1: Exploration intensity boundary
Changes in mutation probability / mutation distribution index alter:
- decision-space step size;
- offspring survival;
- nondominated discovery;
- objective-space spread;
- convergence rate.

The same exploration change can help in one task regime and hurt in another.

### H-M2: Recombination locality boundary
Changes in SBX distribution index alter the locality of recombination and may reverse sign depending on:
- search-stage geometry;
- dimensionality;
- diversity state.

### H-M3: Temporal mediation
Final positive/negative transfer is preceded by a reproducible trajectory signature:
- early exploration gain/loss;
- mid-run diversity retention/collapse;
- late convergence acceleration/stagnation.

These are hypotheses to test, not conclusions to assume.

---

## 3. Core reversal panel

Start from interventions with already replicated sign reversals.

Priority interventions:

1. `mutation_eta_01`
2. `mutation_prob_03`
3. `mutation_prob_04`
4. `crossover_eta_01`

Use task contrasts already supported by Phase 1.5C-R1, including where applicable:

- ZDT1 / ZDT2 / ZDT3 positive regimes;
- ZDT4 negative regime;
- ZDT6 dimension-dependent regimes;
- DTLZ1 negative regimes;
- DTLZ2 negative/neutral regimes.

The exact Stage-A panel must be selected from pre-existing confirmed reversal evidence only.

Do not choose cases based on new trajectory outcomes.

Write:
`configs/phase2m_reversal_panel.json`

For each selected Task × Intervention pair, include:
- source evidence file;
- current state (positive/negative);
- effect size;
- CI;
- seed count;
- paired opposite-sign comparison used in the mechanistic contrast.

---

## 4. Parameterize interventions mechanistically

Absolute mutation probability is difficult to compare across dimensions.

Add derived variables:

### Mutation probability
[
\lambda_m = p_m \times n_{var}
]

Interpretation:
expected number of mutated variables before boundary/repair effects.

Store both:
- absolute `p_m`;
- dimension-normalized `lambda_m`.

### Mutation / crossover eta
Keep:
- `eta_m`;
- `eta_c`.

Also derive a step-locality descriptor from actual offspring:
- normalized decision-space displacement from parent(s).

Do not infer locality only from eta values; measure realized displacement.

---

## 5. Trajectory instrumentation

Current transfer matrix mainly stores final outcomes. Phase 2M requires trajectory-level mechanism data.

For every run record checkpoints at approximately:
- 0%
- 10%
- 20%
- 40%
- 60%
- 80%
- 100%

of the evaluation budget.

At each checkpoint record at minimum:

### Outcome
- unit-normalized HV;
- IGD/IGD+ for benchmark evaluation only.

### Population state
- nondominated ratio;
- objective-space spread;
- decision-space diversity;
- crowding-distance distribution summary;
- distance-to-boundary / boundary-hit rate if meaningful.

### Reproduction dynamics
- mean and quantiles of offspring-parent decision displacement;
- fraction of offspring surviving environmental selection;
- fraction of offspring dominating at least one parent / comparator where well-defined;
- duplicate rate;
- mutation-variable count;
- realized crossover displacement.

### Progress
- delta HV over previous checkpoint;
- delta IGD over previous checkpoint;
- stagnation length;
- best-front turnover / replacement rate.

All metrics must be deterministic given the run output and documented.

---

## 6. Stage A — focused mechanistic replication

Do not immediately rerun the full 390-condition matrix.

Select approximately:
- 4 priority interventions;
- 8–12 task configurations;
- 20 paired seeds.

Each selected intervention must appear in at least one already-confirmed positive and one already-confirmed negative regime.

Baseline and intervention:
- same seed;
- same initial population;
- same evaluation budget.

Run full trajectory instrumentation.

Outputs:
- `results/phase2m_stageA_runs.csv`
- `results/phase2m_stageA_trajectories.csv`
- `results/phase2m_stageA_summary.json`

---

## 7. Mechanism contrast analysis

For each intervention with opposite-sign task regimes, compare trajectories using paired seed statistics.

### Required contrasts

For every checkpoint report:
- paired delta HV;
- paired delta IGD;
- delta diversity;
- delta displacement;
- delta offspring survival;
- delta nondominated ratio;
- delta stagnation.

For each final effect, test which earlier trajectory variables differ between positive and negative regimes.

Do not claim causal mediation from correlation alone.

Call these:
- mechanistic associations;
- trajectory signatures;
- candidate mediators.

Use:
- paired bootstrap CI;
- effect size;
- rank correlation across conditions.

Avoid single-run exemplars as evidence.

---

## 8. Intervention-response phase diagrams

Build dense phase diagrams only after Stage A reproduces the mechanism signal.

### Mutation-probability diagram

Use a pre-registered grid expressed in `lambda_m = p_m * n_var`.

Suggested initial grid:
- 0.5
- 1
- 2
- 3
- 5

Convert to task-specific `p_m = lambda_m / n_var`, clipped only if required by valid probability range.

### Mutation eta diagram

Suggested grid:
- 2
- 5
- 10
- 20
- 40
- 80

### Crossover eta diagram

Suggested grid:
- 5
- 10
- 15
- 20
- 40

For each grid:
- use paired seeds;
- retain equal budgets;
- report final effect and trajectory mechanism variables.

Do not tune grids after inspecting results.

Outputs:
- `results/phase2m_phase_diagram_mutation_probability.csv`
- `results/phase2m_phase_diagram_mutation_eta.csv`
- `results/phase2m_phase_diagram_crossover_eta.csv`

---

## 9. Structural factors

Analyze effect boundaries against pre-specified task factors:

- decision dimension;
- objective count;
- evaluation budget;
- problem family;
- bound-width statistics;
- baseline diversity;
- baseline convergence rate.

Benchmark/problem name may be used as metadata and for descriptive stratification in this mechanistic phase, but not as a substitute for mechanism.

Do not claim a factor explains the reversal unless the direction is reproduced across more than one configuration.

---

## 10. Temporal sign analysis

For each intervention/task pair, classify the trajectory pattern:

1. **consistently beneficial**
2. **consistently harmful**
3. **early-help / late-harm**
4. **early-harm / late-help**
5. **neutral until late divergence**
6. **unstable/noisy**

Define classification rules before inspecting final plots.

Report how often final sign reversals arise from:
- persistent differences;
- temporal crossovers;
- late-stage divergence.

This is important because two interventions can have the same final delta HV for very different reasons.

---

## 11. Negative-transfer boundary atlas

Create a compact atlas where each row is an intervention and each column is a task configuration/regime.

Required fields:
- final sign;
- effect size;
- CI;
- HV/IGD directional agreement;
- dominant trajectory signature;
- dimension;
- objective count;
- budget;
- `lambda_m` or eta value;
- whether independently replicated.

Write:
- `results/phase2m_boundary_atlas.csv`
- `docs/PHASE2M_BOUNDARY_ATLAS.md`

The atlas is descriptive evidence, not a learned predictor.

---

## 12. Statistical requirements

Use task/intervention condition as the primary scientific unit, with paired seeds underneath.

For Stage A:
- 20 paired seeds.

For dense phase-diagram sweeps:
- start with 10 paired seeds;
- extend boundary-adjacent / sign-changing cells to 20 seeds.

Report:
- mean;
- median;
- standard deviation;
- 95% paired bootstrap CI;
- practical threshold `|delta_HV_unit| > 0.01`.

Retain the existing positive/neutral/negative classification rule unless Planner explicitly changes it before new data are inspected.

Multiple comparisons:
- do not claim isolated p-values as discoveries;
- emphasize effect sizes, CIs, replication, and coherent regime structure.

---

## 13. Stage-A Go / No-Go

### GO to full phase diagrams if

1. existing positive/negative signs reproduce under trajectory instrumentation;
2. at least two intervention families show distinct, reproducible trajectory signatures between positive and negative regimes;
3. the same signature is visible across multiple task configurations rather than one benchmark only;
4. HV and IGD remain directionally consistent for the main reversal cases.

### Narrow / redesign if

- final reversal signs fail to reproduce;
- mechanism variables are pure noise across seeds;
- only benchmark identity separates regimes with no interpretable dynamic difference.

If Stage A fails, stop before dense parameter sweeps.

---

## 14. Paper-level claims allowed after Phase 2M

If supported, the paper may claim:

- evolutionary-search interventions exhibit reproducible negative-transfer boundaries;
- the same operator change can reverse sign across task regimes;
- reversal regimes are accompanied by distinct search-dynamic signatures;
- zero/few-shot prediction from passive or one-step task descriptors was not robust in this benchmark;
- a mechanistic phase-diagram view is more faithful than assuming universal transferability.

Do not claim:
- a universal predictor;
- general algorithm selection;
- autonomous self-improvement;
- VETM effectiveness;
- causally identified mediators unless a separate mediation design supports it.

---

## 15. Required deliverables

Commit:

- `docs/PREDICTOR_PATH_CLOSED.md`
- archival reproducibility cleanup;
- trajectory instrumentation;
- `configs/phase2m_reversal_panel.json`;
- Stage-A experiment runner;
- tests;
- `results/phase2m_stageA_runs.csv`;
- `results/phase2m_stageA_trajectories.csv`;
- `results/phase2m_stageA_summary.json`;
- `results/phase2m_boundary_atlas.csv`;
- `docs/PHASE2M_REPORT.md`.

Do **not** run the dense phase-diagram sweep until Stage A has been completed and reviewed by Planner.

Completion reply for Stage A:
- commit hash;
- tests passed;
- selected reversal panel;
- run count;
- reproduced positive/negative conditions;
- checkpoint trajectory summaries;
- candidate mechanism signatures;
- HV/IGD agreement;
- Stage-A GO/NO-GO for dense phase diagrams.

## Executor constraints

- No predictor rescue.
- No Cross Attention.
- No Transformer.
- No LLM/agent layer.
- No model-selection loop.
- No post-hoc task selection from Stage-A outcomes.
- Preserve failures and raw trajectories.
