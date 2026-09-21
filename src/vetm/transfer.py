"""用相同目标任务和随机种子构造受控 transfer label。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .nsga2 import NSGA2Config, run_nsga2
from .problems import ProblemSpec
from .schema import ExperienceRecord


def _score(run: dict) -> float:
    return float(np.mean(np.asarray(run["objectives"], dtype=float)))


def evaluate_parameter_transfer(
    target_problem: ProblemSpec,
    source: ExperienceRecord,
    *,
    seed: int,
    baseline_config: NSGA2Config,
    gain_tolerance: float = 0.0,
) -> tuple[int, float, dict]:
    """复用源经验的 NSGA-II 参数，并与 B0 共享目标任务随机种子。

    signed_gain > 0 表示 transfer 的平均目标值更低；标签只由目标运行结果产生。
    """
    source_params = source.parameters
    transfer_config = replace(
        baseline_config,
        population_size=baseline_config.population_size,
        generations=baseline_config.generations,
        crossover_probability=float(source_params.get("crossover_probability", baseline_config.crossover_probability)),
        mutation_probability=float(source_params.get("mutation_probability", baseline_config.mutation_probability or 1.0 / target_problem.n_var)),
        eta_c=float(source_params.get("eta_c", baseline_config.eta_c)),
        eta_m=float(source_params.get("eta_m", baseline_config.eta_m)),
    )
    baseline = run_nsga2(target_problem, baseline_config, seed=seed)
    transferred = run_nsga2(target_problem, transfer_config, seed=seed)
    signed_gain = _score(baseline) - _score(transferred)
    label = int(signed_gain > abs(gain_tolerance))
    metadata = {
        "baseline_score": _score(baseline),
        "transfer_score": _score(transferred),
        "transfer_parameters": transfer_config.__dict__,
        "target_seed": seed,
        "baseline_function_evaluations": baseline["function_evaluations"],
        "transfer_function_evaluations": transferred["function_evaluations"],
    }
    return label, float(signed_gain), metadata
