"""Phase 1.5 的确定性算法设计干预注册表。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .nsga2 import NSGA2Config

@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    category: str
    parameters: dict[str, Any]
    applicability: str = "real_box_constrained"

    def apply(self, base: NSGA2Config) -> NSGA2Config:
        values = dict(base.__dict__)
        values.update(self.parameters)
        return NSGA2Config(**values)

def intervention_registry() -> list[Intervention]:
    rows: list[Intervention] = []
    for i, value in enumerate((0.05, 0.10, 0.20, 0.30, 0.50), 1):
        rows.append(Intervention(f"mutation_prob_{i:02d}", "mutation", {"mutation_probability": value}))
    for i, value in enumerate((2.0, 5.0, 10.0, 40.0, 80.0), 1):
        rows.append(Intervention(f"mutation_eta_{i:02d}", "mutation", {"eta_m": value}))
    for i, value in enumerate((0.50, 0.70, 1.00), 1):
        rows.append(Intervention(f"crossover_prob_{i:02d}", "crossover", {"crossover_probability": value}))
    for i, value in enumerate((5.0, 10.0, 20.0, 40.0), 1):
        rows.append(Intervention(f"crossover_eta_{i:02d}", "crossover", {"eta_c": value}))
    for i, value in enumerate((3, 4, 5), 1):
        rows.append(Intervention(f"selection_tournament_{i:02d}", "selection", {"tournament_size": value}))
    return rows

def active_interventions(base: NSGA2Config, n_var: int) -> list[Intervention]:
    """按具体任务过滤完全等于 baseline 的 no-op 干预。"""
    baseline = base.mutation_probability if base.mutation_probability is not None else 1.0 / n_var
    active = []
    for item in intervention_registry():
        values = item.parameters
        if "mutation_probability" in values and np.isclose(values["mutation_probability"], baseline):
            continue
        if "crossover_probability" in values and np.isclose(values["crossover_probability"], base.crossover_probability):
            continue
        if "eta_m" in values and np.isclose(values["eta_m"], base.eta_m):
            continue
        if "eta_c" in values and np.isclose(values["eta_c"], base.eta_c):
            continue
        if "tournament_size" in values and values["tournament_size"] == base.tournament_size:
            continue
        active.append(item)
    return active
