"""Phase 1.5 的确定性算法设计干预注册表。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
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
    for i, value in enumerate((5.0, 10.0, 20.0, 40.0), 1):
        rows.append(Intervention(f"mutation_eta_{i:02d}", "mutation", {"eta_m": value}))
    for i, value in enumerate((0.50, 0.70, 0.90, 1.00), 1):
        rows.append(Intervention(f"crossover_prob_{i:02d}", "crossover", {"crossover_probability": value}))
    for i, value in enumerate((5.0, 10.0, 20.0, 40.0), 1):
        rows.append(Intervention(f"crossover_eta_{i:02d}", "crossover", {"eta_c": value}))
    for i, value in enumerate((2, 3, 4, 5), 1):
        rows.append(Intervention(f"selection_tournament_{i:02d}", "selection", {"tournament_size": value}))
    return rows
