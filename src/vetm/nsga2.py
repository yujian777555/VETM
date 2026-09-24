"""小型、可复现的 NSGA-II 实现，用于生成 Phase 1 经验轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .problems import ProblemSpec


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 40
    generations: int = 30
    crossover_probability: float = 0.9
    mutation_probability: float | None = None
    eta_c: float = 15.0
    eta_m: float = 20.0
    tournament_size: int = 2


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(values: np.ndarray) -> list[np.ndarray]:
    n = len(values)
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q]):
                dominated[p].append(q)
            elif dominates(values[q], values[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return [np.asarray(front, dtype=int) for front in fronts if front]


def crowding_distance(values: np.ndarray, front: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        distance.fill(np.inf)
        return distance
    front_values = values[front]
    for objective in range(values.shape[1]):
        order = np.argsort(front_values[:, objective])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = front_values[order[-1], objective] - front_values[order[0], objective]
        if span == 0:
            continue
        distance[order[1:-1]] += (front_values[order[2:], objective] - front_values[order[:-2], objective]) / span
    return distance


def _rank_and_crowding(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.full(len(values), np.iinfo(np.int32).max, dtype=int)
    crowd = np.zeros(len(values), dtype=float)
    for rank, front in enumerate(fast_non_dominated_sort(values)):
        ranks[front] = rank
        crowd[front] = crowding_distance(values, front)
    return ranks, crowd


def _tournament(rng: np.random.Generator, ranks: np.ndarray, crowd: np.ndarray, size: int = 2) -> int:
    candidates = rng.integers(0, len(ranks), size=max(2, int(size)))
    winner = int(candidates[0])
    for candidate in candidates[1:]:
        candidate = int(candidate)
        if ranks[candidate] < ranks[winner] or (ranks[candidate] == ranks[winner] and crowd[candidate] > crowd[winner]):
            winner = candidate
    return winner


def _sbx(a: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray, eta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = a.copy(), b.copy()
    for i in range(len(a)):
        if rng.random() > 0.5 or abs(a[i] - b[i]) < 1e-14:
            continue
        x1, x2 = min(a[i], b[i]), max(a[i], b[i])
        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - lower[i]) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c1[i] = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        beta = 1.0 + 2.0 * (upper[i] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c2[i] = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
    return np.clip(c1, lower, upper), np.clip(c2, lower, upper)


def _polynomial_mutation(x: np.ndarray, lower: np.ndarray, upper: np.ndarray, probability: float, eta: float, rng: np.random.Generator) -> np.ndarray:
    child = x.copy()
    for i in range(len(child)):
        if rng.random() > probability or upper[i] <= lower[i]:
            continue
        y = child[i]
        delta1, delta2 = (y - lower[i]) / (upper[i] - lower[i]), (upper[i] - y) / (upper[i] - lower[i])
        rnd = rng.random()
        power = 1.0 / (eta + 1.0)
        if rnd <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * xy ** (eta + 1.0)
            deltaq = val**power - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xy ** (eta + 1.0)
            deltaq = 1.0 - val**power
        child[i] = np.clip(y + deltaq * (upper[i] - lower[i]), lower[i], upper[i])
    return child


def hypervolume_2d(values: np.ndarray, reference: np.ndarray | None = None) -> float:
    """计算二维最小化问题的简单 hypervolume，供轨迹摘要使用。"""
    if values.size == 0 or values.shape[1] != 2:
        return float("nan")
    ref = np.asarray(reference if reference is not None else np.max(values, axis=0) + 1.0, dtype=float)
    front = values[np.argsort(values[:, 0])]
    hv, previous_y = 0.0, ref[1]
    for x, y in front:
        if y < previous_y:
            hv += max(0.0, ref[0] - x) * (previous_y - y)
            previous_y = y
    return float(hv)


class NSGA2:
    def __init__(self, problem: ProblemSpec, config: NSGA2Config | None = None, seed: int = 0):
        self.problem = problem
        self.config = config or NSGA2Config()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def run(self) -> dict[str, Any]:
        cfg = self.config
        lower, upper = np.asarray(self.problem.lower), np.asarray(self.problem.upper)
        if cfg.population_size < 4 or cfg.generations < 1:
            raise ValueError("population_size >= 4 且 generations >= 1")
        mutation_probability = (1.0 / self.problem.n_var if cfg.mutation_probability is None else cfg.mutation_probability)
        if cfg.tournament_size < 2:
            raise ValueError("tournament_size 至少为 2")
        population = self.rng.uniform(lower, upper, size=(cfg.population_size, self.problem.n_var))
        objectives = self.problem.evaluate(population)
        history: list[dict[str, float | int]] = []
        function_evaluations = cfg.population_size
        for generation in range(cfg.generations + 1):
            ranks, crowd = _rank_and_crowding(objectives)
            first = objectives[ranks == 0]
            summary: dict[str, float | int] = {
                "generation": generation,
                "population_size": cfg.population_size,
                "nondominated_count": int(len(first)),
                "mean_objective": float(np.mean(objectives)),
                "best_objective_sum": float(np.min(np.sum(objectives, axis=1))),
            }
            if objectives.shape[1] == 2:
                summary["hypervolume"] = hypervolume_2d(first)
            history.append(summary)
            if generation == cfg.generations:
                break
            offspring: list[np.ndarray] = []
            while len(offspring) < cfg.population_size:
                p1 = population[_tournament(self.rng, ranks, crowd, cfg.tournament_size)]
                p2 = population[_tournament(self.rng, ranks, crowd, cfg.tournament_size)]
                if self.rng.random() <= cfg.crossover_probability:
                    c1, c2 = _sbx(p1, p2, lower, upper, cfg.eta_c, self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.extend([
                    _polynomial_mutation(c1, lower, upper, mutation_probability, cfg.eta_m, self.rng),
                    _polynomial_mutation(c2, lower, upper, mutation_probability, cfg.eta_m, self.rng),
                ])
            offspring_array = np.asarray(offspring[: cfg.population_size])
            offspring_objectives = self.problem.evaluate(offspring_array)
            function_evaluations += len(offspring_array)
            combined_population = np.vstack([population, offspring_array])
            combined_objectives = np.vstack([objectives, offspring_objectives])
            new_population: list[np.ndarray] = []
            new_objectives: list[np.ndarray] = []
            for front in fast_non_dominated_sort(combined_objectives):
                if len(new_population) + len(front) <= cfg.population_size:
                    new_population.extend(combined_population[front])
                    new_objectives.extend(combined_objectives[front])
                else:
                    distances = crowding_distance(combined_objectives, front)
                    chosen = front[np.argsort(-distances)[: cfg.population_size - len(new_population)]]
                    new_population.extend(combined_population[chosen])
                    new_objectives.extend(combined_objectives[chosen])
                    break
            population, objectives = np.asarray(new_population), np.asarray(new_objectives)
        ranks, _ = _rank_and_crowding(objectives)
        return {
            "population": population,
            "objectives": objectives,
            "front": objectives[ranks == 0],
            "history": history,
            "seed": self.seed,
            "algorithm": "NSGA-II",
            "function_evaluations": function_evaluations,
            "parameters": {
                "population_size": cfg.population_size,
                "generations": cfg.generations,
                "crossover_probability": cfg.crossover_probability,
                "mutation_probability": mutation_probability,
                "eta_c": cfg.eta_c,
                "eta_m": cfg.eta_m,
                "tournament_size": cfg.tournament_size,
            },
        }


def run_nsga2(problem: ProblemSpec, config: NSGA2Config | None = None, seed: int = 0) -> dict[str, Any]:
    return NSGA2(problem, config=config, seed=seed).run()
