"""ZDT/DTLZ 基准问题定义。

实现只依赖 NumPy，并返回最小的、可序列化的任务描述，方便任务级切分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n_var: int
    n_obj: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    evaluate_fn: Callable[[np.ndarray], np.ndarray]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(self.evaluate_fn(np.asarray(x, dtype=float)), dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_obj:
            raise ValueError(f"{self.name} 返回的目标矩阵形状错误: {values.shape}")
        return values

    def features(self) -> dict[str, float | str]:
        return {
            "problem_id": self.name,
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "lower_mean": float(np.mean(self.lower)),
            "upper_mean": float(np.mean(self.upper)),
            "domain_width_mean": float(np.mean(np.asarray(self.upper) - np.asarray(self.lower))),
        }


def _zdt(name: str, n_var: int) -> ProblemSpec:
    if n_var < 2:
        raise ValueError("ZDT 至少需要两个决策变量")

    def evaluate(x: np.ndarray) -> np.ndarray:
        u = np.asarray(x, dtype=float)
        f1 = u[:, 0]
        if name == "ZDT1":
            g = 1.0 + 9.0 * np.mean(u[:, 1:], axis=1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
        elif name == "ZDT2":
            g = 1.0 + 9.0 * np.mean(u[:, 1:], axis=1)
            f2 = g * (1.0 - (f1 / g) ** 2)
        elif name == "ZDT3":
            g = 1.0 + 9.0 * np.mean(u[:, 1:], axis=1)
            f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        elif name == "ZDT4":
            tail = u[:, 1:]
            g = 1.0 + 10.0 * (n_var - 1) + np.sum(tail**2 - 10.0 * np.cos(4.0 * np.pi * tail), axis=1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
        elif name == "ZDT6":
            f1 = 1.0 - np.exp(-4.0 * u[:, 0]) * np.sin(6.0 * np.pi * u[:, 0]) ** 6
            g = 1.0 + 9.0 * np.mean(u[:, 1:], axis=1) ** 0.25
            f2 = g * (1.0 - (f1 / g) ** 2)
        else:
            raise KeyError(name)
        return np.column_stack([f1, f2])

    if name == "ZDT4":
        lower = (0.0,) + (-5.0,) * (n_var - 1)
        upper = (1.0,) + (5.0,) * (n_var - 1)
    else:
        lower = (0.0,) * n_var
        upper = (1.0,) * n_var
    return ProblemSpec(name, n_var, 2, lower, upper, evaluate)


def _dtlz(name: str, n_var: int, n_obj: int) -> ProblemSpec:
    if n_obj < 2 or n_var < n_obj:
        raise ValueError("DTLZ 要求 n_obj >= 2 且 n_var >= n_obj")
    k = n_var - n_obj + 1

    def evaluate(x: np.ndarray) -> np.ndarray:
        u = np.asarray(x, dtype=float)
        tail = u[:, n_obj - 1 :]
        if name == "DTLZ1":
            g = 100.0 * (k + np.sum((tail - 0.5) ** 2 - np.cos(20.0 * np.pi * (tail - 0.5)), axis=1))
            scale = 0.5 * (1.0 + g)
            out = []
            for j in range(n_obj):
                value = scale.copy()
                value *= np.prod(u[:, : n_obj - j - 1], axis=1) if n_obj - j - 1 else 1.0
                if j:
                    value *= 1.0 - u[:, n_obj - j - 1]
                out.append(value)
            return np.column_stack(out)
        if name == "DTLZ2":
            g = np.sum((tail - 0.5) ** 2, axis=1)
            scale = 1.0 + g
            out = []
            for j in range(n_obj):
                value = scale.copy()
                value *= np.prod(np.cos(u[:, : n_obj - j - 1] * np.pi / 2.0), axis=1) if n_obj - j - 1 else 1.0
                if j:
                    value *= np.sin(u[:, n_obj - j - 1] * np.pi / 2.0)
                out.append(value)
            return np.column_stack(out)
        raise KeyError(name)

    return ProblemSpec(name, n_var, n_obj, (0.0,) * n_var, (1.0,) * n_var, evaluate)


def get_problem(name: str, *, n_var: int | None = None, n_obj: int = 3) -> ProblemSpec:
    """返回标准 benchmark；默认 DTLZ 使用 3 个目标和 7 个变量。"""

    normalized = name.upper()
    if normalized.startswith("ZDT"):
        return _zdt(normalized, n_var or 30)
    if normalized.startswith("DTLZ"):
        return _dtlz(normalized, n_var or (n_obj + 4), n_obj)
    raise KeyError(f"未知 benchmark: {name}")


def reference_front(problem: ProblemSpec) -> np.ndarray:
    """由问题定义固定生成参考 Pareto 前沿，不读取算法运行结果。"""
    name = problem.name
    if name.startswith("ZDT"):
        if name == "ZDT3":
            intervals = [(0, .0830015349), (.182228728, .2577623634),
                         (.4093136748, .4538821041), (.6183967944, .6525117038),
                         (.8233317983, .8518328654)]
            x = np.concatenate([np.linspace(a, b, 60) for a, b in intervals])
            y = 1 - np.sqrt(x) - x * np.sin(10 * np.pi * x)
        else:
            x = np.linspace(.2807753191 if name == "ZDT6" else 0, 1, 300)
            y = 1 - (x ** 2 if name in {"ZDT2", "ZDT6"} else np.sqrt(x))
        return np.column_stack([x, y])
    directions = np.vstack([
        np.eye(problem.n_obj),
        np.random.default_rng(1729).dirichlet(np.ones(problem.n_obj), 300),
    ])
    if name == "DTLZ1":
        return .5 * directions
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
