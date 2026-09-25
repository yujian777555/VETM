"""Phase 1.5B 的任务级 HV 校准和健康检查。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from .problems import ProblemSpec, reference_front

@dataclass(frozen=True)
class TaskCalibration:
    problem_id: str
    ideal: np.ndarray
    nadir: np.ndarray
    raw_reference: np.ndarray
    normalized_reference: np.ndarray
    valid: bool

def calibrate_task(problem: ProblemSpec, normalized_reference_value: float = 1.1) -> TaskCalibration:
    if normalized_reference_value <= 1.0:
        raise ValueError("normalized_reference_value 必须大于 1")
    front = np.asarray(reference_front(problem), dtype=float)
    if front.ndim != 2 or front.shape[1] != problem.n_obj or not np.isfinite(front).all():
        raise ValueError("固定参考前沿无效")
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    span = nadir - ideal
    valid = bool(np.isfinite(span).all() and np.all(span > 1e-12))
    if not valid:
        raise ValueError("任务参考前沿的目标范围无效")
    raw_reference = ideal + normalized_reference_value * span
    return TaskCalibration(problem.name, ideal, nadir, raw_reference,
                           np.full(problem.n_obj, normalized_reference_value), valid)

def normalize_objectives(values: np.ndarray, calibration: TaskCalibration) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = calibration.nadir - calibration.ideal
    if values.ndim != 2 or values.shape[1] != len(calibration.ideal):
        raise ValueError("目标矩阵维度与校准不一致")
    if not np.isfinite(values).all():
        raise ValueError("目标矩阵包含非有限值")
    return (values - calibration.ideal) / span

def dominated_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    dominated = np.all(values[:, None] <= values[None, :], axis=2) & np.any(values[:, None] < values[None, :], axis=2)
    return float(np.mean(dominated.any(axis=0)))

def metric_health(
    objective_values: np.ndarray,
    raw_hv: Iterable[float],
    normalized_hv: Iterable[float],
    calibration_valid: bool,
) -> dict:
    values = np.asarray(objective_values, dtype=float)
    raw = np.asarray(list(raw_hv), dtype=float)
    normalized = np.asarray(list(normalized_hv), dtype=float)
    ranges = np.ptp(values, axis=0) if values.ndim == 2 and len(values) else np.asarray([])
    zero_ratio = float(np.mean(np.isclose(normalized, 0.0))) if len(normalized) else 1.0
    finite = bool(np.isfinite(values).all() and np.isfinite(raw).all() and np.isfinite(normalized).all())
    valid = bool(calibration_valid and finite and len(ranges) > 0 and np.all(ranges > 1e-12))
    invalid = bool((not valid) or zero_ratio > 0.95)
    return {
        "zero_HV_ratio": zero_ratio,
        "objective_range": ranges.tolist(),
        "dominated_ratio": dominated_ratio(values) if finite else 1.0,
        "normalization_valid": bool(valid),
        "invalid": invalid,
    }
