"""Phase 1.5 固定参考点 HV、IGD 和成对统计。"""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Iterable
import numpy as np

def _nondominated(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    keep = []
    for i, row in enumerate(p):
        if not np.any(np.all(p <= row, axis=1) & np.any(p < row, axis=1)):
            keep.append(i)
    return p[keep]

def _area_2d(points: np.ndarray, reference: np.ndarray) -> float:
    p = _nondominated(points)
    p = p[np.all(p < reference, axis=1)]
    if not len(p): return 0.0
    order = np.argsort(p[:, 0])
    p = p[order]
    area = 0.0
    previous = reference[1]
    for x, y in p:
        if y < previous:
            area += max(0.0, reference[0] - x) * (previous - y)
            previous = y
    return float(area)

def hypervolume(points: np.ndarray, reference: Iterable[float]) -> float:
    """精确计算二维/三维最小化 HV；参考点必须由任务配置预先提供。"""
    p = _nondominated(np.asarray(points, dtype=float))
    ref = np.asarray(list(reference), dtype=float)
    if p.ndim != 2 or p.shape[1] != len(ref): raise ValueError("points/reference 维度不一致")
    p = p[np.all(p < ref, axis=1)]
    if len(ref) == 2: return _area_2d(p, ref)
    if len(ref) != 3: raise ValueError("pilot 只支持二维和三维精确 HV")
    xs = np.unique(np.r_[p[:, 0], ref[0]])
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        if right <= left: continue
        active = p[p[:, 0] <= left][:, 1:]
        total += (right - left) * _area_2d(active, ref[1:])
    return float(total)

def unit_hypervolume(points: np.ndarray, reference: Iterable[float]) -> float:
    """将 normalized HV 除以参考盒体积，得到跨目标数可比的单位 HV。"""
    ref = np.asarray(list(reference), dtype=float)
    return float(hypervolume(points, ref) / np.prod(ref))

def igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    p = np.asarray(points, dtype=float); r = np.asarray(reference_front, dtype=float)
    if not len(p): return float("inf")
    return float(np.min(np.linalg.norm(r[:, None, :] - p[None, :, :], axis=2), axis=1).mean())

def paired_bootstrap(values: np.ndarray, seed: int = 0, samples: int = 2000) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x): raise ValueError("不能对空样本 bootstrap")
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(samples, len(x)))], axis=1)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "std": float(np.std(x, ddof=1) if len(x) > 1 else 0.0),
            "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)),
            "n": int(len(x))}

def classify_effect(deltas: np.ndarray, tolerance: float = 0.01) -> dict:
    stats = paired_bootstrap(deltas)
    if stats["ci95_low"] > tolerance: state = "positive"
    elif stats["ci95_high"] < -tolerance: state = "negative"
    else: state = "neutral"
    stats["state"] = state; stats["tolerance"] = tolerance
    return stats
