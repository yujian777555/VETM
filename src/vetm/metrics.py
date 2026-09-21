"""验证预测、负迁移和校准的无依赖指标。"""

from __future__ import annotations

import numpy as np


def _binary(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = p >= threshold
    yb = y.astype(bool)
    tp, tn = np.sum(pred & yb), np.sum(~pred & ~yb)
    fp, fn = np.sum(pred & ~yb), np.sum(~pred & yb)
    accuracy = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    positives, negatives = np.sum(y == 1), np.sum(y == 0)
    if positives == 0 or negatives == 0:
        return None
    # 对相同分数使用平均秩，避免把并列概率错误地当作确定排序。
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks_sorted = np.arange(1, len(score) + 1, dtype=float)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j] = np.mean(ranks_sorted[i:j])
        i = j
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = ranks_sorted
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    y, p = np.asarray(y_true).astype(int), np.asarray(probabilities, dtype=float)
    result = _binary(y, p, threshold)
    result.update({"auroc": auroc(y, p), "brier": float(np.mean((p - y) ** 2))})
    return result


def negative_transfer_rate(signed_gain: np.ndarray, *, tolerance: float = 0.0) -> float:
    gains = np.asarray(signed_gain, dtype=float)
    return float(np.mean(gains < -abs(tolerance))) if len(gains) else float("nan")

