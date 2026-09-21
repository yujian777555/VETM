"""不依赖深度学习框架的三个初始 validity validators。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Validator(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Validator":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


class RandomValidator(Validator):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomValidator":
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.rng.random(len(x))


class NearestNeighborValidator(Validator):
    def __init__(self, k: int = 5):
        self.k = max(1, int(k))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestNeighborValidator":
        raw_x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.mean = np.mean(raw_x, axis=0)
        self.scale = np.std(raw_x, axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        self.x = self._normalize(raw_x)
        return self

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale if hasattr(self, "mean") else x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        query = self._normalize(np.asarray(x, dtype=float))
        distances = np.linalg.norm(query[:, None, :] - self.x[None, :, :], axis=2)
        k = min(self.k, len(self.y))
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        weights = 1.0 / (distances[np.arange(len(query))[:, None], nearest] + 1e-8)
        return np.sum(weights * self.y[nearest], axis=1) / np.sum(weights, axis=1)


class MLPValidityPredictor(Validator):
    """单隐藏层 NumPy MLP；用于验证可学习性，不代表最终 VETM 架构。"""

    def __init__(self, hidden_dim: int = 32, epochs: int = 300, learning_rate: float = 0.01, seed: int = 0):
        self.hidden_dim, self.epochs, self.learning_rate, self.seed = hidden_dim, epochs, learning_rate, seed

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPValidityPredictor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.mean, self.scale = np.mean(x, axis=0), np.std(x, axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        x = (x - self.mean) / self.scale
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, 1))
        self.b2 = np.zeros((1, 1))
        for _ in range(self.epochs):
            hidden = np.tanh(x @ self.w1 + self.b1)
            pred = self._sigmoid(hidden @ self.w2 + self.b2)
            grad_out = pred - y
            grad_w2 = hidden.T @ grad_out / len(x)
            grad_b2 = np.mean(grad_out, axis=0, keepdims=True)
            grad_hidden = (grad_out @ self.w2.T) * (1.0 - hidden**2)
            grad_w1 = x.T @ grad_hidden / len(x)
            grad_b1 = np.mean(grad_hidden, axis=0, keepdims=True)
            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(x, dtype=float) - self.mean) / self.scale
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        return self._sigmoid(hidden @ self.w2 + self.b2).ravel()


