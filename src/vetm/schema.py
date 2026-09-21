"""经验记录和迁移样本的稳定 JSON schema。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExperienceRecord:
    experience_id: str
    task_features: dict[str, Any]
    algorithm: str
    parameters: dict[str, Any]
    trajectory: list[dict[str, Any]]
    outcome: dict[str, Any]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        required = {"experience_id", "task_features", "algorithm", "parameters", "trajectory", "outcome", "seed"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"经验记录缺少字段: {sorted(missing)}")
        return cls(**{key: payload[key] for key in required})


@dataclass
class TransferExample:
    source_experience_id: str
    target_task: dict[str, Any]
    features: list[float]
    label: int
    signed_gain: float
    action: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransferExample":
        return cls(**payload)
