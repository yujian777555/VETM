"""经验序列化、特征化和任务实例级数据切分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .problems import ProblemSpec
from .schema import ExperienceRecord, TransferExample


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_from_run(problem: ProblemSpec, run: dict, *, experience_id: str) -> ExperienceRecord:
    objectives = np.asarray(run["objectives"])
    front = np.asarray(run["front"])
    outcome = {
        "success": bool(np.isfinite(objectives).all()),
        "failure_reason": None,
        "final_mean_objective": float(np.mean(objectives)),
        "final_best_objective_sum": float(np.min(np.sum(objectives, axis=1))),
        "nondominated_count": int(len(front)),
        "function_evaluations": int(run.get("function_evaluations", (len(run["history"]) * 2 - 1) * run["parameters"]["population_size"])),
        "final_hypervolume": run["history"][-1].get("hypervolume"),
    }
    return ExperienceRecord(
        experience_id=experience_id,
        task_features=problem.features(),
        algorithm=run["algorithm"],
        parameters=run["parameters"],
        trajectory=run["history"],
        outcome=outcome,
        seed=int(run["seed"]),
    )


def numeric_features(record: ExperienceRecord, target_task: dict) -> list[float]:
    """只读取决策时可用字段；不包含目标结果和迁移标签。"""
    source = record.task_features
    params = record.parameters
    target = target_task
    return [
        float(source.get("n_var", 0)),
        float(source.get("n_obj", 0)),
        float(source.get("lower_mean", 0.0)),
        float(source.get("upper_mean", 0.0)),
        float(source.get("domain_width_mean", 0.0)),
        float(target.get("n_var", 0)),
        float(target.get("n_obj", 0)),
        float(target.get("lower_mean", 0.0)),
        float(target.get("upper_mean", 0.0)),
        float(target.get("domain_width_mean", 0.0)),
        float(params.get("population_size", 0)),
        float(params.get("generations", 0)),
        float(params.get("crossover_probability", 0.0)),
        float(params.get("mutation_probability", 0.0)),
        float(params.get("eta_c", 0.0)),
        float(params.get("eta_m", 0.0)),
        float(record.trajectory[-1].get("nondominated_count", 0) if record.trajectory else 0),
        float(record.trajectory[-1].get("mean_objective", 0.0) if record.trajectory else 0.0),
    ]


def make_transfer_example(record: ExperienceRecord, target_task: dict, *, label: int, signed_gain: float, action: str = "Transfer") -> TransferExample:
    return TransferExample(
        source_experience_id=record.experience_id,
        target_task=target_task,
        features=numeric_features(record, target_task),
        label=int(label),
        signed_gain=float(signed_gain),
        action=action,
        metadata={"source_problem_id": record.task_features.get("problem_id"), "target_problem_id": target_task.get("problem_id")},
    )


def task_instance_split(examples: Sequence[TransferExample], *, train_fraction: float = 0.6, validation_fraction: float = 0.2, seed: int = 0) -> dict[str, list[TransferExample]]:
    """按 target problem_id 切分，保证同一任务实例不会跨 split。"""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction 和 validation_fraction 必须为正且和小于 1")
    groups: dict[str, list[TransferExample]] = {}
    for example in examples:
        key = str(example.target_task.get("problem_id", "unknown"))
        groups.setdefault(key, []).append(example)
    if len(groups) < 3:
        raise ValueError("至少需要三个目标任务组才能创建 train/validation/test")
    task_ids = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(task_ids)
    n = len(task_ids)
    train_end = max(1, int(round(n * train_fraction)))
    validation_end = min(n - 1, train_end + max(1, int(round(n * validation_fraction))))
    assignments = {task: "train" for task in task_ids[:train_end]}
    assignments.update({task: "validation" for task in task_ids[train_end:validation_end]})
    assignments.update({task: "test" for task in task_ids[validation_end:]})
    output = {"train": [], "validation": [], "test": []}
    for task, rows in groups.items():
        output[assignments[task]].extend(rows)
    return output

