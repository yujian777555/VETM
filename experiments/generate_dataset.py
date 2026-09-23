"""生成 Phase 1 经验库和无泄漏 transfer examples。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vetm.dataset import (
    make_transfer_example,
    record_from_run,
    task_group_assignment,
    task_instance_split,
    write_jsonl,
)
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem
from vetm.schema import ExperienceRecord, TransferExample
from vetm.transfer import evaluate_parameter_transfer


def _failure_record(problem_name: str, seed: int, error: Exception) -> ExperienceRecord:
    problem = get_problem(problem_name, n_var=10, n_obj=3)
    return ExperienceRecord(
        experience_id=f"{problem_name.lower()}-failed-{seed}",
        task_features=problem.features(),
        algorithm="NSGA-II",
        parameters={"population_size": 2, "generations": 0},
        trajectory=[],
        outcome={
            "success": False,
            "failure_reason": f"{type(error).__name__}: {error}",
            "function_evaluations": 0,
            "status": "failed",
        },
        seed=seed,
    )


def generate(problems: list[str], seeds: list[int], config: NSGA2Config, output_dir: Path) -> dict:
    experiences: list[dict] = []
    successes: list[ExperienceRecord] = []
    failures = 0
    for problem_index, problem_name in enumerate(problems):
        problem = get_problem(problem_name, n_var=10, n_obj=3)
        for seed in seeds:
            source_config = replace(
                config,
                crossover_probability=0.8 + 0.05 * ((problem_index + seed) % 3),
                eta_m=10.0 + 5.0 * ((problem_index + seed) % 3),
            )
            try:
                run = run_nsga2(problem, source_config, seed=seed)
                record = record_from_run(problem, run, experience_id=f"{problem_name.lower()}-{seed}")
                successes.append(record)
            except Exception as error:
                record = _failure_record(problem_name, seed, error)
                failures += 1
            experiences.append(record.to_dict())
        try:
            run_nsga2(problem, NSGA2Config(population_size=2, generations=1), seed=0)
        except Exception as error:
            experiences.append(_failure_record(problem_name, 0, error).to_dict())
            failures += 1

    assignment = task_group_assignment(problems, seed=0)
    source_tasks = {task for task, split in assignment.items() if split == "train"}
    source_bank = [row for row in successes if row.task_features["problem_id"] in source_tasks]
    examples = []
    for source in source_bank:
        for target_name in problems:
            target = get_problem(target_name, n_var=10, n_obj=3)
            for target_seed in seeds:
                label, gain, metadata = evaluate_parameter_transfer(
                    target, source, seed=target_seed, baseline_config=config
                )
                example = make_transfer_example(source, target.features(), label=label, signed_gain=gain)
                example.metadata.update(metadata)
                examples.append(example.to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(experiences, output_dir / "experiences.jsonl")
    write_jsonl(examples, output_dir / "transfer_examples.jsonl")
    split_rows = task_instance_split(
        [TransferExample.from_dict(row) for row in examples], seed=0
    )
    for name, rows in split_rows.items():
        write_jsonl([row.to_dict() for row in rows], output_dir / f"{name}.jsonl")
    stats = {
        "experience_count": len(experiences),
        "successful_experiences": len(successes),
        "failed_experiences": failures,
        "source_bank_experiences": len(source_bank),
        "source_bank_tasks": sorted(source_tasks),
        "transfer_example_count": len(examples),
        "positive_transfer_examples": int(sum(row["label"] for row in examples)),
        "problems": problems,
        "seeds": seeds,
        "config": config.__dict__,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=40)
    args = parser.parse_args()
    config = NSGA2Config(population_size=args.population_size, generations=args.generations)
    problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2"]
    print(json.dumps(generate(problems, args.seeds, config, Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
