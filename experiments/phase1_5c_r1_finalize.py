"""合并 Stage A/B，评估条件级预测与符号反转。"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.interventions import intervention_registry
from vetm.metrics import evaluate_predictions
from vetm.transfer_matrix import classify_effect
from vetm.validators import MLPValidityPredictor, NearestNeighborValidator

RESULTS = ROOT / "results"
TOLERANCE = 0.01


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    a, p = rankdata(actual), rankdata(predicted)
    if len(a) < 2 or np.std(a) == 0 or np.std(p) == 0:
        return None
    return float(np.corrcoef(a, p)[0, 1])


def feature_matrix(rows: list[dict]) -> np.ndarray:
    registry = {item.intervention_id: item for item in intervention_registry()}
    output = []
    for row in rows:
        n_var = int(row["n_var"])
        baseline = 1.0 / n_var
        intervention = registry[row["intervention_id"]]
        descriptor = intervention.descriptor(
            __import__("vetm.nsga2", fromlist=["NSGA2Config"]).NSGA2Config(
                population_size=50, generations=int(row["budget"]) // 50 - 1
            ), n_var,
        )
        categories = [float(descriptor["category"] == name) for name in ("mutation", "crossover", "selection")]
        parameters = [float(descriptor["parameter"] == name) for name in
                      ("mutation_probability", "eta_m", "crossover_probability", "eta_c", "tournament_size")]
        output.append([
            float(n_var), float(row["n_obj"]), 50.0, float(row["budget"]),
            0.0 if row["problem"] != "ZDT4" else -5.0,
            1.0 if row["problem"] != "ZDT4" else 5.0,
            baseline,
            *categories, *parameters,
            descriptor["absolute_value"], descriptor["signed_change"], descriptor["relative_magnitude"],
        ])
    return np.asarray(output, dtype=float)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x, z = standardize(train_x, test_x)
    center = train_y.mean()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (train_y - center))
    return center + z @ w


def logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, epochs: int = 600) -> np.ndarray:
    x, z = standardize(train_x, test_x)
    x = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(x.shape[1])
    for _ in range(epochs):
        pred = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        gradient = x.T @ (pred - train_y) / len(x)
        gradient[1:] += 0.01 * w[1:]
        w -= 0.08 * gradient
    return 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))


def mlp_regression(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 0) -> np.ndarray:
    x, z = standardize(train_x, test_x)
    mean, scale = train_y.mean(), max(train_y.std(), 1e-12)
    y = (train_y - mean) / scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.1, (x.shape[1], 16)); b1 = np.zeros(16)
    w2 = rng.normal(0, 0.1, 16); b2 = 0.0
    for _ in range(500):
        h = np.tanh(x @ w1 + b1)
        error = h @ w2 + b2 - y
        gw2 = h.T @ error / len(x) + 0.01 * w2
        gb2 = float(error.mean())
        gh = np.outer(error, w2) * (1 - h ** 2)
        gw1 = x.T @ gh / len(x) + 0.01 * w1
        gb1 = gh.mean(axis=0)
        w2 -= 0.01 * gw2; b2 -= 0.01 * gb2
        w1 -= 0.01 * gw1; b1 -= 0.01 * gb1
    return (np.tanh(z @ w1 + b1) @ w2 + b2) * scale + mean


def balanced_accuracy(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    predicted = probabilities >= 0.5
    if not np.any(y == 0) or not np.any(y == 1):
        return None
    return float(((predicted[y == 1]).mean() + (~predicted[y == 0]).mean()) / 2)


def evaluate_split(train: list[dict], test: list[dict]) -> dict:
    if not train or not test:
        return {"available": False}
    train_x, test_x = feature_matrix(train), feature_matrix(test)
    train_y = np.asarray([float(row["mean"]) for row in train])
    test_y = np.asarray([float(row["mean"]) for row in test])
    global_values = defaultdict(list)
    for row in train:
        global_values[row["intervention_id"]].append(float(row["mean"]))
    regression = {
        "zero_gain": np.zeros(len(test)),
        "global_intervention_mean": np.asarray([
            np.mean(global_values[row["intervention_id"]]) if global_values[row["intervention_id"]] else train_y.mean()
            for row in test
        ]),
        "nearest_neighbor": NearestNeighborValidator(k=5).fit(train_x, train_y).predict_proba(test_x),
        "ridge": ridge(train_x, train_y, test_x),
        "small_mlp": mlp_regression(train_x, train_y, test_x),
    }
    negative_train = (train_y < -TOLERANCE).astype(int)
    negative_test = (test_y < -TOLERANCE).astype(int)
    majority = float(negative_train.mean() >= 0.5)
    rng = np.random.default_rng(0)
    classification = {
        "majority": np.full(len(test), majority),
        "random": rng.random(len(test)),
        "nearest_neighbor": NearestNeighborValidator(k=5).fit(train_x, negative_train).predict_proba(test_x),
        "logistic": logistic(train_x, negative_train, test_x),
        "small_mlp": MLPValidityPredictor(hidden_dim=16, epochs=350, learning_rate=0.01, seed=1).fit(
            train_x, negative_train
        ).predict_proba(test_x),
    }
    return {
        "train_conditions": len(train), "test_conditions": len(test),
        "regression": {name: {"mae": float(np.mean(abs(pred - test_y))), "spearman": spearman(test_y, pred)}
                       for name, pred in regression.items()},
        "negative_transfer_detection": {
            name: {**evaluate_predictions(negative_test, pred), "balanced_accuracy": balanced_accuracy(negative_test, pred)}
            for name, pred in classification.items()
        },
    }


def reversal_summary(rows: list[dict]) -> dict:
    by_intervention = defaultdict(list)
    for row in rows:
        by_intervention[row["intervention_id"]].append(row)
    within, cross, dimension, agreement = [], [], [], []
    for intervention_id, conditions in by_intervention.items():
        signs = {row["state"] for row in conditions}
        if not {"positive", "negative"}.issubset(signs):
            continue
        nonneutral = [row for row in conditions if row["state"] != "neutral"]
        if all((row["state"] == "positive" and float(row["mean_delta_IGD"]) < 0) or
               (row["state"] == "negative" and float(row["mean_delta_IGD"]) > 0) for row in nonneutral):
            agreement.append(intervention_id)
        for family in {row["family"] for row in conditions}:
            if {"positive", "negative"}.issubset({row["state"] for row in conditions if row["family"] == family}):
                within.append(intervention_id)
        if {"positive", "negative"}.issubset({row["state"] for row in conditions}) and len({row["family"] for row in nonneutral}) > 1:
            cross.append(intervention_id)
        if len({row["n_var"] for row in nonneutral}) > 1 or len({row["n_obj"] for row in nonneutral}) > 1:
            dimension.append(intervention_id)
    return {"within_family": sorted(set(within)), "cross_family": sorted(set(cross)),
            "dimension_or_objective_count": sorted(set(dimension)), "hv_igd_agree": sorted(set(agreement))}


def main() -> None:
    stage_a = RESULTS / "r1_stage_a"
    stage_b = RESULTS / "r1_stage_b"
    matrix_rows = read_csv(stage_a / "phase1_5C_transfer_matrix.csv") + read_csv(stage_b / "phase1_5C_transfer_matrix.csv")
    matrix_by_key = {(row["task_id"], row["intervention_id"], row["seed"]): row for row in matrix_rows}
    matrix = list(matrix_by_key.values())
    condition_rows = read_csv(stage_a / "phase1_5C_condition_summary.csv") + read_csv(stage_b / "phase1_5C_condition_summary.csv")
    condition_by_key = {(row["task_id"], row["intervention_id"]): row for row in condition_rows}
    conditions = list(condition_by_key.values())
    assert all(row["function_evaluations_equal"] == "True" and row["status"] == "ok" for row in matrix)
    write_csv(RESULTS / "phase1_5C_R1_transfer_matrix.csv", matrix)
    write_csv(RESULTS / "phase1_5C_R1_condition_summary.csv", conditions)
    problems = sorted({row["problem"] for row in conditions})
    splits = {}
    for problem in problems:
        splits["leave_problem_" + problem] = evaluate_split(
            [row for row in conditions if row["problem"] != problem],
            [row for row in conditions if row["problem"] == problem],
        )
    splits["family_ZDT_to_DTLZ"] = evaluate_split(
        [row for row in conditions if row["family"] == "ZDT"],
        [row for row in conditions if row["family"] == "DTLZ"],
    )
    splits["family_DTLZ_to_ZDT"] = evaluate_split(
        [row for row in conditions if row["family"] == "DTLZ"],
        [row for row in conditions if row["family"] == "ZDT"],
    )
    splits["objective_2_to_3"] = evaluate_split(
        [row for row in conditions if int(row["n_obj"]) == 2],
        [row for row in conditions if int(row["n_obj"]) == 3],
    )
    (RESULTS / "phase1_5C_R1_predictability.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    reversals = reversal_summary(conditions)
    stage_b_gate = bool(reversals["within_family"] and reversals["hv_igd_agree"] and any(
        split.get("available", True) and any(
            item["mae"] < split["regression"]["zero_gain"]["mae"] and
            item["mae"] < split["regression"]["global_intervention_mean"]["mae"]
            for name, item in split["regression"].items() if name not in ("zero_gain", "global_intervention_mean")
        ) for split in splits.values()
    ))
    summary = {
        "healthy_task_count": len({row["task_id"] for row in conditions}),
        "condition_count": len(conditions), "paired_run_rows": len(matrix),
        "positive_groups": sum(row["state"] == "positive" for row in conditions),
        "negative_groups": sum(row["state"] == "negative" for row in conditions),
        "neutral_groups": sum(row["state"] == "neutral" for row in conditions),
        "reversals": reversals, "stage_b_gate_observed": stage_b_gate,
        "stage_b": "completed", "phase_2": "NO-GO pending 20-seed candidate confirmation",
    }
    (RESULTS / "phase1_5C_R1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
