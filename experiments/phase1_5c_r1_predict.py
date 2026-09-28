"""在 Stage A 聚合条件上运行 R1 的简单预测基线。"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.interventions import intervention_registry
from vetm.metrics import evaluate_predictions
from vetm.validators import MLPValidityPredictor, NearestNeighborValidator

def rank_corr(a, b):
    def ranks(x):
        order = np.argsort(x); out = np.empty(len(x), float); out[order] = np.arange(len(x)); return out
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0: return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])

def features(row, registry):
    item = registry[row["intervention_id"]]
    p = item.parameters
    return [
        float(row["n_var"]), float(row["n_obj"]), float(row["budget"]),
        float(p.get("mutation_probability", 1.0 / float(row["n_var"]))),
        float(p.get("eta_m", 20.0)), float(p.get("crossover_probability", .9)),
        float(p.get("eta_c", 15.0)), float(p.get("tournament_size", 2)),
    ]

def main():
    stage = ROOT / "results" / "r1_stage_a"
    rows = list(csv.DictReader((stage / "phase1_5C_condition_summary.csv").open(encoding="utf-8")))
    registry = {item.intervention_id: item for item in intervention_registry()}
    train = [row for row in rows if row["family"] == "ZDT"]
    test = [row for row in rows if row["family"] == "DTLZ"]
    x_train = np.asarray([features(row, registry) for row in train], float)
    y_train = np.asarray([float(row["mean"]) for row in train])
    x_test = np.asarray([features(row, registry) for row in test], float)
    y_test = np.asarray([float(row["mean"]) for row in test])
    train_means = defaultdict(list)
    for row in train: train_means[row["intervention_id"]].append(float(row["mean"]))
    global_pred = np.asarray([np.mean(train_means[row["intervention_id"]]) for row in test])
    zero_pred = np.zeros(len(test))
    nn = NearestNeighborValidator(k=5).fit(x_train, y_train)
    nn_pred = nn.predict_proba(x_test)
    mlp_reg = MLPValidityPredictor(hidden_dim=16, epochs=250, learning_rate=.01, seed=0)
    # The classifier is used for sign detection; ridge remains the regression baseline in the report.
    centered = y_train > 0
    mlp_reg.fit(x_train, centered.astype(int))
    negative_y_train = (y_train < -0.01).astype(int)
    negative_y_test = (y_test < -0.01).astype(int)
    nn_negative = NearestNeighborValidator(k=5).fit(x_train, negative_y_train)
    mlp_negative = MLPValidityPredictor(hidden_dim=16, epochs=250, learning_rate=.01, seed=1).fit(x_train, negative_y_train)
    prediction = {
        "split": "family-held-out ZDT train -> DTLZ test",
        "train_conditions": len(train), "test_conditions": len(test),
        "regression": {
            "zero_gain": {"mae": float(np.mean(abs(zero_pred - y_test))), "spearman": rank_corr(zero_pred, y_test)},
            "global_intervention_mean": {"mae": float(np.mean(abs(global_pred - y_test))), "spearman": rank_corr(global_pred, y_test)},
            "nearest_neighbor": {"mae": float(np.mean(abs(nn_pred - y_test))), "spearman": rank_corr(nn_pred, y_test)},
        },
        "negative_transfer_detection": {
            "majority": evaluate_predictions(negative_y_test, np.full(len(test), np.mean(negative_y_train))),
            "nearest_neighbor": evaluate_predictions(negative_y_test, nn_negative.predict_proba(x_test)),
            "mlp": evaluate_predictions(negative_y_test, mlp_negative.predict_proba(x_test)),
        },
        "note": "No sklearn available; NumPy nearest-neighbor and MLP baselines are used. Individual seed rows are not training examples.",
    }
    (ROOT / "results" / "phase1_5C_R1_predictability.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()
