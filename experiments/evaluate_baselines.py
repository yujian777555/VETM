"""在冻结的 test split 上比较三个初始 validity validators。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vetm.dataset import read_jsonl
from vetm.metrics import evaluate_predictions, negative_transfer_rate
from vetm.validators import MLPValidityPredictor, NearestNeighborValidator, RandomValidator


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["features"] for row in rows], dtype=float),
        np.asarray([row["label"] for row in rows], dtype=int),
        np.asarray([row["signed_gain"] for row in rows], dtype=float),
    )


def evaluate(data_dir: Path) -> dict:
    train, test = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "test.jsonl")
    if not train or not test:
        raise ValueError("train/test 数据为空，无法评估")
    x_train, y_train, _ = _matrix(train)
    x_test, y_test, gains = _matrix(test)
    validators = {
        "random": RandomValidator(seed=0),
        "nearest_neighbor": NearestNeighborValidator(k=5),
        "mlp": MLPValidityPredictor(hidden_dim=32, epochs=300, learning_rate=0.01, seed=0),
    }
    results = {}
    for name, validator in validators.items():
        validator.fit(x_train, y_train)
        probabilities = validator.predict_proba(x_test)
        results[name] = evaluate_predictions(y_test, probabilities)
    results["test_negative_transfer_rate"] = negative_transfer_rate(gains)
    output = data_dir / "baseline_results.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    print(json.dumps(evaluate(Path(args.data_dir)), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
