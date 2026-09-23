"""独立数值夹具覆盖指标、评价预算和泄漏边界。"""
from pathlib import Path
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vetm.dataset import record_from_run, task_group_assignment, task_instance_split, numeric_features, write_jsonl
from vetm.metrics import auroc, evaluate_predictions
from vetm.nsga2 import NSGA2Config, run_nsga2, hypervolume_2d
from vetm.problems import get_problem, ProblemSpec
from vetm.schema import ExperienceRecord, TransferExample
from vetm.validators import NearestNeighborValidator, MLPValidityPredictor
from vetm.transfer import evaluate_parameter_transfer


@pytest.mark.parametrize("name,want", [
    ("ZDT1", [0, 1]), ("ZDT2", [0, 1]), ("ZDT3", [0, 1]),
    ("ZDT4", [0, 1]), ("ZDT6", [1, 0]),
])
def test_zdt_known_pareto_endpoints(name, want):
    problem = get_problem(name, n_var=6)
    np.testing.assert_allclose(problem.evaluate(np.zeros((1, 6)))[0], want, atol=1e-12)


@pytest.mark.parametrize("name,total", [("DTLZ1", 0.5), ("DTLZ2", 1.0)])
def test_dtlz_known_front(name, total):
    values = get_problem(name, n_var=7, n_obj=3).evaluate(np.full((1, 7), 0.5))[0]
    measure = values.sum() if name == "DTLZ1" else np.linalg.norm(values)
    assert measure == pytest.approx(total)


def test_actual_objective_calls_not_generation_survival_count():
    calls = []
    original = get_problem("ZDT1", n_var=6)
    def counted(x):
        calls.append(len(x))
        return original.evaluate(x)
    problem = ProblemSpec(original.name, 6, 2, original.lower, original.upper, counted)
    run = run_nsga2(problem, NSGA2Config(population_size=8, generations=3), seed=7)
    assert sum(calls) == 32  # 初始 8 加三批 8 个后代，父代不重新评价
    record = record_from_run(problem, run, experience_id="count")
    assert record.outcome["function_evaluations"] == 32


def test_zero_mutation_is_not_default_mutation():
    run = run_nsga2(get_problem("ZDT1", n_var=6),
                    NSGA2Config(population_size=8, generations=1, mutation_probability=0), 1)
    assert run["parameters"]["mutation_probability"] == 0


def test_reproducible_run_with_bounds():
    problem = get_problem("ZDT4", n_var=6)
    cfg = NSGA2Config(population_size=9, generations=2)
    a, b = run_nsga2(problem, cfg, 9), run_nsga2(problem, cfg, 9)
    np.testing.assert_array_equal(a["objectives"], b["objectives"])
    assert np.all(a["population"] >= problem.lower)
    assert np.all(a["population"] <= problem.upper)


@pytest.mark.parametrize("y,p,want", [
    ([0, 1], [0.1, 0.9], 1.0), ([0, 1], [0.9, 0.1], 0.0),
    ([0, 1], [0.5, 0.5], 0.5), ([1, 0], [0.5, 0.5], 0.5)
])
def test_auc_direction_and_ties(y, p, want):
    assert auroc(np.array(y), np.array(p)) == pytest.approx(want)


def test_undefined_auc_is_json_null_not_nan():
    assert evaluate_predictions(np.array([1, 1]), np.array([0.5, 0.5]))["auroc"] is None


def test_fixed_hv_and_external_points():
    points = np.array([[1, 2], [2, 1], [4, 0], [1, 2]])
    assert hypervolume_2d(points, np.array([3, 3])) == pytest.approx(3.0)


def test_nn_normalizes_training_and_query_identically_on_refit():
    x = np.array([[100, 1000], [200, 2000], [300, 3000]])
    nn = NearestNeighborValidator(k=1).fit(x, np.array([0, 1, 0]))
    assert nn.predict_proba(x)[1] > 0.99
    nn.fit(x + 500, np.array([1, 0, 1]))
    assert nn.predict_proba(x + 500)[1] < 0.01


def test_mlp_learns_simple_separable_fixture():
    x = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0, 0, 1, 1])
    model = MLPValidityPredictor(hidden_dim=8, epochs=400, learning_rate=0.05).fit(x, y)
    assert np.array_equal(model.predict(x), y)


def test_transfer_preserves_target_budget_even_if_source_used_larger_budget():
    problem = get_problem("ZDT1", n_var=6)
    source = record_from_run(problem, run_nsga2(problem, NSGA2Config(
        population_size=12, generations=4), 0), experience_id="source")
    _, _, meta = evaluate_parameter_transfer(problem, source, seed=2,
                      baseline_config=NSGA2Config(population_size=8, generations=2))
    assert meta["baseline_function_evaluations"] == 24
    assert meta["transfer_function_evaluations"] == 24


def test_features_do_not_use_target_outcomes_or_identifiers():
    source = record_from_run(get_problem("ZDT1", n_var=6), run_nsga2(
        get_problem("ZDT1", n_var=6), NSGA2Config(population_size=8, generations=1), 0),
        experience_id="s")
    target = get_problem("ZDT2", n_var=6).features()
    before = numeric_features(source, target)
    target.update(final_igd=-999, label=1, signed_gain=999, seed=999, problem_id="changed")
    assert numeric_features(source, target) == before


def test_too_few_task_groups_rejected():
    row = TransferExample("s", {"problem_id": "ZDT1"}, [1.], 0, 0., "Transfer", {})
    with pytest.raises(ValueError):
        task_instance_split([row])


def test_task_splits_group_all_seeds():
    rows = [TransferExample("s", {"problem_id": p, "n_var": d}, [1.], 0, 0., "Transfer", {})
            for p in ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "DTLZ2"] for d in [6, 10]]
    result = task_instance_split(rows)
    groups = [{r.target_task["problem_id"] for r in result[k]} for k in result]
    assert sum(map(len, result.values())) == 10
    assert all(groups)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_jsonl_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        write_jsonl([{"metric": float("nan")}], tmp_path / "bad.jsonl")



def test_source_bank_uses_same_task_assignment_as_target_split():
    assignment = task_group_assignment(["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2"], seed=0)
    assert {task for task, split in assignment.items() if split == "train"} == {"ZDT1", "ZDT2", "ZDT3", "ZDT6"}
    assert not {"ZDT4", "DTLZ1", "DTLZ2"} & {task for task, split in assignment.items() if split == "train"}
