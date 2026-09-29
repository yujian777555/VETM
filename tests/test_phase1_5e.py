from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem
from vetm.validators import NearestNeighborValidator

def test_injected_initial_population_is_byte_identical_across_branches():
    problem = get_problem("ZDT1", n_var=10)
    initial = np.random.default_rng(200).uniform(problem.lower, problem.upper, size=(8, 10))
    values = problem.evaluate(initial)
    baseline = run_nsga2(problem, NSGA2Config(population_size=8, generations=1), seed=200,
                         initial_population=initial, initial_objectives=values)
    probe = run_nsga2(problem, NSGA2Config(population_size=8, generations=1, mutation_probability=.2), seed=200,
                      initial_population=initial, initial_objectives=values)
    assert baseline["initial_population"].tobytes() == probe["initial_population"].tobytes()
    assert baseline["initial_objectives"].tobytes() == probe["initial_objectives"].tobytes()
    assert baseline["function_evaluations"] == probe["function_evaluations"] == 8

def test_injection_rejects_invalid_shape():
    problem = get_problem("ZDT1", n_var=10)
    with pytest.raises(ValueError):
        run_nsga2(problem, NSGA2Config(population_size=8, generations=1), seed=1,
                  initial_population=np.zeros((7, 10)))

def test_standardized_knn_neighbor_identity_survives_raw_feature_rescaling():
    x = np.array([[0., 100.], [1., 110.], [3., 200.]])
    y = np.array([0., 1., 2.])
    q = np.array([[1.1, 111.]])
    a = NearestNeighborValidator(k=1).fit(x, y).predict_proba(q)
    b = NearestNeighborValidator(k=1).fit(x * np.array([1e6, .001]), y).predict_proba(q * np.array([1e6, .001]))
    np.testing.assert_allclose(a, b)

def test_fingerprint_source_has_no_oracle_calls():
    source = (ROOT / "experiments" / "phase1_5e_fingerprint.py").read_text(encoding="utf-8")
    for forbidden in ("reference_front", "calibrate_task", "igd(", "IGD("):
        assert forbidden not in source
