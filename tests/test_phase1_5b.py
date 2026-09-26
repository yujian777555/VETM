from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vetm.problems import get_problem, reference_front
from vetm.metric_calibration import calibrate_task, metric_health, normalize_objectives
from vetm.interventions import intervention_registry, active_interventions
from vetm.nsga2 import NSGA2Config
from vetm.transfer_matrix import hypervolume, unit_hypervolume


def test_task_calibration_has_ideal_nadir_and_unified_reference():
    task = get_problem("ZDT1", n_var=10)
    calibration = calibrate_task(task)
    assert calibration.ideal.shape == (2,)
    assert calibration.nadir.shape == (2,)
    assert np.all(calibration.nadir > calibration.ideal)
    assert np.allclose(calibration.normalized_reference, [1.1, 1.1])
    front = reference_front(task)
    normalized = normalize_objectives(front, calibration)
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)


def test_raw_and_normalized_hv_are_reportable_with_same_task_calibration():
    task = get_problem("ZDT1", n_var=10)
    calibration = calibrate_task(task)
    points = task.evaluate(np.full((20, task.n_var), 0.5))
    normalized = normalize_objectives(points, calibration)
    assert hypervolume(points, calibration.raw_reference) >= 0
    assert hypervolume(normalized, calibration.normalized_reference) >= 0


def test_metric_health_flags_zero_hv_and_nonfinite_ranges():
    health = metric_health(
        objective_values=np.zeros((4, 2)),
        raw_hv=[0.0, 0.0],
        normalized_hv=[0.0, 0.0],
        calibration_valid=True,
    )
    assert health["zero_HV_ratio"] == 1.0
    assert health["normalization_valid"] is False
    assert health["invalid"] is True


def test_metric_health_threshold_is_ten_percent():
    values = np.tile(np.array([[0.0, 1.0], [1.0, 0.0]]), (5, 1))
    valid = metric_health(values, [1.0] * 9 + [0.0], [1.0] * 9 + [0.0], True)
    invalid = metric_health(values, [1.0] * 8 + [0.0] * 2, [1.0] * 8 + [0.0] * 2, True)
    assert valid["invalid"] is False
    assert invalid["invalid"] is True


def test_unit_hv_normalizes_reference_box_volume():
    points = np.array([[0.0, 0.0]])
    reference = [1.1, 1.1]
    assert unit_hypervolume(points, reference) == pytest.approx(1.0)


def test_active_interventions_remove_task_specific_noops():
    base = NSGA2Config(population_size=50, generations=39)
    active = active_interventions(base, n_var=10)
    ids = {item.intervention_id for item in active}
    assert all(item.parameters.get("crossover_probability") != base.crossover_probability for item in active)
    assert all(item.parameters.get("tournament_size") != base.tournament_size for item in active)
    assert all(item.parameters.get("eta_m") != base.eta_m for item in active)
    assert len(ids) >= 10


def test_phase1_5b_budget_is_at_least_2000():
    cfg = NSGA2Config(population_size=50, generations=39)
    assert cfg.population_size * (cfg.generations + 1) >= 2000
