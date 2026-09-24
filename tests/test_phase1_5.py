from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vetm.interventions import intervention_registry
from vetm.transfer_matrix import hypervolume, igd, paired_bootstrap, classify_effect
from vetm.nsga2 import NSGA2Config


def test_fixed_reference_hv_known_2d_and_3d_fronts():
    assert hypervolume(np.array([[0., 1.], [1., 0.]]), [2., 2.]) == pytest.approx(3.0)
    assert hypervolume(np.array([[0., 0., 0.]]), [1., 1., 1.]) == pytest.approx(1.0)
    assert hypervolume(np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]), [2., 2., 2.]) == pytest.approx(4.0)


def test_hv_requires_fixed_reference_and_igd_is_zero_on_reference_front():
    with pytest.raises(ValueError):
        hypervolume(np.zeros((1, 2)), [1., 1., 1.])
    front = np.array([[0., 1.], [1., 0.]])
    assert igd(front, front) == pytest.approx(0.0)


def test_interventions_have_unique_ids_and_preserve_evaluation_budget():
    registry = intervention_registry()
    assert len(registry) >= 20
    assert len({item.intervention_id for item in registry}) == len(registry)
    base = NSGA2Config(population_size=20, generations=5)
    assert all(item.apply(base).population_size == 20 and item.apply(base).generations == 5 for item in registry)


def test_bootstrap_effect_state_uses_preregistered_tolerance():
    assert classify_effect(np.array([0.2, 0.3, 0.25]), tolerance=0.01)["state"] == "positive"
    assert classify_effect(np.array([-0.2, -0.3, -0.25]), tolerance=0.01)["state"] == "negative"
    assert classify_effect(np.array([0.0, 0.001, -0.001]), tolerance=0.01)["state"] == "neutral"
