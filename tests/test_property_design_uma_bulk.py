from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace
from ase.build import bulk

from experiments.property_design_20260821.uma_bulk import (
    EV_PER_A3_TO_GPA,
    build_volume_batch,
    configure_energy_only_uma,
    group_volume_energies,
    fit_bulk_modulus,
    prediction_energies,
    predict_with_loaded_uma,
    scaled_atoms,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:Using extra keyword arguments on `Field` is deprecated"
)


def harmonic_curve(volumes: np.ndarray, equilibrium_volume: float, bulk_gpa: float):
    curvature = bulk_gpa / (EV_PER_A3_TO_GPA * equilibrium_volume)
    return -10.0 + 0.5 * curvature * (volumes - equilibrium_volume) ** 2


def test_fit_bulk_modulus_recovers_known_positive_curvature():
    volumes = np.linspace(96.0, 104.0, 5)
    energies = harmonic_curve(volumes, equilibrium_volume=100.0, bulk_gpa=240.0)

    result = fit_bulk_modulus(volumes, energies)

    assert result.valid
    assert result.bulk_modulus_gpa == pytest.approx(240.0, rel=1e-8)
    assert result.equilibrium_volume_a3 == pytest.approx(100.0, rel=1e-8)
    assert result.r2 == pytest.approx(1.0)


def test_fit_bulk_modulus_rejects_negative_curvature():
    volumes = np.linspace(96.0, 104.0, 5)
    energies = 2.0 - 0.01 * (volumes - 100.0) ** 2

    result = fit_bulk_modulus(volumes, energies)

    assert not result.valid
    assert result.reason == "nonpositive_curvature"


def test_fit_bulk_modulus_rejects_minimum_outside_sampled_range():
    volumes = np.linspace(96.0, 104.0, 5)
    energies = harmonic_curve(volumes, equilibrium_volume=108.0, bulk_gpa=200.0)

    result = fit_bulk_modulus(volumes, energies)

    assert not result.valid
    assert result.reason == "minimum_outside_sampled_range"


def test_scaled_atoms_preserves_input_and_fractional_coordinates():
    atoms = bulk("Si", "diamond", a=5.43)
    original_cell = atoms.cell.array.copy()
    original_positions = atoms.positions.copy()
    original_scaled = atoms.get_scaled_positions().copy()

    expanded = scaled_atoms(atoms, 1.04)

    assert expanded.get_volume() == pytest.approx(atoms.get_volume() * 1.04)
    assert np.allclose(expanded.get_scaled_positions(), original_scaled)
    assert np.allclose(atoms.cell.array, original_cell)
    assert np.allclose(atoms.positions, original_positions)


def test_fit_bulk_modulus_rejects_nonfinite_energy():
    volumes = np.linspace(96.0, 104.0, 5)
    energies = harmonic_curve(volumes, equilibrium_volume=100.0, bulk_gpa=240.0)
    energies[2] = np.nan

    result = fit_bulk_modulus(volumes, energies)

    assert not result.valid
    assert result.reason == "nonfinite_input"


def test_fit_result_serializes_json_safe_scalars():
    volumes = np.linspace(96.0, 104.0, 5)
    energies = harmonic_curve(volumes, equilibrium_volume=100.0, bulk_gpa=240.0)

    payload = fit_bulk_modulus(volumes, energies).as_dict()

    assert payload["valid"] is True
    assert isinstance(payload["bulk_modulus_gpa"], float)
    assert isinstance(payload["r2"], float)


def test_batch_prediction_alignment_is_preserved_across_structures_and_volumes():
    silicon = bulk("Si", "diamond", a=5.43)
    diamond = bulk("C", "diamond", a=3.57)
    points, atoms_batch = build_volume_batch(
        [("silicon", silicon), ("diamond", diamond)],
        volume_factors=(0.98, 1.00, 1.02),
    )
    predictions = {"energy": np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])}

    energies = prediction_energies(predictions, expected_count=len(points))
    grouped = group_volume_energies(points, energies)

    assert len(atoms_batch) == 6
    assert [point.structure_id for point in points] == [
        "silicon",
        "silicon",
        "silicon",
        "diamond",
        "diamond",
        "diamond",
    ]
    assert grouped["silicon"]["energies_ev"] == [-1.0, -2.0, -3.0]
    assert grouped["diamond"]["energies_ev"] == [-4.0, -5.0, -6.0]
    assert grouped["silicon"]["volume_factors"] == [0.98, 1.0, 1.02]


def test_prediction_alignment_rejects_wrong_system_count():
    with pytest.raises(ValueError, match="expected 2 system energies"):
        prediction_energies({"energy": np.asarray([-1.0])}, expected_count=2)


def test_loaded_predictor_batches_without_reordering_systems():
    import torch

    class FakePredictor:
        def __init__(self):
            self.offset = 0

        def predict(self, batch):
            count = int(batch.natoms.numel())
            values = torch.arange(self.offset, self.offset + count, dtype=torch.float64)
            self.offset += count
            return {"energy": values}

    atoms_batch = [bulk("Si", "diamond", a=5.43) for _ in range(5)]

    energies = predict_with_loaded_uma(FakePredictor(), atoms_batch, batch_size=2)

    assert energies == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_configure_energy_only_removes_unused_tasks_and_gradients():
    energy_task = SimpleNamespace(property="energy")
    force_task = SimpleNamespace(property="forces")
    stress_task = SimpleNamespace(property="stress")
    regress = SimpleNamespace(forces=True, stress=True, hessian=True)
    head_regress = SimpleNamespace(forces=True, stress=True, hessian=True)
    wrapper = SimpleNamespace(
        regress_config=regress,
        head=SimpleNamespace(regress_config=head_regress),
    )
    module = SimpleNamespace(
        _tasks={"omat_energy": energy_task, "omat_forces": force_task, "omat_stress": stress_task},
        _dataset_to_tasks={"omat": [energy_task, force_task, stress_task]},
        output_heads={"energyandforcehead": wrapper},
    )
    predictor = SimpleNamespace(model=SimpleNamespace(module=module))

    returned = configure_energy_only_uma(predictor)

    assert returned is predictor
    assert list(module._tasks) == ["omat_energy"]
    assert module._dataset_to_tasks == {"omat": [energy_task]}
    assert not regress.forces and not regress.stress and not regress.hessian
    assert not head_regress.forces and not head_regress.stress and not head_regress.hessian
