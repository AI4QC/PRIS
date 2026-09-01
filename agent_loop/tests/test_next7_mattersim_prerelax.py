import importlib
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator


prerelax = importlib.import_module("src.next7_mattersim_prerelax")


def _scale_aware_tolerance(scale):
    return 8.0 * np.finfo(np.float64).eps * max(1.0, abs(scale))


def _snapshot(positions, energy=-4.0, forces=None, stress=None):
    atoms = Atoms(
        "H2",
        positions=np.asarray(positions, dtype=float),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    atoms.info["total_energy"] = energy
    atoms.arrays["forces"] = np.asarray(
        forces if forces is not None else [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    atoms.info["stress"] = np.asarray(
        stress if stress is not None else np.zeros(6), dtype=float
    )
    return atoms


def _safe_snapshots():
    positions = {
        0: [[9.90, 0.0, 0.0], [5.0, 0.0, 0.0]],
        2: [[9.95, 0.0, 0.0], [5.0, 0.0, 0.0]],
        4: [[0.05, 0.0, 0.0], [5.0, 0.0, 0.0]],
        8: [[0.10, 0.0, 0.0], [5.0, 0.0, 0.0]],
    }
    energies = {0: -4.00, 2: -4.02, 4: -4.04, 8: -4.06}
    return {
        step: _snapshot(positions[step], energy=energies[step])
        for step in prerelax.SNAPSHOT_STEPS
    }


def test_protocol_constants_are_frozen_and_exact():
    assert prerelax.SNAPSHOT_STEPS == (0, 2, 4, 8)
    assert dict(prerelax.FIRE_PARAMETERS) == {
        "dt": 0.05,
        "dtmax": 0.20,
        "maxstep": 0.05,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
    }
    with pytest.raises(TypeError):
        prerelax.FIRE_PARAMETERS["dt"] = 0.1


def test_summarize_snapshots_computes_numeric_force_stress_and_energy_metrics():
    stress = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, -4.0]])
    snapshots = _safe_snapshots()
    for atoms in snapshots.values():
        atoms.info["total_energy"] = -4.0
        atoms.info["stress"] = stress.copy()

    summaries = prerelax.summarize_snapshots(snapshots)

    summary = summaries[0]
    assert tuple(summaries) == prerelax.SNAPSHOT_STEPS
    assert summary.total_energy_ev == pytest.approx(-4.0)
    assert summary.energy_per_atom_ev == pytest.approx(-2.0)
    assert summary.fmax_ev_per_a == pytest.approx(5.0)
    assert summary.frms_ev_per_a == pytest.approx(np.sqrt(12.5))
    assert summary.stress_frobenius_ev_per_a3 == pytest.approx(np.sqrt(34.0))
    assert summary.stress_max_abs_eigenvalue_ev_per_a3 == pytest.approx(
        2.0 + np.sqrt(5.0)
    )
    assert summary.min_pair_distance_a == pytest.approx(4.9)


def test_summarize_snapshots_reports_adjacent_saved_energy_change_per_atom():
    snapshots = _safe_snapshots()
    for step, energy in {0: -4.0, 2: -3.8, 4: -4.2, 8: -4.0}.items():
        snapshots[step].info["total_energy"] = energy

    summaries = prerelax.summarize_snapshots(snapshots)

    assert [
        summaries[step].energy_change_from_previous_snapshot_ev_per_atom
        for step in prerelax.SNAPSHOT_STEPS
    ] == pytest.approx([0.0, 0.1, -0.2, 0.1])


def test_displacement_from_x0_uses_minimum_image_across_periodic_boundary():
    summaries = prerelax.summarize_snapshots(_safe_snapshots())

    assert summaries[8].rms_displacement_from_x0_a == pytest.approx(
        np.sqrt(0.2**2 / 2.0)
    )
    assert summaries[8].max_displacement_from_x0_a == pytest.approx(0.2)


def test_fail_open_when_atomic_numbers_or_order_differs_from_x0():
    snapshots = _safe_snapshots()
    snapshots[2].set_atomic_numbers([2, 1])

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def test_fail_open_when_cell_differs_from_x0_beyond_absolute_tolerance():
    snapshots = _safe_snapshots()
    changed_cell = snapshots[2].cell.array.copy()
    changed_cell[0, 0] += 2.0e-12
    snapshots[2].set_cell(changed_cell)

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def test_cell_difference_within_absolute_tolerance_is_supported():
    snapshots = _safe_snapshots()
    changed_cell = snapshots[2].cell.array.copy()
    changed_cell[0, 0] += 5.0e-13
    snapshots[2].set_cell(changed_cell)

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_fail_open_when_pbc_differs_from_x0():
    snapshots = _safe_snapshots()
    snapshots[2].pbc = [True, True, False]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def test_summarize_rejects_missing_snapshot():
    snapshots = _safe_snapshots()
    del snapshots[4]

    with pytest.raises(prerelax.SnapshotValidationError, match="missing.*4"):
        prerelax.summarize_snapshots(snapshots)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("forces", np.zeros((2, 2)), "forces.*shape"),
        ("forces", np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]), "forces.*finite"),
        ("stress", np.zeros((2, 2)), "stress.*shape"),
        ("stress", np.array([np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]), "stress.*finite"),
        ("total_energy", np.nan, "total_energy.*finite"),
    ],
)
def test_summarize_rejects_bad_shapes_and_nonfinite_values(field, bad_value, message):
    snapshots = _safe_snapshots()
    if field == "forces":
        snapshots[2].arrays["forces"] = bad_value
    else:
        snapshots[2].info[field] = bad_value

    with pytest.raises(prerelax.SnapshotValidationError, match=message):
        prerelax.summarize_snapshots(snapshots)


def test_summarize_rejects_missing_required_fields():
    snapshots = _safe_snapshots()
    del snapshots[2].info["total_energy"]

    with pytest.raises(prerelax.SnapshotValidationError, match="total_energy.*missing"):
        prerelax.summarize_snapshots(snapshots)


def test_nonnumeric_stress_has_step_and_field_context_and_fails_open():
    snapshots = _safe_snapshots()
    snapshots[2].info["stress"] = ["not-a-number"] * 6

    with pytest.raises(
        prerelax.SnapshotValidationError, match="step 2 stress.*numeric"
    ):
        prerelax.summarize_snapshots(snapshots)
    decision = prerelax.assess_prerelax_support(snapshots)
    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def test_fail_open_when_finite_inputs_overflow_derived_metrics():
    snapshots = _safe_snapshots()
    for atoms in snapshots.values():
        atoms.info["stress"] = np.diag([1.0e200, 1.0e200, 1.0e200])

    with np.errstate(over="ignore", invalid="ignore"):
        decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def test_pack_structures_preserves_order_and_obeys_atom_budget():
    structures = [Atoms(numbers=[1] * size) for size in (2, 3, 4, 1)]

    batches = prerelax.pack_structures(structures, atom_budget=5)

    assert [[len(atoms) for atoms in batch] for batch in batches] == [[2, 3], [4, 1]]
    assert [id(atoms) for batch in batches for atoms in batch] == [
        id(atoms) for atoms in structures
    ]


def test_pack_structures_obeys_optional_structure_cap():
    structures = [Atoms(numbers=[1] * size) for size in (2, 1, 2)]

    batches = prerelax.pack_structures(structures, atom_budget=10, structure_cap=1)

    assert [[len(atoms) for atoms in batch] for batch in batches] == [[2], [1], [2]]


def test_pack_structures_rejects_single_structure_over_atom_budget():
    with pytest.raises(ValueError, match="6.*budget.*5"):
        prerelax.pack_structures([Atoms(numbers=[1] * 6)], atom_budget=5)


def test_supports_safe_complete_snapshots():
    decision = prerelax.assess_prerelax_support(_safe_snapshots())

    assert decision.supported is True
    assert decision.reason == "supported"


@pytest.mark.parametrize("cutoff", [-1, 1, 3, 8.0, False, True, None])
def test_support_cutoff_only_accepts_frozen_snapshot_steps(cutoff):
    with pytest.raises(
        ValueError, match=r"cutoff_step.*\(0, 2, 4, 8\)"
    ):
        prerelax.assess_prerelax_support(
            _safe_snapshots(), cutoff_step=cutoff
        )


@pytest.mark.parametrize("cutoff", prerelax.SNAPSHOT_STEPS[:-1])
def test_support_cutoff_requires_only_its_snapshot_prefix(cutoff):
    snapshots = _safe_snapshots()
    prefix = {
        step: snapshots[step]
        for step in prerelax.SNAPSHOT_STEPS
        if step <= cutoff
    }

    decision = prerelax.assess_prerelax_support(
        prefix, cutoff_step=cutoff
    )

    assert decision.supported is True
    assert decision.reason == "supported"


@pytest.mark.parametrize(
    ("cutoff", "invalid_later_step"), ((0, 2), (2, 4), (4, 8))
)
def test_support_cutoff_does_not_validate_invalid_later_snapshot(
    cutoff, invalid_later_step
):
    snapshots = _safe_snapshots()
    snapshots[invalid_later_step].arrays["forces"][0, 0] = np.nan

    decision = prerelax.assess_prerelax_support(
        snapshots, cutoff_step=cutoff
    )

    assert decision.supported is True
    assert decision.reason == "supported"


def test_fail_open_for_force_above_limit():
    snapshots = _safe_snapshots()
    snapshots[2].arrays["forces"][0] = [20.01, 0.0, 0.0]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "force_limit_exceeded"


def test_support_cutoff_ignores_later_force_limit_violation():
    snapshots = _safe_snapshots()
    snapshots[8].arrays["forces"][0] = [20.01, 0.0, 0.0]

    decisions = {
        cutoff: prerelax.assess_prerelax_support(
            snapshots, cutoff_step=cutoff
        )
        for cutoff in prerelax.SNAPSHOT_STEPS
    }

    assert {
        cutoff: (decision.supported, decision.reason)
        for cutoff, decision in decisions.items()
    } == {
        0: (True, "supported"),
        2: (True, "supported"),
        4: (True, "supported"),
        8: (False, "force_limit_exceeded"),
    }


def test_force_at_mathematical_limit_is_supported():
    snapshots = _safe_snapshots()
    snapshots[2].arrays["forces"][0] = [20.0, 0.0, 0.0]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_force_beyond_scale_aware_tolerance_is_rejected():
    snapshots = _safe_snapshots()
    force_limit = 20.0
    tolerance = _scale_aware_tolerance(force_limit)
    snapshots[2].arrays["forces"][0] = [
        np.nextafter(force_limit + tolerance, np.inf),
        0.0,
        0.0,
    ]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "force_limit_exceeded"


def test_fail_open_for_x8_displacement_above_limit():
    snapshots = _safe_snapshots()
    snapshots[8].positions[0] = [0.50, 0.0, 0.0]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "x8_displacement_limit_exceeded"


def test_support_cutoff_uses_its_own_endpoint_displacement():
    snapshots = _safe_snapshots()
    snapshots[8].positions[0] = [0.50, 0.0, 0.0]

    decisions = {
        cutoff: prerelax.assess_prerelax_support(
            snapshots, cutoff_step=cutoff
        )
        for cutoff in prerelax.SNAPSHOT_STEPS
    }

    assert [decisions[cutoff].supported for cutoff in prerelax.SNAPSHOT_STEPS] == [
        True,
        True,
        True,
        False,
    ]
    assert decisions[8].reason == "x8_displacement_limit_exceeded"


def test_x8_displacement_at_mathematical_limit_is_supported():
    snapshots = _safe_snapshots()
    snapshots[8].positions[0] = [0.30, 0.0, 0.0]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_x8_mic_limit_is_supported_with_large_periodic_input_scale():
    snapshots = _safe_snapshots()
    first_x = {0: 9.01, 2: 9.01, 4: 9.01, 8: 0.01}
    for step, atoms in snapshots.items():
        atoms.set_cell(np.diag([9.4, 9.4, 9.4]))
        atoms.positions[:] = [[first_x[step], 0.0, 0.0], [4.7, 0.0, 0.0]]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_x8_displacement_beyond_scale_aware_tolerance_is_rejected():
    snapshots = _safe_snapshots()
    for atoms in snapshots.values():
        atoms.positions[0] = [0.0, 0.0, 0.0]
    tolerance = _scale_aware_tolerance(10.0)
    snapshots[8].positions[0, 0] = np.nextafter(0.40 + tolerance, np.inf)

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "x8_displacement_limit_exceeded"


def test_fail_open_for_adjacent_saved_energy_increase():
    snapshots = _safe_snapshots()
    snapshots[4].info["total_energy"] = -3.96

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "adjacent_energy_increase"


def test_support_cutoff_ignores_later_adjacent_energy_increase():
    snapshots = _safe_snapshots()
    snapshots[8].info["total_energy"] = snapshots[4].info["total_energy"] + 0.05

    decisions = {
        cutoff: prerelax.assess_prerelax_support(
            snapshots, cutoff_step=cutoff
        )
        for cutoff in prerelax.SNAPSHOT_STEPS
    }

    assert [decisions[cutoff].supported for cutoff in prerelax.SNAPSHOT_STEPS] == [
        True,
        True,
        True,
        False,
    ]
    assert decisions[8].reason == "adjacent_energy_increase"


def test_energy_increase_at_mathematical_limit_is_supported():
    snapshots = _safe_snapshots()
    snapshots[4].info["total_energy"] = snapshots[2].info["total_energy"] + 2 * 0.02

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_energy_limit_is_supported_after_large_baseline_subtraction():
    snapshots = _safe_snapshots()
    snapshots[0].info["total_energy"] = -100.04
    snapshots[2].info["total_energy"] = -100.00
    snapshots[4].info["total_energy"] = -100.00
    snapshots[8].info["total_energy"] = -100.00

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is True
    assert decision.reason == "supported"


def test_energy_increase_beyond_scale_aware_tolerance_is_rejected():
    snapshots = _safe_snapshots()
    tolerance = _scale_aware_tolerance(1.0)
    increase = np.nextafter(0.02 + tolerance, np.inf)
    snapshots[0].info["total_energy"] = 0.0
    snapshots[2].info["total_energy"] = 0.0
    snapshots[4].info["total_energy"] = 2.0 * increase
    snapshots[8].info["total_energy"] = 2.0 * increase

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "adjacent_energy_increase"


def test_energy_increase_gate_is_strictly_greater_than_point_zero_two():
    snapshots = _safe_snapshots()
    snapshots[4].info["total_energy"] = snapshots[2].info["total_energy"] + 2 * (
        0.0200000000005
    )

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "adjacent_energy_increase"


def test_fail_open_for_nonpositive_minimum_pair_distance():
    snapshots = _safe_snapshots()
    for atoms in snapshots.values():
        atoms.positions[:] = [0.0, 0.0, 0.0]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_min_pair_distance"


def test_support_cutoff_ignores_later_invalid_minimum_pair_distance():
    snapshots = _safe_snapshots()
    for atoms in snapshots.values():
        atoms.positions[:] = [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]
    snapshots[8].positions[:] = [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]

    decisions = {
        cutoff: prerelax.assess_prerelax_support(
            snapshots, cutoff_step=cutoff
        )
        for cutoff in prerelax.SNAPSHOT_STEPS
    }

    assert [decisions[cutoff].supported for cutoff in prerelax.SNAPSHOT_STEPS] == [
        True,
        True,
        True,
        False,
    ]
    assert decisions[8].reason == "invalid_min_pair_distance"


def test_fail_open_for_nonfinite_minimum_pair_distance():
    snapshots = {}
    for step in prerelax.SNAPSHOT_STEPS:
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[10.0] * 3, pbc=True)
        atoms.info["total_energy"] = -1.0
        atoms.arrays["forces"] = np.zeros((1, 3))
        atoms.info["stress"] = np.zeros(6)
        snapshots[step] = atoms

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_min_pair_distance"


def test_fail_open_with_stable_reason_for_missing_snapshots():
    snapshots = _safe_snapshots()
    del snapshots[8]

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "missing_snapshots"


def test_fail_open_with_stable_reason_for_invalid_snapshots():
    snapshots = _safe_snapshots()
    snapshots[4].arrays["forces"][0, 0] = np.nan

    decision = prerelax.assess_prerelax_support(snapshots)

    assert decision.supported is False
    assert decision.reason == "invalid_snapshots"


def _labelled_harmonic_structure(atomic_number, displacement):
    atoms = Atoms(
        numbers=[atomic_number, atomic_number],
        positions=[[displacement, 0.0, 0.0], [-displacement, 0.0, 0.0]],
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=[True, True, True],
    )
    atoms.info.update(
        {
            "total_energy": -999.0,
            "stress": np.full(6, 999.0),
            "dft_energy": -123.0,
        }
    )
    atoms.new_array("forces", np.full((2, 3), 999.0))
    atoms.new_array("dft_forces", np.full((2, 3), -123.0))
    atoms.new_array("custom_labels", np.arange(2))
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-456.0,
        forces=np.full((2, 3), -456.0),
        stress=np.full(6, -456.0),
    )
    return atoms


class _RecordingHarmonicPredictor:
    def __init__(self):
        self.calls = []

    def __call__(self, structures):
        call = []
        energies = []
        forces = []
        stresses = []
        for atoms in structures:
            assert atoms.info == {}
            assert set(atoms.arrays) == {"numbers", "positions"}
            assert atoms.calc is None
            call.append(atoms.copy())
            positions = np.asarray(atoms.positions, dtype=float)
            energies.append(0.5 * float(np.sum(positions**2)))
            forces.append(-positions)
            stresses.append(np.zeros(6))
        self.calls.append(call)
        return prerelax.BatchPrediction(
            total_energies_ev=energies,
            forces_ev_per_a=forces,
            stresses_ev_per_a3=stresses,
        )


def test_run_fixed_cell_fire_tracks_sanitized_harmonic_trajectories_in_order():
    structures = [
        _labelled_harmonic_structure(atomic_number, displacement)
        for atomic_number, displacement in ((1, 1.0), (2, 1.2), (3, 1.4))
    ]
    initial_positions = [atoms.positions.copy() for atoms in structures]
    initial_cells = [atoms.cell.array.copy() for atoms in structures]
    predictor = _RecordingHarmonicPredictor()

    run = prerelax.run_fixed_cell_fire(
        structures,
        predictor,
        atom_budget=4,
        structure_cap=2,
    )

    assert run.predictor_forward_calls == 18
    assert run.elapsed_seconds >= 0.0
    assert len(run.results) == 3
    assert [result.snapshots[0].numbers[0] for result in run.results] == [1, 2, 3]
    assert [len(call) for call in predictor.calls] == [2] * 9 + [1] * 9

    call_offsets = (0, 0, 9)
    call_slots = (0, 1, 0)
    for item_index, result in enumerate(run.results):
        assert result.error is None
        assert result.force_evaluations == 9
        assert result.optimizer_updates == 8
        assert result.retry_overhead_force_evaluations == 0
        assert result.retry_overhead_optimizer_updates == 0
        assert tuple(result.snapshots) == prerelax.SNAPSHOT_STEPS
        assert np.array_equal(structures[item_index].positions, initial_positions[item_index])

        trajectory_positions = [
            predictor.calls[call_offsets[item_index] + step][
                call_slots[item_index]
            ].positions
            for step in range(9)
        ]
        assert np.array_equal(trajectory_positions[0], initial_positions[item_index])
        for previous, current in zip(
            trajectory_positions[:-1], trajectory_positions[1:], strict=True
        ):
            assert np.linalg.norm(current - previous) <= 0.05 + 1.0e-12

        for step in prerelax.SNAPSHOT_STEPS:
            snapshot = result.snapshots[step]
            assert np.allclose(snapshot.positions, trajectory_positions[step])
            assert np.array_equal(snapshot.cell.array, initial_cells[item_index])
            assert set(snapshot.info) == {"total_energy", "stress"}
            assert set(snapshot.arrays) == {"numbers", "positions", "forces"}
            assert snapshot.calc is None

        saved_energies = [
            result.snapshots[step].info["total_energy"]
            for step in prerelax.SNAPSHOT_STEPS
        ]
        assert all(
            later < earlier
            for earlier, later in zip(
                saved_energies[:-1], saved_energies[1:], strict=True
            )
        )


@pytest.mark.parametrize("failure_mode", ["exception", "wrong_shape", "nonfinite"])
def test_bad_batch_prediction_restarts_each_structure_from_x0(failure_mode):
    structures = [
        _labelled_harmonic_structure(1, 1.0),
        _labelled_harmonic_structure(2, 1.2),
    ]
    initial_positions = [atoms.positions.copy() for atoms in structures]
    harmonic = _RecordingHarmonicPredictor()
    batch_calls = 0
    all_calls = []

    def predictor(batch):
        nonlocal batch_calls
        all_calls.append([atoms.positions.copy() for atoms in batch])
        if len(batch) > 1:
            batch_calls += 1
            if batch_calls == 3:
                if failure_mode == "exception":
                    raise RuntimeError("synthetic batch failure")
                prediction = harmonic(batch)
                if failure_mode == "wrong_shape":
                    prediction.forces_ev_per_a[0] = np.zeros((len(batch[0]), 2))
                else:
                    prediction.stresses_ev_per_a3[0][0] = np.nan
                return prediction
        return harmonic(batch)

    run = prerelax.run_fixed_cell_fire(structures, predictor, atom_budget=4)

    assert run.predictor_forward_calls == 21
    assert all(result.error is None for result in run.results)
    assert all(result.force_evaluations == 12 for result in run.results)
    assert all(result.optimizer_updates == 10 for result in run.results)
    assert all(
        result.retry_overhead_force_evaluations == 3 for result in run.results
    )
    assert all(
        result.retry_overhead_optimizer_updates == 2 for result in run.results
    )
    assert np.array_equal(all_calls[3][0], initial_positions[0])
    assert np.array_equal(all_calls[12][0], initial_positions[1])


def test_single_prediction_failure_is_fail_open_and_does_not_block_neighbors():
    structures = [
        _labelled_harmonic_structure(1, 1.0),
        _labelled_harmonic_structure(2, 1.2),
        _labelled_harmonic_structure(3, 1.4),
    ]
    harmonic = _RecordingHarmonicPredictor()

    def predictor(batch):
        if len(batch) > 1:
            raise RuntimeError("force per-structure fallback")
        if batch[0].numbers[0] == 2:
            raise RuntimeError("synthetic single failure")
        return harmonic(batch)

    run = prerelax.run_fixed_cell_fire(structures, predictor, atom_budget=6)

    assert run.predictor_forward_calls == 20
    assert [result.error for result in run.results] == [
        None,
        "predictor_failed",
        None,
    ]
    assert run.results[1].snapshots == {}
    assert run.results[1].force_evaluations == 2
    assert run.results[1].optimizer_updates == 0
    assert run.results[1].retry_overhead_force_evaluations == 1
    assert run.results[1].retry_overhead_optimizer_updates == 0
    assert run.results[0].force_evaluations == 10
    assert run.results[2].force_evaluations == 10
    assert run.results[0].retry_overhead_force_evaluations == 1
    assert run.results[2].retry_overhead_force_evaluations == 1
    assert run.results[0].snapshots[0].numbers[0] == 1
    assert run.results[2].snapshots[0].numbers[0] == 3


@pytest.mark.parametrize(
    "bad_kind",
    [
        "length",
        "force_shape",
        "stress_shape",
        "energy_nonfinite",
        "force_nonfinite",
        "stress_nonfinite",
    ],
)
def test_invalid_single_prediction_returns_stable_fail_open_result(bad_kind):
    energies = [0.0]
    forces = [np.zeros((2, 3))]
    stresses = [np.zeros(6)]
    if bad_kind == "length":
        energies.append(1.0)
    elif bad_kind == "force_shape":
        forces[0] = np.zeros((2, 2))
    elif bad_kind == "stress_shape":
        stresses[0] = np.zeros((2, 2))
    elif bad_kind == "energy_nonfinite":
        energies[0] = np.nan
    elif bad_kind == "force_nonfinite":
        forces[0][0, 0] = np.inf
    else:
        stresses[0][0] = np.nan

    def predictor(_batch):
        return prerelax.BatchPrediction(energies, forces, stresses)

    run = prerelax.run_fixed_cell_fire(
        [_labelled_harmonic_structure(1, 1.0)],
        predictor,
        atom_budget=2,
    )

    assert run.predictor_forward_calls == 2
    assert run.results[0].error == "predictor_failed"
    assert run.results[0].snapshots == {}
    assert run.results[0].force_evaluations == 2
    assert run.results[0].optimizer_updates == 0
    assert run.results[0].retry_overhead_force_evaluations == 1
    assert run.results[0].retry_overhead_optimizer_updates == 0


def test_mattersim_predictor_is_lazy_loads_once_and_converts_stress(monkeypatch):
    calls = {"loads": [], "dataloaders": [], "predictions": []}

    class FakePotential:
        def __init__(self):
            self.model = types.SimpleNamespace(
                model_args={"cutoff": 6.25, "threebody_cutoff": 4.75}
            )

        @classmethod
        def from_checkpoint(cls, checkpoint, **kwargs):
            calls["loads"].append((checkpoint, kwargs))
            return cls()

        def predict_properties(self, loader, **kwargs):
            calls["predictions"].append((loader, kwargs))
            count = len(loader)
            return (
                np.arange(count, dtype=float),
                [np.ones((len(atoms), 3)) for atoms in loader],
                np.full((count, 3, 3), 2.0),
            )

    def fake_build_dataloader(structures, **kwargs):
        calls["dataloaders"].append((structures, kwargs))
        return structures

    modules = {
        "mattersim": types.ModuleType("mattersim"),
        "mattersim.datasets": types.ModuleType("mattersim.datasets"),
        "mattersim.datasets.utils": types.ModuleType("mattersim.datasets.utils"),
        "mattersim.datasets.utils.build": types.ModuleType(
            "mattersim.datasets.utils.build"
        ),
        "mattersim.forcefield": types.ModuleType("mattersim.forcefield"),
    }
    modules["mattersim.datasets.utils.build"].build_dataloader = (
        fake_build_dataloader
    )
    modules["mattersim.forcefield"].Potential = FakePotential
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    predictor = prerelax.make_mattersim_predictor(
        "fake-checkpoint.pth", device="cpu", batch_size=7
    )
    structures = [Atoms("H2"), Atoms("He")]
    first = predictor(structures)
    second = predictor(structures[:1])

    assert calls["loads"] == [
        (
            "fake-checkpoint.pth",
            {"device": "cpu", "load_training_state": False},
        )
    ]
    assert len(calls["dataloaders"]) == 2
    assert calls["dataloaders"][0][1] == {
        "cutoff": 6.25,
        "threebody_cutoff": 4.75,
        "batch_size": 7,
        "only_inference": True,
    }
    assert all(
        kwargs == {"include_forces": True, "include_stresses": True}
        for _, kwargs in calls["predictions"]
    )
    assert np.allclose(first.stresses_ev_per_a3, 2.0 * units.GPa)
    assert np.allclose(second.stresses_ev_per_a3, 2.0 * units.GPa)
