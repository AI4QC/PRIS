from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next343_periodic_allocation_redistribution_impedance as n


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def test_frozen_schema_is_one_protected_high_feature() -> None:
    assert n.PROTOCOL == "2026-08-13-next343-periodic-allocation-redistribution-impedance-v1"
    assert n.FEATURE_NAMES == ("pari_allocation_redistribution_protection",)
    assert n.FEATURE_DIRECTIONS == {
        "pari_allocation_redistribution_protection": "protected_high"
    }


def test_exact_allocation_has_unit_protection() -> None:
    result = n.allocation_redistribution_protection(
        observed=[0.25, 0.75],
        target=[0.25, 0.75],
        endpoints=[[0, 1]],
        conductances=[3.0],
    )
    assert result.protection == 1.0
    assert result.global_energy == 0.0
    assert result.local_energy == 0.0
    assert result.maximum_poisson_residual == 0.0


def test_two_site_graph_has_exact_two_thirds_protection() -> None:
    result = n.allocation_redistribution_protection(
        observed=[0.75, 0.25],
        target=[0.25, 0.75],
        endpoints=[[0, 1]],
        conductances=[2.0],
    )
    assert result.global_energy == pytest.approx(0.125)
    assert result.local_energy == pytest.approx(0.25)
    assert result.impedance_ratio == pytest.approx(0.5)
    assert result.protection == pytest.approx(2.0 / 3.0)


def test_global_topology_changes_protection_at_fixed_source_and_local_degree() -> None:
    observed = [0.6, 0.2, 0.2, 0.0]
    target = [0.25, 0.25, 0.25, 0.25]
    cycle = n.allocation_redistribution_protection(
        observed=observed,
        target=target,
        endpoints=[[0, 1], [1, 2], [2, 3], [3, 0]],
        conductances=np.ones(4),
    )
    complete = n.allocation_redistribution_protection(
        observed=observed,
        target=target,
        endpoints=[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        conductances=np.full(6, 2.0 / 3.0),
    )
    assert cycle.local_energy == pytest.approx(complete.local_energy)
    assert cycle.protection != pytest.approx(complete.protection)


def test_kernel_is_conductance_scale_edge_orientation_order_and_gauge_invariant() -> None:
    observed = [0.6, 0.1, 0.3]
    target = [0.2, 0.4, 0.4]
    endpoints = np.asarray([[0, 1], [1, 2], [2, 0]])
    conductances = np.asarray([1.0, 3.0, 2.0])
    reference = n.allocation_redistribution_protection(
        observed=observed, target=target, endpoints=endpoints, conductances=conductances
    )
    order = np.asarray([2, 0, 1])
    reversed_endpoints = endpoints[order, ::-1]
    changed = n.allocation_redistribution_protection(
        observed=observed,
        target=target,
        endpoints=reversed_endpoints,
        conductances=17.0 * conductances[order],
    )
    assert changed.impedance_ratio == pytest.approx(reference.impedance_ratio)
    assert changed.protection == pytest.approx(reference.protection)


def test_kernel_refuses_disconnected_source_and_invalid_allocations() -> None:
    with pytest.raises(ValueError, match="zero graph degree"):
        n.allocation_redistribution_protection(
            observed=[0.5, 0.0, 0.5],
            target=[0.0, 0.5, 0.5],
            endpoints=[[0, 2]],
            conductances=[1.0],
        )
    with pytest.raises(ValueError, match="allocation population differs"):
        n.allocation_redistribution_protection(
            observed=[0.6, 0.6], target=[0.5, 0.5], endpoints=[[0, 1]], conductances=[1.0]
        )


def _feature(atoms: Atoms) -> float:
    result = n.compute_pari_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_pari() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = n.compute_pari_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.edge_count >= 3
        assert result.maximum_poisson_residual <= n.POISSON_RESIDUAL_TOLERANCE
        assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_pari() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int) @ atoms.cell.array,
        scale_atoms=False,
    ); rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    with_calculator = atoms.copy(); with_calculator.calc = Calculator()
    with_metadata = atoms.copy(); with_metadata.info["outcome"] = 1
    with_array = atoms.copy(); with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy(); nonperiodic.pbc = False
    nonfinite = atoms.copy(); nonfinite.positions[0, 0] = np.nan
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_pari_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_boundary_and_builder_interface_are_exact() -> None:
    row = n.compute_pari_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("pari_")) == (
        "pari_allocation_redistribution_protection", "pari_supported", "pari_failure",
        "pari_site_count", "pari_edge_count", "pari_allocation_total_variation",
        "pari_impedance_ratio", "pari_maximum_poisson_residual",
        "pari_volume_tiling_relative_error",
    )
    assert row["pari_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
    parameters = tuple(inspect.signature(n.build_cross_source_pari_features).parameters)
    assert parameters == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "design_path",
        "probe_result_path", "output_dir", "workers", "require_formal_inputs",
    )
    assert not any(
        token in name for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
