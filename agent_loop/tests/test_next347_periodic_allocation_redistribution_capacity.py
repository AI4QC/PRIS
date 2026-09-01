from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next347_periodic_allocation_redistribution_capacity as n


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
    assert n.PROTOCOL == "2026-08-13-next347-periodic-allocation-redistribution-capacity-v1"
    assert n.FEATURE_NAMES == ("parc_allocation_redistribution_protection",)
    assert n.FEATURE_DIRECTIONS == {
        "parc_allocation_redistribution_protection": "protected_high"
    }


def test_exact_allocation_has_unit_protection() -> None:
    result = n.allocation_redistribution_capacity_protection(
        observed=[0.25, 0.75],
        target=[0.25, 0.75],
        endpoints=[[0, 1]],
        conductances=[3.0],
    )
    assert result.protection == 1.0
    assert result.global_energy == 0.0
    assert result.capacity_energy == 0.0
    assert result.maximum_poisson_residual == 0.0


def test_two_site_graph_has_exact_two_thirds_protection() -> None:
    result = n.allocation_redistribution_capacity_protection(
        observed=[0.75, 0.25],
        target=[0.25, 0.75],
        endpoints=[[0, 1]],
        conductances=[2.0],
    )
    assert result.global_energy == pytest.approx(0.125)
    assert result.capacity_energy == pytest.approx(0.25)
    assert result.capacity_ratio == pytest.approx(0.5)
    assert result.protection == pytest.approx(2.0 / 3.0)


def test_self_image_edge_has_two_incident_capacity_contributions() -> None:
    capacity = n.periodic_incident_capacity(
        site_count=2,
        endpoints=[[0, 1], [0, 0], [1, 1]],
        conductances=[2.0, 3.0, 5.0],
    )
    assert capacity.tolist() == pytest.approx([8.0, 12.0])


def test_primitive_quotient_equals_explicit_two_copy_cover() -> None:
    primitive = n.allocation_redistribution_capacity_protection(
        observed=[0.75, 0.25],
        target=[0.25, 0.75],
        endpoints=[[0, 1], [0, 0], [1, 1]],
        conductances=[2.0, 3.0, 5.0],
    )
    cover = n.allocation_redistribution_capacity_protection(
        observed=[0.375, 0.125, 0.375, 0.125],
        target=[0.125, 0.375, 0.125, 0.375],
        endpoints=[[0, 1], [2, 3], [0, 2], [0, 2], [1, 3], [1, 3]],
        conductances=[2.0, 2.0, 3.0, 3.0, 5.0, 5.0],
    )
    assert cover.global_energy == pytest.approx(primitive.global_energy / 2.0)
    assert cover.capacity_energy == pytest.approx(primitive.capacity_energy / 2.0)
    assert cover.capacity_ratio == pytest.approx(primitive.capacity_ratio)
    assert cover.protection == pytest.approx(primitive.protection)


def test_kernel_is_conductance_scale_edge_orientation_order_and_gauge_invariant() -> None:
    observed = [0.6, 0.1, 0.3]
    target = [0.2, 0.4, 0.4]
    endpoints = np.asarray([[0, 1], [1, 2], [2, 0], [0, 0]])
    conductances = np.asarray([1.0, 3.0, 2.0, 0.7])
    reference = n.allocation_redistribution_capacity_protection(
        observed=observed, target=target, endpoints=endpoints, conductances=conductances
    )
    order = np.asarray([3, 2, 0, 1])
    changed = n.allocation_redistribution_capacity_protection(
        observed=observed,
        target=target,
        endpoints=endpoints[order, ::-1],
        conductances=17.0 * conductances[order],
    )
    assert changed.capacity_ratio == pytest.approx(reference.capacity_ratio)
    assert changed.protection == pytest.approx(reference.protection)


def test_kernel_refuses_disconnected_source_and_invalid_populations() -> None:
    with pytest.raises(ValueError, match="Poisson residual differs"):
        n.allocation_redistribution_capacity_protection(
            observed=[0.5, 0.0, 0.5],
            target=[0.0, 0.5, 0.5],
            endpoints=[[0, 2], [1, 1]],
            conductances=[1.0, 1.0],
        )
    with pytest.raises(ValueError, match="allocation population differs"):
        n.allocation_redistribution_capacity_protection(
            observed=[0.6, 0.6],
            target=[0.5, 0.5],
            endpoints=[[0, 1]],
            conductances=[1.0],
        )
    with pytest.raises(ValueError, match="edge population differs"):
        n.periodic_incident_capacity(
            site_count=2,
            endpoints=[[0.0, 0.5]],
            conductances=[1.0],
        )


def _feature(atoms: Atoms) -> float:
    result = n.compute_parc_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_parc() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = n.compute_parc_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.edge_count >= 3
        assert result.minimum_periodic_capacity > 0.0
        assert result.maximum_poisson_residual <= n.POISSON_RESIDUAL_TOLERANCE
        assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_parc() -> None:
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
        result = n.compute_parc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_boundary_and_builder_interface_are_exact() -> None:
    row = n.compute_parc_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("parc_")) == (
        "parc_allocation_redistribution_protection", "parc_supported", "parc_failure",
        "parc_site_count", "parc_edge_count", "parc_allocation_total_variation",
        "parc_capacity_ratio", "parc_minimum_periodic_capacity",
        "parc_maximum_poisson_residual", "parc_volume_tiling_relative_error",
    )
    assert row["parc_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
    parameters = tuple(inspect.signature(n.build_cross_source_parc_features).parameters)
    assert parameters == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "design_path",
        "probe_result_path", "output_dir", "workers", "require_formal_inputs",
    )
    assert not any(
        token in name for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_formal_publication_schema_and_gates_are_frozen() -> None:
    assert n.MANIFEST_NAME == "MANIFEST.json"
    assert n.CATALOGUE_NAME == "NEXT347_PARC_FEATURE_CATALOGUE.json"
    assert n.FEATURE_FILES == {
        "scigen": "next347_scigen_periodic_allocation_redistribution_capacity.parquet",
        "wyformer": "next347_wyformer_periodic_allocation_redistribution_capacity.parquet",
    }
    assert n.EXPECTED_ROWS == {"scigen": 13_470, "wyformer": 5_232}
    assert n.MINIMUM_FORMAL_COVERAGE == 0.90


def test_wyformer_payload_worker_produces_real_row_and_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    structure = n.n267.AseAtomsAdaptor.get_structure(atoms)
    material_id, row = n._compute_wyformer_payload(
        ("wy-real", __import__("json").dumps(structure.as_dict()))
    )
    assert material_id == "wy-real"
    assert row["parc_supported"] is True
    assert np.isfinite(row[n.FEATURE_NAMES[0]])

    bad_id, bad = n._compute_wyformer_payload(("wy-bad", "not-json"))
    assert bad_id == "wy-bad"
    assert bad["parc_supported"] is False
    assert np.isnan(bad[n.FEATURE_NAMES[0]])


def test_label_free_statistics_are_exact_and_ignore_abstentions() -> None:
    table = __import__("pandas").DataFrame(
        {n.FEATURE_NAMES[0]: [0.2, 0.4, np.nan, 0.8, 1.0]}
    )
    statistics = n._label_free_statistics(table)[n.FEATURE_NAMES[0]]
    assert statistics == {
        "minimum": 0.2,
        "q10": 0.2,
        "median": pytest.approx(0.6),
        "q90": 1.0,
        "maximum": 1.0,
        "unique_rounded_10": 4,
    }


def test_builder_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT347 input is missing"):
        n.build_cross_source_parc_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            probe_result_path=tmp_path / "probe",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
