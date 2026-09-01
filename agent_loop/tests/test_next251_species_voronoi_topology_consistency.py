from __future__ import annotations

import inspect
import math

from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next251_species_voronoi_topology_consistency import (
    FEATURE_NAMES,
    PROTOCOL,
    ROBUST_AREA_FRACTION,
    SiteTopology,
    aggregate_svtc_features,
    build_cross_source_svtc_features,
    compute_svtc_features,
    face_topology,
    species_topology_consistency,
)


def test_protocol_identity_is_exact() -> None:
    assert PROTOCOL == "2026-08-09-next251-species-voronoi-topology-consistency-v1"


def test_raw_and_robust_face_topology_are_exact_and_order_invariant() -> None:
    areas = np.asarray([64.0, 32.0, 1.0, 0.5])
    degrees = np.asarray([3, 4, 5, 6])
    raw = face_topology(areas=areas, degrees=degrees, mode="raw")
    robust = face_topology(areas=areas, degrees=degrees, mode="robust")
    reversed_raw = face_topology(
        areas=areas[::-1], degrees=degrees[::-1], mode="raw"
    )
    assert ROBUST_AREA_FRACTION == 1 / 64
    assert raw.signature == (1, 1, 1, 1, 0, 0, 0)
    assert raw.odd_area_fraction == pytest.approx(65.0 / 97.5)
    assert robust.signature == (1, 1, 0, 0, 0, 0, 0)
    assert robust.odd_area_fraction == pytest.approx(2.0 / 3.0)
    assert raw == reversed_raw
    expected_entropy = -sum(
        p * math.log(p) for p in (64 / 97.5, 32 / 97.5, 1 / 97.5, 0.5 / 97.5)
    ) / math.log(7)
    assert raw.degree_entropy == pytest.approx(expected_entropy)
    assert 0.0 <= robust.degree_entropy <= 1.0


def test_face_topology_bins_nine_and_above_together_and_fails_closed() -> None:
    result = face_topology(
        areas=np.asarray([1.0, 2.0, 3.0]),
        degrees=np.asarray([7, 9, 12]),
        mode="raw",
    )
    assert result.signature == (0, 0, 0, 0, 1, 0, 2)
    with pytest.raises(ValueError, match="face population differs"):
        face_topology(areas=[1.0, -1.0], degrees=[4, 5], mode="raw")
    with pytest.raises(ValueError, match="mode differs"):
        face_topology(areas=[1.0], degrees=[4], mode="searched")


def test_species_consistency_is_exact_and_replication_invariant() -> None:
    a = (1, 0, 0, 0, 0, 0, 0)
    b = (0, 1, 0, 0, 0, 0, 0)
    c = (0, 0, 1, 0, 0, 0, 0)
    d = (0, 0, 0, 1, 0, 0, 0)
    signatures = [a, a, b, c, c, d]
    numbers = [8, 8, 8, 14, 14, 6]
    result = species_topology_consistency(
        signatures=signatures, atomic_numbers=numbers
    )
    h_oxygen = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)) / math.log(2)
    h_oxygen_raw = -(
        (2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)
    )
    assert result["species_modal_agreement"] == pytest.approx(5 / 6)
    assert result["species_signature_entropy"] == pytest.approx(0.5 * h_oxygen)
    assert result["species_signature_gini"] == pytest.approx(2 / 9)
    assert result["species_effective_signature_excess"] == pytest.approx(
        0.5 * (1 - math.exp(-h_oxygen_raw))
    )
    replicated = species_topology_consistency(
        signatures=signatures * 3, atomic_numbers=numbers * 3
    )
    assert replicated == pytest.approx(result)


def test_aggregate_schema_and_linear_q90_are_exact() -> None:
    raw = [
        SiteTopology((1, 0, 0, 0, 0, 0, 0), 0.0, 0.0),
        SiteTopology((0, 1, 0, 0, 0, 0, 0), 1.0, 1.0),
    ]
    robust = [
        SiteTopology((1, 0, 0, 0, 0, 0, 0), 0.25, 0.5),
        SiteTopology((1, 0, 0, 0, 0, 0, 0), 0.75, 0.25),
    ]
    features = aggregate_svtc_features(
        raw_sites=raw, robust_sites=robust, atomic_numbers=[8, 8]
    )
    assert tuple(features) == FEATURE_NAMES
    assert features["svtc_raw_odd_area_mean"] == pytest.approx(0.5)
    assert features["svtc_raw_odd_area_q90"] == pytest.approx(0.9)
    assert features["svtc_raw_species_modal_agreement"] == pytest.approx(0.5)
    assert features["svtc_robust_species_modal_agreement"] == pytest.approx(1.0)
    assert np.isfinite(list(features.values())).all()


def test_known_crystal_and_geometry_invariances() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    reference = compute_svtc_features(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    scaled = atoms.copy()
    scaled.set_cell(1.7 * scaled.cell.array, scale_atoms=True)
    supercell = atoms.repeat((2, 1, 1))
    for result in (
        reference,
        compute_svtc_features(rotated),
        compute_svtc_features(scaled),
        compute_svtc_features(supercell),
    ):
        assert result.supported is True, result.failure_reason
        assert tuple(result.features) == FEATURE_NAMES
        np.testing.assert_allclose(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
            rtol=0.0,
            atol=3.0e-10,
        )
    assert reference.features["svtc_raw_odd_area_mean"] == pytest.approx(0.0)
    assert reference.features["svtc_raw_degree_entropy_mean"] == pytest.approx(0.0)


def test_fcc_voronoi_cell_has_twelve_quadrilateral_faces() -> None:
    result = compute_svtc_features(bulk("Cu", "fcc", a=3.6))
    assert result.supported is True, result.failure_reason
    assert result.features["svtc_raw_odd_area_mean"] == pytest.approx(0.0)
    assert result.features["svtc_raw_degree_entropy_mean"] == pytest.approx(0.0)
    assert result.features["svtc_raw_species_modal_agreement"] == pytest.approx(1.0)


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    assert compute_svtc_features(with_calculator).supported is False
    assert compute_svtc_features(with_metadata).supported is False


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_svtc_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "parent_design_path",
        "first_amendment_path",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_builder_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT251 input is missing"):
        build_cross_source_svtc_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            parent_design_path=tmp_path / "parent_design",
            first_amendment_path=tmp_path / "first_amendment",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
