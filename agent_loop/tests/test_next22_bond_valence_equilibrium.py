from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next22_bond_valence_equilibrium import (
    FEATURE_NAMES,
    BondValenceFeatureResult,
    compute_scale_calibrated_bond_valence_features,
    scale_calibrated_bond_valence_features,
)


TETRA = np.asarray(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
)
ENDPOINTS = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int)
CHARGES = np.asarray([4.0, -1.0, -1.0, -1.0, -1.0])


def _result(strengths=None):
    return scale_calibrated_bond_valence_features(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        strengths=np.ones(4) if strengths is None else np.asarray(strengths, float),
        vectors=TETRA,
        parameter_sources=("exact", "exact", "nearest_valence", "brown_generic"),
    )


def test_schema_contains_no_endpoint_or_learned_quantities() -> None:
    forbidden = ("energy", "force", "stress", "dft", "relax", "model", "proxy")
    assert FEATURE_NAMES
    assert not any(token in name.lower() for name in FEATURE_NAMES for token in forbidden)


def test_balanced_tetrahedron_has_zero_scale_calibrated_mismatch() -> None:
    result = _result()
    assert isinstance(result, BondValenceFeatureResult)
    assert result.supported
    assert tuple(result.features) == FEATURE_NAMES
    assert result.features["scbv_mismatch_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["scbv_mismatch_max"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["scbv_parameter_exact_fraction"] == pytest.approx(0.5)
    assert result.features["scbv_parameter_generic_fraction"] == pytest.approx(0.25)


def test_mismatch_and_shape_features_are_strength_scale_invariant() -> None:
    reference = _result([1.0, 1.4, 0.8, 1.2])
    scaled = _result([3.7, 5.18, 2.96, 4.44])
    for name in FEATURE_NAMES:
        if name == "scbv_global_scale":
            assert scaled.features[name] == pytest.approx(
                reference.features[name] / 3.7, rel=1e-12
            )
        elif name not in {"scbv_edge_count", "scbv_site_count"}:
            assert scaled.features[name] == pytest.approx(
                reference.features[name], rel=1e-11, abs=1e-12
            )


def test_one_inconsistent_bond_creates_site_mismatch() -> None:
    result = _result([0.1, 1.0, 1.0, 1.0])
    assert result.supported
    assert result.features["scbv_mismatch_rms"] > 0.1
    assert result.features["scbv_cation_mismatch_rms"] > 0.0
    assert result.features["scbv_anion_mismatch_rms"] > 0.0


def test_radius_generic_parameter_is_disclosed_as_generic() -> None:
    result = scale_calibrated_bond_valence_features(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        strengths=np.ones(4),
        vectors=TETRA,
        parameter_sources=("radius_generic",) * 4,
    )
    assert result.supported
    assert result.features["scbv_parameter_exact_fraction"] == 0.0
    assert result.features["scbv_parameter_generic_fraction"] == 1.0


def test_rigid_rotation_does_not_change_features() -> None:
    theta = 0.41
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    reference = _result([1.0, 1.4, 0.8, 1.2])
    rotated = scale_calibrated_bond_valence_features(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        strengths=np.asarray([1.0, 1.4, 0.8, 1.2]),
        vectors=TETRA @ rotation.T,
        parameter_sources=("exact", "exact", "nearest_valence", "brown_generic"),
    )
    for name in FEATURE_NAMES:
        assert rotated.features[name] == pytest.approx(
            reference.features[name], rel=1e-10, abs=1e-12
        )


def test_invalid_inputs_fail_open() -> None:
    result = scale_calibrated_bond_valence_features(
        charges=[1.0, -1.0],
        endpoints=[[0, 1]],
        strengths=[0.0],
        vectors=[[1.0, 0.0, 0.0]],
        parameter_sources=("exact",),
    )
    assert not result.supported
    assert "strength" in str(result.failure_reason).lower()


def test_structure_entrypoint_uses_one_raw_structure_without_mutation() -> None:
    structure = Structure(
        Lattice.cubic(4.2),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    species = tuple(str(site.specie) for site in structure)
    coordinates = np.asarray(structure.frac_coords).copy()
    result = compute_scale_calibrated_bond_valence_features(
        structure, graph_mode="voronoi"
    )
    assert result.supported, result.failure_reason
    assert tuple(result.features) == FEATURE_NAMES
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.frac_coords, coordinates)
