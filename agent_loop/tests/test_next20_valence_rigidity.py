from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next20_valence_rigidity import (
    FEATURE_NAMES,
    RigidityFeatureResult,
    compute_valence_rigidity_features,
    rigidity_features_from_edges,
)


TETRAHEDRON = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=float,
)


def _tetrahedral_result(*, scale: float = 1.0, radius_sums=None):
    return rigidity_features_from_edges(
        n_sites=5,
        endpoints=np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int),
        vectors=scale * TETRAHEDRON,
        radius_sums=(
            np.full(4, np.sqrt(3.0), dtype=float)
            if radius_sums is None
            else np.asarray(radius_sums, dtype=float)
        ),
        weights=np.ones(4, dtype=float),
    )


def test_feature_schema_contains_no_model_or_endpoint_quantities() -> None:
    forbidden = ("energy", "force", "stress", "dft", "relax", "model", "proxy")
    assert FEATURE_NAMES
    assert not any(token in name.lower() for name in FEATURE_NAMES for token in forbidden)


def test_regular_tetrahedron_has_zero_centered_mismatch_and_imbalance() -> None:
    result = _tetrahedral_result()
    assert isinstance(result, RigidityFeatureResult)
    assert result.supported
    assert tuple(result.features) == FEATURE_NAMES
    assert result.features["sivr_edge_mismatch_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["sivr_site_imbalance_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["sivr_cell_anisotropy"] == pytest.approx(0.0, abs=1e-12)


def test_centered_descriptors_are_uniform_scale_invariant() -> None:
    reference = _tetrahedral_result(scale=1.0)
    scaled = _tetrahedral_result(scale=1.7)
    for name in FEATURE_NAMES:
        if name == "sivr_scale_log_median":
            assert scaled.features[name] - reference.features[name] == pytest.approx(
                np.log(1.7), abs=1e-12
            )
        elif name not in {"sivr_edge_count", "sivr_site_count"}:
            assert scaled.features[name] == pytest.approx(
                reference.features[name], rel=1e-10, abs=1e-12
            )


def test_features_are_rigid_rotation_invariant() -> None:
    theta = 0.713
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference = _tetrahedral_result(
        radius_sums=[np.sqrt(3.0), 1.8, 1.6, 2.0]
    )
    rotated = rigidity_features_from_edges(
        n_sites=5,
        endpoints=np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int),
        vectors=TETRAHEDRON @ rotation.T,
        radius_sums=np.asarray([np.sqrt(3.0), 1.8, 1.6, 2.0]),
        weights=np.ones(4),
    )
    assert reference.supported and rotated.supported
    for name in FEATURE_NAMES:
        assert rotated.features[name] == pytest.approx(
            reference.features[name], rel=1e-9, abs=1e-11
        )


def test_inconsistent_edge_scales_create_mismatch_and_unstable_modes() -> None:
    # The first edge has e > 1 after scale centering, so its radial prestress
    # stiffness is negative under the analytic phi=log(r/r0)^2/2 model.
    result = _tetrahedral_result(radius_sums=[0.5, 1.7, 1.7, 1.7])
    assert result.supported
    assert result.features["sivr_edge_mismatch_rms"] > 0.1
    assert result.features["sivr_site_imbalance_max"] > 0.1
    assert result.features["sivr_negative_mode_fraction"] > 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_sites": 1}, "at least two sites"),
        ({"weights": np.asarray([1.0, 0.0, 1.0, 1.0])}, "weights"),
        ({"radius_sums": np.asarray([1.0, -1.0, 1.0, 1.0])}, "radius"),
    ],
)
def test_invalid_edge_system_fails_open(kwargs, message: str) -> None:
    inputs = {
        "n_sites": 5,
        "endpoints": np.asarray([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int),
        "vectors": TETRAHEDRON,
        "radius_sums": np.full(4, np.sqrt(3.0)),
        "weights": np.ones(4),
    }
    inputs.update(kwargs)
    result = rigidity_features_from_edges(**inputs)
    assert not result.supported
    assert result.features == {}
    assert message in str(result.failure_reason).lower()


def test_structure_entrypoint_is_independent_and_does_not_mutate_input() -> None:
    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    species_before = tuple(str(site.specie) for site in structure)
    coordinates_before = np.asarray(structure.frac_coords).copy()
    result = compute_valence_rigidity_features(
        structure,
        graph_mode="voronoi",
        charge_weight_exponent=0.5,
    )
    assert result.supported, result.failure_reason
    assert tuple(result.features) == FEATURE_NAMES
    assert tuple(str(site.specie) for site in structure) == species_before
    assert np.array_equal(structure.frac_coords, coordinates_before)


def test_structure_entrypoint_rejects_nonfrozen_configuration() -> None:
    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    result = compute_valence_rigidity_features(
        structure,
        graph_mode="unsupported",
        charge_weight_exponent=0.5,
    )
    assert not result.supported
    assert "graph mode" in str(result.failure_reason).lower()
