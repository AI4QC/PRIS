from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next295_positive_contact_force_closure import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    METRIC_NAMES,
    PROTOCOL,
    build_cross_source_pcfc_features,
    compute_pcfc_features,
    positive_contact_force_closure_features,
    positive_equilibrium_fraction,
    site_pcfc_metrics,
)


def _tetrahedron() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)


def _octahedron() -> np.ndarray:
    return np.vstack((np.eye(3), -np.eye(3)))


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray(
            [[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]
        ),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = compute_pcfc_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_metric_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next295-positive-contact-force-closure-v1"
    assert METRIC_NAMES == (
        "uniform_closure",
        "weighted_closure",
        "uniform_equilibrium",
        "weighted_equilibrium",
    )
    assert FEATURE_NAMES == (
        "pcfc_uniform_closure_min",
        "pcfc_uniform_closure_q10",
        "pcfc_uniform_closure_mean",
        "pcfc_weighted_closure_min",
        "pcfc_weighted_closure_q10",
        "pcfc_weighted_closure_mean",
        "pcfc_uniform_equilibrium_min",
        "pcfc_uniform_equilibrium_q10",
        "pcfc_uniform_equilibrium_mean",
        "pcfc_weighted_equilibrium_min",
        "pcfc_weighted_equilibrium_q10",
        "pcfc_weighted_equilibrium_mean",
        "pcfc_locally_enclosed_fraction",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_high" for name in FEATURE_NAMES}


@pytest.mark.parametrize("directions", [_tetrahedron(), _octahedron()])
def test_symmetric_direction_sets_have_exact_positive_equilibrium(
    directions: np.ndarray,
) -> None:
    prior = np.full(len(directions), 1.0 / len(directions))
    assert positive_equilibrium_fraction(directions, prior) == pytest.approx(1.0)
    metrics = site_pcfc_metrics(directions, np.ones(len(directions)))
    assert metrics == pytest.approx(
        {
            "uniform_closure": 1.0,
            "weighted_closure": 1.0,
            "uniform_equilibrium": 1.0,
            "weighted_equilibrium": 1.0,
            "locally_enclosed": 1.0,
        },
        abs=1.0e-12,
    )


def test_one_sided_full_rank_set_separates_closure_from_equilibrium() -> None:
    directions = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    metrics = site_pcfc_metrics(directions, np.ones(4))
    assert 0.0 < metrics["uniform_closure"] < 1.0
    assert metrics["uniform_equilibrium"] == 0.0
    assert metrics["weighted_equilibrium"] == 0.0
    assert metrics["locally_enclosed"] == 0.0


def test_too_few_and_rank_deficient_directions_have_zero_equilibrium() -> None:
    for directions in (
        np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        ),
    ):
        prior = np.full(len(directions), 1.0 / len(directions))
        assert positive_equilibrium_fraction(directions, prior) == 0.0


def test_solver_certificate_respects_frozen_residual_tolerance() -> None:
    directions = np.asarray(
        [
            [-0.8614945220067236, 0.5057566110970450, 0.04513799822809177],
            [0.8614945242112789, -0.5057565964707047, -0.04513812003562824],
            [0.42936353708065395, 0.5699103732924651, 0.7006062513560901],
            [-0.4293635543497018, -0.5699103283309025, -0.7006062773469515],
            [-0.4178473177871051, -0.5875545099622173, 0.6929526079330249],
            [0.41784727913930225, 0.5875545528343642, -0.6929525948861814],
        ]
    )
    prior = np.full(6, 1.0 / 6.0)
    equilibrium = positive_equilibrium_fraction(directions, prior)
    assert 0.9999999 < equilibrium <= 1.0


def test_direct_feature_aggregation_is_bounded_and_exactly_replication_invariant() -> None:
    endpoints = np.asarray(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
        dtype=int,
    )
    vectors = np.asarray(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=float,
    )
    weights = np.asarray([1.0, 2.0, 1.5, 0.5, 0.8, 1.2, 0.9, 1.1])
    original = positive_contact_force_closure_features(
        n_sites=5, endpoints=endpoints, vectors=vectors, weights=weights
    )
    values = np.asarray(list(original.values()), dtype=float)
    assert tuple(original) == FEATURE_NAMES
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()

    replicated = positive_contact_force_closure_features(
        n_sites=10,
        endpoints=np.vstack((endpoints, endpoints + 5)),
        vectors=np.vstack((vectors, vectors)),
        weights=np.concatenate((weights, weights)),
    )
    assert replicated == pytest.approx(original, rel=0.0, abs=0.0)


def test_direct_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="directions differ"):
        site_pcfc_metrics(np.zeros((4, 3)), np.ones(4))
    with pytest.raises(ValueError, match="weights differ"):
        site_pcfc_metrics(_tetrahedron(), [1.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="prior differs"):
        positive_equilibrium_fraction(_tetrahedron(), [0.25, 0.25, 0.25, 0.20])
    with pytest.raises(ValueError, match="endpoints differ"):
        positive_contact_force_closure_features(
            n_sites=4,
            endpoints=[[0, 0]],
            vectors=[[1.0, 0.0, 0.0]],
            weights=[1.0],
        )


def test_standard_ionic_crystals_are_supported_and_bounded() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
    ):
        result = compute_pcfc_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.edge_count > 0
        values = np.asarray([result.features[name] for name in FEATURE_NAMES])
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_geometry_equivalences_preserve_features() -> None:
    atoms = _distorted_nacl()
    reference = _feature_vector(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-8
        )


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    with_array = atoms.copy()
    with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    nonfinite = atoms.copy()
    nonfinite.positions[0, 0] = np.nan
    for changed in (
        with_calculator,
        with_metadata,
        with_array,
        nonperiodic,
        nonfinite,
    ):
        result = compute_pcfc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pcfc_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
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
    with pytest.raises(FileNotFoundError, match="NEXT295 input is missing"):
        build_cross_source_pcfc_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
