from __future__ import annotations

from ase import Atoms
import numpy as np

from src.next55_odac23_analytic_features import (
    ANALYTIC_FEATURE_NAMES,
    NEXT55_FEATURE_NAMES,
    compute_odac23_analytic_features,
)


def _chain() -> Atoms:
    return Atoms(
        "C",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.40, 8.0, 8.0]),
        pbc=True,
    )


def test_analytic_features_are_finite_geometry_only_and_supercell_invariant() -> None:
    primitive = _chain()
    translated = primitive.copy()
    translated.positions += [0.23, -0.31, 0.17]
    repeated = primitive.repeat((2, 1, 1))

    base = compute_odac23_analytic_features(primitive)
    shifted = compute_odac23_analytic_features(translated)
    supercell = compute_odac23_analytic_features(repeated)

    assert base.supported and shifted.supported and supercell.supported
    assert tuple(base.features) == NEXT55_FEATURE_NAMES
    assert np.isfinite(list(base.features.values())).all()
    for name in NEXT55_FEATURE_NAMES:
        assert np.isclose(base.features[name], shifted.features[name], atol=1e-10)
        assert np.isclose(base.features[name], supercell.features[name], atol=1e-10)


def test_hinge_and_metal_donor_features_have_expected_direction() -> None:
    bent = Atoms(
        symbols=["Cu", "O", "C"],
        positions=[[5.0, 5.0, 5.0], [6.8, 5.0, 5.0], [6.8, 6.3, 5.0]],
        cell=np.diag([12.0, 12.0, 12.0]),
        pbc=True,
    )
    result = compute_odac23_analytic_features(bent)

    assert result.supported
    assert result.features["metal_donor_edge_fraction"] > 0.0
    assert result.features["donor_metal_contact_fraction"] > 0.0
    assert result.features["degree2_bend_mean"] > 0.0
    forbidden = ("energy", "force", "stress", "relax", "dft", "label", "target")
    assert not any(token in name for name in ANALYTIC_FEATURE_NAMES for token in forbidden)


def test_metadata_or_calculator_fails_open() -> None:
    invalid = _chain()
    invalid.info["endpoint"] = 1.0

    result = compute_odac23_analytic_features(invalid)

    assert not result.supported
    assert result.features == {}
