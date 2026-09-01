from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from src.next21_normalized_madelung import (
    FEATURE_NAMES,
    MadelungFeatureResult,
    normalized_madelung_features,
)


def _binary_structure(scale: float = 1.0) -> Structure:
    return Structure(
        Lattice.cubic(4.2 * scale),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_schema_has_no_endpoint_or_learned_quantity_names() -> None:
    forbidden = ("energy", "force", "stress", "dft", "relax", "model", "proxy")
    assert FEATURE_NAMES
    assert not any(token in name.lower() for name in FEATURE_NAMES for token in forbidden)


def test_reduced_madelung_is_scale_and_charge_amplitude_invariant() -> None:
    reference = normalized_madelung_features(_binary_structure(), [1.0, -1.0])
    cell_scaled = normalized_madelung_features(
        _binary_structure(1.7), [1.0, -1.0]
    )
    charge_scaled = normalized_madelung_features(
        _binary_structure(), [2.3, -2.3]
    )
    assert isinstance(reference, MadelungFeatureResult)
    assert reference.supported, reference.failure_reason
    assert reference.features["nm_total_reduced"] < 0.0
    for name in FEATURE_NAMES:
        assert cell_scaled.features[name] == pytest.approx(
            reference.features[name], rel=5e-6, abs=1e-8
        )
        assert charge_scaled.features[name] == pytest.approx(
            reference.features[name], rel=5e-6, abs=1e-8
        )


def test_entrypoint_does_not_mutate_structure() -> None:
    structure = _binary_structure()
    species = tuple(str(site.specie) for site in structure)
    coordinates = np.asarray(structure.frac_coords).copy()
    result = normalized_madelung_features(structure, [1.0, -1.0])
    assert result.supported
    assert tuple(result.features) == FEATURE_NAMES
    assert tuple(str(site.specie) for site in structure) == species
    assert np.array_equal(structure.frac_coords, coordinates)


@pytest.mark.parametrize(
    "charges, message",
    [
        ([1.0, 0.0], "neutral"),
        ([1.0], "sites"),
        ([0.0, 0.0], "nonzero"),
        ([np.nan, -1.0], "finite"),
    ],
)
def test_invalid_charge_assignments_fail_open(charges, message: str) -> None:
    result = normalized_madelung_features(_binary_structure(), charges)
    assert not result.supported
    assert result.features == {}
    assert message in str(result.failure_reason).lower()
