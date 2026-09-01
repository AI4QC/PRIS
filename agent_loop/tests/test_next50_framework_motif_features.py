from __future__ import annotations

from ase.build import bulk
import numpy as np

from src.next50_framework_motif_features import (
    COMBINED_FEATURE_NAMES,
    compute_framework_motif_features,
)


def test_combined_framework_motif_features_are_finite_and_geometry_only() -> None:
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    result = compute_framework_motif_features(atoms)

    assert result.supported
    assert tuple(result.features) == COMBINED_FEATURE_NAMES
    assert np.isfinite(list(result.features.values())).all()
    assert result.features["periodic_dimension_max"] == 3.0

    contaminated = atoms.copy()
    contaminated.info["later_state"] = True
    failed = compute_framework_motif_features(contaminated)
    assert not failed.supported
    assert failed.features == {}
