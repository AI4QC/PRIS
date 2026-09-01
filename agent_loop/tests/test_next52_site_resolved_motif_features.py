from __future__ import annotations

from ase.build import bulk
import numpy as np

from src.next52_site_resolved_motif_features import (
    SITE_MOTIF_FEATURE_NAMES,
    compute_site_resolved_motif_features,
)


def test_site_resolved_motif_features_are_finite_and_select_metal_sites() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    result = compute_site_resolved_motif_features(atoms)

    assert result.supported
    assert tuple(result.features) == SITE_MOTIF_FEATURE_NAMES
    assert np.isfinite(list(result.features.values())).all()
    assert np.isclose(result.features["metal_motif_site_fraction"], 0.5)
    assert np.isclose(result.features["donor_motif_site_fraction"], 0.5)

    contaminated = atoms.copy()
    contaminated.info["computed"] = True
    failed = compute_site_resolved_motif_features(contaminated)
    assert not failed.supported
    assert failed.features == {}
