from __future__ import annotations

from ase.build import bulk
import numpy as np

from src.next63_odac23_metal_chemistry_features import (
    METAL_CHEMISTRY_FEATURE_NAMES,
    compute_metal_chemistry_features,
)


def test_metal_chemistry_features_are_finite_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    repeated = atoms.repeat((2, 1, 1))

    base = compute_metal_chemistry_features(atoms)
    supercell = compute_metal_chemistry_features(repeated)

    assert base.supported and supercell.supported
    assert tuple(base.features) == METAL_CHEMISTRY_FEATURE_NAMES
    assert np.isfinite(list(base.features.values())).all()
    assert base.features["halogen_fraction"] == 0.5
    assert base.features["metal_species_count"] == 1.0
    assert base.features["metal_common_oxidation_mean"] == 1.0
    assert base.features["metal_donor_en_gap_mean"] > 2.0
    for name in METAL_CHEMISTRY_FEATURE_NAMES:
        assert np.isclose(base.features[name], supercell.features[name], atol=1e-10)


def test_non_geometry_metadata_fails_open() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["calculated"] = True

    result = compute_metal_chemistry_features(atoms)

    assert not result.supported
    assert result.features == {}
