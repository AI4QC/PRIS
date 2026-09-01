from __future__ import annotations

from ase.build import bulk
import numpy as np

from src.next46_motif_coherence_features import (
    FEATURE_NAMES as GLOBAL_NAMES,
    compute_motif_coherence_features,
)
from src.next52_site_resolved_motif_features import (
    SITE_MOTIF_FEATURE_NAMES as SITE_NAMES,
    compute_site_resolved_motif_features,
)
from src.next58_odac23_shared_motif_features import (
    EXTRA_MOTIF_FEATURE_NAMES,
    SHARED_MOTIF_FEATURE_NAMES,
    compute_shared_motif_features,
)


def test_shared_matrix_reproduces_existing_global_and_site_features() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    shared = compute_shared_motif_features(atoms)
    global_result = compute_motif_coherence_features(atoms)
    site_result = compute_site_resolved_motif_features(atoms)

    assert shared.supported and global_result.supported and site_result.supported
    assert tuple(shared.features) == SHARED_MOTIF_FEATURE_NAMES
    for name in GLOBAL_NAMES:
        assert np.isclose(shared.features[name], global_result.features[name], atol=1e-12)
    for name in SITE_NAMES:
        assert np.isclose(shared.features[name], site_result.features[name], atol=1e-12)
    assert np.isfinite([shared.features[name] for name in EXTRA_MOTIF_FEATURE_NAMES]).all()


def test_shared_motif_boundary_fails_open() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["later_result"] = 1

    result = compute_shared_motif_features(atoms)

    assert not result.supported
    assert result.features == {}
