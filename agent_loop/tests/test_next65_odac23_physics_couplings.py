from __future__ import annotations

import numpy as np

from src.next65_odac23_physics_couplings import (
    INTERACTION_FEATURE_NAMES,
    compute_physics_couplings,
)


def test_couplings_follow_frozen_algebra() -> None:
    row = {
        "atom_density": 0.1,
        "metal_donor_ratio_std": 0.2,
        "metal_donor_distance_q95": 2.5,
        "metal_ligand_ratio_q95": 1.2,
        "bond_orientation_lambda_min": 0.25,
        "heteroatomic_edge_fraction": 0.6,
        "donor_motif_order_strength_min": 0.5,
        "metal_donor_ratio_max": 1.3,
        "motif_order_strength_min": 0.4,
        "degree2_bend_q95": 0.3,
        "volume_per_atom": 10.0,
        "donor_motif_cn_entropy_q95": 0.2,
        "metal_donor_en_gap_q95": 1.5,
    }

    result = compute_physics_couplings(row)

    assert result.supported
    assert tuple(result.features) == INTERACTION_FEATURE_NAMES
    assert np.isclose(result.features["density_metal_donor_strain"], 0.02)
    assert np.isclose(result.features["hetero_directional_confinement"], 2.4)
    assert np.isclose(result.features["metal_strain_to_donor_order"], 0.4)
    assert np.isfinite(list(result.features.values())).all()


def test_nonfinite_input_fails_open() -> None:
    result = compute_physics_couplings({"atom_density": np.nan})

    assert not result.supported
    assert result.features == {}
