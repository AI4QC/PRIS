from __future__ import annotations

import inspect

import experiments.next510_mcpa_label_blind_novelty_probe as p
import src.next510_madelung_charge_potential_antitonicity as n510


def test_prior_universe_direct_controls_and_gates_are_frozen() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert p.DIRECT_CONTROL_NAMES == (
        "recomputed_mvch",
        "recomputed_tbac",
        "recomputed_nm_total_reduced",
        "recomputed_nm_real_reduced",
        "recomputed_nm_reciprocal_reduced",
        "recomputed_nm_point_reduced",
        "recomputed_nm_site_spread",
        "recomputed_nm_site_max",
        "recomputed_nm_site_min",
        "recomputed_nm_site_positive_fraction",
        "recomputed_nm_charge_concentration",
        "recomputed_charge_potential_pearson",
        "recomputed_charge_potential_spearman",
        "recomputed_opposite_sign_potential_order_fraction",
        "recomputed_motif_same_element_dispersion_rms",
        "recomputed_motif_same_element_dispersion_q95",
        "recomputed_motif_same_element_dispersion_max",
        "recomputed_motif_global_dispersion_rms",
    )
    sources = {
        source: {
            "supported": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "unique_rounded_10": 20,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.89},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_probe_gates(sources, 80).values())


def test_probe_interface_has_no_outcome_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_novelty_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in (
            "endpoint", "label", "outcome", "validation", "replication", "relax"
        )
    )


def test_hashes_cover_certificate_direct_controls_and_target() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert "next510_engineering_certificate" in hashes
    assert "src/next21_normalized_madelung.py" in hashes
    assert "src/next505_madelung_valence_class_homogeneity.py" in hashes
    assert "src/next510_madelung_charge_potential_antitonicity.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next510_mcpa_label_blind_novelty_probe.py"


def test_target_adapter_exposes_prior_engine_contract() -> None:
    adapter = p._target_adapter()
    assert adapter.DESIGN_SHA256 == p.DESIGN_SHA256
    assert adapter.FEATURE_NAMES == n510.FEATURE_NAMES
    assert adapter.compute_tbac_features is n510.compute_mcpa_features
