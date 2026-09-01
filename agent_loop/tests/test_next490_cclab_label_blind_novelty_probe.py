from __future__ import annotations

import inspect

import experiments.next490_cclab_label_blind_novelty_probe as p
import src.next490_coordination_conditioned_lewis_acidity_balance as n490


def test_prior_universe_and_direct_controls_are_frozen() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert p.DIRECT_CONTROL_NAMES == (
        "recomputed_clam",
        "recomputed_mvclam",
        "recomputed_eccc",
        "recomputed_cccb",
        "recomputed_sbcc",
        "recomputed_cacc",
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


def test_hashes_cover_engineering_certificate_direct_controls_and_target() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["characteristic_acidity_asset"] == p.ASSET_SHA256
    assert "next490_engineering_certificate" in hashes
    assert "src/next480_site_balanced_characteristic_coordination.py" in hashes
    assert "src/next485_complex_anion_contact_correspondence.py" in hashes
    assert "src/next490_coordination_conditioned_lewis_acidity_balance.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next490_cclab_label_blind_novelty_probe.py"


def test_wrapped_sbcc_engine_adapter_exposes_its_expected_contract() -> None:
    adapter = p._target_adapter()
    assert adapter.DESIGN_SHA256 == p.DESIGN_SHA256
    assert adapter.FEATURE_NAMES == n490.FEATURE_NAMES
    assert adapter.compute_sbcc_features is n490.compute_cclab_features
