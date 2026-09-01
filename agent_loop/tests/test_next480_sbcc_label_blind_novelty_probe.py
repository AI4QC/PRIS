from __future__ import annotations

import inspect

import experiments.next480_sbcc_label_blind_novelty_probe as p


def test_prior_universe_and_direct_controls_are_frozen() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert p.DIRECT_CONTROL_NAMES == (
        "recomputed_clam",
        "recomputed_mvclam",
        "recomputed_eccc",
        "recomputed_cccb",
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
        for token in ("endpoint", "label", "outcome", "validation", "replication", "relax")
    )


def test_hashes_cover_support_certificate_cccb_and_target() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["characteristic_cn_asset"] == p.ASSET_SHA256
    assert "next480_support_certificate" in hashes
    assert "src/next475_characteristic_coordination_bottleneck.py" in hashes
    assert "src/next480_site_balanced_characteristic_coordination.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next480_sbcc_label_blind_novelty_probe.py"
