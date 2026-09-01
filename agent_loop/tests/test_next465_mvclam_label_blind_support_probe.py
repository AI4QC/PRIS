from __future__ import annotations

import inspect

import experiments.next465_mvclam_label_blind_support_probe as p


def test_support_gate_is_frozen_and_ordered_first() -> None:
    assert p.MINIMUM_SUPPORTED == 72
    assert p.evaluate_support_gate({"scigen": 72, "wyformer": 72})
    assert not p.evaluate_support_gate({"scigen": 71, "wyformer": 80})


def test_probe_interface_has_no_prior_features_outcomes_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_support_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in (
            "feature_dir",
            "prior",
            "endpoint",
            "label",
            "outcome",
            "validation",
            "replication",
            "relax",
        )
    )


def test_hashes_cover_asset_core_and_support_probe() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["characteristic_acidity_asset"] == p.ASSET_SHA256
    assert tuple(hashes)[-2:] == (
        "tests/test_next465_mixed_valence_characteristic_lewis_matching.py",
        "tests/test_next465_mvclam_label_blind_support_probe.py",
    )
