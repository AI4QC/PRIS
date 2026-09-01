from __future__ import annotations

import inspect

import experiments.next480_sbcc_label_blind_support_probe as p


def test_support_gate_and_boundary_are_frozen() -> None:
    assert p.MINIMUM_SUPPORTED == 72
    assert p.evaluate_support_gate({"scigen": 72, "wyformer": 80})
    assert not p.evaluate_support_gate({"scigen": 80, "wyformer": 71})
    names = tuple(inspect.signature(p.run_label_blind_support_probe).parameters)
    assert not any(token in name for name in names for token in ("feature", "prior", "label", "outcome", "validation", "replication", "relax"))


def test_hashes_cover_design_asset_core_and_probe() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["characteristic_cn_asset"] == p.ASSET_SHA256
    assert tuple(hashes)[-1] == "tests/test_next480_sbcc_label_blind_support_probe.py"
