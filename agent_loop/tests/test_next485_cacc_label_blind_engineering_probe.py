from __future__ import annotations

import inspect

import experiments.next485_cacc_label_blind_engineering_probe as p


def test_engineering_gates_are_frozen() -> None:
    passing = {
        source: {
            "supported": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "unique_rounded_10": 20,
            "maximum_invariance_error": 0.0,
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_engineering_gates(passing, 80).values())
    passing["scigen"]["unique_rounded_10"] = 19
    assert not p.evaluate_engineering_gates(passing, 80)["nondegenerate"]


def test_probe_interface_has_no_prior_outcome_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_engineering_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in (
            "feature_dir", "prior", "endpoint", "label", "outcome",
            "validation", "replication", "relax",
        )
    )


def test_hashes_cover_both_assets_core_and_probe() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["appendix4_asset"] == p.ASSET_SHA256
    assert tuple(hashes)[-1] == "tests/test_next485_cacc_label_blind_engineering_probe.py"
