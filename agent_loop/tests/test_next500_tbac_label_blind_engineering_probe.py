from __future__ import annotations

import inspect

import experiments.next500_tbac_label_blind_engineering_probe as p


def test_engineering_gates_are_frozen() -> None:
    passing = {
        source: {
            "supported": 72,
            "minimum": 0.0,
            "maximum": 1.0,
            "unique_rounded_10": 20,
            "maximum_invariance_error": 1.0e-8,
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_engineering_gates(passing, 80).values())
    passing["wyformer"]["supported"] = 71
    assert not p.evaluate_engineering_gates(passing, 80)["support"]
    passing["wyformer"]["supported"] = 72
    passing["scigen"]["unique_rounded_10"] = 19
    assert not p.evaluate_engineering_gates(passing, 80)["nondegenerate"]


def test_probe_interface_has_no_prior_outcome_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_engineering_probe).parameters)
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


def test_hashes_cover_design_core_probe_and_tests() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert "src/next500_topological_bond_angular_correspondence.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next500_tbac_label_blind_engineering_probe.py"
