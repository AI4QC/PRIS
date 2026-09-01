from __future__ import annotations

import inspect

import experiments.next515_eccc_cde_label_blind_probe as p


def test_gates_include_shared_domain_identity_and_conservatism() -> None:
    passing = {
        source: {
            "supported": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "unique_rounded_10": 20,
            "maximum_invariance_error": 0.0,
            "shared_supported": 75,
            "maximum_shared_domain_error": 0.0,
            "conservatism_violations": 0,
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_gates(passing, 80).values())
    passing["wyformer"]["maximum_shared_domain_error"] = 1.0e-9
    assert not p.evaluate_gates(passing, 80)["shared_domain_identity"]


def test_probe_interface_has_no_prior_outcome_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in (
            "feature_dir", "prior", "endpoint", "label", "outcome",
            "validation", "replication", "relax",
        )
    )


def test_hashes_cover_design_base_certificate_core_probe_and_tests() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["appendix3_asset"] == p.ASSET_SHA256
    assert "next471_coverage_certificate" in hashes
    assert "src/next470_element_characteristic_coordination_compatibility.py" in hashes
    assert "src/next515_eccc_conservative_domain_extension.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next515_eccc_cde_label_blind_probe.py"
