from __future__ import annotations

import inspect

import experiments.next520_mcpe_label_blind_engineering_probe as p


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


def test_hashes_cover_design_atomic_table_core_probe_and_tests() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["atomic_table"] == p.ATOMIC_TABLE_SHA256
    assert "src/next520_madelung_chemical_potential_equalization.py" in hashes
    assert hashes["first_failed_engineering_probe_result"] == p.FIRST_PROBE_RESULT_SHA256
    assert hashes["supercell_correction_certificate"] == p.CORRECTION_SHA256
    assert tuple(hashes)[-1] == "tests/test_next520_mcpe_label_blind_engineering_probe.py"


def test_corrected_probe_preserves_first_failure_and_boundary() -> None:
    assert p.FIRST_PROBE_RESULT_SHA256 == (
        "eee6b4e2c4a39bf8c5908f01f55d62f54adfa7c63209d9492b846926a8ec1970"
    )
    assert p.CORRECTION_SHA256
    source = inspect.getsource(p.run_label_blind_engineering_probe)
    assert "supersedes_probe_result_sha256" in source
    assert "supercell_charge_normalization_corrected" in source
