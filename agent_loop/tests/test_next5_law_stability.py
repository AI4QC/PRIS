"""Contracts for soft-margin true LOKO stability analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_law_stability import loko_success_gate  # noqa: E402
from next5_law_stability import main as stability_main  # noqa: E402
from next5_law_stability import validate_input_domain_audit  # noqa: E402
from next5_law_stability import validate_reference_contract  # noqa: E402


def test_loko_success_gate_requires_nonnegative_macro_and_bounded_each_kind():
    passed = loko_success_gate([0.01, -0.01, 0.0], [0.02, -0.01, 0.0])
    assert passed["passes"] is True
    failed_macro = loko_success_gate([0.01, -0.02], [0.01, -0.02])
    assert failed_macro["passes"] is False
    failed_kind = loko_success_gate([0.03, -0.021], [0.03, -0.021])
    assert failed_kind["passes"] is False


def test_reference_contract_rejects_wrong_experiment_or_input_hash():
    reference = {
        "protocol": {"experiment": "np-next-20260801e", "floor": 0.98},
        "isolation_audit": {"manifest_sha256": "manifest"},
        "provenance": {"input_sha256": {"real": "abc"}},
    }
    validate_reference_contract(
        reference,
        floor=0.98,
        current_input_sha256={"real": "abc"},
        manifest_sha256="manifest",
    )
    wrong = {**reference, "protocol": {**reference["protocol"], "experiment": "x"}}
    with np.testing.assert_raises_regex(ValueError, "experiment"):
        validate_reference_contract(
            wrong,
            floor=0.98,
            current_input_sha256={"real": "abc"},
            manifest_sha256="manifest",
        )
    with np.testing.assert_raises_regex(ValueError, "input hashes"):
        validate_reference_contract(
            reference,
            floor=0.98,
            current_input_sha256={"real": "changed"},
            manifest_sha256="manifest",
        )


def test_input_domain_audit_must_match_every_descriptor_hash():
    audit = {
        "all_descriptor_keys_within_isolated_domains": True,
        "descriptors": {
            "real_descriptors": {"descriptor_sha256": "a"},
            "bad_descriptors": {"descriptor_sha256": "b"},
        },
    }
    validate_input_domain_audit(
        audit, {"real_descriptors": "a", "bad_descriptors": "b"}
    )
    with np.testing.assert_raises_regex(ValueError, "hash"):
        validate_input_domain_audit(
            audit, {"real_descriptors": "a", "bad_descriptors": "changed"}
        )


def test_stability_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with np.testing.assert_raises_regex(SystemExit, "refusing to overwrite"):
        stability_main(
            [
                "--isolated-dir",
                str(tmp_path / "missing"),
                "--real-descriptors",
                str(tmp_path / "missing-real.parquet"),
                "--bad-descriptors",
                str(tmp_path / "missing-bad.parquet"),
                "--real-sixfam",
                str(tmp_path / "missing-six-real.parquet"),
                "--bad-sixfam",
                str(tmp_path / "missing-six-bad.parquet"),
                "--real-corrected",
                str(tmp_path / "missing-corrected-real.parquet"),
                "--bad-corrected",
                str(tmp_path / "missing-corrected-bad.parquet"),
                "--real-guards",
                str(tmp_path / "missing-guard-real.parquet"),
                "--bad-guards",
                str(tmp_path / "missing-guard-bad.parquet"),
                "--input-domain-audit",
                str(tmp_path / "missing-audit.json"),
                "--reference-report",
                str(tmp_path / "missing-reference.json"),
                "--falsification-report",
                str(tmp_path / "missing-falsification.json"),
                "--out",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "keep"
