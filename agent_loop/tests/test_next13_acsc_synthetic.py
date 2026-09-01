"""End-to-end tests for the dataset-free ACSC-v0 falsification artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


EXPECTED_CASE_ORDER = [
    "positive_uncoupled",
    "subcritical_coupling",
    "coupling_only_saddle",
    "rotated_generalized_saddle",
    "two_scale_inconsistent",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonstandard_json(token: str) -> None:
    raise AssertionError(f"nonstandard JSON token: {token}")


def test_runner_publishes_strict_label_free_manifest(tmp_path: Path) -> None:
    from src import next13_acsc, next13_acsc_synthetic

    output_dir = tmp_path / "acsc-v0-synthetic"
    assert next13_acsc_synthetic.run(output_dir) == output_dir

    assert sorted(path.name for path in output_dir.iterdir()) == ["MANIFEST.json"]
    manifest = json.loads(
        (output_dir / "MANIFEST.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json,
    )
    assert manifest["protocol"] == "2026-08-02-next13-acsc-synthetic-v1"
    assert manifest["version"] == "ACSC-v0"
    assert manifest["evidence_scope"] == {
        "dataset_free": True,
        "label_free": True,
        "engineering_only": True,
        "mattersim_executed": False,
    }
    assert manifest["inputs"] == {
        "datasets": [],
        "labels": [],
        "model_checkpoints": [],
    }
    assert manifest["case_order"] == EXPECTED_CASE_ORDER
    assert all(case["passed"] is True for case in manifest["cases"])
    assert manifest["engineering_pass"] is True
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["source_sha256"] == {
        "src/next13_acsc.py": _sha256(Path(next13_acsc.__file__).resolve()),
        "src/next13_acsc_synthetic.py": _sha256(
            Path(next13_acsc_synthetic.__file__).resolve()
        ),
        "src/next11_phsc.py": _sha256(
            Path(next13_acsc_synthetic.__file__).with_name("next11_phsc.py")
        ),
    }


def test_cases_freeze_incremental_coupling_contract() -> None:
    from src.next13_acsc_synthetic import _build_manifest

    cases = {case["name"]: case for case in _build_manifest()["cases"]}
    positive = cases["positive_uncoupled"]["observed"]
    subcritical = cases["subcritical_coupling"]["observed"]
    saddle = cases["coupling_only_saddle"]["observed"]
    rotated = cases["rotated_generalized_saddle"]["observed"]
    inconsistent = cases["two_scale_inconsistent"]["observed"]

    assert positive["acsc_status"] == "resolved_nonnegative"
    assert subcritical["acsc_status"] == "resolved_nonnegative"
    assert saddle["atomic_lambda_min"] > 0.0
    assert saddle["strain_lambda_min"] > 0.0
    assert saddle["acsc_status"] == "resolved_negative"
    assert saddle["lambda_r_ev_per_atom"] < 0.0
    assert rotated["reference_lambda_r"] == pytest.approx(
        rotated["rotated_lambda_r"], abs=3e-14
    )
    assert inconsistent["acsc_status"] == "near_zero_or_inconsistent"
    assert inconsistent["lambda_h_ev_per_atom"] < 0.0
    assert inconsistent["lambda_h2_ev_per_atom"] > 0.0


def test_second_publication_preserves_existing_output(tmp_path: Path) -> None:
    from src.next13_acsc_synthetic import run

    output_dir = tmp_path / "published"
    run(output_dir)
    original = (output_dir / "MANIFEST.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(output_dir)
    assert (output_dir / "MANIFEST.json").read_bytes() == original


def test_cli_has_no_dataset_label_or_checkpoint_input(tmp_path: Path) -> None:
    from src.next13_acsc_synthetic import main

    output_dir = tmp_path / "cli"
    assert main(["--output-dir", str(output_dir)]) == 0
    for forbidden in ("--dataset", "--labels", "--checkpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main(["--output-dir", str(tmp_path / forbidden[2:]), forbidden, "x"])
        assert exc_info.value.code == 2
