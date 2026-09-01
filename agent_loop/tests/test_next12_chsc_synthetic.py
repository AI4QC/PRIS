"""End-to-end tests for the dataset-free CHSC-v0 synthetic package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


EXPECTED_CASE_ORDER = [
    "positive_quadratic",
    "negative_quadratic",
    "rotated_saddle",
    "semidefinite_zero_ambiguous",
    "quartic_two_scale_inconsistent",
]


def _reject_nonstandard_json(token: str) -> None:
    raise AssertionError(f"nonstandard JSON token: {token}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _cases(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    return {case["name"]: case for case in manifest["cases"]}


def test_runner_publishes_strict_label_free_manifest(tmp_path: Path) -> None:
    from src import next12_chsc, next12_chsc_synthetic

    output_dir = tmp_path / "chsc-v0-synthetic"
    assert next12_chsc_synthetic.run(output_dir) == output_dir

    assert sorted(path.name for path in output_dir.iterdir()) == ["MANIFEST.json"]
    raw = (output_dir / "MANIFEST.json").read_text(encoding="utf-8")
    manifest = json.loads(raw, parse_constant=_reject_nonstandard_json)
    assert manifest["protocol"] == "2026-08-02-next12-chsc-synthetic-v1"
    assert manifest["version"] == "CHSC-v0"
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
    assert manifest["constants"] == {
        "strain_dimension": 6,
        "direction_count": 21,
        "step_strain": 2**-7,
        "tau_multiplier": 64.0,
        "energy_calls_per_structure": 85,
    }
    assert manifest["case_order"] == EXPECTED_CASE_ORDER
    assert [case["name"] for case in manifest["cases"]] == EXPECTED_CASE_ORDER
    assert all(case["passed"] is True for case in manifest["cases"])
    assert manifest["engineering_pass"] is True
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["source_sha256"] == {
        "src/next12_chsc.py": _sha256(Path(next12_chsc.__file__).resolve()),
        "src/next12_chsc_synthetic.py": _sha256(
            Path(next12_chsc_synthetic.__file__).resolve()
        ),
        "src/next11_phsc.py": _sha256(
            Path(next12_chsc_synthetic.__file__).with_name("next11_phsc.py")
        ),
    }
    assert not list(tmp_path.glob(".chsc-v0-synthetic.tmp-*"))


def test_analytic_cases_freeze_sign_rotation_and_inconsistency_contracts() -> None:
    from src.next12_chsc_synthetic import _build_manifest

    cases = _cases(_build_manifest())
    positive = cases["positive_quadratic"]["observed"]
    negative = cases["negative_quadratic"]["observed"]
    rotated = cases["rotated_saddle"]["observed"]
    zero = cases["semidefinite_zero_ambiguous"]["observed"]
    inconsistent = cases["quartic_two_scale_inconsistent"]["observed"]

    assert positive["chsc_status"] == "resolved_nonnegative"
    assert positive["lambda_r"] == pytest.approx(1.0, abs=2e-9)
    assert negative["chsc_status"] == "resolved_negative"
    assert negative["lambda_r"] == pytest.approx(-3.0, abs=2e-9)
    assert rotated["reference_status"] == "resolved_negative"
    assert rotated["rotated_status"] == "resolved_negative"
    assert rotated["reference_lambda_r"] == pytest.approx(
        rotated["rotated_lambda_r"], abs=2e-12
    )
    assert zero["chsc_status"] == "near_zero_or_inconsistent"
    assert abs(zero["lambda_r"]) <= zero["tau_alg"]
    assert inconsistent["chsc_status"] == "near_zero_or_inconsistent"
    assert inconsistent["lambda_h"] < 0.0 < inconsistent["lambda_h2"]
    assert inconsistent["e_num"] > 0.0


def test_engineering_pass_is_conjunction_of_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import next12_chsc_synthetic

    original = next12_chsc_synthetic._build_cases

    def one_failed() -> list[dict[str, object]]:
        cases = original()
        cases[-1] = {**cases[-1], "passed": False}
        return cases

    monkeypatch.setattr(next12_chsc_synthetic, "_build_cases", one_failed)
    manifest = next12_chsc_synthetic._build_manifest()
    assert manifest["engineering_pass"] is False


def test_source_change_before_publish_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import next12_chsc_synthetic

    fake_core = tmp_path / "next12_chsc.py"
    fake_runner = tmp_path / "next12_chsc_synthetic.py"
    fake_phsc = tmp_path / "next11_phsc.py"
    for path, value in ((fake_core, "a\n"), (fake_runner, "b\n"), (fake_phsc, "c\n")):
        path.write_text(value, encoding="utf-8")
    sources = {
        "src/next12_chsc.py": fake_core,
        "src/next12_chsc_synthetic.py": fake_runner,
        "src/next11_phsc.py": fake_phsc,
    }
    monkeypatch.setattr(next12_chsc_synthetic, "_source_paths", lambda: sources)
    original = next12_chsc_synthetic._build_manifest

    def build_then_change(source_paths: dict[str, Path]) -> dict[str, object]:
        manifest = original(source_paths)
        fake_runner.write_text("changed\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(next12_chsc_synthetic, "_build_manifest", build_then_change)
    output_dir = tmp_path / "raced"
    with pytest.raises(RuntimeError, match="source.*changed.*publication"):
        next12_chsc_synthetic.run(output_dir)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".raced.tmp-*"))


def test_second_publication_preserves_existing_output(tmp_path: Path) -> None:
    from src.next12_chsc_synthetic import run

    output_dir = tmp_path / "published"
    run(output_dir)
    original = (output_dir / "MANIFEST.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(output_dir)
    assert (output_dir / "MANIFEST.json").read_bytes() == original


def test_cli_has_no_dataset_label_or_checkpoint_input(tmp_path: Path) -> None:
    from src.next12_chsc_synthetic import main

    output_dir = tmp_path / "cli"
    assert main(["--output-dir", str(output_dir)]) == 0
    for forbidden in ("--dataset", "--labels", "--checkpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main(["--output-dir", str(tmp_path / forbidden[2:]), forbidden, "x"])
        assert exc_info.value.code == 2
