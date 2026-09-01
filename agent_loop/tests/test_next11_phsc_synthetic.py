"""End-to-end tests for the dataset-free PHSC-v0 synthetic package."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path

import pytest


EXPECTED_CASE_ORDER = [
    "positive_quadratic",
    "negative_quadratic",
    "stationary_saddle_lrrc_blind",
    "force_orthogonal_saddle",
    "translation_projection",
    "proper_rotation_covariance",
    "semidefinite_zero_ambiguous",
    "two_scale_inconsistent",
    "antisymmetric_diagnostic",
    "mass_invariance",
]


def _reject_nonstandard_json(token: str) -> None:
    raise AssertionError(f"nonstandard JSON token: {token}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _cases_by_name(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    return {case["name"]: case for case in manifest["cases"]}


def test_runner_publishes_strict_dataset_and_label_free_manifest(tmp_path: Path) -> None:
    from src import next11_phsc, next11_phsc_synthetic, next9_lrrc

    output_dir = tmp_path / "phsc-v0-synthetic"
    published = next11_phsc_synthetic.run(output_dir)

    assert published == output_dir
    assert sorted(path.name for path in output_dir.iterdir()) == ["MANIFEST.json"]
    raw = (output_dir / "MANIFEST.json").read_text(encoding="utf-8")
    manifest = json.loads(raw, parse_constant=_reject_nonstandard_json)

    assert set(manifest) == {
        "protocol",
        "version",
        "evidence_scope",
        "inputs",
        "constants",
        "formulas",
        "source_sha256",
        "case_order",
        "cases",
        "known_limitations",
        "engineering_pass",
        "scientific_improvement_claim",
    }
    assert manifest["protocol"] == "2026-08-02-next11-phsc-synthetic-v1"
    assert manifest["version"] == "PHSC-v0"
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
        "step_fraction": 2**-8,
        "tau_multiplier": 64.0,
    }
    assert manifest["case_order"] == EXPECTED_CASE_ORDER
    assert [case["name"] for case in manifest["cases"]] == EXPECTED_CASE_ORDER
    assert all(
        set(case) == {"name", "expected", "observed", "passed"}
        for case in manifest["cases"]
    )
    assert all(case["passed"] is True for case in manifest["cases"])
    assert manifest["engineering_pass"] is True
    assert manifest["scientific_improvement_claim"] is False
    assert "not a confidence bound" in manifest["formulas"]["e_num"].lower()
    assert any(
        "gamma" in limitation.lower() and "fixed-cell" in limitation.lower()
        for limitation in manifest["known_limitations"]
    )
    assert any(
        "dataset" in limitation.lower() and "scientific" in limitation.lower()
        for limitation in manifest["known_limitations"]
    )

    assert manifest["source_sha256"] == {
        "src/next11_phsc.py": _sha256(Path(next11_phsc.__file__).resolve()),
        "src/next11_phsc_synthetic.py": _sha256(
            Path(next11_phsc_synthetic.__file__).resolve()
        ),
        "src/next9_lrrc.py": _sha256(Path(next9_lrrc.__file__).resolve()),
    }
    assert not list(tmp_path.glob(".phsc-v0-synthetic.tmp-*"))


def test_analytic_cases_freeze_phsc_and_lrrc_blind_spot_contracts() -> None:
    from src.next11_phsc_synthetic import _build_manifest

    cases = _cases_by_name(_build_manifest())

    positive = cases["positive_quadratic"]
    assert positive["expected"] == {
        "phsc_status": "resolved_nonnegative",
        "phsc_negative": False,
        "lambda_min": 1.0,
    }
    assert positive["observed"]["phsc_status"] == "resolved_nonnegative"
    assert positive["observed"]["phsc_negative"] is False
    assert positive["observed"]["lambda_h"] == pytest.approx(1.0, abs=5e-11)
    assert positive["observed"]["lambda_h2"] == pytest.approx(1.0, abs=5e-11)
    assert positive["observed"]["lambda_r"] == pytest.approx(1.0, abs=5e-11)

    negative = cases["negative_quadratic"]
    assert negative["expected"] == {
        "phsc_status": "resolved_negative",
        "phsc_negative": True,
        "lambda_min": -3.0,
    }
    assert negative["observed"]["phsc_status"] == "resolved_negative"
    assert negative["observed"]["lambda_r"] == pytest.approx(-3.0, abs=5e-11)

    stationary = cases["stationary_saddle_lrrc_blind"]
    assert stationary["expected"] == {
        "lrrc_status": "stationary_fallback",
        "lrrc_negative": None,
        "phsc_status": "resolved_negative",
        "phsc_negative": True,
        "lambda_min": -3.0,
    }
    assert stationary["observed"]["lrrc_status"] == "stationary_fallback"
    assert stationary["observed"]["lrrc_negative"] is None
    assert stationary["observed"]["phsc_status"] == "resolved_negative"
    assert stationary["observed"]["lambda_r"] == pytest.approx(-3.0, abs=5e-11)

    orthogonal = cases["force_orthogonal_saddle"]
    assert orthogonal["expected"] == {
        "force_mode_curvature": 3.5,
        "force_negative_mode_dot": 0.0,
        "lrrc_status": "ok",
        "lrrc_negative": False,
        "phsc_status": "resolved_negative",
        "phsc_negative": True,
        "lambda_min": -2.5,
    }
    assert orthogonal["observed"]["lrrc_status"] == "ok"
    assert orthogonal["observed"]["force_negative_mode_dot"] == pytest.approx(
        0.0, abs=2e-15
    )
    assert orthogonal["observed"]["lrrc_negative"] is False
    assert orthogonal["observed"]["lrrc_kappa_r"] == pytest.approx(3.5, abs=5e-11)
    assert orthogonal["observed"]["phsc_status"] == "resolved_negative"
    assert orthogonal["observed"]["lambda_r"] == pytest.approx(-2.5, abs=5e-11)


def test_ambiguous_and_diagnostic_cases_never_become_negative_evidence() -> None:
    from src.next11_phsc_synthetic import _build_manifest

    cases = _cases_by_name(_build_manifest())

    zero = cases["semidefinite_zero_ambiguous"]
    assert zero["observed"]["phsc_status"] == "near_zero_or_inconsistent"
    assert zero["observed"]["phsc_negative"] is False
    assert abs(zero["observed"]["lambda_r"]) <= zero["observed"]["tau_alg"]

    inconsistent = cases["two_scale_inconsistent"]
    assert inconsistent["observed"]["phsc_status"] == "near_zero_or_inconsistent"
    assert inconsistent["observed"]["phsc_negative"] is False
    assert inconsistent["observed"]["lambda_h"] < 0.0
    assert inconsistent["observed"]["lambda_h2"] > 0.0
    assert inconsistent["observed"]["e_num"] > 0.0

    antisymmetric = cases["antisymmetric_diagnostic"]
    assert antisymmetric["observed"]["phsc_status"] == "resolved_nonnegative"
    assert antisymmetric["observed"]["phsc_negative"] is False
    assert antisymmetric["observed"]["antisymmetric_norm_h"] == pytest.approx(7.0)
    assert antisymmetric["observed"]["antisymmetric_norm_h2"] == pytest.approx(7.0)


def test_translation_projection_and_mass_change_do_not_change_primary_decision() -> None:
    from src.next11_phsc_synthetic import _build_manifest

    cases = _cases_by_name(_build_manifest())

    translation = cases["translation_projection"]["observed"]
    assert translation["reference_status"] == "resolved_nonnegative"
    assert translation["contaminated_status"] == translation["reference_status"]
    assert translation["contaminated_lambda_r"] == pytest.approx(
        translation["reference_lambda_r"], abs=5e-12
    )
    assert translation["contaminated_acoustic_residual"] > (
        translation["reference_acoustic_residual"] + 1.0
    )

    rotation = cases["proper_rotation_covariance"]["observed"]
    assert rotation["rotation_determinant"] == pytest.approx(1.0, abs=1e-15)
    assert rotation["reference_status"] == "resolved_negative"
    assert rotation["rotated_status"] == rotation["reference_status"]
    assert rotation["rotated_negative"] is rotation["reference_negative"] is True
    assert rotation["rotated_lambda_r"] == pytest.approx(
        rotation["reference_lambda_r"], abs=5e-12
    )

    mass = cases["mass_invariance"]["observed"]
    assert mass["reference_masses"] != mass["changed_masses"]
    assert mass["reference_status"] == "resolved_negative"
    assert mass["changed_status"] == mass["reference_status"]
    assert mass["changed_negative"] is mass["reference_negative"] is True
    assert mass["changed_lambda_r"] == pytest.approx(
        mass["reference_lambda_r"], abs=5e-12
    )


def test_engineering_pass_is_conjunction_of_every_ordered_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next11_phsc_synthetic

    original_build_cases = next11_phsc_synthetic._build_cases

    def cases_with_one_failed_contract() -> list[dict[str, object]]:
        cases = original_build_cases()
        cases[-1] = {**cases[-1], "passed": False}
        return cases

    monkeypatch.setattr(next11_phsc_synthetic, "_build_cases", cases_with_one_failed_contract)
    manifest = next11_phsc_synthetic._build_manifest()

    assert [case["name"] for case in manifest["cases"]] == EXPECTED_CASE_ORDER
    assert manifest["cases"][-1]["passed"] is False
    assert manifest["engineering_pass"] is False


def test_source_change_before_publish_fails_closed_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_synthetic

    fake_core = tmp_path / "next11_phsc.py"
    fake_runner = tmp_path / "next11_phsc_synthetic.py"
    fake_lrrc = tmp_path / "next9_lrrc.py"
    for path, payload in (
        (fake_core, "core-v1\n"),
        (fake_runner, "runner-v1\n"),
        (fake_lrrc, "lrrc-v1\n"),
    ):
        path.write_text(payload, encoding="utf-8")
    fake_sources = {
        "src/next11_phsc.py": fake_core,
        "src/next11_phsc_synthetic.py": fake_runner,
        "src/next9_lrrc.py": fake_lrrc,
    }
    monkeypatch.setattr(next11_phsc_synthetic, "_source_paths", lambda: fake_sources)
    original_build_manifest = next11_phsc_synthetic._build_manifest

    def build_then_change_source(source_paths: dict[str, Path]) -> dict[str, object]:
        manifest = original_build_manifest(source_paths)
        fake_runner.write_text("runner-v2\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(
        next11_phsc_synthetic, "_build_manifest", build_then_change_source
    )
    output_dir = tmp_path / "source-raced-output"

    with pytest.raises(RuntimeError, match="source.*changed.*publication"):
        next11_phsc_synthetic.run(output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".source-raced-output.tmp-*"))


def test_second_publication_never_changes_existing_output(tmp_path: Path) -> None:
    from src.next11_phsc_synthetic import run

    output_dir = tmp_path / "published"
    run(output_dir)
    original = (output_dir / "MANIFEST.json").read_bytes()

    with pytest.raises(FileExistsError):
        run(output_dir)

    assert (output_dir / "MANIFEST.json").read_bytes() == original
    assert not list(tmp_path.glob(".published.tmp-*"))


def test_missing_no_replace_primitive_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_synthetic

    monkeypatch.setattr(next11_phsc_synthetic.ctypes, "CDLL", lambda *_a, **_k: object())
    output_dir = tmp_path / "unsupported-output"

    with pytest.raises(NotImplementedError, match="atomic no-replace"):
        next11_phsc_synthetic.run(output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".unsupported-output.tmp-*"))


def test_renameat2_eexist_preserves_competing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_synthetic

    output_dir = tmp_path / "raced-output"

    class RacingRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *_args) -> int:
            output_dir.mkdir()
            (output_dir / "competitor.txt").write_text("untouched", encoding="utf-8")
            next11_phsc_synthetic.ctypes.set_errno(errno.EEXIST)
            return -1

    class FakeLibC:
        renameat2 = RacingRenameAt2()

    monkeypatch.setattr(
        next11_phsc_synthetic.ctypes, "CDLL", lambda *_a, **_k: FakeLibC()
    )

    with pytest.raises(FileExistsError):
        next11_phsc_synthetic.run(output_dir)

    assert (output_dir / "competitor.txt").read_text(encoding="utf-8") == "untouched"
    assert not (output_dir / "MANIFEST.json").exists()
    assert not list(tmp_path.glob(".raced-output.tmp-*"))


def test_cli_exposes_no_dataset_label_or_model_input(tmp_path: Path) -> None:
    from src.next11_phsc_synthetic import main

    output_dir = tmp_path / "cli-output"
    assert main(["--output-dir", str(output_dir)]) == 0
    assert (output_dir / "MANIFEST.json").is_file()

    for forbidden in ("--dataset", "--labels", "--checkpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main(["--output-dir", str(tmp_path / forbidden[2:]), forbidden, "x"])
        assert exc_info.value.code == 2
