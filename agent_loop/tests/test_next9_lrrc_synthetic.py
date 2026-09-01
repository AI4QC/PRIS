"""End-to-end tests for the dataset-free LRRC-v0 synthetic runner."""

from __future__ import annotations

import errno
import hashlib
import json
import math
from pathlib import Path

import pytest


EXPECTED_CASE_ORDER = [
    "positive_harmonic",
    "inverted_harmonic",
    "translation_invariance",
    "rotation_invariance",
    "permutation_invariance",
    "pbc_wrapping_invariance",
    "exact_zero_force_saddle",
    "oracle_exception",
    "wrong_shape_force",
    "nonfinite_force",
    "decision_keep_or_negative",
    "decision_reject_or_nonnegative",
    "quota_fixed_ceil_sqrt_n",
    "quota_boundary_ties",
    "quota_abstain_unchanged",
    "quota_rejection_subset",
]

SCALAR_DIAGNOSTIC_KEYS = (
    "d_star",
    "h",
    "kappa_h",
    "kappa_h2",
    "kappa_r",
    "error_proxy",
    "u_num",
)


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


def test_runner_publishes_strict_self_auditing_manifest(tmp_path: Path) -> None:
    from src import next9_lrrc, next9_lrrc_synthetic

    output_dir = tmp_path / "lrrc-v0-synthetic"
    published = next9_lrrc_synthetic.run(output_dir)

    assert published == output_dir
    assert output_dir.is_dir()
    assert sorted(path.name for path in output_dir.iterdir()) == ["MANIFEST.json"]
    raw_manifest = (output_dir / "MANIFEST.json").read_text(encoding="utf-8")
    manifest = json.loads(raw_manifest, parse_constant=_reject_nonstandard_json)

    assert manifest["version"] == "LRRC-v0"
    assert manifest["constants"] == {
        "force_rms_floor": 1e-12,
        "step_fraction": 2**-8,
    }
    assert manifest["case_order"] == EXPECTED_CASE_ORDER
    assert [case["name"] for case in manifest["cases"]] == EXPECTED_CASE_ORDER
    assert all(set(case) == {"name", "expected", "observed", "passed"} for case in manifest["cases"])
    assert all(case["passed"] is True for case in manifest["cases"])
    assert manifest["engineering_pass"] is True
    assert manifest["scientific_improvement_claim"] is False
    assert "u_num" in manifest["formulas"]
    assert "not a confidence bound" in manifest["formulas"]["u_num"].lower()
    assert any(
        "zero-force saddle" in limitation.lower()
        for limitation in manifest["known_limitations"]
    )
    assert any(
        "dataset" in limitation.lower() and "mattersim" in limitation.lower()
        for limitation in manifest["known_limitations"]
    )

    cases = _cases_by_name(manifest)
    analytic_contracts = {
        "positive_harmonic": (3.0, False),
        "inverted_harmonic": (-6.0, True),
    }
    expected_d_star = math.sqrt(2.16)
    expected_h = (2**-8) * expected_d_star
    for name, (analytic_curvature, negative) in analytic_contracts.items():
        case = cases[name]
        assert case["expected"] == {
            "status": "ok",
            "negative": negative,
            "d_star": expected_d_star,
            "h": expected_h,
            "kappa_h": analytic_curvature,
            "kappa_h2": analytic_curvature,
            "kappa_r": analytic_curvature,
            "error_proxy": 0.0,
            "u_num": analytic_curvature,
        }
        assert case["observed"]["status"] == "ok"
        assert case["observed"]["negative"] is negative
        for key in SCALAR_DIAGNOSTIC_KEYS:
            assert case["observed"][key] == pytest.approx(
                case["expected"][key], rel=0.0, abs=1e-10
            )

    reference = cases["positive_harmonic"]["observed"]
    for name in (
        "translation_invariance",
        "rotation_invariance",
        "permutation_invariance",
        "pbc_wrapping_invariance",
    ):
        case = cases[name]
        assert case["expected"]["reference_case"] == "positive_harmonic"
        assert case["expected"]["status"] == reference["status"]
        assert case["expected"]["negative"] is reference["negative"]
        for key in SCALAR_DIAGNOSTIC_KEYS:
            assert case["expected"][key] == reference[key]
            assert case["observed"][key] == pytest.approx(
                reference[key], rel=0.0, abs=1e-10
            )

    expected_statuses = {
        "exact_zero_force_saddle": "stationary_fallback",
        "oracle_exception": "abstain_force_failure",
        "wrong_shape_force": "abstain_invalid_force",
        "nonfinite_force": "abstain_invalid_force",
    }
    for name, status in expected_statuses.items():
        assert cases[name]["expected"] == {"status": status, "negative": None}
        assert cases[name]["observed"] == cases[name]["expected"]

    assert cases["decision_keep_or_negative"]["expected"] == {
        "baseline": "keep",
        "lrrc_negative": True,
        "decision": "reject",
    }
    assert (
        cases["decision_keep_or_negative"]["observed"]
        == cases["decision_keep_or_negative"]["expected"]
    )
    assert cases["decision_reject_or_nonnegative"]["expected"] == {
        "baseline": "reject",
        "lrrc_negative": False,
        "decision": "reject",
    }
    assert (
        cases["decision_reject_or_nonnegative"]["observed"]
        == cases["decision_reject_or_nonnegative"]["expected"]
    )

    assert cases["quota_fixed_ceil_sqrt_n"]["expected"] == {
        "eligible_count": 6,
        "k": 3,
        "boundary_score": 0.2,
        "output_decisions": {
            "q0": "keep",
            "q1": "keep",
            "q2": "keep",
            "q3": "keep",
            "q4": "reject",
            "q5": "reject",
        },
    }
    assert (
        cases["quota_fixed_ceil_sqrt_n"]["observed"]
        == cases["quota_fixed_ceil_sqrt_n"]["expected"]
    )
    assert cases["quota_boundary_ties"]["observed"] == {
        "boundary_ids": ["q1", "q2", "q3"],
        "boundary_decisions": ["keep", "keep", "keep"],
        "eligible_keep_count": 4,
        "kept_beyond_k_due_to_tie": True,
    }
    assert cases["quota_boundary_ties"]["observed"] == cases["quota_boundary_ties"]["expected"]
    assert cases["quota_abstain_unchanged"]["observed"] == {
        "row_id": "q_abstain",
        "input_decision": "abstain",
        "output_decision": "abstain",
    }
    assert cases["quota_abstain_unchanged"]["observed"] == cases["quota_abstain_unchanged"]["expected"]
    assert cases["quota_rejection_subset"]["observed"] == {
        "input_reject_ids": ["q0", "q1", "q2", "q3", "q4", "q5"],
        "output_reject_ids": ["q4", "q5"],
        "is_subset": True,
    }
    assert cases["quota_rejection_subset"]["observed"] == cases["quota_rejection_subset"]["expected"]

    core_path = Path(next9_lrrc.__file__).resolve()
    runner_path = Path(next9_lrrc_synthetic.__file__).resolve()
    assert manifest["source_sha256"] == {
        "src/next9_lrrc.py": _sha256(core_path),
        "src/next9_lrrc_synthetic.py": _sha256(runner_path),
    }
    assert not list(tmp_path.glob(".lrrc-v0-synthetic.tmp-*"))


def test_engineering_pass_is_the_conjunction_of_every_ordered_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next9_lrrc_synthetic

    original_quota_entries = next9_lrrc_synthetic._quota_entries

    def quota_entries_with_one_failed_contract() -> list[dict[str, object]]:
        entries = original_quota_entries()
        entries[-1] = {**entries[-1], "passed": False}
        return entries

    monkeypatch.setattr(
        next9_lrrc_synthetic,
        "_quota_entries",
        quota_entries_with_one_failed_contract,
    )
    manifest = next9_lrrc_synthetic._build_manifest()

    assert [case["name"] for case in manifest["cases"]] == EXPECTED_CASE_ORDER
    assert manifest["cases"][-1]["passed"] is False
    assert manifest["engineering_pass"] is False


def test_source_change_before_publish_fails_closed_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next9_lrrc_synthetic

    fake_core = tmp_path / "next9_lrrc.py"
    fake_runner = tmp_path / "next9_lrrc_synthetic.py"
    fake_core.write_text("core-v1\n", encoding="utf-8")
    fake_runner.write_text("runner-v1\n", encoding="utf-8")
    fake_sources = {
        "src/next9_lrrc.py": fake_core,
        "src/next9_lrrc_synthetic.py": fake_runner,
    }
    monkeypatch.setattr(next9_lrrc_synthetic, "_source_paths", lambda: fake_sources)

    original_build_manifest = next9_lrrc_synthetic._build_manifest

    def build_then_change_source(source_paths: dict[str, Path]) -> dict[str, object]:
        manifest = original_build_manifest(source_paths)
        fake_runner.write_text("runner-v2\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(
        next9_lrrc_synthetic, "_build_manifest", build_then_change_source
    )
    output_dir = tmp_path / "source-raced-output"

    with pytest.raises(RuntimeError, match="source.*changed.*publication"):
        next9_lrrc_synthetic.run(output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".source-raced-output.tmp-*"))


def test_runner_refuses_second_publication_without_changing_existing_output(
    tmp_path: Path,
) -> None:
    from src.next9_lrrc_synthetic import run

    output_dir = tmp_path / "published"
    run(output_dir)
    original = (output_dir / "MANIFEST.json").read_bytes()

    with pytest.raises(FileExistsError):
        run(output_dir)

    assert (output_dir / "MANIFEST.json").read_bytes() == original
    assert not list(tmp_path.glob(".published.tmp-*"))


def test_missing_no_replace_primitive_fails_closed_without_exposing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next9_lrrc_synthetic

    monkeypatch.setattr(next9_lrrc_synthetic.ctypes, "CDLL", lambda *_a, **_k: object())
    output_dir = tmp_path / "unsupported-output"

    with pytest.raises(NotImplementedError, match="atomic no-replace"):
        next9_lrrc_synthetic.run(output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".unsupported-output.tmp-*"))


def test_renameat2_eexist_does_not_replace_competing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next9_lrrc_synthetic

    output_dir = tmp_path / "raced-output"

    class RacingRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *_args) -> int:
            output_dir.mkdir()
            (output_dir / "competitor.txt").write_text("untouched", encoding="utf-8")
            next9_lrrc_synthetic.ctypes.set_errno(errno.EEXIST)
            return -1

    class FakeLibC:
        renameat2 = RacingRenameAt2()

    monkeypatch.setattr(next9_lrrc_synthetic.ctypes, "CDLL", lambda *_a, **_k: FakeLibC())

    with pytest.raises(FileExistsError):
        next9_lrrc_synthetic.run(output_dir)

    assert (output_dir / "competitor.txt").read_text(encoding="utf-8") == "untouched"
    assert not (output_dir / "MANIFEST.json").exists()
    assert not list(tmp_path.glob(".raced-output.tmp-*"))


def test_cli_accepts_only_output_dir_application_argument(tmp_path: Path) -> None:
    from src.next9_lrrc_synthetic import main

    output_dir = tmp_path / "cli-output"
    assert main(["--output-dir", str(output_dir)]) == 0
    assert (output_dir / "MANIFEST.json").is_file()

    with pytest.raises(SystemExit) as exc_info:
        main(["--output-dir", str(tmp_path / "unused"), "--dataset", "forbidden"])
    assert exc_info.value.code == 2
