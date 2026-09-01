from __future__ import annotations

import hashlib
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_waterfill_quotas_preserve_rare_joint_strata_and_total_4096() -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        allocate_stratum_quotas,
    )

    counts = {
        ("elementa", "explicit_violation", "failed"): 19_149,
        ("elementa", "explicit_violation", "fractional"): 75,
        ("elementa", "explicit_violation", "integer"): 16,
        ("elementa", "pass", "failed"): 83_032,
        ("elementa", "pass", "fractional"): 323,
        ("elementa", "pass", "integer"): 32,
        ("lemat", "explicit_violation", "failed"): 155_638,
        ("lemat", "explicit_violation", "fractional"): 1_175,
        ("lemat", "explicit_violation", "integer"): 10_867,
        ("lemat", "pass", "failed"): 90_333,
        ("lemat", "pass", "fractional"): 466,
        ("lemat", "pass", "integer"): 3_665,
    }

    quotas = allocate_stratum_quotas(counts, sample_size=4_096)

    assert sum(quotas.values()) == 4_096
    assert all(0 < quotas[key] <= counts[key] for key in counts)
    assert quotas[("elementa", "explicit_violation", "integer")] == 16
    assert quotas[("elementa", "pass", "integer")] == 32
    assert quotas[("elementa", "explicit_violation", "fractional")] == 75
    assert quotas[("elementa", "pass", "fractional")] == 323


def _synthetic_formal_frame(rows_per_stratum: int = 40) -> pd.DataFrame:
    rows = []
    record_index = 0
    threshold = 2.0 / 3.0
    for provenance, d7, charge in product(
        ("elementa", "lemat"),
        ("explicit_violation", "pass"),
        ("failed", "fractional", "integer"),
    ):
        for within in range(rows_per_stratum):
            direction = 1.0 if d7 == "explicit_violation" else -1.0
            rows.append(
                {
                    "record_index": record_index,
                    "orig_index": 1_000_000 + record_index,
                    "cif_sha256": hashlib.sha256(
                        f"cif-{record_index}".encode()
                    ).hexdigest(),
                    "provenance": provenance,
                    "D7_verdict": d7,
                    "charge_assignment_route": charge,
                    "feature_wyckoff_econ": threshold
                    + direction * (within + 1) * 1e-5,
                    "feature_bl_min": 0.735 + (within - 20) * 1e-4,
                    "feature_bl_mean": 1.081 + (within - 20) * 1e-4,
                    "feature_cn_an_mean": 3.333 + (within - 20) * 1e-3,
                    "feature_madz_range": 31.45 + (within - 20) * 1e-2,
                    "feature_mad_max": 15.17 + (within - 20) * 1e-2,
                    "feature_frac_like_bonds": max(0.0, (within - 20) * 1e-5),
                    "feature_fi": 0.55 + (within - 20) * 1e-3,
                    "feature_bv_rel_mean": 0.7143040821865658
                    + (within - 20) * 1e-4,
                }
            )
            record_index += 1
    return pd.DataFrame(rows)


def test_selection_is_deterministic_jointly_stratified_and_boundary_enriched() -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        select_calibration_rows,
    )

    frame = _synthetic_formal_frame()
    first = select_calibration_rows(frame, sample_size=120, seed="fixed-test-seed")
    second = select_calibration_rows(
        frame.sample(frac=1.0, random_state=91),
        sample_size=120,
        seed="fixed-test-seed",
    )

    assert first.record_index.tolist() == second.record_index.tolist()
    assert first.calibration_index.tolist() == list(range(120))
    assert not first.record_index.duplicated().any()
    counts = first.groupby(
        ["provenance", "D7_verdict", "charge_assignment_route"]
    ).size()
    assert len(counts) == 12
    assert set(counts) == {10}
    assert first.selection_reason.str.startswith("threshold:").any()
    for _, stratum in frame.groupby(
        ["provenance", "D7_verdict", "charge_assignment_route"]
    ):
        nearest = (
            stratum.feature_wyckoff_econ.sub(2.0 / 3.0).abs().idxmin()
        )
        nearest_record = int(frame.loc[nearest, "record_index"])
        assert nearest_record in set(first.record_index)


def test_build_artifacts_separates_compressed_cifs_from_local_expected(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        build_calibration_artifacts,
    )
    from experiments.pu_synthesizability_20260821.remote_full_pool import sha256_file

    formal = _synthetic_formal_frame(rows_per_stratum=4)
    formal["material_id"] = formal.record_index.map(lambda value: f"m-{value}")
    formal["cohort"] = "pu_negative"
    records = []
    for row in formal.itertuples(index=False):
        cif = f"data_{row.record_index}\n_cell_length_a 5"
        formal.loc[
            formal.record_index.eq(row.record_index), "cif_sha256"
        ] = hashlib.sha256(cif.encode()).hexdigest()
        records.append(
            {
                "cohort": "pu_negative",
                "record_index": int(row.record_index),
                "index": int(row.record_index),
                "orig_index": int(row.orig_index),
                "material_id": f"m-{row.record_index}",
                "provenance": row.provenance,
                "cif": cif,
                "pbes_gap": None,
            }
        )

    manifest = build_calibration_artifacts(
        formal,
        records,
        output_dir=tmp_path,
        sample_size=24,
        seed="artifact-test-seed",
        source_audit={"formal": "test-fixture"},
    )

    input_path = tmp_path / "calibration_input.parquet"
    expected_path = tmp_path / "local_expected.parquet"
    input_frame = pd.read_parquet(input_path)
    expected = pd.read_parquet(expected_path)
    assert len(input_frame) == len(expected) == 24
    assert "cif" in input_frame
    assert "cif" not in expected
    assert input_frame.calibration_index.tolist() == list(range(24))
    assert expected.calibration_index.tolist() == list(range(24))
    assert input_frame.record_index.tolist() == expected.record_index.tolist()
    assert input_frame.pool_id.eq("u" + input_frame.orig_index.astype(str)).all()
    assert manifest["sample_size"] == 24
    assert manifest["source_rows"] == 48
    assert manifest["source_audit"] == {"formal": "test-fixture"}
    assert manifest["artifacts"]["calibration_input"]["sha256"] == sha256_file(
        input_path
    )
    assert json.loads((tmp_path / "selection_manifest.json").read_text()) == manifest


def _calibration_result_frame(rows: int = 3) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "calibration_index": np.arange(rows, dtype=np.int64),
            "cohort": "pu_negative",
            "record_index": np.arange(10, 10 + rows, dtype=np.int64),
            "orig_index": np.arange(1_000, 1_000 + rows, dtype=np.int64),
            "material_id": [f"m-{i}" for i in range(rows)],
            "provenance": ["elementa", "lemat", "elementa"][:rows],
            "confidence": ["high"] * rows,
            "CLscore_A": np.linspace(0.01, 0.03, rows),
            "CLscore_B": np.linspace(0.02, 0.04, rows),
            "CLscore_jang": np.linspace(0.03, 0.05, rows),
            "cif_sha256": [hashlib.sha256(f"cif-{i}".encode()).hexdigest() for i in range(rows)],
            "parse_ok": [True] * rows,
            "parse_error": [None] * rows,
            "cif_parse_route": ["default"] * rows,
            "structure_formula": ["NaCl", "SiO2", "Al2O3"][:rows],
            "chemical_system": ["Cl-Na", "O-Si", "Al-O"][:rows],
            "n_elements": [2] * rows,
            "n_sites": np.arange(2, 2 + rows, dtype=np.int64),
            "charge_assignment_route": ["integer", "fractional", "failed"][:rows],
            "feature_error": [None] * rows,
            "minimum_pair_distance_a": np.linspace(1.8, 2.0, rows),
            "wyckoff_econ_symprec_0p1": np.linspace(0.65, 0.68, rows),
            "feature_bl_min": np.linspace(0.72, 0.82, rows),
            "feature_bl_mean": np.linspace(1.02, 1.12, rows),
            "feature_cn_an_mean": np.linspace(3.0, 4.0, rows),
            "feature_madz_range": np.linspace(30.0, 32.0, rows),
            "feature_mad_max": np.linspace(14.0, 16.0, rows),
            "feature_frac_like_bonds": np.linspace(0.0, 2e-4, rows),
            "feature_fi": np.linspace(0.45, 0.6, rows),
            "feature_wyckoff_econ": np.linspace(0.65, 0.68, rows),
            "feature_bv_rel_mean": np.linspace(0.70, 0.73, rows),
        }
    )
    for name in ("D1_735", "D1_804", "D2", "D3", "D4", "D5", "D6", "D7", "D8"):
        result[f"{name}_status"] = "satisfied"
        result[f"{name}_verdict"] = "pass"
    result["D1_status"] = result["D1_804_status"]
    result["D1_verdict"] = result["D1_804_verdict"]
    for name in ("L1", "L1_prime", "L2", "L3", "L4"):
        result[f"{name}_verdict"] = "pass"
    return result


def _write_calibration_input(tmp_path: Path, expected: pd.DataFrame) -> tuple[Path, Path]:
    from experiments.pu_synthesizability_20260821.remote_full_pool import sha256_file

    input_frame = expected[
        ["calibration_index", "cohort", "record_index", "orig_index"]
    ].copy()
    input_frame["cif"] = [f"cif-{value}" for value in input_frame.calibration_index]
    input_path = tmp_path / "calibration_input.parquet"
    input_frame.to_parquet(input_path, index=False)
    selection_manifest = {
        "schema_version": 1,
        "sample_size": len(input_frame),
        "artifacts": {
            "calibration_input": {
                "path": str(input_path.resolve()),
                "bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
            }
        },
    }
    manifest_path = tmp_path / "selection_manifest.json"
    manifest_path.write_text(json.dumps(selection_manifest))
    return input_path, manifest_path


def test_remote_evaluate_is_atomic_resumable_and_uses_full_pool_evaluator_contract(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        remote_evaluate_artifact,
    )

    expected = _calibration_result_frame()
    input_path, selection_manifest_path = _write_calibration_input(tmp_path, expected)
    calls: list[dict[str, object]] = []

    def fake_evaluator(records, **kwargs):
        calls.append({"records": records, **kwargs})
        return expected.copy()

    output_path = tmp_path / "remote_actual.parquet"
    manifest = remote_evaluate_artifact(
        input_path=input_path,
        selection_manifest_path=selection_manifest_path,
        output_path=output_path,
        workers=7,
        chunksize=3,
        src_dir=tmp_path / "src",
        bvparm_path=tmp_path / "bvparm.cif",
        evaluator=fake_evaluator,
        runtime_identity={"python_version": "test-runtime"},
    )

    actual = pd.read_parquet(output_path)
    assert "cif" not in actual
    assert actual.calibration_index.tolist() == [0, 1, 2]
    assert calls[0]["workers"] == 7
    assert calls[0]["chunksize"] == 3
    assert [row["record_index"] for row in calls[0]["records"]] == [10, 11, 12]
    assert manifest["rows"] == 3
    assert manifest["runtime"] == {"python_version": "test-runtime"}
    assert manifest["verdict_gate_ready"] is True

    resumed = remote_evaluate_artifact(
        input_path=input_path,
        selection_manifest_path=selection_manifest_path,
        output_path=output_path,
        workers=7,
        chunksize=3,
        src_dir=tmp_path / "src",
        bvparm_path=tmp_path / "bvparm.cif",
        evaluator=lambda *_args, **_kwargs: pytest.fail("resume re-evaluated rows"),
        runtime_identity={"python_version": "test-runtime"},
    )
    assert resumed == manifest


def test_remote_evaluate_refuses_partial_or_corrupt_resume(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        remote_evaluate_artifact,
    )

    expected = _calibration_result_frame()
    input_path, selection_manifest_path = _write_calibration_input(tmp_path, expected)
    output_path = tmp_path / "remote_actual.parquet"
    output_path.write_bytes(b"partial")

    with pytest.raises(ValueError, match="partial remote calibration output"):
        remote_evaluate_artifact(
            input_path=input_path,
            selection_manifest_path=selection_manifest_path,
            output_path=output_path,
            workers=1,
            chunksize=1,
            src_dir=tmp_path / "src",
            bvparm_path=tmp_path / "bvparm.cif",
            evaluator=lambda *_args, **_kwargs: expected,
            runtime_identity={"python_version": "test-runtime"},
        )


def test_remote_evaluate_freezes_the_production_container_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.pu_synthesizability_20260821.remote_full_pool as full_pool
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        remote_evaluate_artifact,
    )

    expected = _calibration_result_frame()
    input_path, selection_manifest_path = _write_calibration_input(tmp_path, expected)
    captured: dict[str, object] = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return {"container": "frozen-test"}

    monkeypatch.setattr(full_pool, "capture_runtime_identity", fake_runtime)
    manifest = remote_evaluate_artifact(
        input_path=input_path,
        selection_manifest_path=selection_manifest_path,
        output_path=tmp_path / "remote_actual.parquet",
        workers=1,
        chunksize=1,
        src_dir=tmp_path / "src",
        bvparm_path=tmp_path / "bvparm.cif",
        evaluator=lambda *_args, **_kwargs: expected.copy(),
    )

    assert captured["container_sif_path"] == full_pool.PRODUCTION_CONTAINER_SIF
    assert captured["requirements_lock_path"] == full_pool.PRODUCTION_REQUIREMENTS_LOCK
    assert captured["site_packages_path"] == full_pool.PRODUCTION_SITE_PACKAGES
    assert (
        captured["site_packages_manifest_path"]
        == full_pool.PRODUCTION_SITE_PACKAGES_MANIFEST
    )
    assert manifest["runtime"] == {"container": "frozen-test"}


def test_remote_evaluate_revalidates_the_complete_contract_on_resume(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        remote_evaluate_artifact,
    )
    from experiments.pu_synthesizability_20260821.remote_full_pool import sha256_file

    expected = _calibration_result_frame()
    input_path, selection_manifest_path = _write_calibration_input(tmp_path, expected)
    output_path = tmp_path / "remote_actual.parquet"
    remote_evaluate_artifact(
        input_path=input_path,
        selection_manifest_path=selection_manifest_path,
        output_path=output_path,
        workers=1,
        chunksize=1,
        src_dir=tmp_path / "src",
        bvparm_path=tmp_path / "bvparm.cif",
        evaluator=lambda *_args, **_kwargs: expected.copy(),
        runtime_identity={"python_version": "test-runtime"},
    )
    incomplete = pd.read_parquet(output_path).drop(columns=["D7_status"])
    incomplete.to_parquet(output_path, index=False)
    remote_manifest_path = output_path.with_suffix(".manifest.json")
    remote_manifest = json.loads(remote_manifest_path.read_text())
    remote_manifest["output_bytes"] = output_path.stat().st_size
    remote_manifest["output_sha256"] = sha256_file(output_path)
    remote_manifest_path.write_text(json.dumps(remote_manifest))

    with pytest.raises(ValueError, match="omitted comparison fields"):
        remote_evaluate_artifact(
            input_path=input_path,
            selection_manifest_path=selection_manifest_path,
            output_path=output_path,
            workers=1,
            chunksize=1,
            src_dir=tmp_path / "src",
            bvparm_path=tmp_path / "bvparm.cif",
            evaluator=lambda *_args, **_kwargs: pytest.fail("resume re-evaluated"),
            runtime_identity={"python_version": "test-runtime"},
        )


def _write_comparison_manifests(
    tmp_path: Path, local_path: Path, remote_path: Path
) -> tuple[Path, Path]:
    from experiments.pu_synthesizability_20260821.remote_full_pool import sha256_file

    selection_path = tmp_path / "selection_manifest.json"
    selection = {
        "schema_version": 1,
        "sample_size": len(pd.read_parquet(local_path)),
        "artifacts": {
            "calibration_input": {"sha256": "1" * 64},
            "local_expected": {
                "path": str(local_path.resolve()),
                "bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
            },
        },
    }
    selection_path.write_text(json.dumps(selection))
    remote_manifest_path = remote_path.with_suffix(".manifest.json")
    remote_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_sha256": "1" * 64,
                "selection_manifest_sha256": sha256_file(selection_path),
                "rows": len(pd.read_parquet(remote_path)),
                "output_path": str(remote_path.resolve()),
                "output_bytes": remote_path.stat().st_size,
                "output_sha256": sha256_file(remote_path),
                "verdict_gate_ready": True,
            }
        )
    )
    return selection_path, remote_manifest_path


def test_compare_reports_numeric_drift_but_requires_exact_verdicts(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        compare_calibration,
    )

    local = _calibration_result_frame()
    remote = local.copy()
    remote.loc[1, "feature_bl_min"] += 2.5e-8
    local_path = tmp_path / "local_expected.parquet"
    remote_path = tmp_path / "remote_actual.parquet"
    local.to_parquet(local_path, index=False)
    remote.to_parquet(remote_path, index=False)
    selection_path, remote_manifest_path = _write_comparison_manifests(
        tmp_path, local_path, remote_path
    )

    summary = compare_calibration(
        local_expected_path=local_path,
        remote_actual_path=remote_path,
        output_dir=tmp_path / "comparison",
        selection_manifest_path=selection_path,
        remote_manifest_path=remote_manifest_path,
    )

    assert summary["rows"] == 3
    assert summary["verdict_matches"] == summary["verdict_comparisons"]
    assert summary["verdict_match_rate"] == 1.0
    assert summary["verdict_gate_passed"] is True
    assert summary["exact_field_mismatches"] == 0
    numeric = pd.read_csv(tmp_path / "comparison" / "numeric_feature_differences.csv")
    bl_min = numeric.set_index("column").loc["feature_bl_min"]
    assert bl_min["different_rows"] == 1
    assert bl_min["max_abs_difference"] == pytest.approx(2.5e-8)
    details = pd.read_parquet(tmp_path / "comparison" / "numeric_row_differences.parquet")
    assert set(details.column) == {"feature_bl_min"}


def test_compare_writes_evidence_then_fails_on_one_verdict_mismatch(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        compare_calibration,
    )

    local = _calibration_result_frame()
    remote = local.copy()
    remote.loc[2, "D7_verdict"] = "explicit_violation"
    local_path = tmp_path / "local_expected.parquet"
    remote_path = tmp_path / "remote_actual.parquet"
    local.to_parquet(local_path, index=False)
    remote.to_parquet(remote_path, index=False)
    selection_path, remote_manifest_path = _write_comparison_manifests(
        tmp_path, local_path, remote_path
    )

    with pytest.raises(RuntimeError, match="verdict gate failed: 1 of"):
        compare_calibration(
            local_expected_path=local_path,
            remote_actual_path=remote_path,
            output_dir=tmp_path / "comparison",
            selection_manifest_path=selection_path,
            remote_manifest_path=remote_manifest_path,
        )

    summary = json.loads(
        (tmp_path / "comparison" / "comparison_summary.json").read_text()
    )
    assert summary["verdict_gate_passed"] is False
    mismatches = pd.read_parquet(tmp_path / "comparison" / "exact_mismatches.parquet")
    assert (
        (mismatches.calibration_index == 2)
        & (mismatches.column == "D7_verdict")
    ).any()


def test_compare_fails_if_hash_chain_or_numeric_feature_schema_is_incomplete(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        compare_calibration,
    )

    local = _calibration_result_frame()
    remote = local.copy().drop(columns=["feature_bv_rel_mean"])
    local_path = tmp_path / "local_expected.parquet"
    remote_path = tmp_path / "remote_actual.parquet"
    local.to_parquet(local_path, index=False)
    remote.to_parquet(remote_path, index=False)
    selection_path, remote_manifest_path = _write_comparison_manifests(
        tmp_path, local_path, remote_path
    )

    with pytest.raises(ValueError, match="numeric feature columns missing"):
        compare_calibration(
            local_expected_path=local_path,
            remote_actual_path=remote_path,
            output_dir=tmp_path / "comparison-missing",
            selection_manifest_path=selection_path,
            remote_manifest_path=remote_manifest_path,
        )
    remote_manifest = json.loads(remote_manifest_path.read_text())
    remote_manifest["selection_manifest_sha256"] = "0" * 64
    remote_manifest_path.write_text(json.dumps(remote_manifest))
    with pytest.raises(ValueError, match="selection manifest SHA256"):
        compare_calibration(
            local_expected_path=local_path,
            remote_actual_path=remote_path,
            output_dir=tmp_path / "comparison-bad-chain",
            selection_manifest_path=selection_path,
            remote_manifest_path=remote_manifest_path,
        )


def test_compare_exact_parse_formula_site_and_charge_fields_are_a_hard_gate(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        compare_calibration,
    )

    local = _calibration_result_frame()
    remote = local.copy()
    remote.loc[1, "structure_formula"] = "O2Si"
    local_path = tmp_path / "local_expected.parquet"
    remote_path = tmp_path / "remote_actual.parquet"
    local.to_parquet(local_path, index=False)
    remote.to_parquet(remote_path, index=False)
    selection_path, remote_manifest_path = _write_comparison_manifests(
        tmp_path, local_path, remote_path
    )

    with pytest.raises(RuntimeError, match="exact field gate failed: 1"):
        compare_calibration(
            local_expected_path=local_path,
            remote_actual_path=remote_path,
            output_dir=tmp_path / "comparison",
            selection_manifest_path=selection_path,
            remote_manifest_path=remote_manifest_path,
        )
    summary = json.loads(
        (tmp_path / "comparison" / "comparison_summary.json").read_text()
    )
    assert summary["exact_gate_passed"] is False


def _write_tiny_formal_release(tmp_path: Path) -> Path:
    from experiments.pu_synthesizability_20260821.remote_full_pool import sha256_file

    release = tmp_path / "release"
    release.mkdir()
    train = release / "train.csv"
    validation = release / "val.csv"
    train.write_text(
        "index,material_id,cif,pbes_gap\n"
        "0,m-0,cif-0,\n"
        "1,m-1,cif-1,\n"
    )
    validation.write_text(
        "index,material_id,cif,pbes_gap\n"
        "2,m-2,cif-2,\n"
    )
    metadata = release / "meta.tsv"
    pd.DataFrame(
        {
            "index": [0, 1, 2],
            "orig_index": [1_000, 1_001, 1_002],
            "CLscore_A": [0.01, 0.02, 0.03],
            "CLscore_B": [0.02, 0.03, 0.04],
            "CLscore_jang": [0.03, 0.04, 0.05],
            "confidence": ["high"] * 3,
            "provenance": ["elementa", "lemat", "elementa"],
            "license": ["test"] * 3,
        }
    ).to_csv(metadata, sep="\t", index=False)
    formal_dir = tmp_path / "formal"
    formal_dir.mkdir()
    formal = _calibration_result_frame()
    formal["record_index"] = [0, 1, 2]
    formal["index"] = [0, 1, 2]
    formal["license"] = "test"
    part = formal_dir / "part-00000-000000000-000000002.parquet"
    formal.to_parquet(part, index=False)

    def identity(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    input_manifest = {
        "schema_version": 1,
        "cohort": "pu_negative",
        "expected_rows": 3,
        "sources": {
            "metadata": identity(metadata),
            "train": identity(train),
            "validation": identity(validation),
        },
        "pris_implementation": [],
        "newpauling_commit": "test-newpauling",
        "csagent_commit": "test-csagent",
        "verdict_policy": {
            "pass": "retain",
            "explicit_violation": "remove",
            "no_verdict": "retain_and_report_separately",
        },
        "python": "test-python",
    }
    (formal_dir / "input_manifest.json").write_text(json.dumps(input_manifest))
    input_fingerprint = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True).encode()
    ).hexdigest()
    shard_manifest = {
        "input_fingerprint": input_fingerprint,
        "first_record_index": 0,
        "last_record_index": 2,
        "rows": 3,
        "parse_failures": 0,
        "output_sha256": sha256_file(part),
    }
    part.with_suffix(".manifest.json").write_text(json.dumps(shard_manifest))
    (formal_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "cohort": "pu_negative",
                "input_fingerprint": input_fingerprint,
                "rows": 3,
                "parse_failures": 0,
                "n_shards": 1,
                "shards": [shard_manifest],
            }
        )
    )
    return formal_dir


def test_build_from_frozen_release_checks_sources_and_preserves_variant_d1(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        build_from_frozen_release,
    )

    formal_dir = _write_tiny_formal_release(tmp_path)
    output_dir = tmp_path / "calibration"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest = build_from_frozen_release(
            formal_dir=formal_dir,
            output_dir=output_dir,
            sample_size=3,
            expected_rows=3,
            seed="tiny-release-test",
        )

    expected = pd.read_parquet(output_dir / "local_expected.parquet")
    assert set(expected.record_index) == {0, 1, 2}
    assert {"D1_735_status", "D1_804_status"}.issubset(expected.columns)
    assert manifest["source_audit"]["formal_results"]["rows"] == 3
    assert manifest["source_audit"]["raw_sources"]["train"]["sha256"]
    assert manifest["source_audit"]["run_summary"]["rows"] == 3
    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def test_cross_environment_cli_exposes_build_remote_evaluate_and_compare() -> None:
    from experiments.pu_synthesizability_20260821.cross_env_calibration import (
        build_parser,
    )

    build = build_parser().parse_args(
        ["build", "--formal-dir", "/formal", "--output-dir", "/calibration"]
    )
    remote = build_parser().parse_args(
        [
            "remote-evaluate",
            "--input",
            "/calibration/input.parquet",
            "--selection-manifest",
            "/calibration/selection.json",
            "--output",
            "/calibration/remote.parquet",
            "--src-dir",
            "/bundle/src",
            "--bvparm",
            "/bundle/data/bvparm2020.cif",
        ]
    )
    compare = build_parser().parse_args(
        [
            "compare",
            "--local-expected",
            "/calibration/local.parquet",
            "--remote-actual",
            "/calibration/remote.parquet",
            "--selection-manifest",
            "/calibration/selection.json",
            "--remote-manifest",
            "/calibration/remote.manifest.json",
            "--output-dir",
            "/calibration/comparison",
        ]
    )

    assert build.command == "build" and build.sample_size == 4_096
    assert remote.command == "remote-evaluate" and remote.workers == 47
    assert remote.chunksize == 8
    assert compare.command == "compare"
    assert compare.selection_manifest == "/calibration/selection.json"
    assert compare.remote_manifest == "/calibration/remote.manifest.json"
