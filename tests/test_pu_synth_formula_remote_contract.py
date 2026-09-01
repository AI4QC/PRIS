from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821 import remote_full_pool
from experiments.pu_synthesizability_20260821.remote_full_pool import (
    EXPECTED_DUAL_ROUTE_RESULT_COLUMNS,
    EXPECTED_FORMULA_SBATCH_SHA256,
    FORMULA_BUNDLE_RELATIVE_PATHS,
    FORMULA_FEATURE_COLUMNS,
    FORMULA_FEATURE_ORDER,
    FROZEN_FORMULA_RELATIVE_PATHS,
    FROZEN_FORMULA_SHA256,
    PRODUCTION_FORMULA_SBATCH_RELATIVE_PATH,
    load_input_manifest,
    prepare_run,
    run_task,
    sha256_file,
    validate_frozen_inputs,
    verify_full_pool,
)
from experiments.pu_synthesizability_20260821.runner import flatten_pris_result


ROOT = Path(__file__).resolve().parents[1]


def _write_indexed_csv(path: Path, rows: list[list[str]]) -> Path:
    import csv
    import io

    def render(row: list[str]) -> bytes:
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerow(row)
        return stream.getvalue().encode()

    chunks = [render(["index", "material_id", "cif", "pbes_gap"])]
    offsets: list[int] = []
    for row in rows:
        offsets.append(sum(map(len, chunks)))
        chunks.append(render(row))
    path.write_bytes(b"".join(chunks))
    offset_path = path.with_suffix(".offsets.npy")
    np.save(offset_path, np.asarray(offsets, dtype=np.int64))
    return offset_path


def _formula_payload(features: tuple[str, ...]) -> dict[str, object]:
    return {
        "features": list(features),
        "beta": [1.0] * len(features),
        "impute_median": {feature: 0.0 for feature in features},
        "mu": {feature: 0.0 for feature in features},
        "sd": {feature: 1.0 for feature in features},
    }


def _prepare_tiny_dual_route_run(tmp_path: Path) -> tuple[dict, Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    train = inputs / "train.csv"
    val = inputs / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [
            ["0", "train-a", "data_a", ""],
            ["1", "train-b", "data_b", ""],
        ],
    )
    val_offsets = _write_indexed_csv(val, [["2", "val-a", "data_c", ""]])
    counts = inputs / "counts.npy"
    np.save(counts, np.asarray([10, 5, 2, 1], dtype=np.int64))

    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "impl.py").write_text("VALUE = 1\n")
    (bundle / "submit.sbatch").write_text("#!/bin/bash\ntrue\n")
    bvparm = bundle / "data/bvparm.cif"
    bvparm.write_text("Na 1 Cl -1 2.15 0.37\n")
    formula_paths: dict[str, str] = {}
    formula_hashes: dict[str, str] = {}
    for name, features in FORMULA_FEATURE_ORDER.items():
        relative = f"frozen/{name}.json"
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_formula_payload(features), sort_keys=True))
        formula_paths[name] = relative
        formula_hashes[name] = sha256_file(target)

    frozen = {
        "train": {
            "path": str(train),
            "rows": 2,
            "sha256": sha256_file(train),
            "offset_path": str(train_offsets),
            "offset_sha256": sha256_file(train_offsets),
        },
        "val": {
            "path": str(val),
            "rows": 1,
            "sha256": sha256_file(val),
            "offset_path": str(val_offsets),
            "offset_sha256": sha256_file(val_offsets),
        },
        "counts_path": str(counts),
        "counts_sha256": sha256_file(counts),
        "counts": [10, 5, 2, 1],
    }
    run_root = tmp_path / "run"
    manifest = prepare_run(
        run_root=run_root,
        bundle_root=bundle,
        chunk_size=2,
        frozen_inputs=frozen,
        implementation_relative_paths=["impl.py", "submit.sbatch"],
        bvparm_relative_path="data/bvparm.cif",
        expected_bvparm_sha256=sha256_file(bvparm),
        expected_bvparm_rows=1,
        frozen_formula_relative_paths=formula_paths,
        expected_formula_sha256=formula_hashes,
        submission_script_relative_path="submit.sbatch",
        expected_submission_script_sha256=sha256_file(bundle / "submit.sbatch"),
        expected_result_columns=EXPECTED_DUAL_ROUTE_RESULT_COLUMNS,
    )
    return manifest, bvparm, run_root


def _raw_result(record: dict[str, object]) -> dict[str, object]:
    pool_row = int(record["pool_row"])
    if pool_row == 2:
        return flatten_pris_result(
            record,
            None,
            structure_formula=None,
            chemical_system=None,
            n_elements=None,
            n_sites=None,
            cif_sha256="failed",
            elapsed_seconds=0.0,
            parse_error="ValueError: fixture parse failure",
            cif_parse_route="failed",
        )

    if pool_row == 0:
        synthesis = {
            "valence_route": "failed",
            "feature_error": "balance: no single-anion charge solution",
            "historical_size_domain": True,
            "features": {
                "wyckoff_econ_001": 0.25,
                "vol_per_atom": 20.0,
            },
        }
        stability = {
            "valence_route": "failed",
            "feature_error": "guess_oxi: no primary charge solution",
            "historical_size_domain": True,
            "features": {
                "econ_max": 5.0,
                "wyckoff_econ_01": 0.5,
                "econ_min": 2.0,
            },
        }
    else:
        synthesis = {
            "valence_route": "balance",
            "feature_error": None,
            "historical_size_domain": True,
            "features": {
                feature: float(index + 1)
                for index, feature in enumerate(FORMULA_FEATURE_ORDER["S_syn"])
            },
        }
        stability = {
            "valence_route": "guess_oxi",
            "feature_error": None,
            "historical_size_domain": True,
            "features": {
                feature: float(index + 1)
                for index, feature in enumerate(FORMULA_FEATURE_ORDER["S_stab"])
            },
        }
    result = {
        "charge_assignment_route": "failed" if pool_row == 0 else "integer",
        "feature_error": None,
        "minimum_pair_distance_a": 2.0,
        "wyckoff_econ_symprec_0p1": 0.5,
        "features": {},
        "predicates": {},
        "rungs": {},
        "formula_features": {
            "synthesis": synthesis,
            "stability": stability,
        },
    }
    return flatten_pris_result(
        record,
        result,
        structure_formula="NaCl",
        chemical_system="Cl-Na",
        n_elements=2,
        n_sites=2,
        cif_sha256=f"row-{pool_row}",
        elapsed_seconds=0.0,
        cif_parse_route="default",
    )


def _evaluate(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([_raw_result(record) for record in records])


def _rewrite_shard_and_manifest(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["output_bytes"] = path.stat().st_size
    manifest["output_sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_production_bundle_freezes_actual_formula_script_jsons_and_bv_table() -> None:
    script = ROOT / PRODUCTION_FORMULA_SBATCH_RELATIVE_PATH
    assert script.is_file()
    assert sha256_file(script) == EXPECTED_FORMULA_SBATCH_SHA256
    assert "pris_formula_dualroute_fullpool_20260822" in script.read_text()
    assert "pris_formula_fullpool_20260822" not in script.read_text()

    for name, relative in FROZEN_FORMULA_RELATIVE_PATHS.items():
        assert sha256_file(ROOT / relative) == FROZEN_FORMULA_SHA256[name]
    upload_list = (
        ROOT
        / "experiments/pu_synthesizability_20260821/remote_formula_bundle_files.txt"
    ).read_text().splitlines()
    assert upload_list == [*FORMULA_BUNDLE_RELATIVE_PATHS, "data/bvparm2020.cif"]
    assert PRODUCTION_FORMULA_SBATCH_RELATIVE_PATH in FORMULA_BUNDLE_RELATIVE_PATHS
    assert (
        "experiments/pu_synthesizability_20260821/pris_full_pool_remote.sbatch"
        not in FORMULA_BUNDLE_RELATIVE_PATHS
    )


def test_prepare_freezes_dual_formula_contract_and_submission_script(tmp_path: Path) -> None:
    manifest, _, run_root = _prepare_tiny_dual_route_run(tmp_path)
    loaded = load_input_manifest(
        run_root / "input_manifest.json", run_root / "input_manifest.sha256"
    )
    assert loaded == manifest
    validate_frozen_inputs(loaded, full_hash=False)
    assert set(manifest["frozen_formulas"]) == {"S_syn", "S_stab"}
    assert manifest["submission_script"]["sha256"] == sha256_file(
        tmp_path / "bundle/submit.sbatch"
    )
    contract = manifest["result_contract"]
    assert contract["ordered_columns"] == list(EXPECTED_DUAL_ROUTE_RESULT_COLUMNS)
    for name in ("S_syn", "S_stab"):
        assert contract["formula_observation_masks"][name]["feature_order"] == list(
            FORMULA_FEATURE_ORDER[name]
        )
        assert contract["formula_observation_masks"][name]["feature_columns"] == list(
            FORMULA_FEATURE_COLUMNS[name]
        )
        assert contract["formula_observation_masks"][name]["encoding"] == (
            "uint6_bit_i_is_one_iff_raw_feature_i_is_finite"
        )


def test_verify_accepts_legitimate_two_of_six_and_three_of_six_masks(tmp_path: Path) -> None:
    manifest, bvparm, run_root = _prepare_tiny_dual_route_run(tmp_path)
    for task_id in range(2):
        run_task(
            manifest,
            task_id=task_id,
            output_dir=run_root / "results",
            workers=1,
            src_dir=tmp_path,
            bvparm_path=bvparm,
            batch_evaluator=_evaluate,
        )

    success = verify_full_pool(manifest, output_dir=run_root / "results")

    assert success["result_schema"]["columns"] == list(
        EXPECTED_DUAL_ROUTE_RESULT_COLUMNS
    )
    syn = success["formula_observation_masks"]["S_syn"]
    stab = success["formula_observation_masks"]["S_stab"]
    assert syn["mask_counts"] == {"0": 1, "10": 1, "63": 1}
    assert syn["observed_term_count_counts"] == {"0": 1, "2": 1, "6": 1}
    assert stab["mask_counts"] == {"0": 1, "49": 1, "63": 1}
    assert stab["observed_term_count_counts"] == {"0": 1, "3": 1, "6": 1}
    assert success["frozen_formula_sha256"] == {
        name: manifest["frozen_formulas"][name]["sha256"]
        for name in ("S_syn", "S_stab")
    }


def test_verify_rejects_schema_drift_even_when_shard_hash_is_refreshed(tmp_path: Path) -> None:
    manifest, bvparm, run_root = _prepare_tiny_dual_route_run(tmp_path)
    for task_id in range(2):
        run_task(
            manifest,
            task_id=task_id,
            output_dir=run_root / "results",
            workers=1,
            src_dir=tmp_path,
            bvparm_path=bvparm,
            batch_evaluator=_evaluate,
        )
    first = sorted((run_root / "results").glob("part-*.parquet"))[0]
    frame = pd.read_parquet(first).drop(columns=["formula_stab_econ_min"])
    _rewrite_shard_and_manifest(first, frame)

    with pytest.raises(ValueError, match="result schema mismatch in task 0"):
        verify_full_pool(manifest, output_dir=run_root / "results")


def test_verify_rejects_charge_dependent_bits_on_a_failed_formula_route(
    tmp_path: Path,
) -> None:
    manifest, bvparm, run_root = _prepare_tiny_dual_route_run(tmp_path)
    for task_id in range(2):
        run_task(
            manifest,
            task_id=task_id,
            output_dir=run_root / "results",
            workers=1,
            src_dir=tmp_path,
            bvparm_path=bvparm,
            batch_evaluator=_evaluate,
        )
    first = sorted((run_root / "results").glob("part-*.parquet"))[0]
    frame = pd.read_parquet(first)
    frame.loc[frame.pool_row.eq(0), "formula_syn_madz_mean"] = 1.0
    _rewrite_shard_and_manifest(first, frame)

    with pytest.raises(
        ValueError,
        match="S_syn failed route has charge-dependent observed terms in task 0",
    ):
        verify_full_pool(manifest, output_dir=run_root / "results")
