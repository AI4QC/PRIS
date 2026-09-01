from __future__ import annotations
import os

import json
import zlib
from pathlib import Path

import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821.runner import (
    build_shard_manifest,
    evaluate_cif_record,
    flatten_pris_result,
    parse_cif_structure,
    run_record_batch,
    validate_resume_manifest,
)


def test_resume_manifest_requires_identical_inputs_and_evaluator(tmp_path):
    source = tmp_path / "source.csv"
    evaluator = tmp_path / "evaluator.py"
    source.write_text("x\n")
    evaluator.write_text("v1\n")
    manifest = build_shard_manifest(
        cohort="demo",
        shard_id=3,
        source_paths=[source],
        evaluator_path=evaluator,
        start=30,
        stop=40,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    validate_resume_manifest(path, manifest)

    evaluator.write_text("v2\n")
    changed = build_shard_manifest(
        cohort="demo",
        shard_id=3,
        source_paths=[source],
        evaluator_path=evaluator,
        start=30,
        stop=40,
    )
    with pytest.raises(ValueError, match="resume manifest mismatch"):
        validate_resume_manifest(path, changed)


def test_flatten_pris_result_keeps_applicability_and_translates_queue_verdicts():
    record = {
        "cohort": "pu_negative",
        "record_index": 7,
        "material_id": "m7",
        "orig_index": 70,
        "provenance": "lemat",
        "CLscore_A": 0.01,
        "cif": "large payload must not be copied",
    }
    result = {
        "charge_assignment_route": "fractional",
        "feature_error": None,
        "minimum_pair_distance_a": 1.2,
        "wyckoff_econ_symprec_0p1": 0.5,
        "features": {"bl_min": 0.9, "fi": 0.6},
        "predicates": {
            "D1_735": "satisfied",
            "D1_804": "satisfied",
            "D2": "not applicable",
            "D3": "unknown",
            "D4": "violated",
            "D5": "satisfied",
            "D6": "satisfied",
            "D7": "satisfied",
            "D8": "satisfied",
        },
        "rungs": {
            "L1": "pass",
            "L1_prime": "pass",
            "L2": "reject",
            "L3": "reject",
            "L4": "reject",
        },
    }
    got = flatten_pris_result(
        record,
        result,
        structure_formula="NaCl",
        chemical_system="Cl-Na",
        n_elements=2,
        n_sites=2,
        cif_sha256="abc",
        elapsed_seconds=0.2,
    )
    assert "cif" not in got
    assert got["D2_status"] == "not applicable"
    assert got["D2_verdict"] == "pass"
    assert got["D3_verdict"] == "no_verdict"
    assert got["D4_verdict"] == "explicit_violation"
    assert got["L2_verdict"] == "explicit_violation"
    assert got["feature_bl_min"] == pytest.approx(0.9)


def test_flatten_parse_failure_is_no_verdict_not_a_violation():
    got = flatten_pris_result(
        {"cohort": "experimental", "record_index": 1},
        None,
        structure_formula=None,
        chemical_system=None,
        n_elements=None,
        n_sites=None,
        cif_sha256="bad",
        elapsed_seconds=0.01,
        parse_error="ValueError: bad CIF",
    )
    assert got["parse_ok"] is False
    assert got["L1_verdict"] == "no_verdict"
    assert got["L4_verdict"] == "no_verdict"
    assert all(got[f"feature_{name}"] is None for name in (
        "bl_min",
        "bl_mean",
        "cn_an_mean",
        "madz_range",
        "mad_max",
        "frac_like_bonds",
        "fi",
        "wyckoff_econ",
        "bv_rel_mean",
    ))
    assert all(
        got[f"D{i}_verdict"] == "no_verdict" for i in range(1, 9)
    )


def test_evaluate_cif_record_turns_parse_error_into_auditable_no_verdict():
    got = evaluate_cif_record(
        {"cohort": "pu_negative", "record_index": 4, "cif": "not a CIF"},
        src_dir="src",
    )
    assert got["parse_ok"] is False
    assert got["parse_error"].startswith("ValueError:")
    assert got["L2_verdict"] == "no_verdict"


def test_run_record_batch_writes_atomic_parquet_and_accounts_for_every_row(tmp_path):
    output = tmp_path / "shard.parquet"
    summary = run_record_batch(
        [
            {"cohort": "pu_negative", "record_index": 1, "cif": "bad one"},
            {"cohort": "pu_negative", "record_index": 2, "cif": "bad two"},
        ],
        output_path=output,
        src_dir="src",
        workers=1,
    )
    assert output.exists()
    assert not output.with_suffix(".parquet.tmp").exists()
    assert summary["rows"] == 2
    assert summary["parse_failures"] == 2
    assert summary["output_sha256"]


def test_tolerant_parser_recovers_known_frozen_experimental_cif():
    provenance_path = Path(
        os.path.join(Path(os.environ.get("PRIS_FEATURES", "features/")), "provenance.parquet")
    )
    blob_path = Path(
        Path(os.environ.get("PRIS_MATDATA_BLOB", "structures.blob"))
    )
    if not provenance_path.exists() or not blob_path.exists():
        pytest.skip("frozen CSAgent experimental snapshot is not mounted")
    frame = pd.read_parquet(
        provenance_path,
        columns=["source_index", "blob_offset", "blob_length"],
    )
    row = frame[frame.source_index.astype(int).eq(2501)].iloc[0]
    with blob_path.open("rb") as handle:
        handle.seek(int(row.blob_offset))
        cif = zlib.decompress(handle.read(int(row.blob_length))).decode()
    structure, route = parse_cif_structure(cif)
    assert len(structure) == 224
    assert route == "tolerant"
