from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


VERDICT_COLUMNS = ["L2_verdict", "L4_verdict"] + [
    f"D{index}_verdict" for index in range(1, 9)
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _refresh_score_summary(score_path: Path, *, common_rows: int) -> None:
    _write_json(
        score_path.with_suffix(".summary.json"),
        {
            "common_rows": common_rows,
            "a_jang_rows_without_b": 1,
            "output": str(score_path.resolve()),
            "output_bytes": score_path.stat().st_size,
            "output_sha256": _sha256(score_path),
            "source_sha256": {"CLscore_A": "a", "CLscore_B": "b", "CLscore_jang": "j"},
        },
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pool_orig_index = np.asarray(
        [0, 1, 2, 3, 4, 5, 6_120_140, 6_120_141, 6_120_142, 6_120_143, 6_120_144, 6_120_145],
        dtype=np.int64,
    )
    score_path = tmp_path / "full_clscores_common.parquet"
    score = pd.DataFrame(
        {
            "orig_index": pool_orig_index[:10],
            "CLscore_A": np.asarray([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "CLscore_B": np.asarray([0, 0.02, 0.09, 0.21, 0.31, 0.39, 0.52, 0.59, 0.72, 0.79], dtype=np.float32),
            "CLscore_jang": np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        }
    )
    score.to_parquet(score_path, index=False)
    _refresh_score_summary(score_path, common_rows=10)

    run_root = tmp_path / "run"
    result_dir = run_root / "results"
    result_dir.mkdir(parents=True)
    input_manifest: dict[str, object] = {
        "schema_version": 1,
        "chunk_size": 6,
        "total_rows": 12,
        "task_count": 2,
        "sources": [
            {"split": "train", "rows": 6, "sha256": "train-source-sha"},
            {"split": "val", "rows": 6, "sha256": "val-source-sha"},
        ],
        "implementation": [{"path": "frozen.py", "sha256": "code"}],
        "bvparm": {"path": "bvparm2020.cif", "sha256": "bv"},
        "verdict_policy": {
            "pass": "retain",
            "explicit_violation": "remove",
            "no_verdict": "retain_and_report_separately",
        },
    }
    encoded = json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode()
    input_manifest["input_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    input_manifest_path = run_root / "input_manifest.json"
    _write_json(input_manifest_path, input_manifest)
    (run_root / "input_manifest.sha256").write_text(
        f"{_sha256(input_manifest_path)}  input_manifest.json\n"
    )

    states = np.asarray(["pass", "explicit_violation", "no_verdict"] * 4, dtype=object)
    shard_records = []
    for task_id, (start, stop) in enumerate(((0, 6), (6, 12))):
        rows = pd.DataFrame(
            {
                "pool_row": np.arange(start, stop, dtype=np.int64),
                "orig_index": pool_orig_index[start:stop],
                "pool_id": [f"u{value}" for value in pool_orig_index[start:stop]],
                "material_id": [f"m{index}" for index in range(start, stop)],
                "structure_formula": [f"X{index}Y" for index in range(start, stop)],
                "chemical_system": ["X-Y"] * (stop - start),
                "source_split": ["train" if start == 0 else "val"] * (stop - start),
                "n_elements": [2, 2, 2, 3, 3, 3] if start == 0 else [2, 2, 2, 3, 3, 3],
                "n_sites": np.arange(start + 2, stop + 2, dtype=np.int64),
                "parse_ok": [True] * (stop - start),
                "cif_parse_route": ["default"] * (stop - start),
            }
        )
        hashes = []
        for index in range(start, stop):
            # Rows 0 and 1 deliberately share one exact CIF hash.
            identity = "duplicate-0-1" if index in {0, 1} else f"cif-{index}"
            hashes.append(hashlib.sha256(identity.encode()).hexdigest())
        rows["cif_sha256"] = hashes
        for offset, column in enumerate(VERDICT_COLUMNS):
            rows[column] = states[(np.arange(start, stop) + offset) % len(states)]
        # Ensure the highest A/B score is a disagreement example.
        rows.loc[rows.pool_row.eq(9), "L4_verdict"] = "explicit_violation"
        output = result_dir / f"part-{task_id:05d}.parquet"
        rows.to_parquet(output, index=False)
        shard = {
            "input_fingerprint": input_manifest["input_fingerprint"],
            "task_id": task_id,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "output": str(output.resolve()),
            "output_bytes": output.stat().st_size,
            "output_sha256": _sha256(output),
            "parse_failures": 0,
        }
        _write_json(output.with_suffix(".manifest.json"), shard)
        shard_records.append(
            {
                "task_id": task_id,
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "output_sha256": shard["output_sha256"],
            }
        )
    shard_digest = hashlib.sha256(
        json.dumps(shard_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(
        result_dir / "_SUCCESS.json",
        {
            "schema_version": 1,
            "input_fingerprint": input_manifest["input_fingerprint"],
            "total_rows": 12,
            "task_count": 2,
            "parse_failures": 0,
            "tolerant_parses": 0,
            "shard_manifest_digest": shard_digest,
            "source_sha256": {
                "train": "train-source-sha",
                "val": "val-source-sha",
            },
        },
    )
    return score_path, result_dir, input_manifest_path


def _refresh_success(result_dir: Path, input_manifest_path: Path) -> None:
    manifest = json.loads(input_manifest_path.read_text())
    records = []
    parse_failures = 0
    tolerant_parses = 0
    for path in sorted(result_dir.glob("part-*.parquet")):
        shard = json.loads(path.with_suffix(".manifest.json").read_text())
        records.append(
            {
                key: shard[key]
                for key in ("task_id", "start", "stop", "rows", "output_sha256")
            }
        )
        parse_failures += int(shard.get("parse_failures", 0))
        tolerant_parses += int(shard.get("tolerant_parses", 0))
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(
        result_dir / "_SUCCESS.json",
        {
            "schema_version": 1,
            "input_fingerprint": manifest["input_fingerprint"],
            "total_rows": manifest["total_rows"],
            "task_count": manifest["task_count"],
            "parse_failures": parse_failures,
            "tolerant_parses": tolerant_parses,
            "shard_manifest_digest": digest,
            "source_sha256": {
                source["split"]: source["sha256"] for source in manifest["sources"]
            },
        },
    )


def test_equal_frequency_deciles_are_deterministic_and_report_split_ties() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        assign_equal_frequency_bands,
    )

    scores = np.asarray([0.0] * 7 + [0.5, 0.8, 1.0], dtype=np.float32)
    orig_index = np.asarray([9, 4, 7, 1, 5, 3, 8, 0, 2, 6], dtype=np.int64)

    first = assign_equal_frequency_bands(scores, orig_index, n_bands=3)
    second = assign_equal_frequency_bands(scores, orig_index, n_bands=3)

    assert np.array_equal(first.band, second.band)
    counts = np.bincount(first.band, minlength=4)[1:]
    assert counts.max() - counts.min() <= 1
    assert first.boundaries.ties_split_across_boundary.any()
    assert first.band[orig_index.argmin()] == 3


def test_stratified_spearman_excludes_constant_layers() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        stratified_spearman,
    )

    score = np.asarray([0.1, 0.2, 0.3] + list(np.linspace(0, 1, 100)))
    outcome = np.asarray([0.0, 1.0, 2.0] + [0.0] * 100)
    strata = np.asarray([0, 0, 0] + [1] * 100)

    got, per_stratum = stratified_spearman(
        score, outcome, strata, min_group_rows=3
    )

    assert got["n"] == 3
    assert got["n_strata"] == 1
    assert got["excluded_constant_rows"] == 100
    assert got["spearman_rho"] == pytest.approx(1.0)
    constant = per_stratum.loc[per_stratum.stratum.eq(1)].iloc[0]
    assert constant.status == "constant_outcome"
    assert pd.isna(constant.spearman_rho)


def test_stratified_spearman_returns_nan_when_all_layers_constant() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        stratified_spearman,
    )

    got, per_stratum = stratified_spearman(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([0, 0, 0]),
        min_group_rows=3,
    )

    assert got["n"] == 0
    assert got["n_strata"] == 0
    assert got["excluded_constant_rows"] == 3
    assert np.isnan(got["spearman_rho"])
    assert per_stratum.status.tolist() == ["constant_outcome"]


def test_binary_outcomes_receive_true_midrank_not_raw_zero_one() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        _discrete_rank,
    )

    ranked, unique = _discrete_rank(np.asarray([0, 0, 1], dtype=np.uint8))

    assert unique == 2
    assert ranked.tolist() == pytest.approx([1.5, 1.5, 3.0])


def test_batched_stratified_binary_rho_matches_reference_for_unequal_layers(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        _batched_stratified_rows,
        load_pris_common_support,
        load_score_table,
        stratified_spearman,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    data = load_pris_common_support(
        result_dir,
        input_manifest,
        load_score_table(score_path, expected_rows=10),
        expected_pool_rows=12,
    )
    outcome = (data.verdict_codes["L4"] == 1).astype(np.uint8)
    strata = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)

    reference, _ = stratified_spearman(
        data.scores["CLscore_A"], outcome, strata, min_group_rows=2
    )
    batch = _batched_stratified_rows(
        data,
        {"L4:explicit_violation": outcome},
        outcome_names=("L4:explicit_violation",),
        strata=strata,
        scheme="test",
        min_group_rows=2,
    )
    got = next(row for row in batch if row["score_model"] == "CLscore_A")

    assert got["n"] == reference["n"]
    assert got["excluded_constant_rows"] == reference["excluded_constant_rows"]
    assert got["spearman_rho"] == pytest.approx(reference["spearman_rho"])


def test_full_pool_provenance_boundary_is_frozen() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        ELEMENTA_START,
        derive_pool_provenance,
    )

    assert ELEMENTA_START == 6_120_140
    got = derive_pool_provenance(
        np.asarray([0, 6_120_139, 6_120_140, 8_000_000], dtype=np.int64)
    )
    assert got.tolist() == ["lemat", "lemat", "elementa", "elementa"]


@pytest.mark.parametrize("bad_kind", ["duplicate", "missing", "sha"])
def test_score_table_fails_closed_on_duplicate_missing_or_bad_sha(
    tmp_path: Path, bad_kind: str
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_score_table,
    )

    score_path, _, _ = _fixture(tmp_path)
    frame = pd.read_parquet(score_path)
    if bad_kind == "duplicate":
        frame.loc[1, "orig_index"] = frame.loc[0, "orig_index"]
        frame.to_parquet(score_path, index=False)
        _refresh_score_summary(score_path, common_rows=10)
        match = "orig_index.*unique"
    elif bad_kind == "missing":
        frame.loc[1, "CLscore_B"] = np.nan
        frame.to_parquet(score_path, index=False)
        _refresh_score_summary(score_path, common_rows=10)
        match = "scores.*finite"
    else:
        score_path.write_bytes(score_path.read_bytes() + b"tampered")
        match = "score Parquet SHA-256"

    with pytest.raises(ValueError, match=match):
        load_score_table(score_path, expected_rows=10)


def test_production_score_identity_freezes_all_three_raw_sources() -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        FROZEN_SCORE_OUTPUT_SHA256,
        FROZEN_SCORE_SOURCE_ROWS,
        FROZEN_SCORE_SOURCE_SHA256,
        _validate_frozen_score_identity,
    )

    summary = {
        "source_rows": FROZEN_SCORE_SOURCE_ROWS,
        "source_sha256": FROZEN_SCORE_SOURCE_SHA256,
        "common_rows": 8_108_676,
        "a_jang_rows_without_b": 1_221,
        "output_sha256": FROZEN_SCORE_OUTPUT_SHA256,
    }
    _validate_frozen_score_identity(
        summary, actual_output_sha256=FROZEN_SCORE_OUTPUT_SHA256
    )
    bad = json.loads(json.dumps(summary))
    bad["source_sha256"]["CLscore_A"] = "0" * 64
    with pytest.raises(ValueError, match="frozen raw-score SHA-256"):
        _validate_frozen_score_identity(
            bad, actual_output_sha256=FROZEN_SCORE_OUTPUT_SHA256
        )


def test_pris_merge_checks_complete_pool_and_common_score_support(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    scores = load_score_table(score_path, expected_rows=10)
    got = load_pris_common_support(
        result_dir,
        input_manifest,
        scores,
        expected_pool_rows=12,
    )

    assert len(got.orig_index) == 10
    assert got.integrity["pool_rows"] == 12
    assert got.integrity["score_common_rows"] == 10
    assert got.integrity["pool_rows_without_all_three_scores"] == 2
    assert got.integrity["full_pool_unique_cifs"] == 11
    assert got.integrity["full_pool_duplicate_cif_groups"] == 1
    assert got.integrity["full_pool_duplicate_extra_rows"] == 1
    assert got.integrity["common_support_unique_cifs"] == 9
    assert got.unique_cif_weight.sum() == pytest.approx(9.0)
    assert got.unique_cif_weight[:2].tolist() == pytest.approx([0.5, 0.5])
    assert set(got.verdict_codes) == {
        "L2",
        "L4",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
    }
    assert all(set(values).issubset({0, 1, 2}) for values in got.verdict_codes.values())


def test_pris_merge_rejects_duplicate_orig_index_after_valid_shard_rewrite(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    output = sorted(result_dir.glob("part-*.parquet"))[1]
    frame = pd.read_parquet(output)
    frame.loc[0, "orig_index"] = 0
    frame.loc[0, "pool_id"] = "u0"
    frame.to_parquet(output, index=False)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["output_bytes"] = output.stat().st_size
    manifest["output_sha256"] = _sha256(output)
    _write_json(manifest_path, manifest)
    _refresh_success(result_dir, input_manifest)

    with pytest.raises(ValueError, match="orig_index is not unique across the complete pool"):
        load_pris_common_support(
            result_dir,
            input_manifest,
            load_score_table(score_path, expected_rows=10),
            expected_pool_rows=12,
        )


@pytest.mark.parametrize("bad_kind", ["missing", "digest", "source_sha"])
def test_pris_merge_requires_matching_full_pool_success_gate(
    tmp_path: Path, bad_kind: str
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    success_path = result_dir / "_SUCCESS.json"
    if bad_kind == "missing":
        success_path.unlink()
        match = "missing full-pool _SUCCESS"
    else:
        success = json.loads(success_path.read_text())
        if bad_kind == "digest":
            success["shard_manifest_digest"] = "0" * 64
            match = "shard manifest digest"
        else:
            success["source_sha256"]["train"] = "changed"
            match = "source SHA-256"
        _write_json(success_path, success)

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        load_pris_common_support(
            result_dir,
            input_manifest,
            load_score_table(score_path, expected_rows=10),
            expected_pool_rows=12,
        )


@pytest.mark.parametrize(
    "bad_kind",
    ["fractional_pool_row", "string_parse_ok", "n_sites_text", "n_elements_inf"],
)
def test_pris_merge_checks_arrow_types_before_coercion(
    tmp_path: Path, bad_kind: str
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    output = sorted(result_dir.glob("part-*.parquet"))[0]
    frame = pd.read_parquet(output)
    if bad_kind == "fractional_pool_row":
        frame["pool_row"] = frame.pool_row.astype(float) + 0.1
        match = "pool_row.*integer"
    elif bad_kind == "string_parse_ok":
        frame["parse_ok"] = frame.parse_ok.map({True: "True", False: "False"})
        match = "parse_ok.*boolean"
    elif bad_kind == "n_sites_text":
        frame["n_sites"] = "not-a-number"
        match = "n_sites.*numeric"
    else:
        frame["n_elements"] = frame.n_elements.astype(float)
        frame.loc[0, "n_elements"] = np.inf
        match = "n_elements.*finite"
    frame.to_parquet(output, index=False)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["output_bytes"] = output.stat().st_size
    manifest["output_sha256"] = _sha256(output)
    _write_json(manifest_path, manifest)
    _refresh_success(result_dir, input_manifest)

    with pytest.raises(ValueError, match=match):
        load_pris_common_support(
            result_dir,
            input_manifest,
            load_score_table(score_path, expected_rows=10),
            expected_pool_rows=12,
        )


def test_score_agreement_disables_decile_overlap_when_ties_cross_boundaries(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        build_score_agreement,
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    score = pd.read_parquet(score_path)
    score["CLscore_A"] = [0.0] * 7 + [0.1, 0.2, 0.3]
    score.to_parquet(score_path, index=False)
    _refresh_score_summary(score_path, common_rows=10)
    table = load_score_table(score_path, expected_rows=10)
    data = load_pris_common_support(
        result_dir, input_manifest, table, expected_pool_rows=12
    )

    agreement = build_score_agreement(data, n_bands=2).set_index("pair")
    for pair in ("CLscore_A__CLscore_B", "CLscore_A__CLscore_jang"):
        assert agreement.loc[pair, "decile_overlap_status"] == (
            "undefined_due_to_split_score_ties"
        )
        assert pd.isna(agreement.loc[pair, "same_decile_rate"])
        assert pd.isna(agreement.loc[pair, "bottom_decile_jaccard"])


@pytest.mark.parametrize("bad_kind", ["gap", "sha", "input_manifest_sha"])
def test_pris_merge_rejects_gap_or_manifest_sha_tampering(
    tmp_path: Path, bad_kind: str
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        load_pris_common_support,
        load_score_table,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    if bad_kind == "input_manifest_sha":
        input_manifest.with_suffix(".sha256").write_text("0" * 64 + "\n")
        match = "input manifest SHA-256"
    else:
        output = sorted(result_dir.glob("part-*.parquet"))[1]
        if bad_kind == "gap":
            manifest_path = output.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["start"] = 7
            _write_json(manifest_path, manifest)
            match = "do not tile pool_row"
        else:
            output.write_bytes(output.read_bytes() + b"tampered")
            match = "shard Parquet SHA-256"

    with pytest.raises(ValueError, match=match):
        load_pris_common_support(
            result_dir,
            input_manifest,
            load_score_table(score_path, expected_rows=10),
            expected_pool_rows=12,
        )


def test_full_analysis_emits_three_state_source_and_unique_cif_outputs(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.analyze_full_pool import (
        analyze_full_pool,
    )

    score_path, result_dir, input_manifest = _fixture(tmp_path)
    output_dir = tmp_path / "analysis"
    summary = analyze_full_pool(
        score_path=score_path,
        pris_result_dir=result_dir,
        input_manifest_path=input_manifest,
        output_dir=output_dir,
        expected_score_rows=10,
        expected_pool_rows=12,
        n_bands=2,
        min_group_rows=2,
    )

    expected_outputs = {
        "score_deciles.csv",
        "score_deciles_by_provenance.csv",
        "score_deciles_unique_cif_weighted.csv",
        "unique_cif_sensitivity.csv",
        "score_correlations.csv",
        "stratified_correlations.csv",
        "score_agreement.csv",
        "discordant_examples.csv",
        "integrity.json",
        "result_summary.json",
    }
    assert expected_outputs == {path.name for path in output_dir.iterdir()}

    deciles = pd.read_csv(output_dir / "score_deciles.csv")
    assert set(deciles.score_model) == {"CLscore_A", "CLscore_B", "CLscore_jang"}
    assert deciles.groupby("score_model").n.sum().eq(10).all()
    assert (
        deciles.groupby("score_model").n.agg(lambda value: value.max() - value.min())
        <= 1
    ).all()
    for rule in ["L2", "L4", *[f"D{index}" for index in range(1, 9)]]:
        rates = deciles[
            [
                f"{rule}_pass_rate",
                f"{rule}_explicit_violation_rate",
                f"{rule}_no_verdict_rate",
            ]
        ].sum(axis=1)
        assert rates.tolist() == pytest.approx([1.0] * len(rates))
    assert {
        "n_sites_mean",
        "n_sites_median",
        "n_elements_mean",
        "n_elements_median",
        "ties_split_across_boundary",
    }.issubset(deciles.columns)

    by_source = pd.read_csv(output_dir / "score_deciles_by_provenance.csv")
    assert set(by_source.provenance) == {"lemat", "elementa"}
    assert by_source.groupby(["score_model", "provenance"]).n.sum().to_dict() == {
        ("CLscore_A", "elementa"): 4,
        ("CLscore_A", "lemat"): 6,
        ("CLscore_B", "elementa"): 4,
        ("CLscore_B", "lemat"): 6,
        ("CLscore_jang", "elementa"): 4,
        ("CLscore_jang", "lemat"): 6,
    }

    weighted = pd.read_csv(output_dir / "score_deciles_unique_cif_weighted.csv")
    assert weighted.groupby("score_model").weight_sum.sum().tolist() == pytest.approx(
        [9.0, 9.0, 9.0]
    )
    assert {"L4_explicit_violation_rate", "D7_no_verdict_rate"}.issubset(
        weighted.columns
    )

    correlations = pd.read_csv(output_dir / "score_correlations.csv")
    assert {"global", "provenance=lemat", "provenance=elementa"}.issubset(
        set(correlations.scope)
    )
    assert {
        "L2:pass",
        "L2:explicit_violation",
        "L2:no_verdict",
        "D1_D8_explicit_violation_count",
    }.issubset(set(correlations.outcome))
    stratified = pd.read_csv(output_dir / "stratified_correlations.csv")
    assert {
        "provenance+n_elements+site_bin",
        "n_elements+site_bin",
    }.issubset(set(stratified.scheme))
    assert "excluded_constant_rows" in stratified.columns

    agreement = pd.read_csv(output_dir / "score_agreement.csv")
    assert len(agreement) == 3
    assert agreement.pair.is_unique
    examples = pd.read_csv(output_dir / "discordant_examples.csv")
    assert len(examples) == 6
    assert {"high_score_but_L4_explicit_violation", "low_score_but_L4_pass"} == set(
        examples.example_type
    )
    assert "cif" not in examples.columns
    assert "cif_sha256" in examples.columns

    assert summary["provenance"]["elementa_start_orig_index"] == 6_120_140
    assert summary["three_state_policy"]["no_verdict"] == "retained and reported separately"
