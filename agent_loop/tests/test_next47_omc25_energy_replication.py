from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.next47_omc25_energy_replication import (
    EXPECTED_NEXT31_RULE_SHA256,
    extend_refcode_exclusions,
    freeze_replication_protocol,
    summarize_replications,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_protocol_binds_tail_selection_rule_and_unchanged_gates(
    tmp_path: Path,
) -> None:
    rule = tmp_path / "NEXT31_FROZEN_ENERGY_RULE.json"
    rule.write_bytes(b"frozen rule\n")
    exclusions = tmp_path / "refcodes.txt"
    exclusions.write_text("A\nB\n", encoding="utf-8")
    output = tmp_path / "protocol"

    manifest = freeze_replication_protocol(
        frozen_rule_path=rule,
        exclusion_path=exclusions,
        output_dir=output,
        expected_rule_sha256=_sha256(rule),
        expected_exclusion_sha256=_sha256(exclusions),
        expected_exclusion_count=2,
    )

    protocol = json.loads((output / "NEXT47_REPLICATION_PROTOCOL.json").read_text())
    assert protocol["archive_selection"] == {"skip_main": 24, "take_main": 16}
    assert protocol["labels_opened"] is False
    assert protocol["thresholds_refit"] is False
    assert protocol["frozen_rule"]["sha256"] == _sha256(rule)
    assert protocol["gates"]["auc_energy_positive_at_least"] == 0.85
    assert manifest["labels_opened"] is False
    with pytest.raises(FileExistsError):
        freeze_replication_protocol(
            frozen_rule_path=rule,
            exclusion_path=exclusions,
            output_dir=output,
            expected_rule_sha256=_sha256(rule),
            expected_exclusion_sha256=_sha256(exclusions),
            expected_exclusion_count=2,
        )


def test_production_protocol_rejects_any_different_next31_rule(tmp_path: Path) -> None:
    rule = tmp_path / "rule.json"
    rule.write_text("{}\n", encoding="utf-8")
    exclusions = tmp_path / "refcodes.txt"
    exclusions.write_text("A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rule hash"):
        freeze_replication_protocol(
            frozen_rule_path=rule,
            exclusion_path=exclusions,
            output_dir=tmp_path / "protocol",
            expected_rule_sha256=EXPECTED_NEXT31_RULE_SHA256,
            expected_exclusion_sha256=_sha256(exclusions),
            expected_exclusion_count=1,
        )


def test_extend_refcodes_is_label_free_sorted_and_no_replace(tmp_path: Path) -> None:
    previous = tmp_path / "previous.txt"
    previous.write_text("B\nA\n", encoding="utf-8")
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {"material_id": ["m1", "m2"], "csd_refcode": ["C", "A"]}
    ).to_parquet(metadata, index=False)
    metadata_manifest = tmp_path / "MANIFEST.json"
    metadata_manifest.write_text(
        json.dumps(
            {
                "labels_opened": False,
                "endpoint_numeric_fields_parsed": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {metadata.name: _sha256(metadata)},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "extended"

    manifest = extend_refcode_exclusions(
        previous_refcodes_path=previous,
        metadata_path=metadata,
        metadata_manifest_path=metadata_manifest,
        source_shard="data0001",
        output_dir=output,
    )

    assert (output / "refcodes.txt").read_text() == "A\nB\nC\n"
    assert manifest["labels_opened"] is False
    assert manifest["counts"] == {"previous": 2, "new_rows": 2, "union": 3}
    with pytest.raises(FileExistsError):
        extend_refcode_exclusions(
            previous_refcodes_path=previous,
            metadata_path=metadata,
            metadata_manifest_path=metadata_manifest,
            source_shard="data0001",
            output_dir=output,
        )


def test_summarize_replications_recomputes_pooled_metrics_without_refit(
    tmp_path: Path,
) -> None:
    def joined(path: Path, prefix: str, energies: list[float]) -> None:
        pd.DataFrame(
            {
                "material_id": [f"{prefix}-{i}" for i in range(len(energies))],
                "source_shard": [prefix] * len(energies),
                "analytic_supported": [True] * len(energies),
                "next31_risk_score": [0.0, 1.0, 2.0, 3.0],
                "reject": [False, False, False, True],
                "energy_drop_pa": energies,
            }
        ).to_parquet(path, index=False)

    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    joined(first, "first", [0.0, 0.01, 0.04, 0.08])
    joined(second, "second", [0.0, 0.02, 0.05, 0.09])
    output = tmp_path / "summary"

    result = summarize_replications(
        first_joined_path=first,
        second_joined_path=second,
        frozen_rule_path=tmp_path / "missing-is-allowed-in-unit-test.json",
        output_dir=output,
        expected_rule_sha256="unit-test-rule",
        verify_rule_file=False,
    )

    assert result["thresholds_refit"] is False
    assert result["pooled"]["counts"]["rows"] == 8
    assert result["second_confirmation_independent"] is True
    assert result["pooled_is_descriptive_only"] is True
    assert (output / "NEXT47_REPLICATION_SUMMARY.json").is_file()
