#!/usr/bin/env python3
"""Transport the frozen NEXT23 law to a generated x0 cohort without labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    validate_geometry_metadata,
)
from src.next23_apply_rule import (
    _json_bytes,
    _read_json,
    _validate_feature_table,
    _validate_manifest_output_hash,
    _validated_law,
)
from src.next23_relaxation_rule import BASE_TERMS, score_candidate
from src.next24_ssagen_holdout import PROTOCOL as COHORT_PROTOCOL


PROTOCOL = "2026-08-03-next24-label-free-transport-application-v1"
PREDICTIONS_NAME = "next24_generated_transport_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _exact_source_paths(
    selected_sources: Sequence[str],
    feature_paths: Mapping[str, Path],
    feature_manifest_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    expected = set(selected_sources)
    if set(feature_paths) != expected or set(feature_manifest_paths) != expected:
        raise ValueError(
            f"feature paths must contain exactly the selected sources: {sorted(expected)}"
        )
    features = {source: Path(feature_paths[source]).resolve() for source in expected}
    manifests = {
        source: Path(feature_manifest_paths[source]).resolve() for source in expected
    }
    for source in expected:
        if not features[source].is_file() or not manifests[source].is_file():
            raise FileNotFoundError(f"{source} feature input is not a file")
    return features, manifests


def apply_transport_rule(
    *,
    frozen_rule_path: Path,
    rule_manifest_path: Path,
    cohort_metadata_path: Path,
    cohort_manifest_path: Path,
    feature_paths: Mapping[str, Path],
    feature_manifest_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Seal predictions using only sources selected by the frozen law."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    core_paths = {
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "rule_manifest": Path(rule_manifest_path).resolve(),
        "cohort_metadata": Path(cohort_metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
    }
    for role, path in core_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    law, selected_terms, threshold = _validated_law(
        core_paths["frozen_rule"], core_paths["rule_manifest"]
    )
    selected_sources = tuple(
        sorted({BASE_TERMS[key].source for key in selected_terms})
    )
    features_by_source, manifests_by_source = _exact_source_paths(
        selected_sources, feature_paths, feature_manifest_paths
    )
    input_paths = {
        **core_paths,
        **{
            f"{source}_features": features_by_source[source]
            for source in selected_sources
        },
        **{
            f"{source}_manifest": manifests_by_source[source]
            for source in selected_sources
        },
    }
    input_hashes = {role: _sha256(path) for role, path in input_paths.items()}

    cohort_manifest = _read_json(
        core_paths["cohort_manifest"], role="NEXT24 cohort manifest"
    )
    _validate_manifest_output_hash(
        cohort_manifest, core_paths["cohort_metadata"], role="NEXT24 cohort"
    )
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("endpoint_artifacts_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("NEXT24 cohort crossed the label-free boundary")
    metadata = validate_geometry_metadata(
        pd.read_parquet(core_paths["cohort_metadata"])
    )
    ids = metadata["material_id"].astype(str)
    metadata_hash = input_hashes["cohort_metadata"]
    feature_tables = {
        source: _validate_feature_table(
            source=source,
            feature_path=features_by_source[source],
            manifest_path=manifests_by_source[source],
            metadata_hash=metadata_hash,
            expected_ids=ids.tolist(),
        )
        for source in selected_sources
    }

    combined = pd.DataFrame({"material_id": ids})
    for source in selected_sources:
        columns = [
            BASE_TERMS[key].column
            for key in selected_terms
            if BASE_TERMS[key].source == source
        ]
        missing = sorted(set(columns) - set(feature_tables[source].columns))
        if missing:
            raise ValueError(f"{source} features lack selected terms: {missing}")
        combined = combined.merge(
            feature_tables[source].loc[:, ["material_id", *columns]],
            on="material_id",
            how="left",
            validate="one_to_one",
        )
    score, support = score_candidate(
        combined, selected_terms, law["base_parameters"]
    )
    reject = support & (score >= threshold)
    predictions = pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": support,
            "next23_risk_score": score,
            "reject": reject,
            "input_role": "unrelaxed_x0_geometry_only",
        }
    )

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next24_apply_rule.py": Path(__file__).resolve(),
        "src/next23_apply_rule.py": repository_root / "src/next23_apply_rule.py",
        "src/next23_relaxation_rule.py": repository_root
        / "src/next23_relaxation_rule.py",
    }
    source_hashes = {
        relative: _sha256(path) for relative, path in source_paths.items()
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "frozen_cross_source_label_free_transport",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "transported_rule_protocol": law["protocol"],
        "selected_candidate": law["selected_candidate"],
        "selected_terms": list(selected_terms),
        "selected_feature_sources": list(selected_sources),
        "threshold": threshold,
        "thresholds_refit": False,
        "formula_or_parameters_changed": False,
        "input_role": "unrelaxed_x0_geometry_only",
        "blind_labels_opened": False,
        "endpoint_fields_read": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "missing_policy": "fail_open_do_not_reject",
        "counts": {
            "rows": len(predictions),
            "supported": int(support.sum()),
            "rejected": int(reject.sum()),
        },
        "inputs_sha256": {
            role: {"path": str(path), "sha256": input_hashes[role]}
            for role, path in input_paths.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(
            cohort_manifest.get("production_protocol_eligible") is True
        ),
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        predictions_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(predictions_path, index=False)
        if not pd.read_parquet(predictions_path).equals(predictions):
            raise ValueError("NEXT24 prediction prepublication validation failed")
        manifest["outputs_sha256"] = {
            PREDICTIONS_NAME: _sha256(predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in input_paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-rule", required=True, type=Path)
    parser.add_argument("--rule-manifest", required=True, type=Path)
    parser.add_argument("--cohort-metadata", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--sivr-features", required=True, type=Path)
    parser.add_argument("--sivr-manifest", required=True, type=Path)
    parser.add_argument("--scbve-features", required=True, type=Path)
    parser.add_argument("--scbve-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    apply_transport_rule(
        frozen_rule_path=arguments.frozen_rule,
        rule_manifest_path=arguments.rule_manifest,
        cohort_metadata_path=arguments.cohort_metadata,
        cohort_manifest_path=arguments.cohort_manifest,
        feature_paths={
            "sivr": arguments.sivr_features,
            "scbve": arguments.scbve_features,
        },
        feature_manifest_paths={
            "sivr": arguments.sivr_manifest,
            "scbve": arguments.scbve_manifest,
        },
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MANIFEST_NAME", "PREDICTIONS_NAME", "apply_transport_rule", "main"]
