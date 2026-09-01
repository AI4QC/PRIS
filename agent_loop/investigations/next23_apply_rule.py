#!/usr/bin/env python3
"""Apply the frozen NEXT23 law without accepting any endpoint or label input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_relaxation_rule import (
    BASE_TERMS,
    PROTOCOL as RULE_PROTOCOL,
    score_candidate,
)


PROTOCOL = "2026-08-02-next23-label-free-frozen-rule-application-v1"
COHORT_PROTOCOL = "2026-08-02-next23-wbm-relaxation-change-holdout-v1"
PREDICTIONS_NAME = "next23_relaxation_change_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
FORBIDDEN_FEATURE_COLUMN_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
    "endpoint",
    "label",
    "target",
)


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_manifest_output_hash(
    manifest: Mapping[str, object], artifact: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(artifact.name) != _sha256(
        artifact
    ):
        raise ValueError(f"{role} output hash differs")


def _validate_feature_table(
    *,
    source: str,
    feature_path: Path,
    manifest_path: Path,
    metadata_hash: str,
    expected_ids: Sequence[str],
) -> pd.DataFrame:
    manifest = _read_json(manifest_path, role=f"{source} feature manifest")
    _validate_manifest_output_hash(manifest, feature_path, role=source)
    if (
        manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("coordinates_or_cell_modified") is not False
    ):
        raise ValueError(f"{source} manifest crossed the no-DFT geometry contract")
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping) or inputs.get("metadata") != metadata_hash:
        raise ValueError(f"{source} metadata hash differs")
    frame = pd.read_parquet(feature_path)
    forbidden = [
        str(column)
        for column in frame.columns
        if any(
            token in str(column).lower()
            for token in FORBIDDEN_FEATURE_COLUMN_TOKENS
        )
    ]
    if forbidden:
        raise ValueError(f"{source} feature table crossed no-DFT contract: {forbidden}")
    if "material_id" not in frame or frame["material_id"].isna().any():
        raise ValueError(f"{source} feature table lacks material IDs")
    ids = frame["material_id"].astype(str)
    if ids.duplicated().any() or set(ids) != set(expected_ids):
        raise ValueError(f"{source} feature IDs differ from cohort metadata")
    frame = frame.copy()
    frame["material_id"] = ids
    return frame


def _validated_law(
    law_path: Path, manifest_path: Path
) -> tuple[dict[str, object], tuple[str, ...], float]:
    manifest = _read_json(manifest_path, role="rule manifest")
    law = _read_json(law_path, role="frozen rule")
    _validate_manifest_output_hash(manifest, law_path, role="frozen rule")
    if (
        manifest.get("protocol") != RULE_PROTOCOL
        or manifest.get("blind_labels_opened") is not False
        or law.get("protocol") != RULE_PROTOCOL
        or law.get("blind_labels_opened") is not False
        or law.get("eligible") is not True
    ):
        raise ValueError("frozen rule is ineligible or crossed the blind-label boundary")
    selected_terms = law.get("selected_terms")
    threshold = law.get("threshold")
    parameters = law.get("base_parameters")
    if (
        not isinstance(selected_terms, list)
        or not selected_terms
        or len(set(selected_terms)) != len(selected_terms)
        or not isinstance(parameters, Mapping)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("frozen rule parameters are invalid")
    for key in selected_terms:
        if key not in BASE_TERMS or key not in parameters:
            raise ValueError(f"unknown frozen term: {key}")
        expected = BASE_TERMS[key]
        value = parameters[key]
        if (
            not isinstance(value, Mapping)
            or value.get("source") != expected.source
            or value.get("column") != expected.column
            or value.get("direction") != expected.direction
            or not isinstance(value.get("median"), (int, float))
            or not math.isfinite(float(value["median"]))
            or not isinstance(value.get("scale_iqr"), (int, float))
            or not math.isfinite(float(value["scale_iqr"]))
            or float(value["scale_iqr"]) <= 0.0
        ):
            raise ValueError(f"frozen term {key} differs from the catalogue")
    return law, tuple(selected_terms), float(threshold)


def apply_frozen_rule(
    *,
    frozen_rule_path: Path,
    rule_manifest_path: Path,
    cohort_metadata_path: Path,
    cohort_manifest_path: Path,
    sivr_features_path: Path,
    sivr_manifest_path: Path,
    madelung_features_path: Path,
    madelung_manifest_path: Path,
    scbve_features_path: Path,
    scbve_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Score a cohort and publish immutable predictions before label opening."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "rule_manifest": Path(rule_manifest_path).resolve(),
        "cohort_metadata": Path(cohort_metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "sivr_features": Path(sivr_features_path).resolve(),
        "sivr_manifest": Path(sivr_manifest_path).resolve(),
        "madelung_features": Path(madelung_features_path).resolve(),
        "madelung_manifest": Path(madelung_manifest_path).resolve(),
        "scbve_features": Path(scbve_features_path).resolve(),
        "scbve_manifest": Path(scbve_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    law, selected_terms, threshold = _validated_law(
        paths["frozen_rule"], paths["rule_manifest"]
    )

    cohort_manifest = _read_json(paths["cohort_manifest"], role="cohort manifest")
    _validate_manifest_output_hash(
        cohort_manifest, paths["cohort_metadata"], role="cohort metadata"
    )
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("cohort manifest crossed the label-free boundary")
    metadata = pd.read_parquet(paths["cohort_metadata"])
    if "material_id" not in metadata or metadata["material_id"].isna().any():
        raise ValueError("cohort metadata lacks material IDs")
    ids = metadata["material_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("cohort metadata IDs must be unique")
    metadata_hash = input_hashes["cohort_metadata"]
    feature_tables = {
        source: _validate_feature_table(
            source=source,
            feature_path=paths[f"{source}_features"],
            manifest_path=paths[f"{source}_manifest"],
            metadata_hash=metadata_hash,
            expected_ids=ids.tolist(),
        )
        for source in ("sivr", "madelung", "scbve")
    }
    features = pd.DataFrame({"material_id": ids})
    for source in ("sivr", "madelung", "scbve"):
        columns = [
            BASE_TERMS[key].column
            for key in selected_terms
            if BASE_TERMS[key].source == source
        ]
        if not columns:
            continue
        missing = sorted(set(columns) - set(feature_tables[source].columns))
        if missing:
            raise ValueError(f"{source} features lack selected terms: {missing}")
        subset = feature_tables[source].loc[:, ["material_id", *columns]]
        features = features.merge(
            subset, on="material_id", how="left", validate="one_to_one"
        )
    score, support = score_candidate(
        features,
        selected_terms,
        law["base_parameters"],
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
        "src/next23_apply_rule.py": Path(__file__).resolve(),
        "src/next23_relaxation_rule.py": repository_root
        / "src/next23_relaxation_rule.py",
    }
    source_hashes = {
        relative: _sha256(path) for relative, path in source_paths.items()
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "frozen_label_free_analytic_application",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": law["selected_candidate"],
        "selected_terms": list(selected_terms),
        "threshold": threshold,
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
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
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
            raise ValueError("prediction prepublication validation failed")
        manifest["outputs_sha256"] = {
            PREDICTIONS_NAME: _sha256(predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before prediction publication")
        for relative, path in source_paths.items():
            if _sha256(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before prediction publication")
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
    parser.add_argument("--madelung-features", required=True, type=Path)
    parser.add_argument("--madelung-manifest", required=True, type=Path)
    parser.add_argument("--scbve-features", required=True, type=Path)
    parser.add_argument("--scbve-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    apply_frozen_rule(
        frozen_rule_path=args.frozen_rule,
        rule_manifest_path=args.rule_manifest,
        cohort_metadata_path=args.cohort_metadata,
        cohort_manifest_path=args.cohort_manifest,
        sivr_features_path=args.sivr_features,
        sivr_manifest_path=args.sivr_manifest,
        madelung_features_path=args.madelung_features,
        madelung_manifest_path=args.madelung_manifest,
        scbve_features_path=args.scbve_features,
        scbve_manifest_path=args.scbve_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MANIFEST_NAME", "PREDICTIONS_NAME", "apply_frozen_rule", "main"]
