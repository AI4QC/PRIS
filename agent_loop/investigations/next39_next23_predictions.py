#!/usr/bin/env python3
"""Apply the frozen NEXT23 B+E law to NEXT39 step-0 structures only."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next12_pauling_controls import (
    DECISIONS,
    RULES,
    _classical_features,
    _combined_decision,
    _rule_decision,
)
from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next32_inorganic_response_features import compute_inorganic_response_features
from src.next39_omat24_trajectory_cohort import (
    COHORT_NAME,
    GEOMETRY_NAME,
    PROTOCOL as COHORT_PROTOCOL,
)


PROTOCOL = "2026-08-03-next39-next23-frozen-predictions-v1"
RULE_PROTOCOL = "2026-08-02-next23-relaxation-change-rule-freeze-v1"
PREDICTIONS_NAME = "next39_next23_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_RULE_SHA256 = "079c36190d82db4e14b8b8bf8a7db8e3073e85a03003688052ac1bdc58939c89"
TERM_COLUMNS = (
    "voronoi_q0__sivr_cell_anisotropy",
    "scbv_vector_asymmetry_rms",
)
FeatureCalculator = Callable[
    [Atoms], tuple[Mapping[str, object] | None, str | None]
]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} output hash differs")


def _load_step0_cohort(
    *, metadata_path: Path, frames_zip_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms], dict[str, object]]:
    manifest = _strict_json(manifest_path, role="NEXT39 cohort manifest")
    if (
        manifest.get("protocol") != COHORT_PROTOCOL
        or manifest.get("output_role") != "step0_unrelaxed_x0_geometry_only"
        or manifest.get("later_geometry_opened") is not False
        or manifest.get("dft_numeric_fields_parsed") is not False
        or manifest.get("dft_values_read") is not False
        or manifest.get("selection_uses_later_geometry") is not False
        or manifest.get("selection_uses_dft_values") is not False
    ):
        raise ValueError("NEXT39 cohort crossed the prediction boundary")
    if metadata_path.name != COHORT_NAME or frames_zip_path.name != GEOMETRY_NAME:
        raise ValueError("NEXT39 cohort filenames differ")
    _validate_output_hash(manifest, metadata_path, role="cohort metadata")
    _validate_output_hash(manifest, frames_zip_path, role="cohort geometry")
    table = pd.read_parquet(metadata_path)
    required = {
        "material_id",
        "parent_id",
        "trajectory_stem",
        "initial_record_key",
        "latest_record_key",
        "latest_step",
        "natoms",
        "input_role",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"NEXT39 cohort lacks columns: {sorted(required - set(table))}")
    table = table.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        table.material_id.isna().any()
        or table.material_id.astype(str).duplicated().any()
        or table.parent_id.astype(str).duplicated().any()
        or not table.input_role.eq("step0_unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT39 cohort identity or role differs")
    ids = tuple(table.material_id.astype(str))
    loaded, atoms = _load_archive_only(frames_zip_path, ids)
    if loaded != list(ids) or any(
        len(structure) != int(natoms)
        for structure, natoms in zip(atoms, table.natoms, strict=True)
    ):
        raise ValueError("NEXT39 step-0 geometry identity differs")
    return table, atoms, manifest


def _load_frozen_rule(
    *, rule_path: Path, manifest_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    rule = _strict_json(rule_path, role="NEXT23 frozen rule")
    manifest = _strict_json(manifest_path, role="NEXT23 rule manifest")
    _validate_output_hash(manifest, rule_path, role="NEXT23 frozen rule")
    if (
        rule.get("protocol") != RULE_PROTOCOL
        or manifest.get("protocol") != RULE_PROTOCOL
        or manifest.get("blind_labels_opened") is not False
        or rule.get("selected_candidate") != "B+E"
        or rule.get("selected_terms") != ["B", "E"]
        or rule.get("reject_when") != "supported and score >= threshold"
        or rule.get("missing_policy") != "fail_open_do_not_reject"
        or rule.get("dft_or_relaxed_input_used") is not False
        or rule.get("model_or_proxy_potential_used") is not False
        or rule.get("same_composition_candidates_used") is not False
    ):
        raise ValueError("NEXT23 frozen rule contract differs")
    parameters = rule.get("base_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("NEXT23 frozen parameters are missing")
    expected_columns = dict(zip(("B", "E"), TERM_COLUMNS, strict=True))
    for term, column in expected_columns.items():
        parameter = parameters.get(term)
        if (
            not isinstance(parameter, Mapping)
            or parameter.get("column") != column
            or parameter.get("direction") != 1
        ):
            raise ValueError("NEXT23 frozen B+E term definition differs")
        for name in ("median", "scale_iqr"):
            value = parameter.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("NEXT23 frozen robust parameter is invalid")
        if float(parameter["scale_iqr"]) <= 0.0:
            raise ValueError("NEXT23 frozen IQR must be positive")
    threshold = rule.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("NEXT23 frozen threshold is invalid")

    source_hashes = manifest.get("executed_source_sha256")
    repository = Path(__file__).resolve().parents[1]
    required_sources = (
        "src/next20_valence_rigidity.py",
        "src/next21_normalized_madelung.py",
        "src/next22_bond_valence_equilibrium.py",
        "src/next23_relaxation_rule.py",
    )
    if not isinstance(source_hashes, Mapping) or any(
        source_hashes.get(name) != _sha256(repository / name) for name in required_sources
    ):
        raise ValueError("NEXT23 frozen analytic kernel hash differs")
    return rule, manifest


def compute_next23_terms(atoms: Atoms) -> tuple[Mapping[str, object] | None, str | None]:
    """Compute exactly the frozen B and E terms from one unmodified structure."""

    result = compute_inorganic_response_features(atoms)
    failures = []
    for family in ("sivr", "scbve"):
        if not result.family_supported[family]:
            failures.append(result.family_failures[family] or f"{family} unsupported")
    if failures:
        return None, "; ".join(failures)
    return {
        TERM_COLUMNS[0]: result.features["sivr_cell_anisotropy"],
        TERM_COLUMNS[1]: result.features["scbv_vector_asymmetry_rms"],
    }, None


def apply_frozen_next23_rule(
    features: Mapping[str, object], rule: Mapping[str, object]
) -> dict[str, object]:
    """Apply a validated frozen rule with exact fail-open behavior."""

    parameters = rule.get("base_parameters")
    terms = rule.get("selected_terms")
    threshold = rule.get("threshold")
    if not isinstance(parameters, Mapping) or terms != ["B", "E"]:
        raise ValueError("invalid frozen NEXT23 rule")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("invalid frozen NEXT23 threshold")
    score = 0.0
    for term in terms:
        parameter = parameters.get(term)
        if not isinstance(parameter, Mapping):
            raise ValueError("invalid frozen NEXT23 term")
        try:
            value = float(features[str(parameter["column"])])
            median = float(parameter["median"])
            scale = float(parameter["scale_iqr"])
            direction = int(parameter["direction"])
        except (KeyError, TypeError, ValueError):
            return {"supported": False, "score": math.nan, "reject": False}
        if (
            not all(math.isfinite(item) for item in (value, median, scale))
            or scale <= 0.0
            or direction not in {-1, 1}
        ):
            return {"supported": False, "score": math.nan, "reject": False}
        score += direction * (value - median) / scale
    return {
        "supported": True,
        "score": float(score),
        "reject": bool(score >= float(threshold)),
    }


def _pauling_values(
    atoms: Atoms, calculator: FeatureCalculator
) -> dict[str, object]:
    try:
        raw, error = calculator(atoms)
    except Exception as exc:
        raw, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
    values = dict(raw) if isinstance(raw, Mapping) else {}
    row: dict[str, object] = {"pauling_feature_error": error}
    decisions: list[str] = []
    for name, rule in RULES.items():
        value = values.get(str(rule["feature"]), np.nan)
        decision = _rule_decision(
            value,
            operator=str(rule["operator"]),
            threshold=float(rule["threshold"]),
        )
        row[f"{name}_value"] = value
        row[f"{name}_decision"] = decision
        decisions.append(decision)
    row["pauling_p2_p5_decision"] = _combined_decision(decisions)
    return row


def run_next39_predictions(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    frozen_rule_path: Path,
    frozen_rule_manifest_path: Path,
    output_dir: Path,
    term_calculator: FeatureCalculator | None = None,
    pauling_feature_calculator: FeatureCalculator | None = None,
) -> dict[str, object]:
    """Seal NEXT23 and Pauling predictions before any later geometry is opened."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry": Path(frames_zip_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "frozen_rule_manifest": Path(frozen_rule_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT39 prediction input is missing")
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    metadata, structures, _cohort_manifest = _load_step0_cohort(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry"],
        manifest_path=paths["cohort_manifest"],
    )
    rule, _rule_manifest = _load_frozen_rule(
        rule_path=paths["frozen_rule"], manifest_path=paths["frozen_rule_manifest"]
    )
    term_fn = compute_next23_terms if term_calculator is None else term_calculator
    pauling_fn = (
        _classical_features
        if pauling_feature_calculator is None
        else pauling_feature_calculator
    )
    production = (
        term_calculator is None
        and pauling_feature_calculator is None
        and input_hashes["frozen_rule"] == FROZEN_RULE_SHA256
    )

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for upstream, atoms in zip(metadata.to_dict("records"), structures, strict=True):
        try:
            raw_terms, error = term_fn(atoms)
        except Exception as exc:
            raw_terms, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
        terms = dict(raw_terms) if isinstance(raw_terms, Mapping) else {}
        decision = apply_frozen_next23_rule(terms, rule)
        row: dict[str, object] = {
            "material_id": str(upstream["material_id"]),
            "parent_id": str(upstream["parent_id"]),
            "trajectory_stem": str(upstream["trajectory_stem"]),
            "natoms": int(upstream["natoms"]),
            TERM_COLUMNS[0]: terms.get(TERM_COLUMNS[0], np.nan),
            TERM_COLUMNS[1]: terms.get(TERM_COLUMNS[1], np.nan),
            "next23_feature_error": error,
            "next23_supported": bool(decision["supported"]),
            "next23_score": decision["score"],
            "next23_reject": bool(decision["reject"]),
        }
        row.update(_pauling_values(atoms, pauling_fn))
        rows.append(row)
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    if len(table) != len(metadata) or table.material_id.duplicated().any():
        raise RuntimeError("NEXT39 prediction identity accounting differs")
    if bool((table.next23_reject & ~table.next23_supported).any()):
        raise RuntimeError("NEXT39 unsupported rows did not fail open")

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next12_pauling_controls.py",
        "src/next19_valence_transport.py",
        "src/next20_valence_rigidity.py",
        "src/next22_bond_valence_equilibrium.py",
        "src/next23_relaxation_rule.py",
        "src/next32_inorganic_response_features.py",
        "src/next39_omat24_trajectory_cohort.py",
        "src/next39_next23_predictions.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "cross_source_predictions_frozen_before_later_geometry_opening",
        "input_role": "one_step0_unrelaxed_structure_only",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "later_geometry_opened": False,
        "dft_values_read": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "thresholds_refit": False,
        "frozen_rule_sha256": input_hashes["frozen_rule"],
        "frozen_candidate": rule["selected_candidate"],
        "frozen_terms": rule["selected_terms"],
        "frozen_threshold": rule["threshold"],
        "pauling_rules": {name: dict(value) for name, value in RULES.items()},
        "counts": {
            "rows": len(table),
            "next23_supported": int(table.next23_supported.sum()),
            "next23_rejected": int(table.next23_reject.sum()),
            "pauling": {
                decision: int(table.pauling_p2_p5_decision.eq(decision).sum())
                for decision in DECISIONS
            },
        },
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": production,
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        prediction_path = staging / PREDICTIONS_NAME
        table.to_parquet(prediction_path, index=False)
        manifest["outputs_sha256"] = {PREDICTIONS_NAME: _sha256(prediction_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT39 prediction input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT39 prediction source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--frozen-rule", required=True, type=Path)
    parser.add_argument("--frozen-rule-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_next39_predictions(
        metadata_path=args.metadata,
        frames_zip_path=args.frames_zip,
        cohort_manifest_path=args.cohort_manifest,
        frozen_rule_path=args.frozen_rule,
        frozen_rule_manifest_path=args.frozen_rule_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_NAME",
    "PREDICTIONS_NAME",
    "PROTOCOL",
    "apply_frozen_next23_rule",
    "compute_next23_terms",
    "run_next39_predictions",
]
