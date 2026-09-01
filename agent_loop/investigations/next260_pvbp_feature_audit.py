#!/usr/bin/env python3
"""Audit frozen PVBP hypotheses inside the exact NEXT224 rejected cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next227_margin_local_feature_audit as n227
import src.next259_periodic_void_bottleneck_persistence as n259
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next260-pvbp-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT260_PVBP_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT260_PVBP_FEATURE_AUDIT.json"
TABLE_NAME = "next260_pvbp_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = n259.EXPECTED_DESIGN_SHA256
HYPOTHESES = tuple((name, "protected_low") for name in n259.FEATURE_NAMES)
QUANTILES = (1 / 16, 15 / 16)
EXPECTED_NEXT227_SOURCE_SHA256 = (
    "e092b45d8e3d01cb8c5754e7de0bd6908ea00aa49b4286234074db23758d4ef1"
)
EXPECTED_NEXT259_SOURCE_SHA256 = (
    "6affc4a08a3d67d7d2a0039514c3147d9cfdf61cfafdcfd76421f89a8e29e323"
)
NEXT259_SOURCE_PATH = "src/next259_periodic_void_bottleneck_persistence.py"
REQUIRED_STAGES = (*n227.REQUIRED_STAGES, 259)
REQUIRED_DESIGN_STAGES = (*n227.REQUIRED_DESIGN_STAGES, 227, 259)
BOUNDARY_FLAGS = n227.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n227.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next227_design": n227.EXPECTED_INPUT_SHA256["design"],
    "next259_design": EXPECTED_DESIGN_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next259_manifest": (
        "e775f402def2c69071a519e93611cade31411040828d2b84883925e279bf474e"
    ),
    "next259_catalogue": (
        "3490ab51504a870483ffa34ee824c676350b0392e199fb1e00dd252df0fb14c6"
    ),
    "next259_scigen": (
        "fa37b8197e2d95c4e6404636a34779cea56ea326515025105d12299877e77891"
    ),
    "next259_wyformer": (
        "69444686dbea184b0d47d8048cd06bfec903a16de4d73e260e460990827b87f7"
    ),
}


def select_eligible_hypotheses(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    required = {
        "hypothesis",
        "passes_raw_gates",
        "ranking_min_worst_fold_auc",
        "ranking_min_aggregate_auc",
        "ranking_mean_aggregate_auc",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty or required - set(frame.columns):
        raise ValueError("NEXT260 audit ranking table differs")
    result = frame.copy()
    result["eligible_for_search"] = result["passes_raw_gates"].fillna(False).astype(bool)
    eligible = result.loc[result["eligible_for_search"]].sort_values(
        [
            "ranking_min_worst_fold_auc",
            "ranking_min_aggregate_auc",
            "ranking_mean_aggregate_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    leader = None if eligible.empty else eligible.iloc[0].to_dict()
    return result.sort_values("hypothesis", kind="mergesort").reset_index(drop=True), leader


def bounded_protection(
    *, values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    lo = float(q_lo)
    hi = float(q_hi)
    if raw.ndim != 1 or direction not in n227.PROTECTION_DIRECTIONS or not (
        math.isfinite(lo) and math.isfinite(hi) and hi > lo
    ):
        raise ValueError("NEXT260 bounded protection inputs differ")
    result = np.full(raw.shape, np.nan, dtype=float)
    finite = np.isfinite(raw)
    scaled = np.clip((raw[finite] - lo) / (hi - lo), 0.0, 1.0)
    result[finite] = scaled if direction == "protected_high" else 1.0 - scaled
    return result


def _index_pvbp_by_prefixed_material_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    """Return a source-prefixed PVBP table after exact identity validation."""

    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    expected = set(pd.Series(expected_material_ids, dtype=str))
    if set(extra["material_id"]) != expected:
        raise ValueError(f"NEXT260 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n227._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n227.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[227],
    )
    paths["next227_design"] = paths.pop("design")
    paths.update(
        {
            "next259_design": design_paths[259],
            "design": design_path,
            "next259_manifest": roots["next259"] / n259.MANIFEST_NAME,
            "next259_catalogue": roots["next259"] / n259.CATALOGUE_NAME,
            "next259_scigen": roots["next259"] / n259.FEATURE_FILES["scigen"],
            "next259_wyformer": roots["next259"] / n259.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_next259(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    str,
    dict[str, object],
    dict[str, pd.DataFrame],
]:
    base_paths = {key: paths[key] for key in n227.EXPECTED_INPUT_SHA256}
    base_paths["design"] = paths["next227_design"]
    base_hashes = {key: input_hashes[key] for key in n227.EXPECTED_INPUT_SHA256}
    base_hashes["design"] = input_hashes["next227_design"]
    prior = n227._verify_next226(base_paths, base_hashes)
    manifest = json.loads(paths["next259_manifest"].read_text())
    catalogue = json.loads(paths["next259_catalogue"].read_text())
    tables = {
        "scigen": pd.read_parquet(paths["next259_scigen"]),
        "wyformer": pd.read_parquet(paths["next259_wyformer"]),
    }
    expected_outputs = {
        n259.CATALOGUE_NAME: input_hashes["next259_catalogue"],
        n259.FEATURE_FILES["scigen"]: input_hashes["next259_scigen"],
        n259.FEATURE_FILES["wyformer"]: input_hashes["next259_wyformer"],
    }
    if (
        _sha256_file(Path(n227.__file__).resolve()) != EXPECTED_NEXT227_SOURCE_SHA256
        or manifest.get("protocol") != n259.PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(NEXT259_SOURCE_PATH)
        != EXPECTED_NEXT259_SOURCE_SHA256
        or _sha256_file(Path(n259.__file__).resolve()) != EXPECTED_NEXT259_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in n259.BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n259.FEATURE_NAMES)
        or catalogue.get("feature_count") != len(n259.FEATURE_NAMES)
        or catalogue.get("hypothesis_direction") != "protected_low"
        or catalogue.get("quantile_method") != "inverted_cdf"
        or catalogue.get("distance_field") != "unweighted_nearest_atomic_center"
        or catalogue.get("labels_opened") is not False
    ):
        raise ValueError("NEXT260 NEXT259 provenance differs")
    for source, expected_rows in (("scigen", 13_470), ("wyformer", 5_232)):
        table = tables[source]
        supported = table["pvbp_supported"].fillna(False).astype(bool)
        finite = np.column_stack(
            [
                np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                for name in n259.FEATURE_NAMES
            ]
        )
        nodes = pd.to_numeric(table["pvbp_node_count"], errors="coerce")
        edges = pd.to_numeric(table["pvbp_edge_count"], errors="coerce")
        expected_supported = int(manifest["counts"][source]["supported"])
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != expected_supported
            or not finite[supported].all()
            or finite[~supported].any()
            or not (nodes[supported] > 0).all()
            or not (edges[supported] > 0).all()
            or not (nodes[~supported] == 0).all()
            or not (edges[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT260 NEXT259 {source} table differs")
    return (*prior, tables)


def run_pvbp_feature_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Audit frozen PVBP hypotheses without searching a new formula."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT260 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT260 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(stage_dirs[stage]).resolve()
            for stage in REQUIRED_STAGES
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={
            stage: Path(design_paths[stage]).resolve()
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT260 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT260 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        pvbp_tables,
    ) = _verify_next259(paths, input_hashes)
    combined, feature_tables, score, support, endpoint, _ = (
        n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_pvbp_by_prefixed_material_id(
            table=pvbp_tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        for name in n259.FEATURE_NAMES:
            combined.loc[mask, name] = ordered_ids.map(indexed[name]).to_numpy()
    cohort = n227.build_rejected_extreme_cohort(
        score=score,
        support=support,
        endpoint=endpoint,
        threshold=n227.EXPECTED_BASE_THRESHOLD,
    )
    n164 = n227.n226.n225.n222.n215.n214.n164
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    protected = endpoint <= 1.0
    severe = endpoint >= 2.0
    cohort_counts: dict[str, tuple[int, int]] = {}
    for source_name in ("scigen", "wyformer"):
        for fold in (None, 0, 1, 2, 3, 4):
            mask = cohort & (source == source_name)
            if fold is not None:
                mask &= folds == fold
            cell_id = f"{source_name}:{'all' if fold is None else f'fold{fold}'}"
            cohort_counts[cell_id] = (
                int((mask & protected).sum()),
                int((mask & severe).sum()),
            )
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT260 rejected cohort counts differ")
    rows: list[dict[str, object]] = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"NEXT260 {feature} has no finite discovery values")
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        protection = bounded_protection(
            values=values,
            direction=direction,
            q_lo=float(q_lo),
            q_hi=float(q_hi),
        )
        source_results = {}
        for source_name in ("scigen", "wyformer"):
            mask = source == source_name
            source_results[source_name] = n227.audit_one_source(
                values=protection[mask],
                endpoint=endpoint[mask],
                cohort=cohort[mask],
                folds=folds[mask],
                direction="protected_high",
            )
        scigen = source_results["scigen"]
        wyformer = source_results["wyformer"]
        aggregate_aucs = [float(scigen["aggregate_auc"]), float(wyformer["aggregate_auc"])]
        rows.append(
            {
                "hypothesis": f"{feature}__{direction}",
                "feature": feature,
                "direction": direction,
                "q_lo": float(q_lo),
                "q_hi": float(q_hi),
                "passes_raw_gates": bool(
                    scigen["passes_source_gates"] and wyformer["passes_source_gates"]
                ),
                "ranking_min_worst_fold_auc": float(
                    min(scigen["worst_fold_auc"], wyformer["worst_fold_auc"])
                ),
                "ranking_min_aggregate_auc": float(min(aggregate_aucs)),
                "ranking_mean_aggregate_auc": float(np.mean(aggregate_aucs)),
                "scigen_aggregate_auc": scigen["aggregate_auc"],
                "scigen_macro_fold_auc": scigen["macro_fold_auc"],
                "scigen_worst_fold_auc": scigen["worst_fold_auc"],
                "scigen_minimum_cell_coverage": scigen["minimum_cell_coverage"],
                "wyformer_aggregate_auc": wyformer["aggregate_auc"],
                "wyformer_macro_fold_auc": wyformer["macro_fold_auc"],
                "wyformer_worst_fold_auc": wyformer["worst_fold_auc"],
                "wyformer_minimum_cell_coverage": wyformer["minimum_cell_coverage"],
                "source_audits_json": json.dumps(
                    source_results, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    table, selected = select_eligible_hypotheses(pd.DataFrame(rows))
    eligible_table = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible_table["hypothesis"].astype(str))
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "features": [feature for feature, _ in HYPOTHESES],
        "hypotheses": [f"{feature}__{direction}" for feature, direction in HYPOTHESES],
        "hypothesis_count": len(HYPOTHESES),
        "quantiles": list(QUANTILES),
        "quantile_method": "inverted_cdf",
        "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "gates": {
            "minimum_coverage": n227.MINIMUM_COVERAGE,
            "minimum_class_count": n227.MINIMUM_CLASS_COUNT,
            "minimum_aggregate_auc": n227.MINIMUM_AGGREGATE_AUC,
            "minimum_macro_auc": n227.MINIMUM_MACRO_AUC,
            "minimum_worst_auc": n227.MINIMUM_WORST_AUC,
        },
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL,
        "audit_mode": "fixed_next224_rejected_extreme_pvbp_feature_audit",
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "cohort_counts": {
            key: {"protected_rejected": value[0], "severe_rejected": value[1]}
            for key, value in sorted(cohort_counts.items())
        },
        "hypothesis_count": len(table),
        "eligible_hypothesis_count": int(len(eligible_table)),
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected,
        "next261_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next227_margin_local_feature_audit.py": Path(n227.__file__).resolve(),
        NEXT259_SOURCE_PATH: Path(n259.__file__).resolve(),
        "src/next260_pvbp_feature_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(audit_path, audit)
        table.to_parquet(table_path, index=False)
        outputs = [catalogue_path, audit_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "hypothesis_count": len(table),
            "eligible_hypothesis_count": int(len(eligible_table)),
            "eligible_hypothesis_sha256": eligible_sha,
            "next224_frontier_reproduced": True,
            "next261_search_authorized": bool(selected is not None),
            "pvbp_branch_terminated": selected is None,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT260 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT260 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen PVBP features.")
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-discovery-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-discovery-endpoint-dir", type=Path, required=True)
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_pvbp_feature_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES},
        next135_freeze_path=args.next135_freeze_path,
        design_paths={
            stage: getattr(args, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
