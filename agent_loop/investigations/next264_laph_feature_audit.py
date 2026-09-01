#!/usr/bin/env python3
"""Audit frozen LAPH hypotheses inside the exact NEXT224 rejected cohort."""

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
import src.next260_pvbp_feature_audit as n260
import src.next263_local_angular_persistent_homology as n263
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next264-laph-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT264_LAPH_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT264_LAPH_FEATURE_AUDIT.json"
TABLE_NAME = "next264_laph_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = n263.EXPECTED_DESIGN_SHA256
HYPOTHESES = tuple(
    (name, direction)
    for name in n263.FEATURE_NAMES
    for direction in ("protected_low", "protected_high")
)
QUANTILES = (1 / 16, 15 / 16)
EXPECTED_NEXT227_SOURCE_SHA256 = (
    "e092b45d8e3d01cb8c5754e7de0bd6908ea00aa49b4286234074db23758d4ef1"
)
EXPECTED_NEXT263_SOURCE_SHA256 = (
    "f5b849e4b832acc9824c8450fd6bab5f100e474baf16731642020d7e069a008e"
)
NEXT263_SOURCE_PATH = "src/next263_local_angular_persistent_homology.py"
REQUIRED_STAGES = (*n227.REQUIRED_STAGES, 263)
REQUIRED_DESIGN_STAGES = (*n227.REQUIRED_DESIGN_STAGES, 227, 263)
BOUNDARY_FLAGS = n227.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n227.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next227_design": n227.EXPECTED_INPUT_SHA256["design"],
    "next263_design": EXPECTED_DESIGN_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next263_manifest": (
        "70fdfb6b9d83a65406304a74b4244f1190d8fb16886d5343ed1cebfe102359de"
    ),
    "next263_catalogue": (
        "0532cd83de20860dbd46050f1587f6e6ee0c37cb4027ea0b96200e4cfbfcbeb4"
    ),
    "next263_scigen": (
        "70ec9357a7837693a87c2289e955c1b80c88f13a5ec6aa30ee8b3ab3994b3856"
    ),
    "next263_wyformer": (
        "afc52b7314d7c24df8ae8834f248613ef2ee421879747d7b842af9ee9c3d81a7"
    ),
}


def select_eligible_hypotheses(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    try:
        return n260.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT264 audit ranking table differs") from exc


def bounded_protection(
    *, values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    try:
        return n260.bounded_protection(
            values=values, direction=direction, q_lo=q_lo, q_hi=q_hi
        )
    except ValueError as exc:
        raise ValueError("NEXT264 bounded protection inputs differ") from exc


def _index_laph_by_prefixed_material_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    """Return a source-prefixed LAPH table after exact identity validation."""

    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    expected = set(pd.Series(expected_material_ids, dtype=str))
    if set(extra["material_id"]) != expected:
        raise ValueError(f"NEXT264 {source} material identity differs")
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
            "next263_design": design_paths[263],
            "design": design_path,
            "next263_manifest": roots["next263"] / n263.MANIFEST_NAME,
            "next263_catalogue": roots["next263"] / n263.CATALOGUE_NAME,
            "next263_scigen": roots["next263"] / n263.FEATURE_FILES["scigen"],
            "next263_wyformer": roots["next263"] / n263.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_next263(
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
    manifest = json.loads(paths["next263_manifest"].read_text())
    catalogue = json.loads(paths["next263_catalogue"].read_text())
    tables = {
        "scigen": pd.read_parquet(paths["next263_scigen"]),
        "wyformer": pd.read_parquet(paths["next263_wyformer"]),
    }
    expected_outputs = {
        n263.CATALOGUE_NAME: input_hashes["next263_catalogue"],
        n263.FEATURE_FILES["scigen"]: input_hashes["next263_scigen"],
        n263.FEATURE_FILES["wyformer"]: input_hashes["next263_wyformer"],
    }
    if (
        _sha256_file(Path(n227.__file__).resolve()) != EXPECTED_NEXT227_SOURCE_SHA256
        or manifest.get("protocol") != n263.PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(NEXT263_SOURCE_PATH)
        != EXPECTED_NEXT263_SOURCE_SHA256
        or _sha256_file(Path(n263.__file__).resolve())
        != EXPECTED_NEXT263_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in n263.BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n263.FEATURE_NAMES)
        or catalogue.get("feature_count") != len(n263.FEATURE_NAMES)
        or catalogue.get("complex")
        != "complete_vietoris_rips_through_dimension_2"
        or catalogue.get("coefficient_field") != "F2"
        or catalogue.get("distance_grid") != n263.DISTANCE_GRID
        or catalogue.get("quantile_method") != "inverted_cdf"
        or catalogue.get("labels_opened") is not False
    ):
        raise ValueError("NEXT264 NEXT263 provenance differs")
    for source, expected_rows in (("scigen", 13_470), ("wyformer", 5_232)):
        table = tables[source]
        supported = table["laph_supported"].fillna(False).astype(bool)
        finite = np.column_stack(
            [
                np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                for name in n263.FEATURE_NAMES
            ]
        )
        sites = pd.to_numeric(table["laph_site_count"], errors="coerce")
        minimum = pd.to_numeric(table["laph_min_retained_facets"], errors="coerce")
        maximum = pd.to_numeric(table["laph_max_retained_facets"], errors="coerce")
        expected_supported = int(manifest["counts"][source]["supported"])
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != expected_supported
            or not finite[supported].all()
            or finite[~supported].any()
            or not (sites[supported] > 0).all()
            or not (minimum[supported] >= n263.MIN_RETAINED_FACETS).all()
            or not (maximum[supported] <= n263.MAX_RETAINED_FACETS).all()
            or not (sites[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT264 NEXT263 {source} table differs")
    return (*prior, tables)


def run_laph_feature_audit(
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
    """Audit frozen LAPH hypotheses without searching a formula."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT264 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT264 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
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
        raise FileNotFoundError("NEXT264 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT264 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        laph_tables,
    ) = _verify_next263(paths, input_hashes)
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
        indexed = _index_laph_by_prefixed_material_id(
            table=laph_tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        for name in n263.FEATURE_NAMES:
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
        raise ValueError("NEXT264 rejected cohort counts differ")
    rows: list[dict[str, object]] = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"NEXT264 {feature} has no finite discovery values")
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        if not math.isfinite(q_lo) or not math.isfinite(q_hi) or q_hi <= q_lo:
            source_results = {
                source_name: {
                    "passes_source_gates": False,
                    "aggregate_auc": math.nan,
                    "macro_fold_auc": math.nan,
                    "worst_fold_auc": math.nan,
                    "minimum_cell_coverage": float(
                        np.isfinite(values[source == source_name]).mean()
                    ),
                    "reason": "DEGENERATE_NORMALIZATION_RANGE",
                }
                for source_name in ("scigen", "wyformer")
            }
        else:
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
        ranks = [value if math.isfinite(value) else -math.inf for value in aggregate_aucs]
        worst = [float(scigen["worst_fold_auc"]), float(wyformer["worst_fold_auc"])]
        worst = [value if math.isfinite(value) else -math.inf for value in worst]
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
                "ranking_min_worst_fold_auc": float(min(worst)),
                "ranking_min_aggregate_auc": float(min(ranks)),
                "ranking_mean_aggregate_auc": float(np.mean(ranks)),
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
        "features": list(n263.FEATURE_NAMES),
        "directions": ["protected_low", "protected_high"],
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
        "audit_mode": "fixed_next224_rejected_extreme_laph_feature_audit",
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
        "next265_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next227_margin_local_feature_audit.py": Path(n227.__file__).resolve(),
        NEXT263_SOURCE_PATH: Path(n263.__file__).resolve(),
        "src/next264_laph_feature_audit.py": Path(__file__).resolve(),
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
            "next265_search_authorized": bool(selected is not None),
            "laph_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT264 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT264 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    manifest = run_laph_feature_audit(
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
