#!/usr/bin/env python3
"""Audit frozen PSNB on physically isolated discovery outcomes only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping

import numpy as np
import pandas as pd

import src.next268_prv_feature_audit as n268
import src.next368_pbveu_feature_audit as n368
import src.next379_periodic_skeletal_net_bottleneck as n379
import src.next380_psnb_formal_build as n380
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next381-psnb-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT381_PSNB_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT381_PSNB_FEATURE_AUDIT.json"
TABLE_NAME = "next381_psnb_feature_audit.parquet"
HYPOTHESES = tuple(
    (name, n379.FEATURE_DIRECTIONS[name]) for name in n379.FEATURE_NAMES
)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = (*n368.REQUIRED_STAGES, 380)
REQUIRED_DESIGN_STAGES = n368.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n368.BOUNDARY_FLAGS
NEXT379_SOURCE_PATH = "src/next379_periodic_skeletal_net_bottleneck.py"
EXPECTED_NEXT379_SOURCE_SHA256 = "d56a05a5f884b8d3f0d6c3605989c6796197065e2ab4b6bf8c2adee33196b692"
EXPECTED_NEXT380_SOURCE_SHA256 = "0b6868443d0d0c120f1689304669514f4fda70cb46f95f0b9df265a552657b19"
_REPOSITORY = Path(__file__).resolve().parents[1]
_NEXT368_DESIGN_PATH = (
    _REPOSITORY
    / "docs/plans/2026-08-13-next367-next370-periodic-bond-valence-equal-uniformity.md"
)
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n368.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "next368_design": n368.EXPECTED_INPUT_SHA256["design"],
    "design": n379.DESIGN_SHA256,
    "next380_manifest": "7214d8ce532fca61d42addf5d5368362aa6157017bf6fb765323f8b9e53b1411",
    "next380_catalogue": "29d67b3d9f56f9ee02da4bb9f54378afcfe1fbfdb7a80f8de20ef749d8b68dca",
    "next380_scigen": "6f28dfd04a7b5ff2254d8c31eb0ffcae71fa6054ae2cf1041c762cab74a48277",
    "next380_wyformer": "21d5cafdc9f8fec266fcfac77d02d1efd01c9bab6724154138346a8402da2d0b",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT381 audit ranking table differs") from exc


def bounded_protection(
    *, values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT381 bounded protection inputs differ")
    try:
        return n268.bounded_protection(
            values=values, direction=direction, q_lo=q_lo, q_hi=q_hi
        )
    except ValueError as exc:
        raise ValueError("NEXT381 bounded protection inputs differ") from exc


def _index_psnb_by_prefixed_material_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT381 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n368._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={stage: design_paths[stage] for stage in REQUIRED_DESIGN_STAGES},
        design_path=_NEXT368_DESIGN_PATH,
    )
    paths["next368_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next380_manifest": roots["next380"] / n380.MANIFEST_NAME,
            "next380_catalogue": roots["next380"] / n380.CATALOGUE_NAME,
            "next380_scigen": roots["next380"] / n380.FEATURE_FILES["scigen"],
            "next380_wyformer": roots["next380"] / n380.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_next380(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
):
    prior_paths = {key: paths[key] for key in n368.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next368_design"]
    prior_hashes = {key: input_hashes[key] for key in n368.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next368_design"]
    prior = n368._verify_next367(prior_paths, prior_hashes)
    manifest = json.loads(paths["next380_manifest"].read_text())
    catalogue = json.loads(paths["next380_catalogue"].read_text())
    tables = {
        "scigen": pd.read_parquet(paths["next380_scigen"]),
        "wyformer": pd.read_parquet(paths["next380_wyformer"]),
    }
    expected_outputs = {
        n380.CATALOGUE_NAME: input_hashes["next380_catalogue"],
        n380.FEATURE_FILES["scigen"]: input_hashes["next380_scigen"],
        n380.FEATURE_FILES["wyformer"]: input_hashes["next380_wyformer"],
    }
    if (
        _sha256_file(Path(n379.__file__).resolve()) != EXPECTED_NEXT379_SOURCE_SHA256
        or _sha256_file(Path(n380.__file__).resolve()) != EXPECTED_NEXT380_SOURCE_SHA256
        or manifest.get("protocol") != n380.PROTOCOL
        or manifest.get("next381_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(NEXT379_SOURCE_PATH)
        != EXPECTED_NEXT379_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n379.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n379.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("graph")
        != "ordinary periodic Voronoi facets with mutual local solid-angle salience"
        or catalogue.get("formula")
        != "q10 site threshold for entry into a translation-rank-3 atomic contact component"
        or catalogue.get("minimum_formal_coverage") != n380.MINIMUM_FORMAL_COVERAGE
        or catalogue.get("directions_frozen_before_outcome") is not True
    ):
        raise ValueError("NEXT381 NEXT380 provenance differs")
    for source, expected_rows in n380.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["psnb_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(table[n379.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
        sites = pd.to_numeric(table["psnb_site_count"], errors="coerce")
        faces = pd.to_numeric(table["psnb_directed_face_count"], errors="coerce")
        undirected = pd.to_numeric(
            table["psnb_undirected_edge_count"], errors="coerce"
        )
        rank3_sites = pd.to_numeric(
            table["psnb_rank3_site_count"], errors="coerce"
        )
        reverse_error = pd.to_numeric(
            table["psnb_maximum_reverse_angle_error"], errors="coerce"
        )
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n380.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all()
            or np.isfinite(values[~supported]).any()
            or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
            or not (sites[supported] >= 1).all()
            or not (undirected[supported] >= 1).all()
            or not (faces[supported] == 2 * undirected[supported]).all()
            or not (
                (rank3_sites[supported] >= 0)
                & (rank3_sites[supported] <= sites[supported])
            ).all()
            or not (
                np.isfinite(reverse_error[supported])
                & (reverse_error[supported] <= n379.REVERSE_ANGLE_TOLERANCE)
            ).all()
            or not (sites[~supported] == 0).all()
            or not (faces[~supported] == 0).all()
            or not (undirected[~supported] == 0).all()
            or not (rank3_sites[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT381 NEXT380 {source} table differs")
    return (*prior[:-1], tables)


def run_psnb_feature_audit(
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
    """Audit exactly the one frozen PSNB hypothesis."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT381 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT381 design path universe differs")
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
        design_paths={stage: Path(design_paths[stage]).resolve() for stage in REQUIRED_DESIGN_STAGES},
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT381 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT381 formal input identity differs: {differing}")
    (
        eligible, eligible214, primary_key, base_start_key, formula214,
        current_key, formula222, psnb_tables,
    ) = _verify_next380(paths, input_hashes)
    n227 = n268.n227
    combined, feature_tables, score, support, endpoint, _ = n227._reconstruct_next224_frontier(
        paths=paths,
        eligible=eligible,
        eligible214=eligible214,
        primary_key=primary_key,
        base_start_key=base_start_key,
        formula214=formula214,
        current_key=current_key,
        formula222=formula222,
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_psnb_by_prefixed_material_id(
            table=psnb_tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        combined.loc[mask, n379.FEATURE_NAMES[0]] = ordered_ids.map(
            indexed[n379.FEATURE_NAMES[0]]
        ).to_numpy()
    cohort = n227.build_rejected_extreme_cohort(
        score=score, support=support, endpoint=endpoint,
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
                int((mask & protected).sum()), int((mask & severe).sum())
            )
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT381 rejected cohort counts differ")

    rows: list[dict[str, object]] = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"NEXT381 {feature} has no finite discovery values")
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        if not math.isfinite(q_lo) or not math.isfinite(q_hi) or q_hi <= q_lo:
            source_results = {
                name: {
                    "passes_source_gates": False,
                    "aggregate_auc": math.nan,
                    "macro_fold_auc": math.nan,
                    "worst_fold_auc": math.nan,
                    "minimum_cell_coverage": float(np.isfinite(values[source == name]).mean()),
                    "reason": "DEGENERATE_NORMALIZATION_RANGE",
                }
                for name in ("scigen", "wyformer")
            }
        else:
            protection_values = bounded_protection(
                values=values, direction=direction, q_lo=float(q_lo), q_hi=float(q_hi)
            )
            source_results = {
                name: n227.audit_one_source(
                    values=protection_values[source == name],
                    endpoint=endpoint[source == name],
                    cohort=cohort[source == name],
                    folds=folds[source == name],
                    direction="protected_high",
                )
                for name in ("scigen", "wyformer")
            }
        scigen, wyformer = source_results["scigen"], source_results["wyformer"]
        aggregate = [float(scigen["aggregate_auc"]), float(wyformer["aggregate_auc"])]
        ranks = [value if math.isfinite(value) else -math.inf for value in aggregate]
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
        "features": list(n379.FEATURE_NAMES),
        "feature_directions": n379.FEATURE_DIRECTIONS,
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
        "audit_mode": "fixed_next224_rejected_extreme_psnb_audit",
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
        "next382_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        "src/next368_pbveu_feature_audit.py": Path(n368.__file__).resolve(),
        NEXT379_SOURCE_PATH: Path(n379.__file__).resolve(),
        "src/next380_psnb_formal_build.py": Path(n380.__file__).resolve(),
        "src/next381_psnb_feature_audit.py": Path(__file__).resolve(),
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
            "next382_search_authorized": bool(selected is not None),
            "psnb_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT381 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT381 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "HYPOTHESES", "QUANTILES", "REQUIRED_DESIGN_STAGES", "REQUIRED_STAGES",
    "_index_psnb_by_prefixed_material_id", "bounded_protection",
    "run_psnb_feature_audit", "select_eligible_hypotheses",
]
