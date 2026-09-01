#!/usr/bin/env python3
"""Audit frozen RFDR on physically isolated discovery outcomes only."""

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
import src.next351_periodic_deviatoric_strain_rigidity as n351
import src.next355_radical_facet_deviatoric_rigidity as n355
import src.next355_rfdr_formal_build as n355b
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-13-next356-rfdr-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT356_RFDR_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT356_RFDR_FEATURE_AUDIT.json"
TABLE_NAME = "next356_rfdr_feature_audit.parquet"
HYPOTHESES = tuple((name, n355.FEATURE_DIRECTIONS[name]) for name in n355.FEATURE_NAMES)
QUANTILES = n268.QUANTILES
REQUIRED_STAGES = (*n268.REQUIRED_STAGES, 355)
REQUIRED_DESIGN_STAGES = n268.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n268.BOUNDARY_FLAGS
EXPECTED_NEXT268_SOURCE_SHA256 = "36af0e632c8c3aae93f78271f837f693d49eff6ae88cdcad8edde42b3e0eaf64"
EXPECTED_NEXT355_SOURCE_SHA256 = "431eb8323e12e61ad93c316ac599bd0d4bd122784151fe1354d954d11256d8a3"
NEXT355_SOURCE_PATH = "src/next355_radical_facet_deviatoric_rigidity.py"
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n268.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "next268_design": n268.EXPECTED_INPUT_SHA256["design"],
    "design": n355.DESIGN_SHA256,
    "next355_manifest": "14d13c66766be04a693d4463aa29caa57889b3b4ca67a3d0afbc609a7e0a17bc",
    "next355_catalogue": "a64baa9b1f90ca331ad1dd20ae98e7250dc5e678aeffa3f78ba5dec70d5a65b8",
    "next355_scigen": "fd20dbb4964b20ce82b874a906a85c6266f9adb8e8ef42282a9af1c256776d5b",
    "next355_wyformer": "f566cc858dd37894dd490a2f4c36907ff83e94e6d07a38ab75d8bd3aa874c019",
}


def select_eligible_hypotheses(frame: pd.DataFrame):
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT356 audit ranking table differs") from exc


def bounded_protection(*, values: object, direction: str, q_lo: float, q_hi: float) -> np.ndarray:
    if direction != "protected_high":
        raise ValueError("NEXT356 bounded protection inputs differ")
    try:
        return n268.bounded_protection(values=values, direction=direction, q_lo=q_lo, q_hi=q_hi)
    except ValueError as exc:
        raise ValueError("NEXT356 bounded protection inputs differ") from exc


def _index_rfdr_by_prefixed_material_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    if set(extra["material_id"]) != set(pd.Series(expected_material_ids, dtype=str)):
        raise ValueError(f"NEXT356 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(*, roots: Mapping[str, Path], next135_freeze_path: Path,
           design_paths: Mapping[int, Path], design_path: Path) -> dict[str, Path]:
    paths = n268._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={stage: design_paths[stage] for stage in n268.REQUIRED_DESIGN_STAGES},
        design_path=design_paths[267],
    )
    paths["next268_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next355_manifest": roots["next355"] / n355b.MANIFEST_NAME,
            "next355_catalogue": roots["next355"] / n355b.CATALOGUE_NAME,
            "next355_scigen": roots["next355"] / n355b.FEATURE_FILES["scigen"],
            "next355_wyformer": roots["next355"] / n355b.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_next355(paths: Mapping[str, Path], input_hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n268.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next268_design"]
    prior_hashes = {key: input_hashes[key] for key in n268.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next268_design"]
    prior = n268._verify_next267(prior_paths, prior_hashes)
    manifest = json.loads(paths["next355_manifest"].read_text())
    catalogue = json.loads(paths["next355_catalogue"].read_text())
    tables = {
        "scigen": pd.read_parquet(paths["next355_scigen"]),
        "wyformer": pd.read_parquet(paths["next355_wyformer"]),
    }
    expected_outputs = {
        n355b.CATALOGUE_NAME: input_hashes["next355_catalogue"],
        n355b.FEATURE_FILES["scigen"]: input_hashes["next355_scigen"],
        n355b.FEATURE_FILES["wyformer"]: input_hashes["next355_wyformer"],
    }
    if (
        _sha256_file(Path(n268.__file__).resolve()) != EXPECTED_NEXT268_SOURCE_SHA256
        or _sha256_file(Path(n355.__file__).resolve()) != EXPECTED_NEXT355_SOURCE_SHA256
        or manifest.get("protocol") != n355b.PROTOCOL
        or manifest.get("next356_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(NEXT355_SOURCE_PATH)
        != EXPECTED_NEXT355_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n355.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n355.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != 1
        or catalogue.get("graph")
        != "NEXT339 all reciprocal radius-weighted radical facets with A/d weights"
        or catalogue.get("formula")
        != "lambda_min(H0^-1/2 D.T(I-UU+)D H0^-1/2)"
        or catalogue.get("strain_space") != "five-dimensional symmetric trace-free"
        or catalogue.get("minimum_formal_coverage") != n355b.MINIMUM_FORMAL_COVERAGE
        or catalogue.get("directions_frozen_before_outcome") is not True
    ):
        raise ValueError("NEXT356 NEXT355 provenance differs")
    for source, expected_rows in n355b.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["rfdr_supported"].fillna(False).astype(bool)
        values = pd.to_numeric(table[n355.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
        sites = pd.to_numeric(table["rfdr_site_count"], errors="coerce")
        edges = pd.to_numeric(table["rfdr_edge_count"], errors="coerce")
        area = pd.to_numeric(table["rfdr_minimum_facet_area"], errors="coerce")
        reciprocal = pd.to_numeric(table["rfdr_maximum_reciprocal_area_relative_error"], errors="coerce")
        residual = pd.to_numeric(table["rfdr_maximum_orthogonality_residual"], errors="coerce")
        tiling = pd.to_numeric(table["rfdr_volume_tiling_relative_error"], errors="coerce")
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n355b.MINIMUM_FORMAL_COVERAGE
            or not np.isfinite(values[supported]).all()
            or np.isfinite(values[~supported]).any()
            or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
            or not (sites[supported] >= 1).all()
            or not (edges[supported] >= 5).all()
            or not (area[supported] > 0.0).all()
            or not (np.isfinite(reciprocal[supported]) & (reciprocal[supported] <= n355.RECIPROCAL_AREA_RELATIVE_TOLERANCE)).all()
            or not (np.isfinite(residual[supported]) & (residual[supported] <= n351.ORTHOGONALITY_TOLERANCE)).all()
            or not (np.isfinite(tiling[supported]) & (tiling[supported] <= n355.VOLUME_TILING_RELATIVE_TOLERANCE)).all()
            or not (sites[~supported] == 0).all()
            or not (edges[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT356 NEXT355 {source} table differs")
    return (*prior[:-1], tables)


def run_rfdr_feature_audit(
    *, scigen_feature_dir: Path, scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path, wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path], next135_freeze_path: Path,
    design_paths: Mapping[int, Path], design_path: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Audit exactly the one frozen RFDR hypothesis."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT356 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT356 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(stage_dirs[stage]).resolve() for stage in REQUIRED_STAGES},
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots, next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={stage: Path(design_paths[stage]).resolve() for stage in REQUIRED_DESIGN_STAGES},
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT356 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
                           if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name))
        raise ValueError(f"NEXT356 formal input identity differs: {differing}")
    (eligible, eligible214, primary_key, base_start_key, formula214,
     current_key, formula222, rfdr_tables) = _verify_next355(paths, input_hashes)
    n227 = n268.n227
    combined, feature_tables, score, support, endpoint, _ = n227._reconstruct_next224_frontier(
        paths=paths, eligible=eligible, eligible214=eligible214,
        primary_key=primary_key, base_start_key=base_start_key,
        formula214=formula214, current_key=current_key, formula222=formula222,
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    for source_name in ("scigen", "wyformer"):
        indexed = _index_rfdr_by_prefixed_material_id(
            table=rfdr_tables[source_name], source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        combined.loc[mask, n355.FEATURE_NAMES[0]] = ordered_ids.map(
            indexed[n355.FEATURE_NAMES[0]]
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
            cohort_counts[cell_id] = (int((mask & protected).sum()), int((mask & severe).sum()))
    if cohort_counts != n227.EXPECTED_COHORT_COUNTS:
        raise ValueError("NEXT356 rejected cohort counts differ")

    rows: list[dict[str, object]] = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"NEXT356 {feature} has no finite discovery values")
        q_lo, q_hi = np.quantile(finite, QUANTILES, method="inverted_cdf")
        if not math.isfinite(q_lo) or not math.isfinite(q_hi) or q_hi <= q_lo:
            source_results = {
                name: {"passes_source_gates": False, "aggregate_auc": math.nan,
                       "macro_fold_auc": math.nan, "worst_fold_auc": math.nan,
                       "minimum_cell_coverage": float(np.isfinite(values[source == name]).mean()),
                       "reason": "DEGENERATE_NORMALIZATION_RANGE"}
                for name in ("scigen", "wyformer")
            }
        else:
            protection_values = bounded_protection(
                values=values, direction=direction, q_lo=float(q_lo), q_hi=float(q_hi)
            )
            source_results = {
                name: n227.audit_one_source(
                    values=protection_values[source == name], endpoint=endpoint[source == name],
                    cohort=cohort[source == name], folds=folds[source == name],
                    direction="protected_high",
                ) for name in ("scigen", "wyformer")
            }
        scigen, wyformer = source_results["scigen"], source_results["wyformer"]
        aggregate = [float(scigen["aggregate_auc"]), float(wyformer["aggregate_auc"])]
        ranks = [value if math.isfinite(value) else -math.inf for value in aggregate]
        worst = [float(scigen["worst_fold_auc"]), float(wyformer["worst_fold_auc"])]
        worst = [value if math.isfinite(value) else -math.inf for value in worst]
        rows.append(
            {
                "hypothesis": f"{feature}__{direction}", "feature": feature,
                "direction": direction, "q_lo": float(q_lo), "q_hi": float(q_hi),
                "passes_raw_gates": bool(scigen["passes_source_gates"] and wyformer["passes_source_gates"]),
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
                "source_audits_json": json.dumps(source_results, sort_keys=True, separators=(",", ":")),
            }
        )
    table, selected = select_eligible_hypotheses(pd.DataFrame(rows))
    eligible_table = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible_table["hypothesis"].astype(str))
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()
    catalogue = {
        "protocol": PROTOCOL, "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "features": list(n355.FEATURE_NAMES), "feature_directions": n355.FEATURE_DIRECTIONS,
        "hypotheses": [f"{feature}__{direction}" for feature, direction in HYPOTHESES],
        "hypothesis_count": len(HYPOTHESES), "quantiles": list(QUANTILES),
        "quantile_method": "inverted_cdf", "normalization_population": "ALL_FINITE_COMBINED_DISCOVERY",
        "gates": {"minimum_coverage": n227.MINIMUM_COVERAGE,
                  "minimum_class_count": n227.MINIMUM_CLASS_COUNT,
                  "minimum_aggregate_auc": n227.MINIMUM_AGGREGATE_AUC,
                  "minimum_macro_auc": n227.MINIMUM_MACRO_AUC,
                  "minimum_worst_auc": n227.MINIMUM_WORST_AUC},
        "validation_outputs_opened": False, "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL, "audit_mode": "fixed_next224_rejected_extreme_rfdr_audit",
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "cohort_counts": {key: {"protected_rejected": value[0], "severe_rejected": value[1]}
                          for key, value in sorted(cohort_counts.items())},
        "hypothesis_count": len(table), "eligible_hypothesis_count": int(len(eligible_table)),
        "eligible_hypotheses": eligible_names, "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected, "next357_search_authorized": bool(selected is not None),
        "new_formula_searched": False, "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        NEXT355_SOURCE_PATH: Path(n355.__file__).resolve(),
        "src/next355_rfdr_formal_build.py": Path(n355b.__file__).resolve(),
        "src/next356_rfdr_feature_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path, audit_path, table_path = staging / CATALOGUE_NAME, staging / AUDIT_NAME, staging / TABLE_NAME
        _write_json(catalogue_path, catalogue); _write_json(audit_path, audit); table.to_parquet(table_path, index=False)
        outputs = [catalogue_path, audit_path, table_path]
        manifest = {
            "protocol": PROTOCOL, "hypothesis_count": len(table),
            "eligible_hypothesis_count": int(len(eligible_table)),
            "eligible_hypothesis_sha256": eligible_sha, "next224_frontier_reproduced": True,
            "next357_search_authorized": bool(selected is not None),
            "rfdr_branch_terminated": selected is None, "new_formula_searched": False,
            "new_formula_selected": False, "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True, **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False, "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT356 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT356 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "HYPOTHESES", "QUANTILES", "REQUIRED_DESIGN_STAGES", "REQUIRED_STAGES",
    "_index_rfdr_by_prefixed_material_id", "bounded_protection",
    "run_rfdr_feature_audit", "select_eligible_hypotheses",
]
