#!/usr/bin/env python3
"""Audit frozen PBVHL hypotheses on physically isolated discovery outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next268_prv_feature_audit as n268
import src.next307_periodic_bond_valence_hodge_loop as n307
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next308-pbvhl-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT308_PBVHL_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT308_PBVHL_FEATURE_AUDIT.json"
TABLE_NAME = "next308_pbvhl_feature_audit.parquet"
HYPOTHESES = tuple((name, n307.FEATURE_DIRECTIONS[name]) for name in n307.FEATURE_NAMES)
QUANTILES = n268.QUANTILES
EXPECTED_NEXT268_SOURCE_SHA256 = (
    "36af0e632c8c3aae93f78271f837f693d49eff6ae88cdcad8edde42b3e0eaf64"
)
EXPECTED_NEXT307_SOURCE_SHA256 = (
    "49373c0246c1a997a56320bf1416419a6c91d77a01763bf8bbce7bfbec317a2f"
)
NEXT307_SOURCE_PATH = "src/next307_periodic_bond_valence_hodge_loop.py"
REQUIRED_STAGES = (*n268.REQUIRED_STAGES, 307)
REQUIRED_DESIGN_STAGES = n268.REQUIRED_DESIGN_STAGES
BOUNDARY_FLAGS = n268.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n268.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next268_design": n268.EXPECTED_INPUT_SHA256["design"],
    "design": n307.EXPECTED_DESIGN_SHA256,
    "next307_manifest": "aeb402568f532fe65b4e44c5c0c59faf863194626839aae15e55727f6ed997a5",
    "next307_catalogue": "fe61fd353d6d0283860b2a75c1526aa5b0dbdd506b3d111ff6973b766b9ae730",
    "next307_scigen": "225085504036bc74eca31987a97ed7c40f7410463738b98b994154b51332b23c",
    "next307_wyformer": "3e5d1058349e80e1de46670014536f6790659dd8e4fb2c1e0476c7643510b52b",
}


def select_eligible_hypotheses(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    try:
        return n268.select_eligible_hypotheses(frame)
    except ValueError as exc:
        raise ValueError("NEXT308 audit ranking table differs") from exc


def bounded_protection(
    *, values: object, direction: str, q_lo: float, q_hi: float
) -> np.ndarray:
    if direction != "protected_low":
        raise ValueError("NEXT308 bounded protection inputs differ")
    try:
        return n268.bounded_protection(
            values=values, direction=direction, q_lo=q_lo, q_hi=q_hi
        )
    except ValueError as exc:
        raise ValueError("NEXT308 bounded protection inputs differ") from exc


def _index_pbvhl_by_prefixed_material_id(
    *, table: pd.DataFrame, source: str, expected_material_ids: object
) -> pd.DataFrame:
    extra = table.copy()
    extra["material_id"] = source + ":" + extra["material_id"].astype(str)
    expected = set(pd.Series(expected_material_ids, dtype=str))
    if set(extra["material_id"]) != expected:
        raise ValueError(f"NEXT308 {source} material identity differs")
    return extra.set_index("material_id")


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n268._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n268.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[267],
    )
    paths["next268_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next307_manifest": roots["next307"] / n307.MANIFEST_NAME,
            "next307_catalogue": roots["next307"] / n307.CATALOGUE_NAME,
            "next307_scigen": roots["next307"] / n307.FEATURE_FILES["scigen"],
            "next307_wyformer": roots["next307"] / n307.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_next307(paths: Mapping[str, Path], input_hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n268.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next268_design"]
    prior_hashes = {key: input_hashes[key] for key in n268.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next268_design"]
    prior = n268._verify_next267(prior_paths, prior_hashes)
    manifest = json.loads(paths["next307_manifest"].read_text())
    catalogue = json.loads(paths["next307_catalogue"].read_text())
    tables = {
        "scigen": pd.read_parquet(paths["next307_scigen"]),
        "wyformer": pd.read_parquet(paths["next307_wyformer"]),
    }
    expected_outputs = {
        n307.CATALOGUE_NAME: input_hashes["next307_catalogue"],
        n307.FEATURE_FILES["scigen"]: input_hashes["next307_scigen"],
        n307.FEATURE_FILES["wyformer"]: input_hashes["next307_wyformer"],
    }
    if (
        _sha256_file(Path(n268.__file__).resolve()) != EXPECTED_NEXT268_SOURCE_SHA256
        or _sha256_file(Path(n307.__file__).resolve()) != EXPECTED_NEXT307_SOURCE_SHA256
        or manifest.get("protocol") != n307.PROTOCOL
        or manifest.get("next308_audit_authorized") is not True
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("internal_validation_geometry_opened") is not False
        or manifest.get("internal_replication_geometry_opened") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(NEXT307_SOURCE_PATH)
        != EXPECTED_NEXT307_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("feature_names") != list(n307.FEATURE_NAMES)
        or catalogue.get("feature_directions") != n307.FEATURE_DIRECTIONS
        or catalogue.get("feature_count") != len(n307.FEATURE_NAMES)
        or catalogue.get("loop_projection")
        != "bond_valence_minus_row_space_projection"
        or catalogue.get("quantile_method") != "inverted_cdf"
        or catalogue.get("directions_frozen_before_outcome") is not True
    ):
        raise ValueError("NEXT308 NEXT307 provenance differs")
    for source, expected_rows in n307.EXPECTED_ROWS.items():
        table = tables[source]
        supported = table["pbvhl_supported"].fillna(False).astype(bool)
        values = np.column_stack(
            [
                pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                for name in n307.FEATURE_NAMES
            ]
        )
        finite = np.isfinite(values)
        sites = pd.to_numeric(table["pbvhl_site_count"], errors="coerce")
        edges = pd.to_numeric(table["pbvhl_edge_count"], errors="coerce")
        ranks = pd.to_numeric(table["pbvhl_incidence_rank"], errors="coerce")
        cycles = pd.to_numeric(table["pbvhl_cycle_dimension"], errors="coerce")
        divergence = pd.to_numeric(table["pbvhl_loop_divergence_max"], errors="coerce")
        if (
            len(table) != expected_rows
            or table["material_id"].astype(str).duplicated().any()
            or int(supported.sum()) != int(manifest["counts"][source]["supported"])
            or float(supported.mean()) < n307.MINIMUM_FORMAL_COVERAGE
            or not finite[supported].all()
            or finite[~supported].any()
            or not (values[supported] >= 0.0).all()
            or not (values[supported, 0] <= 1.0 + 1.0e-12).all()
            or not (sites[supported] >= 2).all()
            or not (edges[supported] > ranks[supported]).all()
            or not (cycles[supported] == edges[supported] - ranks[supported]).all()
            or not (cycles[supported] > 0).all()
            or not np.isfinite(divergence[supported]).all()
            or not (divergence[supported] <= 1.0e-8).all()
            or not (sites[~supported] == 0).all()
            or not (edges[~supported] == 0).all()
            or not (ranks[~supported] == 0).all()
            or not (cycles[~supported] == 0).all()
        ):
            raise ValueError(f"NEXT308 NEXT307 {source} table differs")
    return (*prior[:-1], tables)


def run_pbvhl_feature_audit(
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
    """Audit exactly the four frozen PBVHL hypotheses."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT308 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT308 design path universe differs")
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
        raise FileNotFoundError("NEXT308 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT308 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        pbvhl_tables,
    ) = _verify_next307(paths, input_hashes)
    n227 = n268.n227
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
        indexed = _index_pbvhl_by_prefixed_material_id(
            table=pbvhl_tables[source_name],
            source=source_name,
            expected_material_ids=feature_tables[source_name]["material_id"],
        )
        mask = source == source_name
        ordered_ids = combined.loc[mask, "material_id"].astype(str)
        for name in n307.FEATURE_NAMES:
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
        raise ValueError("NEXT308 rejected cohort counts differ")

    rows: list[dict[str, object]] = []
    for feature, direction in HYPOTHESES:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"NEXT308 {feature} has no finite discovery values")
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
            protection_values = bounded_protection(
                values=values,
                direction=direction,
                q_lo=float(q_lo),
                q_hi=float(q_hi),
            )
            source_results = {
                source_name: n227.audit_one_source(
                    values=protection_values[source == source_name],
                    endpoint=endpoint[source == source_name],
                    cohort=cohort[source == source_name],
                    folds=folds[source == source_name],
                    direction="protected_high",
                )
                for source_name in ("scigen", "wyformer")
            }
        scigen = source_results["scigen"]
        wyformer = source_results["wyformer"]
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
        "features": list(n307.FEATURE_NAMES),
        "feature_directions": n307.FEATURE_DIRECTIONS,
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
        "audit_mode": "fixed_next224_rejected_extreme_pbvhl_audit",
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
        "next309_search_authorized": bool(selected is not None),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next268_prv_feature_audit.py": Path(n268.__file__).resolve(),
        NEXT307_SOURCE_PATH: Path(n307.__file__).resolve(),
        "src/next308_pbvhl_feature_audit.py": Path(__file__).resolve(),
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
            "next309_search_authorized": bool(selected is not None),
            "pbvhl_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT308 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT308 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
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
    arguments = parser.parse_args(argv)
    manifest = run_pbvhl_feature_audit(
        scigen_feature_dir=arguments.scigen_feature_dir,
        scigen_discovery_endpoint_dir=arguments.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=arguments.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=arguments.wyformer_discovery_endpoint_dir,
        stage_dirs={stage: getattr(arguments, f"next{stage}_dir") for stage in REQUIRED_STAGES},
        next135_freeze_path=arguments.next135_freeze_path,
        design_paths={
            stage: getattr(arguments, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=arguments.design_path,
        output_dir=arguments.output_dir,
        require_formal_inputs=not arguments.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HYPOTHESES",
    "NEXT307_SOURCE_PATH",
    "REQUIRED_DESIGN_STAGES",
    "REQUIRED_STAGES",
    "bounded_protection",
    "run_pbvhl_feature_audit",
    "select_eligible_hypotheses",
]
