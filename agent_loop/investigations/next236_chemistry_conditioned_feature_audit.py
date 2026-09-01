#!/usr/bin/env python3
"""Audit endpoint-blind chemistry-conditioned raw x0 certificates."""

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
import src.next235_final_stagewise_margin_local_broad_diagnostic as n235
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next236-chemistry-conditioned-feature-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT236_CHEMISTRY_CONDITIONED_FEATURE_CATALOGUE.json"
AUDIT_NAME = "NEXT236_CHEMISTRY_CONDITIONED_FEATURE_AUDIT.json"
TABLE_NAME = "next236_chemistry_conditioned_feature_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "d6161caceac82da781754abcc0a1a0b114baf347e37097330a172ce4f029dc94"
)
CONDITIONERS = (
    "geom_electronegativity_mean",
    "geom_electronegativity_range",
    "geom_atomic_number_mean",
    "geom_covalent_radius_mean",
)
STRATUM_QUANTILES = (1 / 4, 1 / 2, 3 / 4)
CUTOFF_QUANTILES = (1 / 16, 15 / 16)
EXPECTED_HYPOTHESIS_COUNT = 242 * len(CONDITIONERS) * 2
EXPECTED_FRONTIER_FAILED_COUNT = 5
EXPECTED_FRONTIER_SHORTFALL = 0.12339543654931197
REQUIRED_STAGES = (*n227.REQUIRED_STAGES, 235)
REQUIRED_DESIGN_STAGES = (*n227.REQUIRED_DESIGN_STAGES, 227)
BOUNDARY_FLAGS = n227.BOUNDARY_FLAGS
EXPECTED_NEXT235_SOURCE_SHA256 = (
    "7e87d8dcb83cce7f19ba35f4c8f4cc2736b7f98581d5b3817d2ed1c8b1e1e7e0"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n227.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next227_design": n227.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next235_manifest": (
        "12b819b5128b0d271aefb11bafbf5a7e4bc21a4d149b426785f686507c1ce240"
    ),
    "next235_diagnostic": (
        "d12bed44442e5543b7f9bdb7e0e1c2ddea5bf0a930fb744495ddf15d8ebef510"
    ),
    "next235_table": (
        "995f3780caefaaaf742540c88902594cbef3f3a99c0dd17b5cf6c8b6526bd514"
    ),
}


select_auditable_features = n227.select_auditable_features
audit_one_source = n227.audit_one_source
build_rejected_extreme_cohort = n227.build_rejected_extreme_cohort


def assign_lower_inclusive_strata(values: object, edges: object) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    cuts = np.asarray(edges, dtype=float)
    if (
        raw.ndim != 1
        or cuts.shape != (3,)
        or np.any(~np.isfinite(cuts))
        or np.any(np.diff(cuts) <= 0.0)
    ):
        raise ValueError("NEXT236 stratum inputs differ")
    strata = np.full(raw.shape, -1, dtype=np.int8)
    finite = np.isfinite(raw)
    strata[finite] = np.searchsorted(cuts, raw[finite], side="left").astype(
        np.int8
    )
    return strata


def fit_conditioned_cutoffs(
    *, values: object, conditioner: object, stratum_edges: object
) -> dict[str, object]:
    raw = np.asarray(values, dtype=float)
    chemistry = np.asarray(conditioner, dtype=float)
    edges = np.asarray(stratum_edges, dtype=float)
    if raw.ndim != 1 or chemistry.shape != raw.shape:
        raise ValueError("NEXT236 conditioned cutoff population differs")
    strata = assign_lower_inclusive_strata(chemistry, edges)
    q_lo: list[float] = []
    q_hi: list[float] = []
    counts: list[int] = []
    for stratum in range(4):
        mask = (strata == stratum) & np.isfinite(raw)
        counts.append(int(mask.sum()))
        if not mask.any():
            q_lo.append(float("nan"))
            q_hi.append(float("nan"))
            continue
        lo, hi = np.quantile(raw[mask], CUTOFF_QUANTILES, method="inverted_cdf")
        if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and hi > lo):
            q_lo.append(float("nan"))
            q_hi.append(float("nan"))
        else:
            q_lo.append(float(lo))
            q_hi.append(float(hi))
    return {
        "stratum_edges": edges.tolist(),
        "q_lo_by_stratum": q_lo,
        "q_hi_by_stratum": q_hi,
        "finite_target_count_by_stratum": counts,
        "quantile_method": "inverted_cdf",
    }


def chemistry_conditioned_protection(
    *,
    values: object,
    conditioner: object,
    direction: str,
    model: Mapping[str, object],
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    chemistry = np.asarray(conditioner, dtype=float)
    if raw.ndim != 1 or chemistry.shape != raw.shape:
        raise ValueError("NEXT236 conditioned protection population differs")
    if direction not in n227.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT236 conditioned protection direction differs")
    edges = np.asarray(model.get("stratum_edges"), dtype=float)
    q_lo = np.asarray(model.get("q_lo_by_stratum"), dtype=float)
    q_hi = np.asarray(model.get("q_hi_by_stratum"), dtype=float)
    if q_lo.shape != (4,) or q_hi.shape != (4,):
        raise ValueError("NEXT236 conditioned protection model differs")
    strata = assign_lower_inclusive_strata(chemistry, edges)
    protection = np.full(raw.shape, np.nan, dtype=float)
    for stratum in range(4):
        mask = (strata == stratum) & np.isfinite(raw)
        lo = q_lo[stratum]
        hi = q_hi[stratum]
        if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and hi > lo):
            continue
        scaled = np.clip((raw[mask] - lo) / (hi - lo), 0.0, 1.0)
        protection[mask] = scaled if direction == "protected_high" else 1.0 - scaled
    return protection


def select_conditioned_hypotheses(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    required = {
        "hypothesis",
        "feature",
        "conditioner",
        "direction",
        "passes_raw_gates",
        "ranking_min_worst_fold_auc",
        "ranking_min_aggregate_auc",
        "ranking_mean_aggregate_auc",
    }
    if rows.empty or required - set(rows.columns):
        raise ValueError("NEXT236 conditioned hypothesis table differs")
    table = rows.copy()
    passing = table["passes_raw_gates"].fillna(False).astype(bool)
    pass_count = passing.groupby(
        [table["feature"].astype(str), table["conditioner"].astype(str)]
    ).transform("sum")
    table["opposite_direction_veto_passed"] = pass_count.eq(1)
    table["eligible_for_search"] = passing & pass_count.eq(1)
    eligible = table.loc[table["eligible_for_search"]].sort_values(
        [
            "ranking_min_worst_fold_auc",
            "ranking_min_aggregate_auc",
            "ranking_mean_aggregate_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table.sort_values("hypothesis", kind="mergesort").reset_index(drop=True), selected


def _ranking_auc(value: object) -> float:
    """Place unavailable low-coverage AUCs below every evaluable hypothesis."""

    return -1.0 if value is None else float(value)


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
            "design": design_path,
            "next235_manifest": roots["next235"] / n235.MANIFEST_NAME,
            "next235_diagnostic": roots["next235"] / n235.DIAGNOSTIC_NAME,
            "next235_table": roots["next235"] / n235.TABLE_NAME,
        }
    )
    return paths


def _verify_next235_comparison(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> None:
    manifest = json.loads(paths["next235_manifest"].read_text())
    diagnostic = json.loads(paths["next235_diagnostic"].read_text())
    table = pd.read_parquet(paths["next235_table"])
    closest = diagnostic["global_closest"]
    expected_outputs = {
        n235.DIAGNOSTIC_NAME: input_hashes["next235_diagnostic"],
        n235.TABLE_NAME: input_hashes["next235_table"],
    }
    if (
        manifest.get("protocol") != n235.PROTOCOL
        or manifest.get("final_stagewise_margin_local_branch_closed") is not True
        or manifest.get("continuation_requires_new_preoutcome_freeze") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next235_final_stagewise_margin_local_broad_diagnostic.py"
        )
        != EXPECTED_NEXT235_SOURCE_SHA256
        or _sha256_file(Path(n235.__file__).resolve())
        != EXPECTED_NEXT235_SOURCE_SHA256
        or int(closest["failed_constraint_count"]) != EXPECTED_FRONTIER_FAILED_COUNT
        or not math.isclose(
            float(closest["normalized_shortfall_sum"]),
            EXPECTED_FRONTIER_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or len(table) != n235.EXPECTED_CANDIDATE_COUNT
        or n235.candidate_key_sha256(table) != n235.EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT236 NEXT235 comparison provenance differs")


def run_chemistry_conditioned_feature_audit(
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
    """Audit all frozen chemistry-conditioned certificates on discovery."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT236 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT236 design path universe differs")
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
        design_paths={stage: Path(design_paths[stage]).resolve() for stage in REQUIRED_DESIGN_STAGES},
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT236 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT236 formal input identity differs: {differing}")
    prior_paths = {key: paths[key] for key in n227.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next227_design"]
    prior_hashes = {key: input_hashes[key] for key in n227.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next227_design"]
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
    ) = n227._verify_next226(prior_paths, prior_hashes)
    _verify_next235_comparison(paths, input_hashes)
    combined, _, score, support, endpoint, _ = n227._reconstruct_next224_frontier(
        paths=prior_paths,
        eligible=eligible,
        eligible214=eligible214,
        primary_key=primary_key,
        base_start_key=base_start_key,
        formula214=formula214,
        current_key=current_key,
        formula222=formula222,
    )
    cohort = build_rejected_extreme_cohort(
        score=score,
        support=support,
        endpoint=endpoint,
        threshold=n227.EXPECTED_BASE_THRESHOLD,
    )
    n164 = n227.n226.n225.n222.n215.n214.n164
    folds = n164.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    source = combined["source_dataset"].astype(str).to_numpy()
    feature_names = select_auditable_features(combined)
    feature_sha = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    if (
        len(feature_names) != n227.EXPECTED_FEATURE_COUNT
        or feature_sha != n227.EXPECTED_FEATURE_NAME_SHA256
        or any(name not in feature_names for name in CONDITIONERS)
    ):
        raise ValueError("NEXT236 frozen feature universe differs")

    conditioner_models: dict[str, dict[str, object]] = {}
    for conditioner_name in CONDITIONERS:
        chemistry = pd.to_numeric(
            combined[conditioner_name], errors="coerce"
        ).to_numpy(float)
        finite = chemistry[np.isfinite(chemistry)]
        edges = np.quantile(finite, STRATUM_QUANTILES, method="inverted_cdf")
        if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("NEXT236 conditioner strata differ")
        conditioner_models[conditioner_name] = {
            "stratum_edges": [float(value) for value in edges],
            "finite_count": int(len(finite)),
        }

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        values = pd.to_numeric(combined[feature], errors="coerce").to_numpy(float)
        for conditioner_name in CONDITIONERS:
            chemistry = pd.to_numeric(
                combined[conditioner_name], errors="coerce"
            ).to_numpy(float)
            edges = conditioner_models[conditioner_name]["stratum_edges"]
            model = fit_conditioned_cutoffs(
                values=values, conditioner=chemistry, stratum_edges=edges
            )
            for direction in n227.PROTECTION_DIRECTIONS:
                protection = chemistry_conditioned_protection(
                    values=values,
                    conditioner=chemistry,
                    direction=direction,
                    model=model,
                )
                source_results = {}
                for source_name in ("scigen", "wyformer"):
                    mask = source == source_name
                    source_results[source_name] = audit_one_source(
                        values=protection[mask],
                        endpoint=endpoint[mask],
                        cohort=cohort[mask],
                        folds=folds[mask],
                        direction="protected_high",
                    )
                scigen = source_results["scigen"]
                wyformer = source_results["wyformer"]
                aggregate_aucs = [
                    _ranking_auc(scigen["aggregate_auc"]),
                    _ranking_auc(wyformer["aggregate_auc"]),
                ]
                hypothesis = f"{feature}__{conditioner_name}__{direction}"
                rows.append(
                    {
                        "hypothesis": hypothesis,
                        "feature": feature,
                        "conditioner": conditioner_name,
                        "direction": direction,
                        "passes_raw_gates": bool(
                            scigen["passes_source_gates"]
                            and wyformer["passes_source_gates"]
                        ),
                        "ranking_min_worst_fold_auc": float(
                            min(
                                _ranking_auc(scigen["worst_fold_auc"]),
                                _ranking_auc(wyformer["worst_fold_auc"]),
                            )
                        ),
                        "ranking_min_aggregate_auc": float(min(aggregate_aucs)),
                        "ranking_mean_aggregate_auc": float(np.mean(aggregate_aucs)),
                        "scigen_aggregate_auc": scigen["aggregate_auc"],
                        "scigen_macro_fold_auc": scigen["macro_fold_auc"],
                        "scigen_worst_fold_auc": scigen["worst_fold_auc"],
                        "wyformer_aggregate_auc": wyformer["aggregate_auc"],
                        "wyformer_macro_fold_auc": wyformer["macro_fold_auc"],
                        "wyformer_worst_fold_auc": wyformer["worst_fold_auc"],
                        "stratum_edges_json": json.dumps(
                            model["stratum_edges"], separators=(",", ":")
                        ),
                        "q_lo_by_stratum_json": json.dumps(
                            model["q_lo_by_stratum"], separators=(",", ":")
                        ),
                        "q_hi_by_stratum_json": json.dumps(
                            model["q_hi_by_stratum"], separators=(",", ":")
                        ),
                        "finite_target_count_by_stratum_json": json.dumps(
                            model["finite_target_count_by_stratum"],
                            separators=(",", ":"),
                        ),
                        "source_audits_json": json.dumps(
                            source_results, sort_keys=True, separators=(",", ":")
                        ),
                    }
                )
    raw_table = pd.DataFrame(rows)
    if len(raw_table) != EXPECTED_HYPOTHESIS_COUNT:
        raise RuntimeError("NEXT236 hypothesis count differs")
    table, selected = select_conditioned_hypotheses(raw_table)
    eligible_table = table.loc[table["eligible_for_search"]]
    eligible_names = sorted(eligible_table["hypothesis"].astype(str))
    eligible_sha = hashlib.sha256("\n".join(eligible_names).encode()).hexdigest()

    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": n227.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_threshold": n227.EXPECTED_BASE_THRESHOLD,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "feature_name_sha256": feature_sha,
        "conditioners": CONDITIONERS,
        "conditioner_models": conditioner_models,
        "stratum_quantiles": STRATUM_QUANTILES,
        "cutoff_quantiles": CUTOFF_QUANTILES,
        "hypothesis_count": len(table),
        "normalization_fit_uses_endpoint": False,
        "source_identity_used_by_executable_formula": False,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    audit = {
        "protocol": PROTOCOL,
        "audit_mode": "next224_rejected_extreme_chemistry_conditioned_certificate_audit",
        "hypothesis_count": len(table),
        "raw_gate_passing_count": int(table["passes_raw_gates"].sum()),
        "eligible_hypothesis_count": int(len(eligible_table)),
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_sha256": eligible_sha,
        "selected_hypothesis": selected,
        "next237_search_authorized": bool(selected is not None),
        "next235_comparison_verified": True,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_or_replication_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next227_margin_local_feature_audit.py": Path(n227.__file__).resolve(),
        "src/next235_final_stagewise_margin_local_broad_diagnostic.py": Path(n235.__file__).resolve(),
        "src/next236_chemistry_conditioned_feature_audit.py": Path(__file__).resolve(),
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
            "feature_count": len(feature_names),
            "conditioner_count": len(CONDITIONERS),
            "hypothesis_count": len(table),
            "eligible_hypothesis_count": int(len(eligible_table)),
            "eligible_hypothesis_sha256": eligible_sha,
            "next224_frontier_reproduced": True,
            "next235_comparison_verified": True,
            "next237_search_authorized": bool(selected is not None),
            "chemistry_conditioned_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT236 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT236 source changed before publication")
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
    manifest = run_chemistry_conditioned_feature_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES},
        next135_freeze_path=args.next135_freeze_path,
        design_paths={stage: getattr(args, f"next{stage}_design_path") for stage in REQUIRED_DESIGN_STAGES},
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "assign_lower_inclusive_strata",
    "chemistry_conditioned_protection",
    "fit_conditioned_cutoffs",
    "run_chemistry_conditioned_feature_audit",
    "select_conditioned_hypotheses",
]


if __name__ == "__main__":
    raise SystemExit(main())
