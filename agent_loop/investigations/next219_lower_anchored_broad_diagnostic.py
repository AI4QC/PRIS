#!/usr/bin/env python3
"""Diagnose frozen NEXT218 AUC+SAFE, non-BROAD anchored candidates."""

from __future__ import annotations

import argparse
from collections import Counter
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

import src.next218_lower_anchored_relief_search as n218
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next219-lower-anchored-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT219_LOWER_ANCHORED_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next219_lower_anchored_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n218.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT218_SOURCE_SHA256 = (
    "5112645c106c13d65ee233d3a3e70dda635528b6d0df893fd976f76f892d0ab2"
)
EXPECTED_CANDIDATE_COUNT = 89
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "4e8efea493197b340e36b9b8c0d2974f9ab575fab2a81c09eef0fdbf25acc428"
)
EXPECTED_BASE_FAILED_COUNT = n218.n217.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n218.n217.EXPECTED_BASE_SHORTFALL
SEARCH_WORKERS = n218.SEARCH_WORKERS
BOUNDARY_FLAGS = n218.BOUNDARY_FLAGS
REQUIRED_STAGES = n218.REQUIRED_STAGES + (218,)
REQUIRED_DESIGN_STAGES = n218.REQUIRED_DESIGN_STAGES + (218,)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n218.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next218_design": n218.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next218_manifest": (
        "8de7e58dff2cc0147bc568f50cea289871114a70490b93cf0f53f68b6a6b06ac"
    ),
    "next218_catalogue": (
        "00ec8d27f4251f4b16b27ed17e253fe41ef8ce677ed097720961751a333c5bbc"
    ),
    "next218_evaluation": (
        "084665480ee3292b02cd0923ec2792aa20dd04a7e025f9f03348d0ad2bc6c1fc"
    ),
    "next218_formula": (
        "09803f8a4e6ad32c988953cf04193b836d4263852c1b8158490925bffe49cb65"
    ),
    "next218_search": (
        "fffd0a42beed0fcc0e5e13c1d7eead280ba7f0ebc4f7c2064b71b92ce3130ba5"
    ),
}


select_diagnostic_candidates = n218.n217.select_diagnostic_candidates
candidate_key_sha256 = n218.n217.candidate_key_sha256
select_closest_residual = n218.n217.select_closest_residual


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n218._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n218.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[218],
    )
    paths["next218_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next218_manifest": roots["next218"] / n218.MANIFEST_NAME,
            "next218_catalogue": roots["next218"] / n218.CATALOGUE_NAME,
            "next218_evaluation": roots["next218"] / n218.EVALUATION_NAME,
            "next218_formula": roots["next218"] / n218.FORMULA_NAME,
            "next218_search": roots["next218"] / n218.SEARCH_NAME,
        }
    )
    return paths


def _verify_next218(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Verify NEXT218 and return its exact diagnostic population."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next218_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next218_design"]
    eligible, eligible214, primary_key, start_key, formula214 = (
        n218._verify_next217(prior_paths, prior_hashes)
    )
    manifest = json.loads(paths["next218_manifest"].read_text())
    catalogue = json.loads(paths["next218_catalogue"].read_text())
    evaluation = json.loads(paths["next218_evaluation"].read_text())
    formula = json.loads(paths["next218_formula"].read_text())
    published_all = pd.read_parquet(paths["next218_search"])
    expected_outputs = {
        n218.CATALOGUE_NAME: input_hashes["next218_catalogue"],
        n218.EVALUATION_NAME: input_hashes["next218_evaluation"],
        n218.FORMULA_NAME: input_hashes["next218_formula"],
        n218.SEARCH_NAME: input_hashes["next218_search"],
    }
    counts = evaluation.get("counts", {})
    if (
        manifest.get("protocol") != n218.PROTOCOL
        or manifest.get("candidate_count") != n218.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_hypothesis_count")
        != n218.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != n218.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next219_diagnostic_authorized") is not True
        or manifest.get("next219_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next219_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next218_lower_anchored_relief_search.py"
        )
        != EXPECTED_NEXT218_SOURCE_SHA256
        or _sha256_file(Path(n218.__file__).resolve())
        != EXPECTED_NEXT218_SOURCE_SHA256
        or catalogue.get("protocol") != n218.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next218_design"]
        or catalogue.get("candidate_count") != n218.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("active_scores_cannot_cross_lower_boundary") is not True
        or catalogue.get("normalization_fit_uses_endpoint") is not False
        or evaluation.get("protocol") != n218.PROTOCOL
        or evaluation.get("candidate_count") != n218.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next219_diagnostic_authorized") is not True
        or evaluation.get("next219_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next219_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or counts.get("passes_source_auc_gates") != EXPECTED_CANDIDATE_COUNT
        or counts.get("passes_safe_all_cells") != EXPECTED_CANDIDATE_COUNT
        or counts.get("passes_broad_all_cells") != 0
        or counts.get("passes_all_discovery_gates") != 0
        or counts.get("passes_auc_and_safe_but_not_broad")
        != EXPECTED_CANDIDATE_COUNT
        or formula.get("protocol") != n218.PROTOCOL
        or formula.get("dft_values_used_by_executable_formula") is not False
        or formula.get("learned_energy_force_stress_proxy_used") is not False
        or formula.get("model_or_proxy_potential_used") is not False
        or formula.get("physical_relaxation_executed") is not False
        or len(published_all) != n218.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT219 NEXT218 provenance differs")
    published = select_diagnostic_candidates(published_all)
    if (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(published) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT219 frozen diagnostic population differs")
    return (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    )


def run_lower_anchored_broad_diagnostic(
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
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT219 BROAD residual diagnostic."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT219 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT219 design path universe differs")
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
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT219 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT219 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT219 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    ) = _verify_next218(paths, input_hashes)
    combined, _, base_score, base_support, endpoint = (
        n218.n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=start_key,
            formula=formula214,
        )
    )
    next214_table = pd.read_parquet(paths["next214_search"])
    accepted = next214_table.loc[
        next214_table["depth"].eq(3)
        & next214_table["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & next214_table["proposed_amplitude_fraction"].eq(0.0625)
    ]
    if len(accepted) != 1:
        raise ValueError("NEXT219 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    all_specs = n218.build_anchored_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if (
        len(all_specs) != n218.EXPECTED_CANDIDATE_COUNT
        or {str(spec["candidate_key"]) for spec in all_specs}
        != set(published_all["candidate_key"].astype(str))
    ):
        raise ValueError("NEXT219 NEXT218 candidate universe differs")
    diagnostic_keys = set(published["candidate_key"].astype(str))
    selected_specs = [
        spec for spec in all_specs if str(spec["candidate_key"]) in diagnostic_keys
    ]
    if len(selected_specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT219 selected specification universe differs")
    combined_virtual, terms, runtime = n218.materialize_anchored_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=selected_specs,
    )
    rerun = (
        n218.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel(
            features=combined_virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=runtime,
            workers=search_workers,
        )
    )
    n218.n215.n214.n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
    )
    folds = n218.n215.n214.n164.assign_group_folds(
        combined_virtual["reduced_formula"].astype(str).to_numpy()
    )
    sources = combined_virtual["source_dataset"].astype(str).to_numpy()
    cells = n218.n215.n214.n164.build_source_fold_cells(
        source=sources, folds=folds
    )
    pauling_by_cell = {
        str(cell["cell_id"]): n218.n215.n214.n164._pauling_baseline(
            combined_virtual.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    spec_by_key = {str(spec["candidate_key"]): spec for spec in selected_specs}
    published_by_key = published.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n218.n215.n214.n164._term_risk(combined_virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT219 candidate support differs from NEXT214")
        tables = n218.n215.n214.n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT219 candidate has no threshold table")
        residual = n218.n215.n214.n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT219 contradicts NEXT218 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "hypothesis": spec["hypothesis"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "amplitude_fraction": float(spec["amplitude_fraction"]),
                "q_lo": spec["q_lo"],
                "q_hi": spec["q_hi"],
                "anchored_active_rows": int(row["anchored_active_rows"]),
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(residual["best_threshold"]),
                "failed_constraint_count": int(residual["failed_constraint_count"]),
                "normalized_shortfall_sum": float(
                    residual["normalized_shortfall_sum"]
                ),
                "eligible_threshold_count": int(
                    residual["eligible_threshold_count"]
                ),
                "failures_json": json.dumps(
                    residual["failures"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    per_candidate = pd.DataFrame(records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)
    base_rows = per_candidate.loc[per_candidate["hypothesis"].isna()]
    if (
        len(base_rows) != 1
        or int(base_rows.iloc[0]["failed_constraint_count"])
        != EXPECTED_BASE_FAILED_COUNT
        or not math.isclose(
            float(base_rows.iloc[0]["normalized_shortfall_sum"]),
            EXPECTED_BASE_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("NEXT219 NEXT214 baseline residual differs")
    closest = select_closest_residual(per_candidate)
    closest_new = select_closest_residual(
        per_candidate.loc[per_candidate["hypothesis"].notna()]
    )
    closest_shortfall = float(closest["normalized_shortfall_sum"])
    closest_failures = int(closest["failed_constraint_count"])
    closest_improves = bool(
        closest_failures < EXPECTED_BASE_FAILED_COUNT
        or (
            closest_failures == EXPECTED_BASE_FAILED_COUNT
            and closest_shortfall + 1.0e-12 < EXPECTED_BASE_SHORTFALL
        )
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "hypothesis": None if pd.isna(row["hypothesis"]) else str(row["hypothesis"]),
            "feature": None if pd.isna(row["feature"]) else str(row["feature"]),
            "direction": None if pd.isna(row["direction"]) else str(row["direction"]),
            "amplitude_fraction": float(row["amplitude_fraction"]),
            "q_lo": None if pd.isna(row["q_lo"]) else float(row["q_lo"]),
            "q_hi": None if pd.isna(row["q_hi"]) else float(row["q_hi"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(row["normalized_shortfall_sum"]),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next218_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": candidate_key_sha256(published),
        "next218_record_population_reproduced": True,
        "next218_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "closest_new_candidate": residual_record(closest_new),
        "next214_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next214_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next214": (
            EXPECTED_BASE_SHORTFALL - closest_shortfall
        ),
        "improves_over_next214_global_residual": closest_improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "lower_anchored_relief_branch_closed": not closest_improves,
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(
            n218.n215.n214.n164.__file__
        ).resolve(),
        "src/next218_lower_anchored_relief_search.py": Path(n218.__file__).resolve(),
        "src/next219_lower_anchored_broad_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        table_path = staging / TABLE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(table_path, index=False)
        outputs = [diagnostic_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": len(per_candidate),
            "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
            "next218_record_population_reproduced": True,
            "next218_candidate_universe_reproduced": True,
            "next218_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": closest_improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "lower_anchored_relief_branch_closed": not closest_improves,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in outputs
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT219 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT219 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--scigen-discovery-endpoint-dir", type=Path, required=True
    )
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--wyformer-discovery-endpoint-dir", type=Path, required=True
    )
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_lower_anchored_broad_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={
            stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES
        },
        next135_freeze_path=args.next135_freeze_path,
        design_paths={
            stage: getattr(args, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "candidate_key_sha256",
    "run_lower_anchored_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
