#!/usr/bin/env python3
"""Diagnose the frozen NEXT231 eligible BROAD residual population."""

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

import src.next231_stagewise_margin_local_search as n231
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next232-stagewise-margin-local-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT232_STAGEWISE_MARGIN_LOCAL_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next232_stagewise_margin_local_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n231.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT231_SOURCE_SHA256 = (
    "a0edc4dae393073012eb0918fb6c7145188aad36e923ec08f948c820e7d77e8c"
)
EXPECTED_CANDIDATE_COUNT = 450
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "5a0f44bccbd7fcb57f7a79caf0c1dc6e25acd689cd5943cfd61f08c93254f199"
)
EXPECTED_BASE_FAILED_COUNT = n231.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n231.EXPECTED_BASE_SHORTFALL
SEARCH_WORKERS = n231.SEARCH_WORKERS
BOUNDARY_FLAGS = n231.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n231.REQUIRED_STAGES, 231)
REQUIRED_DESIGN_STAGES = (*n231.REQUIRED_DESIGN_STAGES, 231)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n231.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next231_design": n231.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next231_manifest": (
        "228cd6b44ad7aab90bac4af4b2471bd0db0f854b83082ec9b41883caec5e9c0d"
    ),
    "next231_catalogue": (
        "7770217c7f5ee959162604218579c68b34b4b90db449c79f123118f312f6beef"
    ),
    "next231_evaluation": (
        "f086878314744dafd2e7467b4d4a84286fa40549ad387dcb7cf0d2e00cf174c7"
    ),
    "next231_formula": (
        "05f2d9c926129aa3242943751d3e5a03a38a9a81ad25368cceecc9335e9b7ad7"
    ),
    "next231_search": (
        "7296756ee110b2abc73d47aa514131af9f8183a95a2ed73bc5c86b53583db7e7"
    ),
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_key",
        "eligible_new_candidate",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if required - set(published.columns):
        raise ValueError("NEXT232 published candidate columns differ")
    mask = (
        published["eligible_new_candidate"].fillna(False).astype(bool)
        & published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(frame["candidate_key"].astype(str))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    required = {
        "candidate_key",
        "failed_constraint_count",
        "normalized_shortfall_sum",
    }
    if frame.empty or required - set(frame.columns):
        raise ValueError("NEXT232 residual population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n231._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n231.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[231],
    )
    paths["next231_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next231_manifest": roots["next231"] / n231.MANIFEST_NAME,
            "next231_catalogue": roots["next231"] / n231.CATALOGUE_NAME,
            "next231_evaluation": roots["next231"] / n231.EVALUATION_NAME,
            "next231_formula": roots["next231"] / n231.FORMULA_NAME,
            "next231_search": roots["next231"] / n231.SEARCH_NAME,
        }
    )
    return paths


def _verify_next231(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...], tuple[str, ...], str, str, dict[str, object], str,
    dict[str, object], tuple[str, ...], pd.DataFrame, dict[str, object],
    tuple[str, ...], pd.DataFrame,
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next231_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next231_design"]
    prior = n231._verify_next230(prior_paths, prior_hashes)
    manifest = json.loads(paths["next231_manifest"].read_text())
    catalogue = json.loads(paths["next231_catalogue"].read_text())
    evaluation = json.loads(paths["next231_evaluation"].read_text())
    formula = json.loads(paths["next231_formula"].read_text())
    table = pd.read_parquet(paths["next231_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n231.CATALOGUE_NAME: input_hashes["next231_catalogue"],
        n231.EVALUATION_NAME: input_hashes["next231_evaluation"],
        n231.FORMULA_NAME: input_hashes["next231_formula"],
        n231.SEARCH_NAME: input_hashes["next231_search"],
    }
    if (
        manifest.get("protocol") != n231.PROTOCOL
        or manifest.get("candidate_count") != n231.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count")
        != n231.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("next232_diagnostic_authorized") is not True
        or manifest.get("next232_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next232_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next231_stagewise_margin_local_search.py"
        )
        != EXPECTED_NEXT231_SOURCE_SHA256
        or _sha256_file(Path(n231.__file__).resolve())
        != EXPECTED_NEXT231_SOURCE_SHA256
        or any(manifest.get(k) is not value for k, value in BOUNDARY_FLAGS.items())
        or catalogue.get("candidate_count") != n231.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next232_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next232_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind")
        != "two_stage_triangular_margin_local_x0_no_dft_score"
        or any(
            formula.get(key) is not False
            for key in (
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            )
        )
        or len(table) != n231.EXPECTED_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT232 NEXT231 provenance differs")
    return (*prior, selected)


def run_stagewise_margin_local_broad_diagnostic(
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
    """Reproduce NEXT231 and compute its unchanged discovery BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT232 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT232 design path universe differs")
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
        raise ValueError("NEXT232 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT232 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT232 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        eligible227,
        published228,
        closest229,
        eligible230,
        published231,
    ) = _verify_next231(paths, input_hashes)
    combined, _, base_score, base_support, endpoint, base_spec = (
        n231.n230._reconstruct_next229_frontier(
            paths=paths,
            eligible=eligible,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
            eligible227=eligible227,
            published228=published228,
            closest229=closest229,
        )
    )
    base_key = str(closest229["candidate_key"])
    all_specs = n231.build_stagewise_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible230,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    keys = set(published231["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT232 selected specification universe differs")
    virtual, terms, runtime, activity = n231.materialize_stagewise_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n231.n228.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    rerun = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    n164 = n231.n228.n223.n222.n215.n214.n164
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published231
    )
    folds = n164.assign_group_folds(
        combined["reduced_formula"].astype(str).to_numpy()
    )
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = n164.build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): n164._pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    published_by_key = published231.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT232 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT232 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT232 contradicts NEXT231 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "first_stage_hypothesis": base_spec["hypothesis"],
                "hypothesis": spec["hypothesis"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "q_lo": float(spec["q_lo"]),
                "q_hi": float(spec["q_hi"]),
                "local_width_fraction": float(spec["local_width_fraction"]),
                "amplitude_fraction": float(spec["amplitude_fraction"]),
                "local_active_rows": activity[key]["rows"],
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
    if (
        len(per_candidate) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(per_candidate) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise RuntimeError("NEXT232 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n231.n228.n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
            "normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        },
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "first_stage_hypothesis": str(row["first_stage_hypothesis"]),
            "second_stage_hypothesis": str(row["hypothesis"]),
            "second_stage_feature": str(row["feature"]),
            "second_stage_direction": str(row["direction"]),
            "second_stage_q_lo": float(row["q_lo"]),
            "second_stage_q_hi": float(row["q_hi"]),
            "second_stage_local_width_fraction": float(
                row["local_width_fraction"]
            ),
            "second_stage_amplitude_fraction": float(row["amplitude_fraction"]),
            "second_stage_active_rows": int(row["local_active_rows"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(row["normalized_shortfall_sum"]),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next231_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next231_record_population_reproduced": True,
        "next231_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next229_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next229_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next229": (
            EXPECTED_BASE_SHORTFALL
            - float(closest["normalized_shortfall_sum"])
        ),
        "strict_improvement_over_next229_diagnostic": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "stagewise_margin_local_branch_closed": not improves,
        "continuation_requires_new_preoutcome_freeze": improves,
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
            n164.__file__
        ).resolve(),
        "src/next231_stagewise_margin_local_search.py": Path(
            n231.__file__
        ).resolve(),
        "src/next232_stagewise_margin_local_broad_diagnostic.py": Path(
            __file__
        ).resolve(),
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
            "next231_record_population_reproduced": True,
            "next231_candidate_universe_reproduced": True,
            "next231_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "strict_improvement_over_next229_diagnostic": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "stagewise_margin_local_branch_closed": not improves,
            "continuation_requires_new_preoutcome_freeze": improves,
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
            raise RuntimeError("NEXT232 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT232 source changed before publication")
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
    manifest = run_stagewise_margin_local_broad_diagnostic(
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
    "run_stagewise_margin_local_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
