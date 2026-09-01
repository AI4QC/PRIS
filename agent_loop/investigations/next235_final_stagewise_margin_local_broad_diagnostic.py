#!/usr/bin/env python3
"""Diagnose the frozen NEXT234 eligible BROAD residual population."""

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

import src.next234_final_stagewise_margin_local_search as n234
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next235-final-stagewise-margin-local-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT235_FINAL_STAGEWISE_MARGIN_LOCAL_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next235_final_stagewise_margin_local_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n234.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT234_SOURCE_SHA256 = (
    "a79bbcb94d7acb05ca5599b6a1efd9f57f218f88309b87fe5a026eaea8ffe400"
)
EXPECTED_CANDIDATE_COUNT = 484
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "e8c235fe788f99644d300e53e85e3426c7fbddadd0b125376917e87aacf0330a"
)
EXPECTED_BASE_FAILED_COUNT = n234.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n234.EXPECTED_BASE_SHORTFALL
SEARCH_WORKERS = n234.SEARCH_WORKERS
BOUNDARY_FLAGS = n234.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n234.REQUIRED_STAGES, 234)
REQUIRED_DESIGN_STAGES = (*n234.REQUIRED_DESIGN_STAGES, 234)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n234.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next234_design": n234.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next234_manifest": (
        "c94eeab304ebe3ad8ea5e2f20cbf13428955b4cb7c521b73e8ef2d8befb0fa60"
    ),
    "next234_catalogue": (
        "ac7758062f4a2eedab5401e4c46e349903f254b04f848734d894d2709749f3ca"
    ),
    "next234_evaluation": (
        "88f96e7e0fa0990204f5f7b5ccbfeafd19f5b44b6f78a9f04b3f75dddae58f51"
    ),
    "next234_formula": (
        "1988ccd6c910ef411dd9bd38f17d8c5aa97be46cd6f375565c0d27404dbe7318"
    ),
    "next234_search": (
        "7089f8ffb88b81a7b8f52fbb48e55250cdd8ce226961ad94e70b875a943008ee"
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
        raise ValueError("NEXT235 published candidate columns differ")
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
        raise ValueError("NEXT235 residual population differs")
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
    paths = n234._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n234.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[234],
    )
    paths["next234_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next234_manifest": roots["next234"] / n234.MANIFEST_NAME,
            "next234_catalogue": roots["next234"] / n234.CATALOGUE_NAME,
            "next234_evaluation": roots["next234"] / n234.EVALUATION_NAME,
            "next234_formula": roots["next234"] / n234.FORMULA_NAME,
            "next234_search": roots["next234"] / n234.SEARCH_NAME,
        }
    )
    return paths


def _verify_next234(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], pd.DataFrame]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next234_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next234_design"]
    eligible233 = n234._verify_next233(prior_paths, prior_hashes)
    manifest = json.loads(paths["next234_manifest"].read_text())
    catalogue = json.loads(paths["next234_catalogue"].read_text())
    evaluation = json.loads(paths["next234_evaluation"].read_text())
    formula = json.loads(paths["next234_formula"].read_text())
    table = pd.read_parquet(paths["next234_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n234.CATALOGUE_NAME: input_hashes["next234_catalogue"],
        n234.EVALUATION_NAME: input_hashes["next234_evaluation"],
        n234.FORMULA_NAME: input_hashes["next234_formula"],
        n234.SEARCH_NAME: input_hashes["next234_search"],
    }
    if (
        manifest.get("protocol") != n234.PROTOCOL
        or manifest.get("candidate_count") != n234.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count")
        != n234.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("next235_diagnostic_authorized") is not True
        or manifest.get("next235_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next235_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next234_final_stagewise_margin_local_search.py"
        )
        != EXPECTED_NEXT234_SOURCE_SHA256
        or _sha256_file(Path(n234.__file__).resolve())
        != EXPECTED_NEXT234_SOURCE_SHA256
        or any(manifest.get(k) is not value for k, value in BOUNDARY_FLAGS.items())
        or catalogue.get("candidate_count") != n234.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next235_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next235_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind")
        != "three_stage_triangular_margin_local_x0_no_dft_score"
        or any(
            formula.get(key) is not False
            for key in (
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            )
        )
        or len(table) != n234.EXPECTED_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT235 NEXT234 provenance differs")
    return eligible233, selected


def run_final_stagewise_margin_local_broad_diagnostic(
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
    """Reproduce NEXT234 and compute its unchanged discovery BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT235 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT235 design path universe differs")
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
        raise ValueError("NEXT235 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT235 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT235 formal input identity differs: {differing}")
    eligible233, published234 = _verify_next234(paths, input_hashes)
    combined, base_score, base_support, endpoint, _ = (
        n234.n233._verify_and_reconstruct_next232(paths, input_hashes)
    )
    diagnostic232 = json.loads(paths["next232_diagnostic"].read_text())
    base_key = str(diagnostic232["global_closest"]["candidate_key"])
    base_spec = json.loads(base_key)
    all_specs = n234.build_final_stagewise_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible233,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    keys = set(published234["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT235 selected specification universe differs")
    virtual, terms, runtime, activity = n234.materialize_final_stagewise_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n234.n228.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
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
    n164 = n234.n228.n223.n222.n215.n214.n164
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published234
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
    published_by_key = published234.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT235 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT235 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT235 contradicts NEXT234 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "second_stage_base_hypothesis": base_spec["hypothesis"],
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
        raise RuntimeError("NEXT235 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n234.n228.n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
            "normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        },
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "second_stage_base_hypothesis": str(
                row["second_stage_base_hypothesis"]
            ),
            "third_stage_hypothesis": str(row["hypothesis"]),
            "third_stage_feature": str(row["feature"]),
            "third_stage_direction": str(row["direction"]),
            "third_stage_q_lo": float(row["q_lo"]),
            "third_stage_q_hi": float(row["q_hi"]),
            "third_stage_local_width_fraction": float(
                row["local_width_fraction"]
            ),
            "third_stage_amplitude_fraction": float(row["amplitude_fraction"]),
            "third_stage_active_rows": int(row["local_active_rows"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(row["normalized_shortfall_sum"]),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next234_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next234_record_population_reproduced": True,
        "next234_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next232_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next232_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next232": (
            EXPECTED_BASE_SHORTFALL
            - float(closest["normalized_shortfall_sum"])
        ),
        "strict_improvement_over_next232_diagnostic": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "final_stagewise_margin_local_branch_closed": True,
        "continuation_requires_new_preoutcome_freeze": False,
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
        "src/next234_final_stagewise_margin_local_search.py": Path(
            n234.__file__
        ).resolve(),
        "src/next235_final_stagewise_margin_local_broad_diagnostic.py": Path(
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
            "next234_record_population_reproduced": True,
            "next234_candidate_universe_reproduced": True,
            "next234_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "strict_improvement_over_next232_diagnostic": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "final_stagewise_margin_local_branch_closed": True,
            "continuation_requires_new_preoutcome_freeze": False,
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
            raise RuntimeError("NEXT235 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT235 source changed before publication")
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
    manifest = run_final_stagewise_margin_local_broad_diagnostic(
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
    "run_final_stagewise_margin_local_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
