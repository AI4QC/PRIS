#!/usr/bin/env python3
"""Diagnose the frozen NEXT223 eligible BROAD residual population."""

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

import src.next223_dual_evidence_consensus_search as n223
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next224-dual-evidence-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT224_DUAL_EVIDENCE_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next224_dual_evidence_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n223.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT223_SOURCE_SHA256 = (
    "aaf4843f5461f40218cb4c52f2f2e481e43447a5ad5219e59f118851f8470219"
)
EXPECTED_CANDIDATE_COUNT = 5147
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "b8dbdb72a27296eda51c076c1ad36a8ad0360515f185f83a510ecb497f80caf3"
)
EXPECTED_BASE_FAILED_COUNT = n223.EXPECTED_NEXT222_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n223.EXPECTED_NEXT222_SHORTFALL
SEARCH_WORKERS = n223.SEARCH_WORKERS
BOUNDARY_FLAGS = n223.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n223.REQUIRED_STAGES, 223)
REQUIRED_DESIGN_STAGES = (*n223.REQUIRED_DESIGN_STAGES, 223)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n223.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next223_design": n223.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next223_manifest": (
        "f69149512ed38de3a162e012830be545da81a8cbfa1134888ab06db5fc7a51bb"
    ),
    "next223_catalogue": (
        "5317af5b690b209fb20d1d3ec38623b8a64dac8d80f35ea62893bfbdb8c62b65"
    ),
    "next223_evaluation": (
        "82d20ec1f147d4fee031797cc9b95c28da7e56f880c10b05585ae464673fc8cd"
    ),
    "next223_formula": (
        "b71992d6b2e0b09bca75c7147cd5d1c39a600201ee8a62085ed674f04e2679f4"
    ),
    "next223_search": (
        "2c4a9de4fe230b509681ace0095727e289bbeebe5c098fe91611cbe77a0512b1"
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
        raise ValueError("NEXT224 published candidate columns differ")
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
    if frame.empty:
        raise ValueError("NEXT224 residual population is empty")
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
    paths = n223._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n223.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[223],
    )
    paths["next223_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next223_manifest": roots["next223"] / n223.MANIFEST_NAME,
            "next223_catalogue": roots["next223"] / n223.CATALOGUE_NAME,
            "next223_evaluation": roots["next223"] / n223.EVALUATION_NAME,
            "next223_formula": roots["next223"] / n223.FORMULA_NAME,
            "next223_search": roots["next223"] / n223.SEARCH_NAME,
        }
    )
    return paths


def _verify_next223(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    str,
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next223_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next223_design"]
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
    ) = n223._verify_next222(prior_paths, prior_hashes)
    manifest = json.loads(paths["next223_manifest"].read_text())
    catalogue = json.loads(paths["next223_catalogue"].read_text())
    evaluation = json.loads(paths["next223_evaluation"].read_text())
    formula = json.loads(paths["next223_formula"].read_text())
    table = pd.read_parquet(paths["next223_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n223.CATALOGUE_NAME: input_hashes["next223_catalogue"],
        n223.EVALUATION_NAME: input_hashes["next223_evaluation"],
        n223.FORMULA_NAME: input_hashes["next223_formula"],
        n223.SEARCH_NAME: input_hashes["next223_search"],
    }
    if (
        manifest.get("protocol") != n223.PROTOCOL
        or manifest.get("candidate_count") != n223.EXPECTED_TOTAL_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count")
        != n223.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("diagonal_control_count") != n223.EXPECTED_CONTROL_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("next224_diagnostic_authorized") is not True
        or manifest.get("next224_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next224_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next223_dual_evidence_consensus_search.py"
        )
        != EXPECTED_NEXT223_SOURCE_SHA256
        or _sha256_file(Path(n223.__file__).resolve())
        != EXPECTED_NEXT223_SOURCE_SHA256
        or catalogue.get("candidate_count") != n223.EXPECTED_TOTAL_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next224_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next224_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind")
        != "allocated_dual_evidence_consensus_x0_no_dft_score"
        or len(table) != n223.EXPECTED_TOTAL_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT224 NEXT223 provenance differs")
    return (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
        selected,
    )


def run_dual_evidence_broad_diagnostic(
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
    """Reproduce NEXT223 and compute its frozen discovery-only BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT224 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT224 design path universe differs")
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
        raise ValueError("NEXT224 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT224 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT224 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        _,
        published,
    ) = _verify_next223(paths, input_hashes)
    combined, _, base_score, base_support, endpoint = (
        n223.n222.n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=base_start_key,
            formula=formula214,
        )
    )
    current_delta = n223._reconstruct_next222_delta(
        features=combined,
        base_score=base_score,
        support=base_support,
        formula=formula222,
    )
    next214_table = pd.read_parquet(paths["next214_search"])
    accepted214 = next214_table.loc[
        next214_table["depth"].eq(3)
        & next214_table["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & next214_table["proposed_amplitude_fraction"].eq(0.0625)
    ]
    if len(accepted214) != 1:
        raise ValueError("NEXT224 NEXT214 base identity differs")
    initial_specs = n223.n222.n220.build_signed_candidate_specs(
        base_candidate_key=str(accepted214.iloc[0]["candidate_key"]),
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    normalizations = {
        str(spec["hypothesis"]): dict(spec)
        for spec in initial_specs
        if spec["hypothesis"] is not None
    }
    all_specs = n223.build_dual_evidence_candidate_specs(
        current_path_key=current_key,
        current_terms=[dict(value) for value in formula222["terms"]],
        normalizations=normalizations,
    )
    keys = set(published["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT224 selected specification universe differs")
    virtual, terms, runtime, activity = n223.materialize_dual_evidence_candidates(
        features=combined,
        base_score=base_score,
        current_delta=current_delta,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
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
    n164 = n223.n222.n215.n214.n164
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
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
    published_by_key = published.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT224 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT224 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT224 contradicts NEXT223 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "protection_hypothesis": spec["protection_hypothesis"],
                "protection_feature": spec["protection_feature"],
                "risk_hypothesis": spec["risk_hypothesis"],
                "risk_feature": spec["risk_feature"],
                "beta_fraction": float(spec["beta_fraction"]),
                "protection_budget_fraction": float(
                    spec["protection_budget_fraction"]
                ),
                "pair_active_rows": activity[key]["rows"],
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(residual["best_threshold"]),
                "failed_constraint_count": int(
                    residual["failed_constraint_count"]
                ),
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
        raise RuntimeError("NEXT224 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
            "normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        },
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "protection_hypothesis": str(row["protection_hypothesis"]),
            "protection_feature": str(row["protection_feature"]),
            "risk_hypothesis": str(row["risk_hypothesis"]),
            "risk_feature": str(row["risk_feature"]),
            "beta_fraction": float(row["beta_fraction"]),
            "protection_budget_fraction": float(
                row["protection_budget_fraction"]
            ),
            "pair_active_rows": int(row["pair_active_rows"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(
                row["normalized_shortfall_sum"]
            ),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next223_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next223_record_population_reproduced": True,
        "next223_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next222_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next222_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next222": (
            EXPECTED_BASE_SHORTFALL
            - float(closest["normalized_shortfall_sum"])
        ),
        "improves_over_next222_global_residual": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "dual_evidence_branch_closed": not improves,
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
        "src/next223_dual_evidence_consensus_search.py": Path(
            n223.__file__
        ).resolve(),
        "src/next224_dual_evidence_broad_diagnostic.py": Path(__file__).resolve(),
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
            "next223_record_population_reproduced": True,
            "next223_candidate_universe_reproduced": True,
            "next223_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "dual_evidence_branch_closed": not improves,
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
            raise RuntimeError("NEXT224 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT224 source changed before publication")
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
        parser.add_argument(
            f"--next{stage}-design-path", type=Path, required=True
        )
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_dual_evidence_broad_diagnostic(
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
    "run_dual_evidence_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
