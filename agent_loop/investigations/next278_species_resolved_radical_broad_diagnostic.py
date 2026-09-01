#!/usr/bin/env python3
"""Diagnose the frozen NEXT277 species-resolved radical BROAD residual."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next270_prv_broad_diagnostic as n270
import src.next277_species_resolved_radical_margin_local_search as n277
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next278-species-resolved-radical-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT278_SPECIES_RESOLVED_RADICAL_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next278_species_resolved_radical_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n277.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT270_SOURCE_SHA256 = (
    "c950ae130e820b2e799abb284c8bcf079afbf923379fde7e586b2a3328b2cf6d"
)
EXPECTED_NEXT277_SOURCE_SHA256 = (
    "593fef46cdc7c367fe7f8587e81e175307cf1bf61e07601848c3b3bfc66f0487"
)
EXPECTED_CANDIDATE_COUNT = 18
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "4e8a9cbaf75f9ae9721de15610d3be0ed3f55d214f5f54038c8ee60f2143fbbe"
)
EXPECTED_NEXT277_SEARCH_SHA256 = (
    "3742a137e2c338f2be4d6db10ce2a6dcfa568a5d538696307fec96dd70db1b5c"
)
NEXT270_REFERENCE_FAILED_COUNT = 5
NEXT270_REFERENCE_SHORTFALL = 0.0955435292756307
SEARCH_WORKERS = n277.SEARCH_WORKERS
BOUNDARY_FLAGS = n277.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n277.REQUIRED_STAGES, 277)
REQUIRED_DESIGN_STAGES = (*n277.REQUIRED_DESIGN_STAGES, 277)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n277.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next277_design": n277.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next277_manifest": (
        "b6f61f08c67df47ed8e463448f1dcdd0391d0f11c84296438cdb3ebaab6350c9"
    ),
    "next277_catalogue": (
        "065233a9196c2ef8f4190b4e25a3b337cb001a756d02e945844bb77a910cfea8"
    ),
    "next277_evaluation": (
        "a0f38ce4a38eb4d212b96ccfde4aa4406b9d297dd833228fe90b1f6cb41d4252"
    ),
    "next277_formula": (
        "8d92978a0b673bdc8edd6c1eb1115b9a35c3dd276bbad7be053b405308bd4c7d"
    ),
    "next277_search": EXPECTED_NEXT277_SEARCH_SHA256,
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    try:
        return n270.select_diagnostic_candidates(published)
    except ValueError as exc:
        raise ValueError("NEXT278 published candidate columns differ") from exc


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    try:
        return n270.candidate_key_sha256(frame)
    except ValueError as exc:
        raise ValueError("NEXT278 candidate identity table differs") from exc


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    try:
        return n270.select_closest_residual(frame)
    except ValueError as exc:
        raise ValueError("NEXT278 residual population differs") from exc


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n277._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n277.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[277],
    )
    paths["next277_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next277_manifest": roots["next277"] / n277.MANIFEST_NAME,
            "next277_catalogue": roots["next277"] / n277.CATALOGUE_NAME,
            "next277_evaluation": roots["next277"] / n277.EVALUATION_NAME,
            "next277_formula": roots["next277"] / n277.FORMULA_NAME,
            "next277_search": roots["next277"] / n277.SEARCH_NAME,
        }
    )
    return paths


def _verify_next277(paths: Mapping[str, Path], input_hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n277.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next277_design"]
    prior_hashes = {key: input_hashes[key] for key in n277.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next277_design"]
    prior = n277._verify_next276(prior_paths, prior_hashes)
    manifest = json.loads(paths["next277_manifest"].read_text())
    catalogue = json.loads(paths["next277_catalogue"].read_text())
    evaluation = json.loads(paths["next277_evaluation"].read_text())
    formula = json.loads(paths["next277_formula"].read_text())
    table = pd.read_parquet(paths["next277_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n277.CATALOGUE_NAME: input_hashes["next277_catalogue"],
        n277.EVALUATION_NAME: input_hashes["next277_evaluation"],
        n277.FORMULA_NAME: input_hashes["next277_formula"],
        n277.SEARCH_NAME: input_hashes["next277_search"],
    }
    if (
        _sha256_file(Path(n270.__file__).resolve()) != EXPECTED_NEXT270_SOURCE_SHA256
        or _sha256_file(Path(n277.__file__).resolve())
        != EXPECTED_NEXT277_SOURCE_SHA256
        or manifest.get("protocol") != n277.PROTOCOL
        or manifest.get("candidate_count") != n277.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count")
        != n277.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("next278_diagnostic_authorized") is not True
        or manifest.get("next278_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next278_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next277_species_resolved_radical_margin_local_search.py"
        )
        != EXPECTED_NEXT277_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("candidate_count") != n277.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next278_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next278_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind")
        != "species_resolved_radical_triangular_margin_local_x0_no_dft_score"
        or formula.get("selected") is not True
        or any(formula.get(key) is None for key in ("hypothesis", "feature", "direction"))
        or any(
            formula.get(key) is not False
            for key in (
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            )
        )
        or len(table) != n277.EXPECTED_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT278 NEXT277 provenance differs")
    return (*prior, selected)


def run_species_resolved_radical_broad_diagnostic(
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
    """Reproduce NEXT277 and compute its unchanged discovery BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT278 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT278 design path universe differs")
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
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT278 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT278 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT278 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        prvs_tables,
        eligible276,
        published277,
    ) = _verify_next277(paths, input_hashes)
    combined, feature_tables, base_score, base_support, endpoint, _ = (
        n277.n227._reconstruct_next224_frontier(
            paths=paths,
            eligible=eligible_prior,
            eligible214=eligible214,
            primary_key=primary_key,
            base_start_key=base_start_key,
            formula214=formula214,
            current_key=current_key,
            formula222=formula222,
        )
    )
    n277._attach_prvs_features(
        combined=combined,
        feature_tables=feature_tables,
        prvs_tables=prvs_tables,
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    all_specs = n277.build_prvs_candidate_specs(
        base_candidate_key=base_key,
        eligible_table=eligible276,
    )
    keys = set(published277["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT278 selected specification universe differs")
    virtual, terms, runtime, activity = n277.materialize_prvs_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n277.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
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
    n164 = n277.n223.n222.n215.n214.n164
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published277
    )
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
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
    published_by_key = published277.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT278 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score,
            supported=support,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT278 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT278 contradicts NEXT277 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "hypothesis": str(spec["hypothesis"]),
                "feature": str(spec["feature"]),
                "direction": str(spec["direction"]),
                "q_lo": float(spec["q_lo"]),
                "q_hi": float(spec["q_hi"]),
                "local_width_fraction": float(spec["local_width_fraction"]),
                "amplitude_fraction": float(spec["amplitude_fraction"]),
                "local_active_rows": activity[key]["rows"],
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(residual["best_threshold"]),
                "failed_constraint_count": int(residual["failed_constraint_count"]),
                "normalized_shortfall_sum": float(residual["normalized_shortfall_sum"]),
                "eligible_threshold_count": int(residual["eligible_threshold_count"]),
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
        raise RuntimeError("NEXT278 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n277.n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": NEXT270_REFERENCE_FAILED_COUNT,
            "normalized_shortfall_sum": NEXT270_REFERENCE_SHORTFALL,
        },
    )

    def residual_record(row: pd.Series) -> dict[str, object]:
        return {
            "candidate_key": str(row["candidate_key"]),
            "hypothesis": str(row["hypothesis"]),
            "feature": str(row["feature"]),
            "direction": str(row["direction"]),
            "q_lo": float(row["q_lo"]),
            "q_hi": float(row["q_hi"]),
            "local_width_fraction": float(row["local_width_fraction"]),
            "amplitude_fraction": float(row["amplitude_fraction"]),
            "local_active_rows": int(row["local_active_rows"]),
            "safe_threshold": float(row["safe_threshold"]),
            "best_threshold": float(row["best_threshold"]),
            "failed_constraint_count": int(row["failed_constraint_count"]),
            "normalized_shortfall_sum": float(row["normalized_shortfall_sum"]),
            "failures": json.loads(str(row["failures_json"])),
        }

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next277_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next277_record_population_reproduced": True,
        "next277_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next270_reference_failed_constraint_count": NEXT270_REFERENCE_FAILED_COUNT,
        "next270_reference_normalized_shortfall_sum": NEXT270_REFERENCE_SHORTFALL,
        "normalized_shortfall_reduction_from_next270": (
            NEXT270_REFERENCE_SHORTFALL
            - float(closest["normalized_shortfall_sum"])
        ),
        "strict_improvement_over_next270_diagnostic": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "species_resolved_radical_certificate_branch_closed": bool(not improves),
        "continuation_requires_new_preoutcome_freeze": bool(improves),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(n164.__file__).resolve(),
        "src/next277_species_resolved_radical_margin_local_search.py": Path(
            n277.__file__
        ).resolve(),
        "src/next278_species_resolved_radical_broad_diagnostic.py": Path(
            __file__
        ).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
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
            "next277_record_population_reproduced": True,
            "next277_candidate_universe_reproduced": True,
            "next277_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "strict_improvement_over_next270_diagnostic": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "species_resolved_radical_certificate_branch_closed": bool(not improves),
            "continuation_requires_new_preoutcome_freeze": bool(improves),
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
            raise RuntimeError("NEXT278 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT278 source changed before publication")
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
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_species_resolved_radical_broad_diagnostic(
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
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
