#!/usr/bin/env python3
"""Diagnose the frozen NEXT313 PCAE BROAD residual without a new search."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next262_pvbp_broad_diagnostic as n262
import src.next313_pcae_margin_local_search as n313
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next314-pcae-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT314_PCAE_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next314_pcae_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n313.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT262_SOURCE_SHA256 = (
    "2e074b9575985fadf1e680b367c72ab2e3c9b21bbd996ad3e1af1f00b73f7421"
)
EXPECTED_NEXT313_SOURCE_SHA256 = (
    "aa27e34acb9ca67eeeb43055e399d0f81a7c416744d769de414f6a0cf9d0a34e"
)
EXPECTED_CANDIDATE_COUNT = 27
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "edb91d965e401cca1a2c132ee59adb4dad5f466ee2f63dc4bf7b50464431c27e"
)
EXPECTED_NEXT313_SEARCH_SHA256 = (
    "b35457fe45f409f1fdd67059ba088d4b56bc52aaace45648fac9c6e95528d1de"
)
NEXT235_REFERENCE_FAILED_COUNT = n262.NEXT235_REFERENCE_FAILED_COUNT
NEXT235_REFERENCE_SHORTFALL = n262.NEXT235_REFERENCE_SHORTFALL
SEARCH_WORKERS = n313.SEARCH_WORKERS
BOUNDARY_FLAGS = n313.BOUNDARY_FLAGS
REQUIRED_STAGES = (*n313.REQUIRED_STAGES, 313)
REQUIRED_DESIGN_STAGES = (*n313.REQUIRED_DESIGN_STAGES, 313)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n313.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next313_design": n313.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next313_manifest": "435d6e8fcd786e0c6f8d7b2e535656f2903310031a76156ec5ec6a23809cb8e3",
    "next313_catalogue": "6f7f836c9086d9161b15924ed866b8e152bc26187458a3063848d0c765fff786",
    "next313_evaluation": "11d130334937282142e74985c057e60f5e6b3084d96e413e92b8d4fcdc6303c2",
    "next313_formula": "b5d904e14740b27b27a169c27ef895e009dff696ce1b0aeaef8682652d70abb6",
    "next313_search": EXPECTED_NEXT313_SEARCH_SHA256,
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    try:
        return n262.select_diagnostic_candidates(published)
    except ValueError as exc:
        raise ValueError("NEXT314 published candidate columns differ") from exc


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    try:
        return n262.candidate_key_sha256(frame)
    except ValueError as exc:
        raise ValueError("NEXT314 candidate identity table differs") from exc


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    try:
        return n262.select_closest_residual(frame)
    except ValueError as exc:
        raise ValueError("NEXT314 residual population differs") from exc


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n313._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={stage: design_paths[stage] for stage in n313.REQUIRED_DESIGN_STAGES},
        design_path=design_paths[313],
    )
    paths["next313_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next313_manifest": roots["next313"] / n313.MANIFEST_NAME,
            "next313_catalogue": roots["next313"] / n313.CATALOGUE_NAME,
            "next313_evaluation": roots["next313"] / n313.EVALUATION_NAME,
            "next313_formula": roots["next313"] / n313.FORMULA_NAME,
            "next313_search": roots["next313"] / n313.SEARCH_NAME,
        }
    )
    return paths


def _verify_next313(paths: Mapping[str, Path], input_hashes: Mapping[str, str]):
    prior_paths = {key: paths[key] for key in n313.EXPECTED_INPUT_SHA256}
    prior_paths["design"] = paths["next313_design"]
    prior_hashes = {key: input_hashes[key] for key in n313.EXPECTED_INPUT_SHA256}
    prior_hashes["design"] = input_hashes["next313_design"]
    prior = n313._verify_next312(prior_paths, prior_hashes)
    manifest = json.loads(paths["next313_manifest"].read_text())
    catalogue = json.loads(paths["next313_catalogue"].read_text())
    evaluation = json.loads(paths["next313_evaluation"].read_text())
    formula = json.loads(paths["next313_formula"].read_text())
    table = pd.read_parquet(paths["next313_search"])
    selected = select_diagnostic_candidates(table)
    expected_outputs = {
        n313.CATALOGUE_NAME: input_hashes["next313_catalogue"],
        n313.EVALUATION_NAME: input_hashes["next313_evaluation"],
        n313.FORMULA_NAME: input_hashes["next313_formula"],
        n313.SEARCH_NAME: input_hashes["next313_search"],
    }
    if (
        manifest.get("protocol") != n313.PROTOCOL
        or manifest.get("candidate_count") != n313.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_new_candidate_count") != n313.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("next314_diagnostic_authorized") is not True
        or manifest.get("next314_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next314_candidate_key_sha256") != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next313_pcae_margin_local_search.py"
        )
        != EXPECTED_NEXT313_SOURCE_SHA256
        or _sha256_file(Path(n313.__file__).resolve()) != EXPECTED_NEXT313_SOURCE_SHA256
        or _sha256_file(Path(n262.__file__).resolve()) != EXPECTED_NEXT262_SOURCE_SHA256
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or catalogue.get("candidate_count") != n313.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next314_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next314_candidate_key_sha256") != EXPECTED_CANDIDATE_KEY_SHA256
        or formula.get("kind") != "pcae_triangular_margin_local_x0_no_dft_score"
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
        or len(table) != n313.EXPECTED_CANDIDATE_COUNT
        or len(selected) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(selected) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT314 NEXT313 provenance differs")
    return (*prior, selected)


def run_pcae_broad_diagnostic(
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
    """Reproduce NEXT313 and compute its unchanged discovery BROAD residual."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT314 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT314 design path universe differs")
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
            stage: Path(design_paths[stage]).resolve() for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT314 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT314 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT314 formal input identity differs: {differing}")
    (
        eligible_prior,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        pcae_tables,
        eligible312,
        published313,
    ) = _verify_next313(paths, input_hashes)
    combined, feature_tables, base_score, base_support, endpoint, _ = (
        n313.n227._reconstruct_next224_frontier(
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
    n313._attach_pcae_features(
        combined=combined,
        feature_tables=feature_tables,
        pcae_tables=pcae_tables,
    )
    diagnostic224 = json.loads(paths["next224_diagnostic"].read_text())
    base_key = str(diagnostic224["global_closest"]["candidate_key"])
    all_specs = n313.build_pcae_candidate_specs(
        base_candidate_key=base_key,
        eligible_table=eligible312,
    )
    keys = set(published313["candidate_key"].astype(str))
    specs = [spec for spec in all_specs if str(spec["candidate_key"]) in keys]
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT314 selected specification universe differs")
    virtual, terms, runtime, activity = n313.materialize_pcae_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    evaluator = (
        n313.n223.n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
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
    n164 = n313.n223.n222.n215.n214.n164
    n164._verify_reproduction(rerun=rerun["candidate_records"], published=published313)
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
    published_by_key = published313.set_index("candidate_key", drop=False)
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for term in terms:
        key = str(term["physical_candidate_key"])
        spec = spec_by_key[key]
        row = published_by_key.loc[key]
        score, support = n164._term_risk(virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT314 candidate support differs from NEXT214")
        tables = n164._threshold_tables(
            score=score,
            supported=support,
            endpoint=endpoint,
            cells=cells,
        )
        if tables is None:
            raise RuntimeError("NEXT314 candidate has no threshold table")
        residual = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT314 contradicts NEXT313 BROAD result")
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
        raise RuntimeError("NEXT314 residual population differs")
    closest = select_closest_residual(per_candidate)
    improves = n313.n223.n222.strictly_improves(
        closest,
        {
            "failed_constraint_count": NEXT235_REFERENCE_FAILED_COUNT,
            "normalized_shortfall_sum": NEXT235_REFERENCE_SHORTFALL,
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
        "diagnostic_mode": "offline_discovery_label_next313_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "next313_record_population_reproduced": True,
        "next313_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "next235_reference_failed_constraint_count": NEXT235_REFERENCE_FAILED_COUNT,
        "next235_reference_normalized_shortfall_sum": NEXT235_REFERENCE_SHORTFALL,
        "normalized_shortfall_reduction_from_next235": (
            NEXT235_REFERENCE_SHORTFALL - float(closest["normalized_shortfall_sum"])
        ),
        "strict_improvement_over_next235_diagnostic": improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "pcae_certificate_branch_closed": bool(not improves),
        "continuation_requires_new_preoutcome_freeze": bool(improves),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(n164.__file__).resolve(),
        "src/next313_pcae_margin_local_search.py": Path(n313.__file__).resolve(),
        "src/next314_pcae_broad_diagnostic.py": Path(__file__).resolve(),
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
            "next313_record_population_reproduced": True,
            "next313_candidate_universe_reproduced": True,
            "next313_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": improves,
            "strict_improvement_over_next235_diagnostic": improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "pcae_certificate_branch_closed": bool(not improves),
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
            raise RuntimeError("NEXT314 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT314 source changed before publication")
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
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = run_pcae_broad_diagnostic(
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
        search_workers=arguments.search_workers,
        require_formal_inputs=not arguments.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_DESIGN_STAGES",
    "REQUIRED_STAGES",
    "candidate_key_sha256",
    "run_pcae_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]
