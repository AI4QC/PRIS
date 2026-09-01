#!/usr/bin/env python3
"""Diagnose frozen NEXT220 AUC+SAFE, non-BROAD signed candidates."""

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

import src.next220_signed_redistribution_search as n220
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next221-signed-redistribution-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT221_SIGNED_REDISTRIBUTION_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next221_signed_redistribution_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = n220.EXPECTED_DESIGN_SHA256
EXPECTED_NEXT220_SOURCE_SHA256 = (
    "1f539d8ba77831098880983ca868f7cdbeafe271575e4b4cf26f690f98623c5d"
)
EXPECTED_CANDIDATE_COUNT = 39
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "61230aa643a4ffb80fd3daa9943e81f5c3597e955222517c5f5cd4ffdef99fbd"
)
EXPECTED_BASE_FAILED_COUNT = n220.EXPECTED_BASE_FAILED_COUNT
EXPECTED_BASE_SHORTFALL = n220.EXPECTED_BASE_SHORTFALL
SEARCH_WORKERS = n220.SEARCH_WORKERS
BOUNDARY_FLAGS = n220.BOUNDARY_FLAGS
REQUIRED_STAGES = n220.REQUIRED_STAGES + (220,)
REQUIRED_DESIGN_STAGES = n220.REQUIRED_DESIGN_STAGES + (220,)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n220.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next220_design": n220.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next220_manifest": (
        "6f948b60409b250bfcc00db468ff32921ef149d2a53d8054f899b1ac36996b0f"
    ),
    "next220_catalogue": (
        "c21046ba196b8bc89958b338c4197cb385cd030064eb79b43a1d3b7e5481e463"
    ),
    "next220_evaluation": (
        "d6cae6b4ea4c242c89fd240a3d26623cee553b6c23cc27b705eaf175de56a2fc"
    ),
    "next220_formula": (
        "03e33a02563021ecefbd49aa8cfc0060211fb90dc899410a07e1145404387976"
    ),
    "next220_search": (
        "60090fcbd1092053f4552f88e9a7821382285f3269ddae7eda48eb2fa73c2335"
    ),
}


select_diagnostic_candidates = n220.n219.select_diagnostic_candidates
candidate_key_sha256 = n220.n219.candidate_key_sha256
select_closest_residual = n220.n219.select_closest_residual


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n220._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n220.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[220],
    )
    paths["next220_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next220_manifest": roots["next220"] / n220.MANIFEST_NAME,
            "next220_catalogue": roots["next220"] / n220.CATALOGUE_NAME,
            "next220_evaluation": roots["next220"] / n220.EVALUATION_NAME,
            "next220_formula": roots["next220"] / n220.FORMULA_NAME,
            "next220_search": roots["next220"] / n220.SEARCH_NAME,
        }
    )
    return paths


def _verify_next220(
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
    """Verify NEXT220 and return its exact diagnostic population."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next220_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next220_design"]
    eligible, eligible214, primary_key, start_key, formula214 = (
        n220._verify_next219(prior_paths, prior_hashes)
    )
    manifest = json.loads(paths["next220_manifest"].read_text())
    catalogue = json.loads(paths["next220_catalogue"].read_text())
    evaluation = json.loads(paths["next220_evaluation"].read_text())
    formula = json.loads(paths["next220_formula"].read_text())
    published_all = pd.read_parquet(paths["next220_search"])
    expected_outputs = {
        n220.CATALOGUE_NAME: input_hashes["next220_catalogue"],
        n220.EVALUATION_NAME: input_hashes["next220_evaluation"],
        n220.FORMULA_NAME: input_hashes["next220_formula"],
        n220.SEARCH_NAME: input_hashes["next220_search"],
    }
    counts = evaluation.get("counts", {})
    if (
        manifest.get("protocol") != n220.PROTOCOL
        or manifest.get("candidate_count") != n220.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_hypothesis_count")
        != n220.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next221_diagnostic_authorized") is not True
        or manifest.get("next221_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next221_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next220_signed_redistribution_search.py"
        )
        != EXPECTED_NEXT220_SOURCE_SHA256
        or _sha256_file(Path(n220.__file__).resolve())
        != EXPECTED_NEXT220_SOURCE_SHA256
        or catalogue.get("protocol") != n220.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next220_design"]
        or catalogue.get("candidate_count") != n220.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("signed_zero_centered_redistribution") is not True
        or catalogue.get("normalization_fit_uses_endpoint") is not False
        or evaluation.get("protocol") != n220.PROTOCOL
        or evaluation.get("candidate_count") != n220.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next221_diagnostic_authorized") is not True
        or evaluation.get("next221_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next221_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or counts.get("passes_source_auc_gates") != n220.EXPECTED_CANDIDATE_COUNT
        or counts.get("passes_safe_all_cells") != EXPECTED_CANDIDATE_COUNT
        or counts.get("passes_broad_all_cells") != 0
        or counts.get("passes_all_discovery_gates") != 0
        or counts.get("passes_auc_and_safe_but_not_broad")
        != EXPECTED_CANDIDATE_COUNT
        or formula.get("protocol") != n220.PROTOCOL
        or formula.get("dft_values_used_by_executable_formula") is not False
        or formula.get("learned_energy_force_stress_proxy_used") is not False
        or formula.get("model_or_proxy_potential_used") is not False
        or formula.get("physical_relaxation_executed") is not False
        or len(published_all) != n220.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT221 NEXT220 provenance differs")
    published = select_diagnostic_candidates(published_all)
    if (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(published) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT221 frozen diagnostic population differs")
    return (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    )


def run_signed_redistribution_broad_diagnostic(
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
    """Run the frozen discovery-only NEXT221 BROAD residual diagnostic."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT221 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT221 design path universe differs")
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
        raise ValueError("NEXT221 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT221 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT221 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    ) = _verify_next220(paths, input_hashes)
    combined, _, base_score, base_support, endpoint = (
        n220.n215._reconstruct_next214_final(
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
        raise ValueError("NEXT221 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    all_specs = n220.build_signed_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if (
        len(all_specs) != n220.EXPECTED_CANDIDATE_COUNT
        or {str(spec["candidate_key"]) for spec in all_specs}
        != set(published_all["candidate_key"].astype(str))
    ):
        raise ValueError("NEXT221 NEXT220 candidate universe differs")
    diagnostic_keys = set(published["candidate_key"].astype(str))
    selected_specs = [
        spec for spec in all_specs if str(spec["candidate_key"]) in diagnostic_keys
    ]
    if len(selected_specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT221 selected specification universe differs")
    combined_virtual, terms, runtime = n220.materialize_signed_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=selected_specs,
    )
    rerun = (
        n220.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel(
            features=combined_virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=runtime,
            workers=search_workers,
        )
    )
    n220.n215.n214.n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
    )
    folds = n220.n215.n214.n164.assign_group_folds(
        combined_virtual["reduced_formula"].astype(str).to_numpy()
    )
    sources = combined_virtual["source_dataset"].astype(str).to_numpy()
    cells = n220.n215.n214.n164.build_source_fold_cells(
        source=sources, folds=folds
    )
    pauling_by_cell = {
        str(cell["cell_id"]): n220.n215.n214.n164._pauling_baseline(
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
        score, support = n220.n215.n214.n164._term_risk(combined_virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT221 candidate support differs from NEXT214")
        tables = n220.n215.n214.n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT221 candidate has no threshold table")
        residual = n220.n215.n214.n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT221 contradicts NEXT220 BROAD result")
        for failure in residual["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "hypothesis": spec["hypothesis"],
                "feature": spec["feature"],
                "direction": spec["direction"],
                "beta_fraction": float(spec["beta_fraction"]),
                "q_lo": spec["q_lo"],
                "q_hi": spec["q_hi"],
                "redistribution_active_rows": int(
                    row["redistribution_active_rows"]
                ),
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
        raise ValueError("NEXT221 NEXT214 baseline residual differs")
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
            "beta_fraction": float(row["beta_fraction"]),
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
        "diagnostic_mode": "offline_discovery_label_next220_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": candidate_key_sha256(published),
        "next220_record_population_reproduced": True,
        "next220_candidate_universe_reproduced": True,
        "global_closest": residual_record(closest),
        "closest_new_candidate": residual_record(closest_new),
        "next214_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next214_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next214": (
            EXPECTED_BASE_SHORTFALL - closest_shortfall
        ),
        "improves_over_next214_global_residual": closest_improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "signed_redistribution_branch_closed": not closest_improves,
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
            n220.n215.n214.n164.__file__
        ).resolve(),
        "src/next220_signed_redistribution_search.py": Path(n220.__file__).resolve(),
        "src/next221_signed_redistribution_broad_diagnostic.py": Path(
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
            "next220_record_population_reproduced": True,
            "next220_candidate_universe_reproduced": True,
            "next220_all_gate_candidate_count": 0,
            "strict_residual_improvement_observed": closest_improves,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "signed_redistribution_branch_closed": not closest_improves,
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
            raise RuntimeError("NEXT221 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT221 source changed before publication")
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
    manifest = run_signed_redistribution_broad_diagnostic(
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
    "run_signed_redistribution_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
