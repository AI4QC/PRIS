#!/usr/bin/env python3
"""Diagnose the frozen NEXT216 AUC+SAFE, non-BROAD candidate population."""

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

import src.next216_repair_band_relief_search as n216
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next217-repair-band-broad-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT217_REPAIR_BAND_BROAD_DIAGNOSTIC.json"
TABLE_NAME = "next217_repair_band_broad_diagnostic.parquet"
EXPECTED_DESIGN_SHA256 = (
    "78ff5b477c8a016eee0ce8ba28aaa3c47c977ef55ff157d209e2bb2ddbf58a06"
)
EXPECTED_NEXT216_SOURCE_SHA256 = (
    "9aa58d9807400ad0729ed5f589eb5a7d34c82b36e6867a29d8f0a7d5afeb2050"
)
EXPECTED_CANDIDATE_COUNT = 88
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "550ba336091545f8040ac62cc9b4e7f426fb196757e2ad8d815b00cb1cc90c35"
)
EXPECTED_BASE_FAILED_COUNT = 6
EXPECTED_BASE_SHORTFALL = n216.n215.EXPECTED_NEXT214_SHORTFALL
SEARCH_WORKERS = n216.SEARCH_WORKERS
BOUNDARY_FLAGS = n216.BOUNDARY_FLAGS
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n216.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next216_design": n216.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next216_manifest": (
        "c3a53bb248c612a00fe493b7bb3ae7fb695cf80253e11f0c296ffceaa9738838"
    ),
    "next216_catalogue": (
        "dd180b13be58860511ef55bb0c4ea99ff213f75bf81fd5f7d73ce187ade989c5"
    ),
    "next216_evaluation": (
        "d31ee14913f3a993d64a862f46bd8e7f9a6e17ed046e3af02fe9832921eea914"
    ),
    "next216_formula": (
        "feda3a7baf4ff690bcad2468619b7c8a9d6ccc6c0f643d6ee8b36909c3c107dd"
    ),
    "next216_search": (
        "93ebafe13f1c66ea1560c0ddb76b2c01d8b04dc6cb1c4a208c4ae2a60213905c"
    ),
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    """Select the exact AUC+SAFE/non-BROAD candidate population."""

    required = {
        "candidate_key",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if (
        not isinstance(published, pd.DataFrame)
        or required - set(published.columns)
        or published["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT217 published candidate schema differs")
    mask = (
        published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)


def candidate_key_sha256(frame: pd.DataFrame) -> str:
    """Hash sorted unique candidate keys with frozen newline joining."""

    if (
        not isinstance(frame, pd.DataFrame)
        or "candidate_key" not in frame.columns
        or frame["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT217 candidate identity table differs")
    return hashlib.sha256(
        "\n".join(sorted(frame["candidate_key"].astype(str))).encode()
    ).hexdigest()


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    """Return the frozen lexicographically closest BROAD residual."""

    required = {
        "candidate_key",
        "failed_constraint_count",
        "normalized_shortfall_sum",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty or required - set(frame.columns):
        raise ValueError("NEXT217 closest residual population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    next205_design_path: Path,
    next207_design_path: Path,
    next208_design_path: Path,
    next209_design_path: Path,
    next210_design_path: Path,
    next211_design_path: Path,
    next212_design_path: Path,
    next213_design_path: Path,
    next214_design_path: Path,
    next215_design_path: Path,
    next216_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n216._paths(
        roots,
        freeze_path,
        next202_design_path,
        next205_design_path,
        next207_design_path,
        next208_design_path,
        next209_design_path,
        next210_design_path,
        next211_design_path,
        next212_design_path,
        next213_design_path,
        next214_design_path,
        next215_design_path,
        next216_design_path,
    )
    paths["next216_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next216_manifest": roots["next216"] / n216.MANIFEST_NAME,
            "next216_catalogue": roots["next216"] / n216.CATALOGUE_NAME,
            "next216_evaluation": roots["next216"] / n216.EVALUATION_NAME,
            "next216_formula": roots["next216"] / n216.FORMULA_NAME,
            "next216_search": roots["next216"] / n216.SEARCH_NAME,
        }
    )
    return paths


def _verify_next216(
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
    """Verify NEXT216 and return its exact diagnostic population."""

    prior_paths = dict(paths)
    prior_paths["design"] = paths["next216_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next216_design"]
    eligible, eligible214, primary_key, start_key, formula214 = (
        n216._verify_next215(prior_paths, prior_hashes)
    )
    manifest = json.loads(paths["next216_manifest"].read_text())
    catalogue = json.loads(paths["next216_catalogue"].read_text())
    evaluation = json.loads(paths["next216_evaluation"].read_text())
    formula = json.loads(paths["next216_formula"].read_text())
    published_all = pd.read_parquet(paths["next216_search"])
    expected_outputs = {
        n216.CATALOGUE_NAME: input_hashes["next216_catalogue"],
        n216.EVALUATION_NAME: input_hashes["next216_evaluation"],
        n216.FORMULA_NAME: input_hashes["next216_formula"],
        n216.SEARCH_NAME: input_hashes["next216_search"],
    }
    counts = evaluation.get("counts", {})
    if (
        manifest.get("protocol") != n216.PROTOCOL
        or manifest.get("candidate_count") != n216.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_hypothesis_count") != n216.EXPECTED_ELIGIBLE_COUNT
        or manifest.get("eligible_hypothesis_sha256")
        != n216.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("next217_diagnostic_authorized") is not True
        or manifest.get("next217_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("next217_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next216_repair_band_relief_search.py"
        )
        != EXPECTED_NEXT216_SOURCE_SHA256
        or _sha256_file(Path(n216.__file__).resolve())
        != EXPECTED_NEXT216_SOURCE_SHA256
        or catalogue.get("protocol") != n216.PROTOCOL
        or catalogue.get("design_sha256") != input_hashes["next216_design"]
        or catalogue.get("candidate_count") != n216.EXPECTED_CANDIDATE_COUNT
        or catalogue.get("normalization_fit_uses_endpoint") is not False
        or catalogue.get("base_support_unchanged") is not True
        or evaluation.get("protocol") != n216.PROTOCOL
        or evaluation.get("candidate_count") != n216.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("next217_diagnostic_authorized") is not True
        or evaluation.get("next217_candidate_count") != EXPECTED_CANDIDATE_COUNT
        or evaluation.get("next217_candidate_key_sha256")
        != EXPECTED_CANDIDATE_KEY_SHA256
        or counts.get("passes_source_auc_gates") != 88
        or counts.get("passes_safe_all_cells") != 89
        or counts.get("passes_broad_all_cells") != 0
        or counts.get("passes_all_discovery_gates") != 0
        or counts.get("passes_auc_and_safe_but_not_broad") != 88
        or formula.get("protocol") != n216.PROTOCOL
        or formula.get("dft_values_used_by_executable_formula") is not False
        or formula.get("learned_energy_force_stress_proxy_used") is not False
        or formula.get("model_or_proxy_potential_used") is not False
        or formula.get("physical_relaxation_executed") is not False
        or len(published_all) != n216.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT217 NEXT216 provenance differs")
    published = select_diagnostic_candidates(published_all)
    if (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_key_sha256(published) != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT217 frozen diagnostic population differs")
    return (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    )


def run_repair_band_broad_diagnostic(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path,
    next113_dir: Path, next114_dir: Path, next116_dir: Path,
    next117_dir: Path, next120_dir: Path, next121_dir: Path,
    next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    next134_dir: Path, next163_dir: Path, next164_dir: Path,
    next168_dir: Path, next173_dir: Path, next179_dir: Path,
    next180_dir: Path, next181_dir: Path, next182_dir: Path,
    next183_dir: Path, next184_dir: Path, next185_dir: Path,
    next186_dir: Path, next188_dir: Path, next190_dir: Path,
    next192_dir: Path, next194_dir: Path, next199_dir: Path,
    next200_dir: Path, next201_dir: Path, next202_dir: Path,
    next203_dir: Path, next204_dir: Path, next205_dir: Path,
    next206_dir: Path, next207_dir: Path, next208_dir: Path,
    next209_dir: Path, next210_dir: Path, next211_dir: Path,
    next212_dir: Path, next213_dir: Path, next214_dir: Path,
    next215_dir: Path, next216_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, next213_design_path: Path,
    next214_design_path: Path, next215_design_path: Path,
    next216_design_path: Path, design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT217 BROAD residual diagnostic."""

    early_stages = (
        (98, next98_dir), (110, next110_dir), (111, next111_dir),
        (113, next113_dir), (114, next114_dir), (116, next116_dir),
        (117, next117_dir), (120, next120_dir), (121, next121_dir),
        (122, next122_dir), (124, next124_dir), (125, next125_dir),
        (129, next129_dir), (130, next130_dir), (133, next133_dir),
        (134, next134_dir), (163, next163_dir), (164, next164_dir),
        (168, next168_dir), (173, next173_dir), (179, next179_dir),
        (180, next180_dir), (181, next181_dir), (182, next182_dir),
        (183, next183_dir), (184, next184_dir), (185, next185_dir),
        (186, next186_dir), (188, next188_dir), (190, next190_dir),
        (192, next192_dir),
    )
    later_stages = (
        (194, next194_dir), (199, next199_dir), (200, next200_dir),
        (201, next201_dir), (202, next202_dir), (203, next203_dir),
        (204, next204_dir), (205, next205_dir), (206, next206_dir),
        (207, next207_dir), (208, next208_dir), (209, next209_dir),
        (210, next210_dir), (211, next211_dir), (212, next212_dir),
        (213, next213_dir), (214, next214_dir), (215, next215_dir),
        (216, next216_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (*early_stages, *later_stages)
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(),
        Path(next208_design_path).resolve(),
        Path(next209_design_path).resolve(),
        Path(next210_design_path).resolve(),
        Path(next211_design_path).resolve(),
        Path(next212_design_path).resolve(),
        Path(next213_design_path).resolve(),
        Path(next214_design_path).resolve(),
        Path(next215_design_path).resolve(),
        Path(next216_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT217 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT217 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT217 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        start_key,
        formula214,
        published_all,
        published,
    ) = _verify_next216(paths, input_hashes)
    combined, _, base_score, base_support, endpoint = (
        n216.n215._reconstruct_next214_final(
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
        raise ValueError("NEXT217 NEXT214 base identity differs")
    base_candidate_key = str(accepted.iloc[0]["candidate_key"])
    all_specs = n216.build_candidate_specs(
        base_candidate_key=base_candidate_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
    )
    if (
        len(all_specs) != n216.EXPECTED_CANDIDATE_COUNT
        or {str(spec["candidate_key"]) for spec in all_specs}
        != set(published_all["candidate_key"].astype(str))
    ):
        raise ValueError("NEXT217 NEXT216 candidate universe differs")
    diagnostic_keys = set(published["candidate_key"].astype(str))
    selected_specs = [
        spec for spec in all_specs if str(spec["candidate_key"]) in diagnostic_keys
    ]
    if len(selected_specs) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("NEXT217 selected specification universe differs")
    combined_virtual, terms, runtime = n216.materialize_repair_band_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=selected_specs,
    )
    rerun = (
        n216.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel(
            features=combined_virtual,
            endpoint=endpoint,
            old_terms=terms,
            optional_terms=[],
            candidate_specs=runtime,
            workers=search_workers,
        )
    )
    n216.n215.n214.n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
    )
    folds = n216.n215.n214.n164.assign_group_folds(
        combined_virtual["reduced_formula"].astype(str).to_numpy()
    )
    sources = combined_virtual["source_dataset"].astype(str).to_numpy()
    cells = n216.n215.n214.n164.build_source_fold_cells(
        source=sources, folds=folds
    )
    pauling_by_cell = {
        str(cell["cell_id"]): n216.n215.n214.n164._pauling_baseline(
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
        score, support = n216.n215.n214.n164._term_risk(combined_virtual, term)
        if not np.array_equal(support, base_support):
            raise RuntimeError("NEXT217 candidate support differs from NEXT214")
        tables = n216.n215.n214.n164._threshold_tables(
            score=score, supported=support, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT217 candidate has no threshold table")
        residual = n216.n215.n214.n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if residual["passes_broad"]:
            raise RuntimeError("NEXT217 contradicts NEXT216 BROAD result")
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
                "relief_active_rows": int(row["relief_active_rows"]),
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
        raise ValueError("NEXT217 NEXT214 baseline residual differs")
    closest = select_closest_residual(per_candidate)
    closest_shortfall = float(closest["normalized_shortfall_sum"])
    closest_failures = int(closest["failed_constraint_count"])
    closest_improves = bool(
        closest_failures < EXPECTED_BASE_FAILED_COUNT
        or (
            closest_failures == EXPECTED_BASE_FAILED_COUNT
            and closest_shortfall + 1.0e-12 < EXPECTED_BASE_SHORTFALL
        )
    )
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next216_broad_residual",
        "candidate_count": len(per_candidate),
        "candidate_key_sha256": candidate_key_sha256(published),
        "next216_record_population_reproduced": True,
        "next216_candidate_universe_reproduced": True,
        "global_closest": {
            "candidate_key": str(closest["candidate_key"]),
            "hypothesis": (
                None if pd.isna(closest["hypothesis"])
                else str(closest["hypothesis"])
            ),
            "feature": (
                None if pd.isna(closest["feature"])
                else str(closest["feature"])
            ),
            "direction": (
                None if pd.isna(closest["direction"])
                else str(closest["direction"])
            ),
            "amplitude_fraction": float(closest["amplitude_fraction"]),
            "q_lo": (
                None if pd.isna(closest["q_lo"]) else float(closest["q_lo"])
            ),
            "q_hi": (
                None if pd.isna(closest["q_hi"]) else float(closest["q_hi"])
            ),
            "safe_threshold": float(closest["safe_threshold"]),
            "best_threshold": float(closest["best_threshold"]),
            "failed_constraint_count": closest_failures,
            "normalized_shortfall_sum": closest_shortfall,
            "failures": json.loads(str(closest["failures_json"])),
        },
        "next214_reference_failed_constraint_count": EXPECTED_BASE_FAILED_COUNT,
        "next214_reference_normalized_shortfall_sum": EXPECTED_BASE_SHORTFALL,
        "normalized_shortfall_reduction_from_next214": (
            EXPECTED_BASE_SHORTFALL - closest_shortfall
        ),
        "improves_over_next214_global_residual": closest_improves,
        "failure_frequency": dict(sorted(frequency.items())),
        "repair_band_relief_branch_closed": True,
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
            n216.n215.n214.n164.__file__
        ).resolve(),
        "src/next216_repair_band_relief_search.py": Path(n216.__file__).resolve(),
        "src/next217_repair_band_broad_diagnostic.py": Path(__file__).resolve(),
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
            "next216_record_population_reproduced": True,
            "next216_candidate_universe_reproduced": True,
            "next216_all_gate_candidate_count": 0,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "repair_band_relief_branch_closed": True,
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
            raise RuntimeError("NEXT217 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT217 source changed before publication")
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
    early_stages = (
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    later_stages = (
        194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216,
    )
    for stage in early_stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_repair_band_broad_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in early_stages + later_stages
        },
        next135_freeze_path=args.next135_freeze_path,
        **{
            f"next{stage}_design_path": getattr(args, f"next{stage}_design_path")
            for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216)
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
    "run_repair_band_broad_diagnostic",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
