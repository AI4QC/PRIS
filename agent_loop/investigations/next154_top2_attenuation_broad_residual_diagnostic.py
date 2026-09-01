#!/usr/bin/env python3
"""Diagnose BROAD residuals of frozen NEXT153 AUC+SAFE12 candidates."""

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
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next153_top2_attenuation_homotopy_search as n153
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds
from src.next98_cross_source_discovery_search import (
    _pauling_baseline,
    _threshold_tables,
    build_source_fold_cells,
)


PROTOCOL = "2026-08-08-next154-top2-attenuation-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT154_TOP2_ATTENUATION_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next154_top2_attenuation_broad_residual_by_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "c0d99a65ecbb90759d3c1e4f3b0efc95ab8fc933f5f618e83b44ad2d98ae19b8"
EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT = 22
EXPECTED_CANDIDATE_KEY_SHA256 = "6b065381ec7fc40d60cc1074eb477480ffedac09faa51250ba2fe6d714a83838"
EXPECTED_INPUT_SHA256 = {
    **n135.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next153_manifest": "f682e9732cee58245c05d0a82ee34fe5598ffe09a92a5018ca0ec92fd41a50ff",
    "next153_catalogue": "b1848d6aa13b9047a731d34abfed887a1f183b854d0d3ce2df671c2c3f898f56",
    "next153_evaluation": "95d73de69dc16ea2c073f5d6a7edf77875a7774f79a2864fd9b583ed841cf699",
    "next153_search": "9c22d915625ae57cc1df9591bd7b46c3fb76d61a9783a3320510350a38eaa5ad",
}


def select_auc_safe_candidates(records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_key",
        "safe_threshold",
        "top2_attenuation",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
    }
    if required - set(records.columns) or records["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT154 published candidate schema differs")
    selected = records.loc[
        records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
    ].copy()
    selected["safe_threshold"] = pd.to_numeric(
        selected["safe_threshold"], errors="coerce"
    )
    selected["top2_attenuation"] = pd.to_numeric(
        selected["top2_attenuation"], errors="coerce"
    )
    if selected.empty or not np.isfinite(
        selected[["safe_threshold", "top2_attenuation"]].to_numpy(float)
    ).all():
        raise ValueError("NEXT154 published thresholds differ")
    return selected.sort_values("candidate_key").reset_index(drop=True)


def _closest(records: pd.DataFrame) -> pd.Series:
    return records.sort_values(
        [
            "failed_constraint_count",
            "normalized_shortfall_sum",
            "best_threshold",
            "candidate_key",
        ]
    ).iloc[0]


def summarize_by_gamma(records: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for gamma, frame in records.groupby("top2_attenuation", sort=True):
        best = _closest(frame)
        minimum_count = int(best["failed_constraint_count"])
        summary[f"gamma={gamma:g}"] = {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": minimum_count,
            "minimum_normalized_shortfall_sum_at_best_count": float(
                frame.loc[
                    frame["failed_constraint_count"].eq(minimum_count),
                    "normalized_shortfall_sum",
                ].min()
            ),
            "closest_candidate_key": str(best["candidate_key"]),
        }
    return summary


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n135._paths(roots, freeze_path)
    paths.update(
        {
            "design": design_path,
            "next153_manifest": roots["next153"] / n153.MANIFEST_NAME,
            "next153_catalogue": roots["next153"] / n153.CATALOGUE_NAME,
            "next153_evaluation": roots["next153"] / n153.EVALUATION_NAME,
            "next153_search": roots["next153"] / n153.SEARCH_NAME,
        }
    )
    return paths


def _verify_reproduction(
    *, rerun: Sequence[Mapping[str, object]], published: pd.DataFrame
) -> None:
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
        "safe_threshold",
    )
    booleans = (
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
        "passes_all_discovery_gates",
    )
    prior = published.set_index("candidate_key", drop=False)
    if {str(record["candidate_key"]) for record in rerun} != set(prior.index.astype(str)):
        raise RuntimeError("NEXT154 rerun identities differ")
    for record in rerun:
        row = prior.loc[str(record["candidate_key"])]
        if any(
            not math.isclose(
                float(record[name]), float(row[name]), rel_tol=0.0, abs_tol=1.0e-12
            )
            for name in metrics
        ) or any(bool(record[name]) != bool(row[name]) for name in booleans):
            raise RuntimeError("NEXT154 does not reproduce NEXT153 AUC+SAFE result")


def run_top2_attenuation_broad_residual_diagnostic(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next110_dir: Path,
    next111_dir: Path,
    next113_dir: Path,
    next114_dir: Path,
    next116_dir: Path,
    next117_dir: Path,
    next120_dir: Path,
    next121_dir: Path,
    next122_dir: Path,
    next124_dir: Path,
    next125_dir: Path,
    next129_dir: Path,
    next130_dir: Path,
    next133_dir: Path,
    next134_dir: Path,
    next153_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = n153.SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (98, next98_dir),
                (110, next110_dir),
                (111, next111_dir),
                (113, next113_dir),
                (114, next114_dir),
                (116, next116_dir),
                (117, next117_dir),
                (120, next120_dir),
                (121, next121_dir),
                (122, next122_dir),
                (124, next124_dir),
                (125, next125_dir),
                (129, next129_dir),
                (130, next130_dir),
                (133, next133_dir),
                (134, next134_dir),
                (153, next153_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT154 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT154 formal input identity differs: {differing}")
    manifest153 = json.loads(paths["next153_manifest"].read_text())
    outputs153 = manifest153.get("outputs_sha256", {})
    if (
        manifest153.get("protocol") != n153.PROTOCOL
        or manifest153.get("top2_attenuation_homotopy_branch_terminated") is not True
        or manifest153.get("opened_validation_outputs_used") is not False
        or manifest153.get("dft_values_used_by_executable_formula") is not False
        or manifest153.get("executed_source_sha256", {}).get(
            "src/next153_top2_attenuation_homotopy_search.py"
        )
        != _sha256_file(Path(n153.__file__).resolve())
        or outputs153.get(n153.SEARCH_NAME) != input_hashes["next153_search"]
    ):
        raise ValueError("NEXT154 prior provenance differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1
    )
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n153.build_candidate_specs(bases=bases, physical_term_ids=physical_ids)

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:"
                    + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:"
                    + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT154 endpoint row accounting differs")
    combined, virtual_terms, runtime = n153.materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=specs
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0]) for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    published_all = pd.read_parquet(paths["next153_search"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT154 published universe differs")
    published = select_auc_safe_candidates(published_all)
    candidate_sha = hashlib.sha256(
        "\n".join(published["candidate_key"].astype(str)).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(published) != EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
        or published["top2_attenuation"].value_counts().sort_index().to_dict()
        != {0.0: 11, 0.1: 11}
    ):
        raise ValueError("NEXT154 frozen diagnostic population differs")
    selected_keys = set(published["candidate_key"].astype(str))
    rerun_runtime = [
        spec for spec in runtime if str(spec["candidate_key"]) in selected_keys
    ]
    rerun = n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=rerun_runtime,
        workers=search_workers,
    )
    _verify_reproduction(rerun=rerun["candidate_records"], published=published)

    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): _pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        key = str(row["candidate_key"])
        score, supported = _term_risk(
            combined, virtual_by_id[virtual_by_candidate[key]]
        )
        tables = _threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT154 candidate has no threshold table")
        diagnostic = diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT154 contradicts NEXT153 BROAD result")
        for failure in diagnostic["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "top2_attenuation": float(row["top2_attenuation"]),
                "safe_threshold": float(row["safe_threshold"]),
                "best_threshold": float(diagnostic["best_threshold"]),
                "failed_constraint_count": int(
                    diagnostic["failed_constraint_count"]
                ),
                "normalized_shortfall_sum": float(
                    diagnostic["normalized_shortfall_sum"]
                ),
                "eligible_threshold_count": int(
                    diagnostic["eligible_threshold_count"]
                ),
                "failures_json": json.dumps(
                    diagnostic["failures"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    per_candidate = pd.DataFrame(records)
    global_closest = _closest(per_candidate)
    by_gamma = summarize_by_gamma(per_candidate)
    gamma0 = by_gamma["gamma=0"]
    gamma01 = by_gamma["gamma=0.1"]
    residual_reduced = bool(
        gamma01["minimum_failed_constraint_count"]
        < gamma0["minimum_failed_constraint_count"]
        or (
            gamma01["minimum_failed_constraint_count"]
            == gamma0["minimum_failed_constraint_count"]
            and gamma01["minimum_normalized_shortfall_sum_at_best_count"]
            < gamma0["minimum_normalized_shortfall_sum_at_best_count"]
        )
    )
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next153_broad_constraint_residual",
        "candidate_count": int(len(per_candidate)),
        "candidate_key_sha256": candidate_sha,
        "by_gamma": by_gamma,
        "gamma0p1_residual_reduced_vs_gamma0": residual_reduced,
        "global_closest": {
            "candidate_key": str(global_closest["candidate_key"]),
            "top2_attenuation": float(global_closest["top2_attenuation"]),
            "safe_threshold": float(global_closest["safe_threshold"]),
            "best_threshold": float(global_closest["best_threshold"]),
            "failed_constraint_count": int(
                global_closest["failed_constraint_count"]
            ),
            "normalized_shortfall_sum": float(
                global_closest["normalized_shortfall_sum"]
            ),
            "failures": json.loads(str(global_closest["failures_json"])),
        },
        "failure_frequency_at_per_candidate_optima": dict(frequency.most_common()),
        "new_formula_searched": False,
        "validation_or_replication_opened": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next128_broad_residual_diagnostic.py": Path(
            diagnose_broad_threshold_tables.__code__.co_filename
        ).resolve(),
        "src/next153_top2_attenuation_homotopy_search.py": Path(n153.__file__).resolve(),
        "src/next154_top2_attenuation_broad_residual_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        table_path = staging / PER_CANDIDATE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(table_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(len(per_candidate)),
            "gamma0p1_residual_reduced_vs_gamma0": residual_reduced,
            "new_formula_searched": False,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                DIAGNOSTIC_NAME: _sha256_file(diagnostic_path),
                PER_CANDIDATE_NAME: _sha256_file(table_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT154 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT154 source changed before publication")
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
    stages = (
        98,
        110,
        111,
        113,
        114,
        116,
        117,
        120,
        121,
        122,
        124,
        125,
        129,
        130,
        133,
        134,
        153,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=n153.SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_top2_attenuation_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in stages
        },
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CANDIDATE_KEY_SHA256",
    "EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT",
    "run_top2_attenuation_broad_residual_diagnostic",
    "select_auc_safe_candidates",
    "summarize_by_gamma",
]
