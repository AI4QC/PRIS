#!/usr/bin/env python3
"""Diagnose BROAD residuals for frozen NEXT177 discovery candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next176_weighted_rigidity_broad_residual_diagnostic as n176
import src.next177_weighted_rigidity_amplitude_extension_search as n177
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next178-amplitude-extension-broad-residual-diagnostic-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT178_AMPLITUDE_EXTENSION_BROAD_RESIDUAL_DIAGNOSTIC.json"
PER_CANDIDATE_NAME = "next178_amplitude_extension_broad_residual_per_candidate.parquet"
EXPECTED_DESIGN_SHA256 = "7d40259c8ba2de5c0822567bec08f82259b7856e060f825f4ae58d38ec55d2cf"
EXPECTED_CANDIDATE_COUNT = 8
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "1947c3c1a3cbb17a1472cc0f0dc665c2bc731da4f6dc07f6cb11c5d48a753b1c"
)
EXPECTED_FEATURE_COUNTS = {
    "BASE": 1,
    "pwldr_crystalnn_tightness_min": 4,
    "pwldr_crystalnn_tightness_q10": 3,
}
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n177.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next177_manifest": "76cf14d68eff4203e875a75511ddd475d1fb30b34cca7f5f8a9952c0adbde929",
    "next177_catalogue": "2cca14219c4b29a04e1a5b0b97e209ef4e984ac1899ddb98d2e9404146e8b77a",
    "next177_evaluation": "e79ecffc27ba0e9357d7c1eee7a2b43a03fd8075f4a341b2560e1ad8727d208b",
    "next177_formula": "55f637ac4568e3292fc60c3a6c6cc9172bb392ba65975af23f4b7f0f6fc68e88",
    "next177_search": "808f77fbb3835a1e39643518effe8e08046576cbd68d0188fcf57318ad325113",
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    """Select the exact AUC+SAFE, non-BROAD diagnostic population."""

    required = {
        "candidate_key",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if (
        required - set(published.columns)
        or published["candidate_key"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT178 published candidate schema differs")
    mask = (
        published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values("candidate_key").reset_index(drop=True)


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n177._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next177_manifest": roots["next177"] / n177.MANIFEST_NAME,
            "next177_catalogue": roots["next177"] / n177.CATALOGUE_NAME,
            "next177_evaluation": roots["next177"] / n177.EVALUATION_NAME,
            "next177_formula": roots["next177"] / n177.FORMULA_NAME,
            "next177_search": roots["next177"] / n177.SEARCH_NAME,
        }
    )
    return paths


def run_amplitude_extension_broad_residual_diagnostic(
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
    next163_dir: Path,
    next164_dir: Path,
    next168_dir: Path,
    next173_dir: Path,
    next174_dir: Path,
    next175_dir: Path,
    next176_dir: Path,
    next177_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT178 BROAD residual diagnostic."""

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
                (163, next163_dir),
                (164, next164_dir),
                (168, next168_dir),
                (173, next173_dir),
                (174, next174_dir),
                (175, next175_dir),
                (176, next176_dir),
                (177, next177_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT178 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT178 formal input identity differs: {differing}")

    manifest177 = json.loads(paths["next177_manifest"].read_text())
    evaluation177 = json.loads(paths["next177_evaluation"].read_text())
    expected_outputs177 = {
        n177.CATALOGUE_NAME: input_hashes["next177_catalogue"],
        n177.EVALUATION_NAME: input_hashes["next177_evaluation"],
        n177.FORMULA_NAME: input_hashes["next177_formula"],
        n177.SEARCH_NAME: input_hashes["next177_search"],
    }
    if (
        manifest177.get("protocol") != n177.PROTOCOL
        or manifest177.get("candidate_count") != n177.EXPECTED_CANDIDATE_COUNT
        or manifest177.get("base_endpoint_reproduced") is not True
        or manifest177.get("passes_all_cross_source_discovery_gates") is not False
        or manifest177.get("freeze_authorized") is not False
        or manifest177.get("weighted_rigidity_amplitude_extension_branch_terminated")
        is not True
        or manifest177.get("opened_validation_outputs_used") is not False
        or manifest177.get("scigen_replication_endpoint_opened") is not False
        or manifest177.get("wyformer_replication_endpoint_opened") is not False
        or manifest177.get("dft_calculation_executed") is not False
        or manifest177.get("dft_values_used_by_executable_formula") is not False
        or manifest177.get("learned_energy_force_stress_proxy_used") is not False
        or manifest177.get("physical_relaxation_executed") is not False
        or manifest177.get("outputs_sha256") != expected_outputs177
        or manifest177.get("executed_source_sha256", {}).get(
            "src/next177_weighted_rigidity_amplitude_extension_search.py"
        )
        != _sha256_file(Path(n177.__file__).resolve())
        or evaluation177.get("protocol") != n177.PROTOCOL
        or evaluation177.get("candidate_count") != n177.EXPECTED_CANDIDATE_COUNT
        or evaluation177.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation177.get("freeze_authorized") is not False
    ):
        raise ValueError("NEXT178 NEXT177 provenance differs")

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
    weighted_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next173_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        weighted_frames.append(table)
    combined = extended.merge(
        pd.concat(weighted_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_candidate_key = str(
        diagnostic164.get("global_closest", {}).get("candidate_key", "")
    )
    if (
        hashlib.sha256(base_candidate_key.encode()).hexdigest()
        != n177.EXPECTED_BASE_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT178 base candidate identity differs")
    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_base_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_candidate_key
    ]
    if len(selected_base_specs) != 1:
        raise ValueError("NEXT178 base reconstruction differs")
    combined, base_terms, base_runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_base_specs,
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT178 base materialization differs")
    base_score, base_support = _term_risk(combined, base_terms[0])

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
    combined = combined.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(
        float
    )
    if not np.isfinite(endpoint).all() or len(combined) != len(base_score):
        raise ValueError("NEXT178 endpoint row accounting differs")

    specs = n177.build_candidate_specs(base_candidate_key=base_candidate_key)
    combined, virtual_terms, runtime = n177.materialize_extended_amplitude_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0])
        for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    published_all = pd.read_parquet(paths["next177_search"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT178 published universe differs")
    published = select_diagnostic_candidates(published_all)
    candidate_sha = hashlib.sha256(
        "\n".join(published["candidate_key"].astype(str)).encode()
    ).hexdigest()
    feature_counts = (
        published["weighted_rigidity_feature"]
        .fillna("BASE")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if require_formal_inputs and (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
        or feature_counts != EXPECTED_FEATURE_COUNTS
    ):
        raise ValueError("NEXT178 frozen diagnostic population differs")
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
    n164._verify_reproduction(rerun=rerun["candidate_records"], published=published)

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
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        key = str(row["candidate_key"])
        score, supported = _term_risk(
            combined, virtual_by_id[virtual_by_candidate[key]]
        )
        tables = n164._threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT178 candidate has no threshold table")
        diagnostic = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT178 contradicts NEXT177 BROAD result")
        for failure in diagnostic["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "weighted_rigidity_feature": row["weighted_rigidity_feature"],
                "attenuation": float(row["attenuation"]),
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
    global_closest = n176._closest(per_candidate)
    base_rows = per_candidate.loc[per_candidate["weighted_rigidity_feature"].isna()]
    if len(base_rows) != 1:
        raise RuntimeError("NEXT178 base residual count differs")
    base_residual = base_rows.iloc[0]
    by_feature = {}
    grouped = per_candidate.assign(
        weighted_rigidity_feature=per_candidate["weighted_rigidity_feature"].fillna(
            "BASE"
        )
    ).groupby("weighted_rigidity_feature", sort=True)
    for feature, frame in grouped:
        closest = n176._closest(frame)
        by_feature[str(feature)] = {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": int(
                closest["failed_constraint_count"]
            ),
            "minimum_normalized_shortfall_sum_at_best_count": float(
                closest["normalized_shortfall_sum"]
            ),
            "closest_attenuation": float(closest["attenuation"]),
            "closest_candidate_key": str(closest["candidate_key"]),
        }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next177_broad_constraint_residual",
        "candidate_count": int(len(per_candidate)),
        "candidate_key_sha256": candidate_sha,
        "feature_counts": feature_counts,
        "base_residual": {
            "best_threshold": float(base_residual["best_threshold"]),
            "failed_constraint_count": int(base_residual["failed_constraint_count"]),
            "normalized_shortfall_sum": float(
                base_residual["normalized_shortfall_sum"]
            ),
            "failures": json.loads(str(base_residual["failures_json"])),
        },
        "by_feature": by_feature,
        "global_closest": {
            "candidate_key": str(global_closest["candidate_key"]),
            "weighted_rigidity_feature": global_closest[
                "weighted_rigidity_feature"
            ],
            "attenuation": float(global_closest["attenuation"]),
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
        "failure_frequency": dict(sorted(frequency.items())),
        "new_formula_searched": False,
        "new_formula_selected": False,
        "validation_outputs_opened": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(n164.__file__).resolve(),
        "src/next177_weighted_rigidity_amplitude_extension_search.py": Path(n177.__file__).resolve(),
        "src/next178_amplitude_extension_broad_residual_diagnostic.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        diagnostic_path = staging / DIAGNOSTIC_NAME
        table_path = staging / PER_CANDIDATE_NAME
        _write_json(diagnostic_path, summary)
        per_candidate.to_parquet(table_path, index=False)
        outputs = [diagnostic_path, table_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(len(per_candidate)),
            "candidate_key_sha256": candidate_sha,
            "next177_records_reproduced": True,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "amplitude_extension_broad_residual_diagnosed": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT178 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT178 source changed before publication")
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
        163,
        164,
        168,
        173,
        174,
        175,
        176,
        177,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_amplitude_extension_broad_residual_diagnostic(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "select_diagnostic_candidates",
    "run_amplitude_extension_broad_residual_diagnostic",
]


if __name__ == "__main__":
    raise SystemExit(main())
