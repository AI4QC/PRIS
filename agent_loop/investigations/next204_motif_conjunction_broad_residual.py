#!/usr/bin/env python3
"""Diagnose BROAD residuals for frozen motif-conjunction exceptions."""

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

import src.next164_interior_attenuation_broad_residual as n164
import src.next203_motif_conjunction_exception_search as n203
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next204-motif-conjunction-broad-residual-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT204_MOTIF_CONJUNCTION_BROAD_RESIDUAL.json"
PER_CANDIDATE_NAME = "next204_motif_conjunction_broad_residual_per_candidate.parquet"
EXPECTED_DESIGN_SHA256 = n203.EXPECTED_DESIGN_SHA256
EXPECTED_CANDIDATE_COUNT = 190
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "5b671c0fd3f3d28aca3f87b6bbbcea522ef380c5044efb3695fb31b2c26ce3e9"
)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n203.EXPECTED_INPUT_SHA256,
    "next203_manifest": "deb84793328aa00dfdf80cbf72d37719edbdcf3c8aebb4260c86aa1bb8bdf4cc",
    "next203_catalogue": "e2749e4e64dfa2d026269b1333c82d07f5469641c6a82bd4744bb0c31fa8f2fd",
    "next203_evaluation": "3c5d6929ff34a0917aed6b262e17deee242fe8693d17ccfbb796f69cbb366098",
    "next203_formula": "2ea9f2e8e63f8a9dbe345aa1eb6cba270f031c5dc267f73693a3318dd65297d4",
    "next203_search": "b1cc81268a23f169a3e8dd0c7399ae561e87c94c673e3a788c6d8d723f3cc893",
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
        raise ValueError("NEXT204 published candidate schema differs")
    mask = (
        published["passes_source_auc_gates"].fillna(False).astype(bool)
        & published["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~published["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    return published.loc[mask].sort_values("candidate_key").reset_index(drop=True)


def select_closest_residual(frame: pd.DataFrame) -> pd.Series:
    """Select the closest residual under the frozen diagnostic ordering."""

    required = {
        "candidate_key",
        "failed_constraint_count",
        "normalized_shortfall_sum",
    }
    if frame.empty or required - set(frame.columns):
        raise ValueError("NEXT204 closest population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n203._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next203_manifest": roots["next203"] / n203.MANIFEST_NAME,
            "next203_catalogue": roots["next203"] / n203.CATALOGUE_NAME,
            "next203_evaluation": roots["next203"] / n203.EVALUATION_NAME,
            "next203_formula": roots["next203"] / n203.FORMULA_NAME,
            "next203_search": roots["next203"] / n203.SEARCH_NAME,
        }
    )
    return paths


def _verify_next203(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> list[str]:
    eligible = n203._verify_next202(paths, input_hashes)
    manifest = json.loads(paths["next203_manifest"].read_text())
    evaluation = json.loads(paths["next203_evaluation"].read_text())
    expected_outputs = {
        n203.CATALOGUE_NAME: input_hashes["next203_catalogue"],
        n203.EVALUATION_NAME: input_hashes["next203_evaluation"],
        n203.FORMULA_NAME: input_hashes["next203_formula"],
        n203.SEARCH_NAME: input_hashes["next203_search"],
    }
    if (
        manifest.get("protocol") != n203.PROTOCOL
        or manifest.get("candidate_count") != n203.EXPECTED_CANDIDATE_COUNT
        or manifest.get("eligible_hypothesis_count") != len(eligible)
        or manifest.get("base_endpoint_reproduced") is not True
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("motif_conjunction_exception_search_branch_terminated")
        is not True
        or manifest.get("opened_validation_outputs_used") is not False
        or manifest.get("scigen_replication_endpoint_opened") is not False
        or manifest.get("wyformer_replication_endpoint_opened") is not False
        or manifest.get("dft_calculation_executed") is not False
        or manifest.get("dft_values_used_by_executable_formula") is not False
        or manifest.get("learned_energy_force_stress_proxy_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or manifest.get("executed_source_sha256", {}).get(
            "src/next203_motif_conjunction_exception_search.py"
        )
        != _sha256_file(Path(n203.__file__).resolve())
        or evaluation.get("protocol") != n203.PROTOCOL
        or evaluation.get("candidate_count") != n203.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("freeze_authorized") is not False
    ):
        raise ValueError("NEXT204 NEXT203 provenance differs")
    return eligible


def run_motif_conjunction_broad_residual(
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
    next179_dir: Path,
    next180_dir: Path,
    next181_dir: Path,
    next182_dir: Path,
    next183_dir: Path,
    next184_dir: Path,
    next185_dir: Path,
    next186_dir: Path,
    next188_dir: Path,
    next190_dir: Path,
    next192_dir: Path,
    next194_dir: Path,
    next199_dir: Path,
    next200_dir: Path,
    next201_dir: Path,
    next202_dir: Path,
    next203_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT204 BROAD residual diagnostic."""

    stage_values = (
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
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in stage_values},
        "next194": Path(next194_dir).resolve(),
        "next199": Path(next199_dir).resolve(),
        "next200": Path(next200_dir).resolve(),
        "next201": Path(next201_dir).resolve(),
        "next202": Path(next202_dir).resolve(),
        "next203": Path(next203_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT204 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT204 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT204 formal input identity differs: {differing}")
    eligible_names = _verify_next203(paths, input_hashes)

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n203.EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT204 base candidate identity differs")

    combined, _, old_terms, mhcr_terms = (
        n203.n202.n200.n194.n130._join_label_free_features(paths)
    )
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    combined = combined.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    combined = pd.concat(
        [
            combined.reset_index(drop=True),
            n203.n202.n200.n194.n135.materialize_conjunctive_features(
                combined
            ).reset_index(drop=True),
        ],
        axis=1,
    )
    closure_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next179_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        closure_frames.append(table)
    combined = combined.merge(
        pd.concat(closure_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    motif_columns = [
        "material_id", "motif_weight_sum_min", *n203.n202.SECONDARY_FEATURES
    ]
    motif_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(
            paths[f"next199_{source}_features"]
        )[motif_columns].copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        motif_frames.append(table)
    motif_table = pd.concat(motif_frames, ignore_index=True)
    combined = combined.merge(
        motif_table, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(motif_table):
        raise ValueError("NEXT204 motif row accounting differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n203.n202.n200.n194.n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n203.n202.n200.n194.n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n203.n202.n200.n194.n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT204 base reconstruction differs")
    combined, base_terms, base_runtime = (
        n203.n202.n200.n194.n163.materialize_candidates(
            features=combined,
            physical_terms=physical_terms,
            specs=selected_specs,
        )
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT204 base materialization differs")
    base_score, base_support = n203.n202.n200.n194.n87._term_risk(
        combined, base_terms[0]
    )

    raw_weakest = pd.to_numeric(
        combined["motif_weight_sum_min"], errors="coerce"
    ).to_numpy(float)
    weakest_by_floor = {
        floor: n203.n202.weakest_site_confidence(
            raw_weakest, floor_threshold=floor
        )
        for _, floor in n203.n202.FLOOR_LEVELS
    }
    clean_by_feature = {
        feature: n203.n202.secondary_cleanliness(
            pd.to_numeric(combined[feature], errors="coerce").to_numpy(float),
            feature=feature,
        )
        for feature in n203.n202.SECONDARY_FEATURES
    }
    certificates = {}
    for hypothesis in eligible_names:
        secondary, floor, conjunction, _ = n203.n202.HYPOTHESES[hypothesis]
        certificates[hypothesis] = n203.n202.motif_conjunction_certificate(
            weakest_site=weakest_by_floor[floor],
            secondary=clean_by_feature[secondary],
            conjunction=conjunction,
        )

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
                    "_endpoint": n203.n202.n200.n194.n130.n125.n121.prior._endpoint_numeric(
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
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(endpoint) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT204 endpoint row accounting differs")

    specs = n203.build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible_names
    )
    combined, virtual_terms, runtime = n203.materialize_motif_exception_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        certificates=certificates,
        specs=specs,
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0])
        for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    published_all = pd.read_parquet(paths["next203_search"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT204 published universe differs")
    published = select_diagnostic_candidates(published_all)
    candidate_sha = hashlib.sha256(
        "\n".join(published["candidate_key"].astype(str)).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT204 frozen diagnostic population differs")
    selected_keys = set(published["candidate_key"].astype(str))
    rerun_runtime = [
        spec for spec in runtime if str(spec["candidate_key"]) in selected_keys
    ]
    rerun = n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=rerun_runtime,
        workers=search_workers,
    )
    n164._verify_reproduction(
        rerun=rerun["candidate_records"], published=published
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
    records: list[dict[str, object]] = []
    frequency: Counter[str] = Counter()
    for _, row in published.iterrows():
        key = str(row["candidate_key"])
        score, supported = n203.n202.n200.n194.n87._term_risk(
            combined, virtual_by_id[virtual_by_candidate[key]]
        )
        tables = n164._threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT204 candidate has no threshold table")
        diagnostic = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT204 contradicts NEXT203 BROAD result")
        for failure in diagnostic["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "certificate_hypothesis": row["certificate_hypothesis"],
                "secondary_feature": row["secondary_feature"],
                "floor_threshold": row["floor_threshold"],
                "conjunction": row["conjunction"],
                "certificate_cutoff": float(row["certificate_cutoff"]),
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
    global_closest = select_closest_residual(per_candidate)
    base_rows = per_candidate.loc[per_candidate["certificate_hypothesis"].isna()]
    if len(base_rows) != 1:
        raise RuntimeError("NEXT204 base residual count differs")
    base_residual = base_rows.iloc[0]
    by_hypothesis = {}
    grouped = per_candidate.assign(
        certificate_hypothesis=per_candidate["certificate_hypothesis"].fillna(
            "BASE"
        )
    ).groupby("certificate_hypothesis", sort=True)
    for hypothesis, frame in grouped:
        closest = select_closest_residual(frame)
        by_hypothesis[str(hypothesis)] = {
            "candidate_count": int(len(frame)),
            "minimum_failed_constraint_count": int(
                closest["failed_constraint_count"]
            ),
            "minimum_normalized_shortfall_sum_at_best_count": float(
                closest["normalized_shortfall_sum"]
            ),
            "closest_certificate_cutoff": float(closest["certificate_cutoff"]),
            "closest_candidate_key": str(closest["candidate_key"]),
        }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next203_broad_constraint_residual",
        "candidate_count": int(len(per_candidate)),
        "candidate_key_sha256": candidate_sha,
        "base_residual": {
            "best_threshold": float(base_residual["best_threshold"]),
            "failed_constraint_count": int(base_residual["failed_constraint_count"]),
            "normalized_shortfall_sum": float(
                base_residual["normalized_shortfall_sum"]
            ),
            "failures": json.loads(str(base_residual["failures_json"])),
        },
        "by_hypothesis": by_hypothesis,
        "global_closest": {
            "candidate_key": str(global_closest["candidate_key"]),
            "certificate_hypothesis": global_closest["certificate_hypothesis"],
            "secondary_feature": global_closest["secondary_feature"],
            "floor_threshold": global_closest["floor_threshold"],
            "conjunction": global_closest["conjunction"],
            "certificate_cutoff": float(global_closest["certificate_cutoff"]),
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
        "src/next164_interior_attenuation_broad_residual.py": Path(
            n164.__file__
        ).resolve(),
        "src/next203_motif_conjunction_exception_search.py": Path(
            n203.__file__
        ).resolve(),
        "src/next204_motif_conjunction_broad_residual.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
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
            "next203_records_reproduced": True,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "motif_conjunction_broad_residual_diagnosed": True,
            "motif_conjunction_exception_branch_closed": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
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
            raise RuntimeError("NEXT204 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT204 source changed before publication")
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
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next194-dir", type=Path, required=True)
    parser.add_argument("--next199-dir", type=Path, required=True)
    parser.add_argument("--next200-dir", type=Path, required=True)
    parser.add_argument("--next201-dir", type=Path, required=True)
    parser.add_argument("--next202-dir", type=Path, required=True)
    parser.add_argument("--next203-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_motif_conjunction_broad_residual(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next194_dir=args.next194_dir,
        next199_dir=args.next199_dir,
        next200_dir=args.next200_dir,
        next201_dir=args.next201_dir,
        next202_dir=args.next202_dir,
        next203_dir=args.next203_dir,
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "run_motif_conjunction_broad_residual",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
