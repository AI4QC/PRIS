#!/usr/bin/env python3
"""Diagnose BROAD residuals for frozen signed-local closure candidates."""

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
import src.next192_signed_safe_margin_audit as n192
import src.next194_signed_local_closure_audit as n194
import src.next195_signed_local_closure_search as n195
import src.next87_scigen_sparse_law_search as n87
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next196-signed-local-closure-broad-residual-v1"
MANIFEST_NAME = "MANIFEST.json"
DIAGNOSTIC_NAME = "NEXT196_SIGNED_LOCAL_CLOSURE_BROAD_RESIDUAL.json"
PER_CANDIDATE_NAME = "next196_signed_local_closure_broad_residual_per_candidate.parquet"
EXPECTED_DESIGN_SHA256 = (
    "305e1d871a6c427e64b073a97bc032024f41beddb0c255ad0ea41dd6a187a6dc"
)
EXPECTED_CANDIDATE_COUNT = 37
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "c9019f11352eb4cbf44731ac6d4b182dfc8339a9aaf552e40a56631e340a91fb"
)
SEARCH_WORKERS = 4
EXPECTED_INPUT_SHA256 = {
    **n195.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next195_manifest": "4d831ead0f2b255ac176302edd421496c53947cb34d6d64ed7c599e0ab7c757d",
    "next195_catalogue": "388b6090b6f85a11b6ecc2dd34c2caa7c7a57f3f9e9e5017e4813c8562badccf",
    "next195_evaluation": "b7f7aeffd8910a793a9b6b6d04c202896a1f8a446fac8cc3342b8087b7f9c62d",
    "next195_formula": "e4a5d48daeeed48bec4f0dae937ba861d421a29c07d39f1e2c451bc734c1af5b",
    "next195_search": "a3ef422a9e1b76e64d6e979885a95308c6d87b11bef559264509c334ff596629",
}


def select_diagnostic_candidates(published: pd.DataFrame) -> pd.DataFrame:
    """Select the exact AUC+SAFE, non-BROAD diagnostic population."""

    required = {
        "candidate_key",
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
    }
    if required - set(published.columns) or published["candidate_key"].astype(str).duplicated().any():
        raise ValueError("NEXT196 published candidate schema differs")
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
        raise ValueError("NEXT196 closest population differs")
    return frame.sort_values(
        ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
        kind="mergesort",
    ).iloc[0]


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n195._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next195_manifest": roots["next195"] / n195.MANIFEST_NAME,
            "next195_catalogue": roots["next195"] / n195.CATALOGUE_NAME,
            "next195_evaluation": roots["next195"] / n195.EVALUATION_NAME,
            "next195_formula": roots["next195"] / n195.FORMULA_NAME,
            "next195_search": roots["next195"] / n195.SEARCH_NAME,
        }
    )
    return paths


def run_signed_local_closure_broad_residual(
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
    next195_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT196 BROAD residual diagnostic."""

    stage_values = (
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
        (179, next179_dir),
        (180, next180_dir),
        (181, next181_dir),
        (182, next182_dir),
        (183, next183_dir),
        (184, next184_dir),
        (185, next185_dir),
        (186, next186_dir),
        (188, next188_dir),
        (190, next190_dir),
        (192, next192_dir),
        (194, next194_dir),
        (195, next195_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in stage_values},
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
        raise FileNotFoundError("NEXT196 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT196 formal input identity differs: {differing}")

    manifest195 = json.loads(paths["next195_manifest"].read_text())
    evaluation195 = json.loads(paths["next195_evaluation"].read_text())
    expected_outputs195 = {
        n195.CATALOGUE_NAME: input_hashes["next195_catalogue"],
        n195.EVALUATION_NAME: input_hashes["next195_evaluation"],
        n195.FORMULA_NAME: input_hashes["next195_formula"],
        n195.SEARCH_NAME: input_hashes["next195_search"],
    }
    if (
        manifest195.get("protocol") != n195.PROTOCOL
        or manifest195.get("candidate_count") != n195.EXPECTED_CANDIDATE_COUNT
        or manifest195.get("eligible_hypothesis_count") != n195.EXPECTED_ELIGIBLE_COUNT
        or manifest195.get("base_endpoint_reproduced") is not True
        or manifest195.get("passes_all_cross_source_discovery_gates") is not False
        or manifest195.get("freeze_authorized") is not False
        or manifest195.get("signed_local_closure_search_branch_terminated") is not True
        or manifest195.get("opened_validation_outputs_used") is not False
        or manifest195.get("scigen_replication_endpoint_opened") is not False
        or manifest195.get("wyformer_replication_endpoint_opened") is not False
        or manifest195.get("dft_calculation_executed") is not False
        or manifest195.get("dft_values_used_by_executable_formula") is not False
        or manifest195.get("learned_energy_force_stress_proxy_used") is not False
        or manifest195.get("physical_relaxation_executed") is not False
        or manifest195.get("outputs_sha256") != expected_outputs195
        or manifest195.get("executed_source_sha256", {}).get(
            "src/next195_signed_local_closure_search.py"
        )
        != _sha256_file(Path(n195.__file__).resolve())
        or evaluation195.get("protocol") != n195.PROTOCOL
        or evaluation195.get("candidate_count") != n195.EXPECTED_CANDIDATE_COUNT
        or evaluation195.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation195.get("freeze_authorized") is not False
    ):
        raise ValueError("NEXT196 NEXT195 provenance differs")

    audit194 = json.loads(paths["next194_audit"].read_text())
    eligible_names = [str(value) for value in audit194["eligible_hypotheses"]]
    if len(eligible_names) != n195.EXPECTED_ELIGIBLE_COUNT:
        raise ValueError("NEXT196 eligible certificate universe differs")

    combined, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    combined = combined.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    combined = pd.concat(
        [
            combined.reset_index(drop=True),
            n135.materialize_conjunctive_features(combined).reset_index(drop=True),
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
        on="material_id",
        how="inner",
        validate="one_to_one",
    )

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n195.EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT196 base candidate identity differs")
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
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_base_specs) != 1:
        raise ValueError("NEXT196 base reconstruction differs")
    combined, base_terms, base_runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_base_specs,
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT196 base materialization differs")
    base_score, base_support = n87._term_risk(combined, base_terms[0])
    family_means = n192.complementary_safe_family_means(
        features=combined,
        physical_terms=physical_terms,
        base_spec=selected_base_specs[0],
        base_support=base_support,
    )
    signed_local = family_means["local_geometry"]
    certificates = {}
    for hypothesis in eligible_names:
        closure_feature, conjunction, _ = n194.HYPOTHESES[hypothesis]
        certificates[hypothesis] = n194.signed_local_closure_certificate(
            closure=pd.to_numeric(
                combined[closure_feature], errors="coerce"
            ).to_numpy(float),
            signed_local_safety=signed_local,
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
    if len(combined) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT196 endpoint row accounting differs")

    specs = n195.build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible_names
    )
    combined, virtual_terms, runtime = n195.materialize_signed_local_closure_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        certificates=certificates,
        specs=specs,
    )
    virtual_by_candidate = {
        str(spec["candidate_key"]): str(spec["base_term_ids"][0]) for spec in runtime
    }
    virtual_by_id = {str(term["term_id"]): term for term in virtual_terms}
    published_all = pd.read_parquet(paths["next195_search"])
    if set(published_all["candidate_key"].astype(str)) != set(virtual_by_candidate):
        raise ValueError("NEXT196 published universe differs")
    published = select_diagnostic_candidates(published_all)
    candidate_sha = hashlib.sha256(
        "\n".join(published["candidate_key"].astype(str)).encode()
    ).hexdigest()
    if require_formal_inputs and (
        len(published) != EXPECTED_CANDIDATE_COUNT
        or candidate_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT196 frozen diagnostic population differs")
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
        score, supported = n87._term_risk(
            combined, virtual_by_id[virtual_by_candidate[key]]
        )
        tables = n164._threshold_tables(
            score=score, supported=supported, endpoint=endpoint, cells=cells
        )
        if tables is None:
            raise RuntimeError("NEXT196 candidate has no threshold table")
        diagnostic = n164.diagnose_broad_threshold_tables(
            tables=tables,
            cells=cells,
            pauling_by_cell=pauling_by_cell,
            safe_threshold=float(row["safe_threshold"]),
        )
        if diagnostic["passes_broad"] or bool(row["passes_broad_all_cells"]):
            raise RuntimeError("NEXT196 contradicts NEXT195 BROAD result")
        for failure in diagnostic["failures"]:
            frequency[f"{failure['cell_id']}::{failure['component']}"] += 1
        records.append(
            {
                "candidate_key": key,
                "certificate_hypothesis": row["certificate_hypothesis"],
                "closure_feature": row["closure_feature"],
                "conjunction": row["conjunction"],
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
    global_closest = select_closest_residual(per_candidate)
    base_rows = per_candidate.loc[per_candidate["certificate_hypothesis"].isna()]
    if len(base_rows) != 1:
        raise RuntimeError("NEXT196 base residual count differs")
    base_residual = base_rows.iloc[0]
    by_hypothesis = {}
    grouped = per_candidate.assign(
        certificate_hypothesis=per_candidate["certificate_hypothesis"].fillna("BASE")
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
            "closest_attenuation": float(closest["attenuation"]),
            "closest_candidate_key": str(closest["candidate_key"]),
        }
    summary = {
        "protocol": PROTOCOL,
        "diagnostic_mode": "offline_discovery_label_next195_broad_constraint_residual",
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
            "closure_feature": global_closest["closure_feature"],
            "conjunction": global_closest["conjunction"],
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
        "src/next164_interior_attenuation_broad_residual.py": Path(
            n164.__file__
        ).resolve(),
        "src/next195_signed_local_closure_search.py": Path(n195.__file__).resolve(),
        "src/next196_signed_local_closure_broad_residual.py": Path(__file__).resolve(),
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
            "next195_records_reproduced": True,
            "new_formula_searched": False,
            "new_formula_selected": False,
            "signed_local_closure_broad_residual_diagnosed": True,
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
            raise RuntimeError("NEXT196 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT196 source changed before publication")
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
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        188,
        190,
        192,
        194,
        195,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_signed_local_closure_broad_residual(
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
    "run_signed_local_closure_broad_residual",
    "select_closest_residual",
    "select_diagnostic_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
