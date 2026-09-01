#!/usr/bin/env python3
"""Frozen audit of strong-neighborhood directional-closure hypotheses."""

from __future__ import annotations

import argparse
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
import src.next151_violation_multiplicity_audit as n151
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next169_periodic_local_directional_rigidity_audit as n169
import src.next174_weighted_local_directional_rigidity_audit as n174
import src.next179_strong_neighborhood_directional_closure as n179
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next180-strong-neighborhood-directional-closure-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT180_STRONG_NEIGHBORHOOD_DIRECTIONAL_CLOSURE_AUDIT.json"
TABLE_NAME = "next180_strong_neighborhood_directional_closure_audit.parquet"
EXPECTED_DESIGN_SHA256 = n179.EXPECTED_DESIGN_SHA256
EXPECTED_CANDIDATE_KEY_SHA256 = n174.EXPECTED_CANDIDATE_KEY_SHA256
SAFE_THRESHOLD = n174.SAFE_THRESHOLD
BROAD_THRESHOLD = n174.BROAD_THRESHOLD
MIN_FULL_SUPPORT = n174.MIN_FULL_SUPPORT
HYPOTHESES = {
    f"{mode}_{suffix}__high": (f"psndc_{mode}_{suffix}", 1)
    for mode in n179.GRAPH_MODES
    for suffix in n179.FEATURE_SUFFIXES
}
EXPECTED_INPUT_SHA256 = {
    **n174.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next179_manifest": "68a6f2de167ad23356524204bf84af4c768ebabccbad034dbfb9e62194cea3a7",
    "next179_catalogue": "12da40c842e40ced321020401adbb8a251d66de45b3eee6b3130caccf9a4a8dd",
    "next179_scigen_features": "75cd8aa8708fbdbd92c7e9e5965540126f26c2bea23b6419038ecde54c2670a8",
    "next179_wyformer_features": "253cead7a35faf7d314013d9ac3be80143a4f405a30ec05c1f2327cc560f0dfa",
}


def eligibility_from_metrics(
    *,
    scigen_full_support: float,
    wyformer_full_support: float,
    scigen_shell_worst_auc: float,
    scigen_shell_evaluable_folds: int,
    wyformer_shell_pooled_auc: float,
    scigen_full_pooled_auc: float,
    wyformer_full_pooled_auc: float,
) -> bool:
    """Apply the unchanged frozen NEXT169 audit gates."""

    return n169.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_strong_closure_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible hypotheses with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if (
        required - set(records.columns)
        or records["hypothesis"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT180 audit record schema differs")
    table = records.sort_values(
        [
            "eligible_for_search",
            "ranking_min_auc",
            "ranking_mean_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"].fillna(False).astype(bool)]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n174._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next179_manifest": roots["next179"] / n179.MANIFEST_NAME,
            "next179_catalogue": roots["next179"] / n179.CATALOGUE_NAME,
            "next179_scigen_features": roots["next179"] / n179.SCIGEN_NAME,
            "next179_wyformer_features": roots["next179"] / n179.WYFORMER_NAME,
        }
    )
    return paths


def run_strong_neighborhood_directional_closure_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT180 audit and publish atomically."""

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
                (179, next179_dir),
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
        raise FileNotFoundError("NEXT180 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT180 formal input identity differs: {differing}")

    manifest164 = json.loads(paths["next164_manifest"].read_text())
    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    closest = diagnostic164.get("global_closest", {})
    candidate_key = str(closest.get("candidate_key", ""))
    if (
        manifest164.get("protocol") != n164.PROTOCOL
        or manifest164.get("opened_validation_outputs_used") is not False
        or manifest164.get("dft_values_used_by_executable_formula") is not False
        or manifest164.get("outputs_sha256", {}).get(n164.DIAGNOSTIC_NAME)
        != input_hashes["next164_diagnostic"]
        or hashlib.sha256(candidate_key.encode()).hexdigest()
        != EXPECTED_CANDIDATE_KEY_SHA256
        or not np.isclose(
            float(closest.get("safe_threshold", np.nan)),
            SAFE_THRESHOLD,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(closest.get("best_threshold", np.nan)),
            BROAD_THRESHOLD,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError("NEXT180 closest-candidate provenance differs")
    manifest179 = json.loads(paths["next179_manifest"].read_text())
    if (
        manifest179.get("protocol") != n179.PROTOCOL
        or manifest179.get("labels_or_endpoints_opened") is not False
        or manifest179.get("validation_geometry_opened") is not False
        or manifest179.get("replication_geometry_opened") is not False
        or manifest179.get("dft_calculation_executed") is not False
        or manifest179.get("dft_values_used_by_features") is not False
        or manifest179.get("learned_energy_force_stress_proxy_used") is not False
        or manifest179.get("physical_relaxation_executed") is not False
        or manifest179.get("outputs_sha256", {}).get(n179.CATALOGUE_NAME)
        != input_hashes["next179_catalogue"]
        or manifest179.get("outputs_sha256", {}).get(n179.SCIGEN_NAME)
        != input_hashes["next179_scigen_features"]
        or manifest179.get("outputs_sha256", {}).get(n179.WYFORMER_NAME)
        != input_hashes["next179_wyformer_features"]
        or manifest179.get("executed_source_sha256", {}).get(
            "src/next179_strong_neighborhood_directional_closure.py"
        )
        != _sha256_file(Path(n179.__file__).resolve())
    ):
        raise ValueError("NEXT180 strong-closure provenance differs")

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
    closure_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next179_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        closure_frames.append(table)
    extended = extended.merge(
        pd.concat(closure_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n163.build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    selected_specs = [
        spec for spec in specs if str(spec["candidate_key"]) == candidate_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT180 candidate reconstruction differs")

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
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(
        float
    )
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT180 endpoint row accounting differs")
    combined, virtual_terms, runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(virtual_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT180 score materialization differs")
    published_score, published_support = _term_risk(combined, virtual_terms[0])

    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    extremes = published_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = (
        extremes
        & (published_score >= BROAD_THRESHOLD)
        & (published_score < SAFE_THRESHOLD)
    )
    populations = {
        "scigen_shell": shell & (sources == "scigen"),
        "wyformer_shell": shell & (sources == "wyformer"),
        "scigen_full": extremes & (sources == "scigen"),
        "wyformer_full": extremes & (sources == "wyformer"),
    }
    population_counts = {
        name: {
            "rows": int(mask.sum()),
            "protected": int((mask & (endpoint <= 1.0)).sum()),
            "severe": int((mask & (endpoint >= 2.0)).sum()),
        }
        for name, mask in populations.items()
    }

    records = []
    for hypothesis in sorted(HYPOTHESES):
        column, direction = HYPOTHESES[hypothesis]
        values = pd.to_numeric(combined[column], errors="coerce").to_numpy(float)
        evaluations = {}
        for population, mask in populations.items():
            result = n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if result is None:
                raise ValueError("NEXT180 AUC population differs")
            evaluations[population] = result
        support = {
            source: float(np.isfinite(values[sources == source]).mean())
            for source in ("scigen", "wyformer")
        }
        scigen_worst = evaluations["scigen_shell"]["worst_auc"]
        key_aucs = [
            0.0 if scigen_worst is None else float(scigen_worst),
            float(evaluations["wyformer_shell"]["pooled_auc"]),
            float(evaluations["scigen_full"]["pooled_auc"]),
            float(evaluations["wyformer_full"]["pooled_auc"]),
        ]
        eligible = eligibility_from_metrics(
            scigen_full_support=support["scigen"],
            wyformer_full_support=support["wyformer"],
            scigen_shell_worst_auc=key_aucs[0],
            scigen_shell_evaluable_folds=int(
                evaluations["scigen_shell"]["evaluable_folds"]
            ),
            wyformer_shell_pooled_auc=key_aucs[1],
            scigen_full_pooled_auc=key_aucs[2],
            wyformer_full_pooled_auc=key_aucs[3],
        )
        record: dict[str, object] = {
            "hypothesis": hypothesis,
            "feature": column,
            "direction": direction,
            "eligible_for_search": eligible,
            "scigen_full_support": support["scigen"],
            "wyformer_full_support": support["wyformer"],
            "ranking_min_auc": float(min(key_aucs)),
            "ranking_mean_auc": float(np.mean(key_aucs)),
        }
        for population, result in evaluations.items():
            for metric in (
                "pooled_auc",
                "macro_auc",
                "worst_auc",
                "evaluable_folds",
                "protected",
                "severe",
            ):
                record[f"{population}_{metric}"] = result[metric]
            record[f"{population}_fold_aucs_json"] = json.dumps(
                result["fold_aucs"], separators=(",", ":")
            )
        records.append(record)

    audit_table, selected = select_strong_closure_hypothesis(pd.DataFrame(records))
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "population_counts": population_counts,
        "hypotheses": {
            name: {"feature": value[0], "direction": value[1]}
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": {
            "minimum_full_support": MIN_FULL_SUPPORT,
            "scigen_shell_worst_auc": 0.55,
            "wyformer_shell_pooled_auc": 0.55,
            "full_source_pooled_auc": 0.50,
            "required_scigen_shell_folds": 5,
        },
        "eligible_hypotheses": eligible_names,
        "selected_hypothesis": selected,
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
        "src/next169_periodic_local_directional_rigidity_audit.py": Path(n169.__file__).resolve(),
        "src/next179_strong_neighborhood_directional_closure.py": Path(n179.__file__).resolve(),
        "src/next180_strong_neighborhood_directional_closure_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(audit_path, audit)
        audit_table.to_parquet(table_path, index=False)
        terminated = selected is None
        manifest = {
            "protocol": PROTOCOL,
            "eligible_hypothesis_count": len(eligible_names),
            "strong_neighborhood_directional_closure_branch_terminated": terminated,
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
                AUDIT_NAME: _sha256_file(audit_path),
                TABLE_NAME: _sha256_file(table_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT180 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT180 source changed before publication")
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
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_strong_neighborhood_directional_closure_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "HYPOTHESES",
    "eligibility_from_metrics",
    "run_strong_neighborhood_directional_closure_audit",
    "select_strong_closure_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
