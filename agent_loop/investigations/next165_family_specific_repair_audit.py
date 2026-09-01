#!/usr/bin/env python3
"""Audit family-specific concentration in the frozen NEXT164 repair shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next151_violation_multiplicity_audit as n151
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds


PROTOCOL = "2026-08-08-next165-family-specific-repair-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT165_FAMILY_SPECIFIC_REPAIR_AUDIT.json"
TABLE_NAME = "next165_family_specific_repair_audit.parquet"
EXPECTED_DESIGN_SHA256 = "884005cf158cabdd489c7c3901178d321d37170f26647d62fc0029256023b1d0"
EXPECTED_CANDIDATE_KEY_SHA256 = "1d0ea8331f38aa69cfdedbe664d5ceb46c14e166e121bae92d9e14dd4fc6109e"
SAFE_THRESHOLD = 0.5415470292150686
BROAD_THRESHOLD = 0.21976295573076796
EXPECTED_INPUT_SHA256 = {
    **n164.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next164_manifest": "21489c49397e98fc3b5c1f7e4b45d4ee8b4bd2108e2fc4fbeffa17b4b9b5eb58",
    "next164_diagnostic": "12176c0ab6069ee48b1eeaf0dcc04ba7a3b24ed85f785a49221ad7bc703ad20d",
    "next164_per_candidate": "8487d76fbf345aab2778428be383c3006907e4e5c74f92837f3041c0dd146278",
}


FAMILY_PREFIXES = {
    "local_geometry": ("cov_", "scbv_", "sivr_"),
    "charge_flow_feasibility": ("cmvo_", "hcid_"),
    "valence_transport": ("bvtbd_", "bvtc_"),
    "contact_robustness": ("mhcr_",),
}
FIXED_DIRECTIONS = {
    **{
        name: 1
        for family in FAMILY_PREFIXES
        for name in (
            f"{family}_share",
            f"{family}_margin",
            f"{family}_is_dominant",
        )
    },
    "largest_family_share": 1,
    "effective_family_count": -1,
    "normalized_family_entropy": -1,
}


def family_repair_statistics(
    contributions: object, term_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return the frozen family-concentration catalogue."""

    values = np.asarray(contributions, dtype=float)
    if (
        values.ndim != 2
        or len(term_ids) != values.shape[1]
        or np.any(~np.isfinite(values))
        or np.any(values < -1.0e-12)
    ):
        raise ValueError("NEXT165 contribution matrix differs")
    values = np.maximum(values, 0.0)
    members: dict[str, list[int]] = {family: [] for family in FAMILY_PREFIXES}
    for index, term_id in enumerate(term_ids):
        matches = [
            family
            for family, prefixes in FAMILY_PREFIXES.items()
            if str(term_id).startswith(prefixes)
        ]
        if len(matches) != 1:
            raise ValueError("NEXT165 term-to-family assignment differs")
        members[matches[0]].append(index)
    if any(not indices for indices in members.values()):
        raise ValueError("NEXT165 family coverage differs")

    capped_means = np.column_stack(
        [
            np.minimum(values[:, indices], 0.5).mean(axis=1)
            for indices in members.values()
        ]
    )
    total = capped_means.sum(axis=1)
    shares = np.divide(
        capped_means,
        total[:, None],
        out=np.zeros_like(capped_means),
        where=total[:, None] > 0.0,
    )
    largest = capped_means.max(axis=1)
    result: dict[str, np.ndarray] = {}
    for index, family in enumerate(FAMILY_PREFIXES):
        other_max = np.delete(capped_means, index, axis=1).max(axis=1)
        result[f"{family}_share"] = shares[:, index]
        result[f"{family}_margin"] = np.maximum(
            capped_means[:, index] - other_max, 0.0
        )
        result[f"{family}_is_dominant"] = (
            (capped_means[:, index] > 0.0)
            & (capped_means[:, index] == largest)
        ).astype(float)
    result["largest_family_share"] = shares.max(axis=1)
    squared = np.square(capped_means).sum(axis=1)
    result["effective_family_count"] = np.divide(
        np.square(total), squared, out=np.zeros_like(total), where=squared > 0.0
    )
    log_shares = np.zeros_like(shares)
    positive = shares > 0.0
    log_shares[positive] = np.log(shares[positive])
    result["normalized_family_entropy"] = -(
        shares * log_shares
    ).sum(axis=1) / np.log(float(len(FAMILY_PREFIXES)))
    if set(result) != set(FIXED_DIRECTIONS):
        raise RuntimeError("NEXT165 statistic schema differs")
    return result


def select_family_repair_statistic(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Apply the frozen eligibility-first ranking to audit records."""

    required = {
        "statistic",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["statistic"].astype(str).duplicated().any():
        raise ValueError("NEXT165 audit record schema differs")
    table = records.sort_values(
        [
            "eligible_for_search",
            "ranking_min_auc",
            "ranking_mean_auc",
            "statistic",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"].fillna(False).astype(bool)]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n164._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next164_manifest": roots["next164"] / n164.MANIFEST_NAME,
            "next164_diagnostic": roots["next164"] / n164.DIAGNOSTIC_NAME,
            "next164_per_candidate": roots["next164"] / n164.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_family_specific_repair_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run and atomically publish the frozen NEXT165 discovery audit."""

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
        raise FileNotFoundError("NEXT165 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT165 formal input identity differs: {differing}")

    manifest164 = json.loads(paths["next164_manifest"].read_text())
    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    outputs164 = manifest164.get("outputs_sha256", {})
    closest = diagnostic164.get("global_closest", {})
    candidate_key = str(closest.get("candidate_key", ""))
    if (
        manifest164.get("protocol") != n164.PROTOCOL
        or manifest164.get("opened_validation_outputs_used") is not False
        or manifest164.get("dft_calculation_executed") is not False
        or manifest164.get("dft_values_used_by_executable_formula") is not False
        or manifest164.get("learned_energy_force_stress_proxy_used") is not False
        or manifest164.get("physical_relaxation_executed") is not False
        or outputs164.get(n164.DIAGNOSTIC_NAME)
        != input_hashes["next164_diagnostic"]
        or outputs164.get(n164.PER_CANDIDATE_NAME)
        != input_hashes["next164_per_candidate"]
        or manifest164.get("executed_source_sha256", {}).get(
            "src/next164_interior_attenuation_broad_residual.py"
        )
        != _sha256_file(Path(n164.__file__).resolve())
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
        or not np.isclose(
            float(closest.get("dominant_family_attenuation", np.nan)),
            0.075,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(closest.get("coordination_protection_weight", np.nan)),
            0.0,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(closest.get("packing_protection_weight", np.nan)),
            0.5,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError("NEXT165 prior provenance or closest candidate differs")

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
    physical_by_id = {str(term["term_id"]): dict(term) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n163.build_candidate_specs(
        bases=bases, physical_term_ids=set(physical_by_id)
    )
    selected_specs = [
        spec for spec in specs if str(spec["candidate_key"]) == candidate_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT165 candidate reconstruction differs")
    spec = selected_specs[0]

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
        raise ValueError("NEXT165 endpoint row accounting differs")

    combined, virtual_terms, runtime = n163.materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=selected_specs
    )
    if len(virtual_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT165 score materialization differs")
    published_score, published_support = _term_risk(combined, virtual_terms[0])
    contribution_columns = []
    contribution_support = np.ones(len(combined), dtype=bool)
    contribution_terms = []
    for term_id, weight in zip(
        spec["base_term_ids"], spec["base_weights"], strict=True
    ):
        risk, supported = _term_risk(combined, physical_by_id[str(term_id)])
        contribution_columns.append(float(weight) * risk)
        contribution_support &= supported
        contribution_terms.append(
            {"term_id": str(term_id), "weight": float(weight)}
        )
    if not np.array_equal(contribution_support, published_support):
        raise ValueError("NEXT165 contribution support differs from published score")
    contributions = np.column_stack(contribution_columns)
    statistics = family_repair_statistics(
        contributions, [str(term_id) for term_id in spec["base_term_ids"]]
    )

    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
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
    if min(
        population_counts[name][label]
        for name in populations
        for label in ("protected", "severe")
    ) < 1:
        raise ValueError("NEXT165 audit population differs")

    records = []
    for name in sorted(statistics):
        direction = FIXED_DIRECTIONS[name]
        evaluations = {}
        for population, mask in populations.items():
            result = n151._evaluate_auc(
                values=statistics[name][mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if result is None:
                raise ValueError("NEXT165 AUC population differs")
            evaluations[population] = result
        key_aucs = [
            float(evaluations["scigen_shell"]["worst_auc"]),
            float(evaluations["wyformer_shell"]["pooled_auc"]),
            float(evaluations["scigen_full"]["pooled_auc"]),
            float(evaluations["wyformer_full"]["pooled_auc"]),
        ]
        eligible = bool(
            key_aucs[0] >= 0.55
            and key_aucs[1] >= 0.55
            and key_aucs[2] >= 0.50
            and key_aucs[3] >= 0.50
            and int(evaluations["scigen_shell"]["evaluable_folds"]) == 5
        )
        record: dict[str, object] = {
            "statistic": name,
            "direction": direction,
            "eligible_for_search": eligible,
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
    audit_table, selected = select_family_repair_statistic(pd.DataFrame(records))
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "statistic"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "base_term_count": len(contribution_terms),
        "base_terms": contribution_terms,
        "family_prefixes": FAMILY_PREFIXES,
        "population_counts": population_counts,
        "fixed_directions": FIXED_DIRECTIONS,
        "eligibility_gates": {
            "scigen_shell_worst_auc": 0.55,
            "wyformer_shell_pooled_auc": 0.55,
            "full_source_pooled_auc": 0.50,
            "required_scigen_shell_folds": 5,
        },
        "eligible_statistics": eligible_names,
        "selected_statistic": selected,
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
        "src/next163_interior_family_attenuation_search.py": Path(
            n163.__file__
        ).resolve(),
        "src/next165_family_specific_repair_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(audit_path, audit)
        audit_table.to_parquet(table_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "eligible_statistic_count": len(eligible_names),
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
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT165 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT165 source changed before publication")
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
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_family_specific_repair_audit(
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
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAMILY_PREFIXES",
    "FIXED_DIRECTIONS",
    "family_repair_statistics",
    "run_family_specific_repair_audit",
    "select_family_repair_statistic",
]
