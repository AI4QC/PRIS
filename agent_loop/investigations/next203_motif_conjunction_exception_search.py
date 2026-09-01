#!/usr/bin/env python3
"""Search discrete protected exceptions from eligible motif conjunctions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next182_local_family_closure_attenuation_search as n182
import src.next202_motif_conjunction_audit as n202
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next203-motif-conjunction-exception-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT203_MOTIF_CONJUNCTION_EXCEPTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT203_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT203_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next203_motif_conjunction_exception_search.parquet"
EXPECTED_DESIGN_SHA256 = n202.EXPECTED_DESIGN_SHA256
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n202.n201.EXPECTED_BASE_CANDIDATE_KEY_SHA256
BROAD_THRESHOLD = n202.BROAD_THRESHOLD
SAFE_THRESHOLD = n202.SAFE_THRESHOLD
INTERVAL_FOLD_RATIO = BROAD_THRESHOLD / SAFE_THRESHOLD
CERTIFICATE_CUTOFFS = (
    1.0 / 16.0,
    1.0 / 8.0,
    3.0 / 16.0,
    1.0 / 4.0,
    3.0 / 8.0,
    1.0 / 2.0,
    5.0 / 8.0,
    3.0 / 4.0,
    7.0 / 8.0,
)
EXPECTED_ELIGIBLE_COUNT = 21
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(CERTIFICATE_CUTOFFS)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "base_score_if_outside_frozen_repair_interval_or_certificate_missing_or_"
    "certificate_below_cutoff_else_base_score*(broad_threshold/safe_threshold)"
)
EXPECTED_INPUT_SHA256 = {
    **n202.EXPECTED_INPUT_SHA256,
    "next202_manifest": "40247b89229088420b3d534c325d4ed71d5d3f33b92b67045def7892292b1fc9",
    "next202_audit": "a29f08876ccd0ce393f094fdf0214fd8ea8cf96816a1c8b03b2be8a00c35021c",
    "next202_table": "5152c267fa28b3053713699ec2386760dfeba93310cbc08cfd49c20eb74d2ec3",
}


def discrete_motif_exception_score(
    *,
    base_score: object,
    base_support: object,
    certificate: object,
    certificate_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold certified repair-interval rows strictly below BROAD."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(certificate, dtype=float)
    cutoff = float(certificate_cutoff)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or cutoff not in {0.0, *CERTIFICATE_CUTOFFS}
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT203 base score or certificate cutoff differs")
    finite = np.isfinite(values)
    if np.any((values[finite] < -1.0e-12) | (values[finite] > 1.0 + 1.0e-12)):
        raise ValueError("NEXT203 certificate is outside [0,1]")
    active = (
        support
        & finite
        & (score >= BROAD_THRESHOLD)
        & (score < SAFE_THRESHOLD)
        & (values >= cutoff)
        & (cutoff > 0.0)
    )
    corrected = score.copy()
    corrected[active] = score[active] * INTERVAL_FOLD_RATIO
    if np.any(corrected[active] >= BROAD_THRESHOLD):
        raise RuntimeError("NEXT203 interval fold did not land below BROAD")
    return corrected, support.copy(), active


def build_candidate_specs(
    *, base_candidate_key: str, eligible_hypotheses: Sequence[str]
) -> list[dict[str, object]]:
    """Build one base plus the exact eligible-certificate/cutoff product."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT203 base candidate key must be nonempty")
    eligible = tuple(sorted(str(value) for value in eligible_hypotheses))
    if (
        len(eligible) != EXPECTED_ELIGIBLE_COUNT
        or len(set(eligible)) != len(eligible)
        or any(name not in n202.HYPOTHESES for name in eligible)
    ):
        raise ValueError("NEXT203 eligible hypothesis universe differs")
    pairs: list[tuple[str | None, float]] = [(None, 0.0)]
    pairs.extend(
        (hypothesis, cutoff)
        for hypothesis in eligible
        for cutoff in CERTIFICATE_CUTOFFS
    )
    specs = []
    for hypothesis, cutoff in pairs:
        definition = None if hypothesis is None else n202.HYPOTHESES[hypothesis]
        payload = {
            "base_candidate_key": base_candidate_key,
            "broad_threshold": BROAD_THRESHOLD,
            "certificate_cutoff": cutoff,
            "certificate_hypothesis": hypothesis,
            "conjunction": None if definition is None else definition[2],
            "floor_threshold": None if definition is None else definition[1],
            "interval_fold_ratio": INTERVAL_FOLD_RATIO,
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "safe_threshold": SAFE_THRESHOLD,
            "score_composition": SCORE_COMPOSITION,
            "secondary_feature": None if definition is None else definition[0],
        }
        specs.append(
            {
                **payload,
                "candidate_key": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    if (
        len(specs) != EXPECTED_CANDIDATE_COUNT
        or len({str(spec["candidate_key"]) for spec in specs})
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError("NEXT203 candidate universe differs")
    return specs


def materialize_motif_exception_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    certificates: Mapping[str, object],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every exception score as an exactly recoverable virtual term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    if score.shape != (len(features),) or support.shape != score.shape:
        raise ValueError("NEXT203 base score shape differs")
    certificate_arrays = {
        str(name): np.asarray(values, dtype=float)
        for name, values in certificates.items()
    }
    if (
        len(certificate_arrays) != EXPECTED_ELIGIBLE_COUNT
        or any(name not in n202.HYPOTHESES for name in certificate_arrays)
        or any(values.shape != score.shape for values in certificate_arrays.values())
    ):
        raise ValueError("NEXT203 certificate table differs")
    expected_specs = build_candidate_specs(
        base_candidate_key=(
            str(specs[0].get("base_candidate_key", "")) if specs else ""
        ),
        eligible_hypotheses=tuple(certificate_arrays),
    )
    if [str(spec.get("candidate_key", "")) for spec in specs] != [
        str(spec["candidate_key"]) for spec in expected_specs
    ]:
        raise ValueError("NEXT203 candidate specs differ")

    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for raw in specs:
        spec = dict(raw)
        key = str(spec["candidate_key"])
        hypothesis = spec["certificate_hypothesis"]
        certificate = (
            np.full(len(features), np.nan)
            if hypothesis is None
            else certificate_arrays[str(hypothesis)]
        )
        corrected, corrected_support, _ = discrete_motif_exception_score(
            base_score=score,
            base_support=support,
            certificate=certificate,
            certificate_cutoff=float(spec["certificate_cutoff"]),
        )
        maximum = (
            float(np.max(corrected[corrected_support]))
            if corrected_support.any()
            else 0.0
        )
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[corrected_support] = np.sinh(
            corrected[corrected_support] / divisor
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next203_virtual_candidate__{digest}"
        column = f"_{virtual_id}_value"
        columns[column] = encoded
        terms.append(
            {
                "term_id": virtual_id,
                "feature": column,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next203_motif_conjunction_exception",
                "encoding": "asinh_sinh_exact_discrete_motif_exception_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [virtual_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
    )


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n202._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next202_manifest": roots["next202"] / n202.MANIFEST_NAME,
            "next202_audit": roots["next202"] / n202.AUDIT_NAME,
            "next202_table": roots["next202"] / n202.TABLE_NAME,
        }
    )
    return paths


def _verify_next202(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> list[str]:
    n202._verify_prior_boundaries(paths, input_hashes)
    manifest = json.loads(paths["next202_manifest"].read_text())
    audit = json.loads(paths["next202_audit"].read_text())
    eligible = audit.get("eligible_hypotheses")
    if (
        manifest.get("protocol") != n202.PROTOCOL
        or manifest.get("hypothesis_count") != len(n202.HYPOTHESES)
        or manifest.get("eligible_hypothesis_count") != EXPECTED_ELIGIBLE_COUNT
        or manifest.get("next203_search_authorized") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("opened_validation_outputs_used") is not False
        or manifest.get("scigen_replication_endpoint_opened") is not False
        or manifest.get("wyformer_replication_endpoint_opened") is not False
        or manifest.get("dft_calculation_executed") is not False
        or manifest.get("dft_values_used_by_executable_formula") is not False
        or manifest.get("learned_energy_force_stress_proxy_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or manifest.get("outputs_sha256")
        != {
            n202.AUDIT_NAME: input_hashes["next202_audit"],
            n202.TABLE_NAME: input_hashes["next202_table"],
        }
        or manifest.get("executed_source_sha256", {}).get(
            "src/next202_motif_conjunction_audit.py"
        )
        != _sha256_file(Path(n202.__file__).resolve())
    ):
        raise ValueError("NEXT203 NEXT202 provenance differs")
    if (
        audit.get("protocol") != n202.PROTOCOL
        or not isinstance(eligible, list)
        or len(eligible) != EXPECTED_ELIGIBLE_COUNT
        or len(set(str(name) for name in eligible)) != EXPECTED_ELIGIBLE_COUNT
        or any(str(name) not in n202.HYPOTHESES for name in eligible)
        or audit.get("selected_hypothesis") is None
        or audit.get("new_formula_searched") is not False
        or audit.get("validation_or_replication_opened") is not False
    ):
        raise ValueError("NEXT203 NEXT202 audit boundary differs")
    return sorted(str(name) for name in eligible)


def run_motif_conjunction_exception_search(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT203 search and publish atomically."""

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
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT203 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT203 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT203 formal input identity differs: {differing}")
    eligible_names = _verify_next202(paths, input_hashes)

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT203 base candidate identity differs")

    combined, feature_tables, old_terms, mhcr_terms = n202.n200.n194.n130._join_label_free_features(paths)
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
            n202.n200.n194.n135.materialize_conjunctive_features(combined).reset_index(drop=True),
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
        "material_id", "motif_weight_sum_min", *n202.SECONDARY_FEATURES
    ]
    motif_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next199_{source}_features"])[motif_columns].copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        motif_frames.append(table)
    motif_table = pd.concat(motif_frames, ignore_index=True)
    combined = combined.merge(
        motif_table, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(motif_table):
        raise ValueError("NEXT203 motif row accounting differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n202.n200.n194.n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n202.n200.n194.n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n202.n200.n194.n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT203 base reconstruction differs")
    combined, base_terms, base_runtime = n202.n200.n194.n163.materialize_candidates(
        features=combined, physical_terms=physical_terms, specs=selected_specs
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT203 base materialization differs")
    base_score, base_support = n202.n200.n194.n87._term_risk(combined, base_terms[0])

    raw_weakest = pd.to_numeric(
        combined["motif_weight_sum_min"], errors="coerce"
    ).to_numpy(float)
    weakest_by_floor = {
        floor: n202.weakest_site_confidence(raw_weakest, floor_threshold=floor)
        for _, floor in n202.FLOOR_LEVELS
    }
    clean_by_feature = {
        feature: n202.secondary_cleanliness(
            pd.to_numeric(combined[feature], errors="coerce").to_numpy(float),
            feature=feature,
        )
        for feature in n202.SECONDARY_FEATURES
    }
    certificates = {}
    for hypothesis in eligible_names:
        secondary, floor, conjunction, _ = n202.HYPOTHESES[hypothesis]
        certificates[hypothesis] = n202.motif_conjunction_certificate(
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
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n202.n200.n194.n130.n125.n121.prior._endpoint_numeric(
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
        raise ValueError("NEXT203 endpoint row accounting differs")

    specs = build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible_names
    )
    combined, virtual_terms, runtime = materialize_motif_exception_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        certificates=certificates,
        specs=specs,
    )
    started = time.perf_counter()
    result = n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT203 evaluator count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "base_candidate_key": base_key,
                "certificate_hypothesis": spec["certificate_hypothesis"],
                "certificate_cutoff": float(spec["certificate_cutoff"]),
                "secondary_feature": spec["secondary_feature"],
                "floor_threshold": spec["floor_threshold"],
                "conjunction": spec["conjunction"],
                "interval_fold_ratio": INTERVAL_FOLD_RATIO,
                "broad_threshold_frozen": BROAD_THRESHOLD,
                "safe_threshold_frozen": SAFE_THRESHOLD,
                "missing_policy": "TERM_OFF_KEEP_BASE",
                "score_composition": SCORE_COMPOSITION,
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "certificate_cutoff" not in selected["record"]:
        decorate(selected["record"])
    base_records = [
        record
        for record in result["candidate_records"]
        if record["certificate_hypothesis"] is None
    ]
    if len(base_records) != 1:
        raise RuntimeError("NEXT203 base candidate differs")
    n182.n181.n175.n170._verify_base_reproduction(
        record=base_records[0],
        published=pd.read_parquet(paths["next163_search"]),
        candidate_key=base_key,
    )
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    prior = json.loads(paths["next163_evaluation"].read_text())
    formula = {
        "protocol": PROTOCOL,
        "kind": "discrete_motif_conjunction_protected_exception_no_dft_score",
        "base_candidate_key": base_key,
        "base_formula": prior["selected_formula"],
        "certificate_hypothesis": selected_spec["certificate_hypothesis"],
        "secondary_feature": selected_spec["secondary_feature"],
        "floor_threshold": selected_spec["floor_threshold"],
        "conjunction": selected_spec["conjunction"],
        "certificate_cutoff": float(selected_spec["certificate_cutoff"]),
        "interval_fold_ratio": INTERVAL_FOLD_RATIO,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_threshold": SAFE_THRESHOLD,
        "interval_policy": "BROAD_INCLUSIVE_SAFE_EXCLUSIVE_ON_ORIGINAL_BASE_SCORE",
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    records = pd.DataFrame(result["candidate_records"])
    counts = {}
    grouped = records.assign(
        certificate_hypothesis=records["certificate_hypothesis"].fillna("BASE")
    ).groupby(["certificate_hypothesis", "certificate_cutoff"], sort=True)
    for (hypothesis, cutoff), frame in grouped:
        counts[f"hypothesis={hypothesis},cutoff={float(cutoff):g}"] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(frame["passes_all_discovery_gates"].sum()),
        }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "base_endpoint_reproduced": True,
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_count": len(eligible_names),
        "certificate_cutoff_grid": CERTIFICATE_CUTOFFS,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "interval_fold_ratio": INTERVAL_FOLD_RATIO,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_threshold": SAFE_THRESHOLD,
        "score_composition": SCORE_COMPOSITION,
        "base_support_unchanged": True,
        "outside_interval_exactly_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next163_interior_family_attenuation_search.py": Path(
            n202.n200.n194.n163.__file__
        ).resolve(),
        "src/next202_motif_conjunction_audit.py": Path(n202.__file__).resolve(),
        "src/next203_motif_conjunction_exception_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(
            evaluation_path,
            {
                "protocol": PROTOCOL,
                "evaluation_mode": "fixed_motif_conjunction_exception_search",
                "base_endpoint_reproduced": True,
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_hypothesis_and_cutoff": counts,
                "selected_record": selected["record"],
                "selected_formula": formula,
                "selected_safe": selected["safe"],
                "selected_safe_diagnostic": selected["safe_diagnostic"],
                "selected_broad": selected["broad"],
                "selected_source_diagnostics": selected["source_diagnostics"],
                "pauling_by_cell": result["pauling_by_cell"],
                "cells": result["cells"],
                "passes_all_cross_source_discovery_gates": passes,
                "freeze_authorized": passes,
                "requires_unopened_internal_validation_before_claim": True,
            },
        )
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "eligible_hypothesis_count": len(eligible_names),
            "search_workers": search_workers,
            "base_endpoint_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "motif_conjunction_exception_search_branch_terminated": not passes,
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
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                CATALOGUE_NAME: _sha256_file(catalogue_path),
                EVALUATION_NAME: _sha256_file(evaluation_path),
                FORMULA_NAME: _sha256_file(formula_path),
                SEARCH_NAME: _sha256_file(search_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT203 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT203 source changed before publication")
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
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_motif_conjunction_exception_search(
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
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BROAD_THRESHOLD",
    "CERTIFICATE_CUTOFFS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "SAFE_THRESHOLD",
    "build_candidate_specs",
    "discrete_motif_exception_score",
    "materialize_motif_exception_candidates",
    "run_motif_conjunction_exception_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
