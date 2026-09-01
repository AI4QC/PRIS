#!/usr/bin/env python3
"""Frozen discovery-only search for a discrete protected exception."""

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

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next163_interior_family_attenuation_search as n163
import src.next194_signed_local_closure_audit as n194
import src.next195_signed_local_closure_search as n195
import src.next196_signed_local_closure_broad_residual as n196
import src.next87_scigen_sparse_law_search as n87
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next197-discrete-protected-exception-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT197_DISCRETE_PROTECTED_EXCEPTION_CATALOGUE.json"
EVALUATION_NAME = "NEXT197_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT197_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next197_discrete_protected_exception_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "e1d7c6eaf2b55f48b526ee7f7349690333cfca174e3f5db132026d8fca3f1508"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n195.EXPECTED_BASE_CANDIDATE_KEY_SHA256
BROAD_THRESHOLD = n195.BROAD_THRESHOLD
SAFE_THRESHOLD = n195.SAFE_THRESHOLD
INTERVAL_FOLD_RATIO = BROAD_THRESHOLD / SAFE_THRESHOLD
CERTIFICATE_CUTOFFS = (
    1 / 16,
    1 / 8,
    3 / 16,
    1 / 4,
    3 / 8,
    1 / 2,
    5 / 8,
    3 / 4,
    7 / 8,
)
EXPECTED_ELIGIBLE_HYPOTHESES = tuple(
    sorted(
        (
            "psndc_crystalnn_closure_min__signed_local_safe__product__high",
            "psndc_crystalnn_closure_min__signed_local_safe__minimum__high",
            "psndc_crystalnn_closure_q10__signed_local_safe__product__high",
            "psndc_crystalnn_closure_q10__signed_local_safe__minimum__high",
            "psndc_crystalnn_volume_q10__signed_local_safe__product__high",
            "psndc_crystalnn_volume_q10__signed_local_safe__minimum__high",
        )
    )
)
EXPECTED_ELIGIBLE_COUNT = len(EXPECTED_ELIGIBLE_HYPOTHESES)
EXPECTED_CANDIDATE_COUNT = 1 + EXPECTED_ELIGIBLE_COUNT * len(CERTIFICATE_CUTOFFS)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "base_score_if_outside_frozen_repair_interval_or_certificate_missing_or_"
    "certificate_below_cutoff_else_base_score*(broad_threshold/safe_threshold)"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n196.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "design": EXPECTED_DESIGN_SHA256,
    "next196_manifest": "f29db616ff6801459d2e7991ed186588eec5187ac1c9594ebb3ce55f28ad2e51",
    "next196_diagnostic": "467279e885c5d7bc8d4a2fc1338b3124daad007a94082f32656cba4d6de9d0b1",
    "next196_table": "256d3951171adfc772673cfebaed81e35fea5ebb998901d323cb0da02ecba012",
}


def discrete_protected_exception_score(
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
        raise ValueError("NEXT197 base score or certificate cutoff differs")
    finite = np.isfinite(values)
    if np.any((values[finite] < -1.0e-12) | (values[finite] > 1.0 + 1.0e-12)):
        raise ValueError("NEXT197 certificate is outside [0,1]")
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
        raise RuntimeError("NEXT197 interval fold did not land below BROAD")
    return corrected, support.copy(), active


def build_candidate_specs(
    *, base_candidate_key: str, eligible_hypotheses: Sequence[str]
) -> list[dict[str, object]]:
    """Build one base plus the exact certificate-cutoff product."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT197 base candidate key must be nonempty")
    eligible = tuple(sorted(str(value) for value in eligible_hypotheses))
    if (
        not eligible
        or len(set(eligible)) != len(eligible)
        or any(name not in n194.HYPOTHESES for name in eligible)
    ):
        raise ValueError("NEXT197 eligible hypothesis universe differs")
    pairs: list[tuple[str | None, float]] = [(None, 0.0)]
    pairs.extend(
        (hypothesis, cutoff)
        for hypothesis in eligible
        for cutoff in CERTIFICATE_CUTOFFS
    )
    specs: list[dict[str, object]] = []
    for hypothesis, cutoff in pairs:
        definition = None if hypothesis is None else n194.HYPOTHESES[hypothesis]
        payload = {
            "base_candidate_key": base_candidate_key,
            "broad_threshold": BROAD_THRESHOLD,
            "certificate_cutoff": cutoff,
            "certificate_hypothesis": hypothesis,
            "closure_feature": None if definition is None else definition[0],
            "conjunction": None if definition is None else definition[1],
            "interval_fold_ratio": INTERVAL_FOLD_RATIO,
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "safe_threshold": SAFE_THRESHOLD,
            "score_composition": SCORE_COMPOSITION,
            "signed_local_normalization": "clip(safe_local_geometry/0.5,0,1)",
        }
        specs.append(
            {
                **payload,
                "candidate_key": json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    expected = 1 + len(eligible) * len(CERTIFICATE_CUTOFFS)
    if (
        len(specs) != expected
        or len({str(spec["candidate_key"]) for spec in specs}) != expected
    ):
        raise RuntimeError("NEXT197 candidate universe differs")
    return specs


def materialize_discrete_exception_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    certificates: Mapping[str, object],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every folded score as an exactly recoverable virtual term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    if score.shape != (len(features),) or support.shape != score.shape:
        raise ValueError("NEXT197 base score shape differs")
    certificate_arrays = {
        str(name): np.asarray(values, dtype=float)
        for name, values in certificates.items()
    }
    if (
        not certificate_arrays
        or any(name not in n194.HYPOTHESES for name in certificate_arrays)
        or any(values.shape != score.shape for values in certificate_arrays.values())
    ):
        raise ValueError("NEXT197 certificate table differs")
    expected_specs = build_candidate_specs(
        base_candidate_key=(str(specs[0].get("base_candidate_key", "")) if specs else ""),
        eligible_hypotheses=tuple(certificate_arrays),
    )
    if [str(spec.get("candidate_key", "")) for spec in specs] != [
        str(spec["candidate_key"]) for spec in expected_specs
    ]:
        raise ValueError("NEXT197 candidate specs differ")

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
        corrected, corrected_support, _ = discrete_protected_exception_score(
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
        encoded[corrected_support] = np.sinh(corrected[corrected_support] / divisor)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        virtual_id = f"next197_virtual_candidate__{digest}"
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
                "group": "next197_discrete_protected_exception",
                "encoding": "asinh_sinh_exact_discrete_interval_fold_score",
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
    paths = n196._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next196_manifest": roots["next196"] / n196.MANIFEST_NAME,
            "next196_diagnostic": roots["next196"] / n196.DIAGNOSTIC_NAME,
            "next196_table": roots["next196"] / n196.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_discrete_protected_exception_search(
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
    next196_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT197 search and publish atomically."""

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
        (196, next196_dir),
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
        raise FileNotFoundError("NEXT197 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT197 formal input identity differs: {differing}")

    manifest196 = json.loads(paths["next196_manifest"].read_text())
    diagnostic196 = json.loads(paths["next196_diagnostic"].read_text())
    expected_outputs196 = {
        n196.DIAGNOSTIC_NAME: input_hashes["next196_diagnostic"],
        n196.PER_CANDIDATE_NAME: input_hashes["next196_table"],
    }
    if (
        manifest196.get("protocol") != n196.PROTOCOL
        or manifest196.get("candidate_count") != n196.EXPECTED_CANDIDATE_COUNT
        or manifest196.get("next195_records_reproduced") is not True
        or manifest196.get("signed_local_closure_broad_residual_diagnosed")
        is not True
        or manifest196.get("new_formula_searched") is not False
        or manifest196.get("opened_validation_outputs_used") is not False
        or manifest196.get("scigen_replication_endpoint_opened") is not False
        or manifest196.get("wyformer_replication_endpoint_opened") is not False
        or manifest196.get("dft_calculation_executed") is not False
        or manifest196.get("dft_values_used_by_executable_formula") is not False
        or manifest196.get("learned_energy_force_stress_proxy_used") is not False
        or manifest196.get("physical_relaxation_executed") is not False
        or manifest196.get("outputs_sha256") != expected_outputs196
        or manifest196.get("executed_source_sha256", {}).get(
            "src/next196_signed_local_closure_broad_residual.py"
        )
        != _sha256_file(Path(n196.__file__).resolve())
        or diagnostic196.get("protocol") != n196.PROTOCOL
        or diagnostic196.get("candidate_count") != n196.EXPECTED_CANDIDATE_COUNT
        or diagnostic196.get("new_formula_searched") is not False
        or diagnostic196.get("validation_outputs_opened") is not False
    ):
        raise ValueError("NEXT197 NEXT196 provenance differs")

    audit194 = json.loads(paths["next194_audit"].read_text())
    table194 = pd.read_parquet(paths["next194_table"])
    eligible_names = tuple(sorted(str(value) for value in audit194["eligible_hypotheses"]))
    table_eligible = tuple(
        sorted(
            table194.loc[
                table194["eligible_for_search"].fillna(False).astype(bool),
                "hypothesis",
            ].astype(str)
        )
    )
    if (
        eligible_names != EXPECTED_ELIGIBLE_HYPOTHESES
        or table_eligible != EXPECTED_ELIGIBLE_HYPOTHESES
    ):
        raise ValueError("NEXT197 eligible certificate universe differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT197 base candidate identity differs")

    combined, feature_tables, old_terms, mhcr_terms = n130._join_label_free_features(paths)
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
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT197 base reconstruction differs")
    combined, base_terms, base_runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT197 base materialization differs")
    base_score, base_support = n87._term_risk(combined, base_terms[0])
    family_means = n195.n192.complementary_safe_family_means(
        features=combined,
        physical_terms=physical_terms,
        base_spec=selected_specs[0],
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
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
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
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(endpoint) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT197 endpoint row accounting differs")

    specs = build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible_names
    )
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT197 formal candidate count differs")
    candidate_key_sha256 = hashlib.sha256(
        "\n".join(sorted(str(spec["candidate_key"]) for spec in specs)).encode()
    ).hexdigest()
    combined, virtual_terms, runtime = materialize_discrete_exception_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        certificates=certificates,
        specs=specs,
    )
    started = time.perf_counter()
    result = n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT197 evaluator count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "base_candidate_key": base_key,
                "certificate_hypothesis": spec["certificate_hypothesis"],
                "closure_feature": spec["closure_feature"],
                "conjunction": spec["conjunction"],
                "certificate_cutoff": float(spec["certificate_cutoff"]),
                "interval_fold_ratio": INTERVAL_FOLD_RATIO,
                "broad_threshold": BROAD_THRESHOLD,
                "safe_threshold_frozen": SAFE_THRESHOLD,
                "missing_policy": "TERM_OFF_KEEP_BASE",
                "score_composition": SCORE_COMPOSITION,
                "signed_local_normalization": spec["signed_local_normalization"],
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
        raise RuntimeError("NEXT197 base candidate differs")
    n195.n182.n181.n175.n170._verify_base_reproduction(
        record=base_records[0],
        published=pd.read_parquet(paths["next163_search"]),
        candidate_key=base_key,
    )
    records = pd.DataFrame(result["candidate_records"])
    passes_any = bool(records["passes_all_discovery_gates"].any())
    if passes_any != bool(selected["record"]["passes_all_discovery_gates"]):
        raise RuntimeError("NEXT197 evaluator selection differs from passing population")
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    prior = json.loads(paths["next163_evaluation"].read_text())
    formula = {
        "protocol": PROTOCOL,
        "kind": "discrete_signed_local_strong_closure_protected_exception",
        "base_candidate_key": base_key,
        "base_formula": prior["selected_formula"],
        "certificate_hypothesis": selected_spec["certificate_hypothesis"],
        "closure_feature": selected_spec["closure_feature"],
        "conjunction": selected_spec["conjunction"],
        "signed_local_definition": "mean_selected_local_terms(min(0.5,weight*max(0,-signed_frozen_z)))",
        "signed_local_normalization": selected_spec["signed_local_normalization"],
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
    grouped_counts = {}
    grouped = records.assign(
        certificate_hypothesis=records["certificate_hypothesis"].fillna("BASE")
    ).groupby(["certificate_hypothesis", "certificate_cutoff"], sort=True)
    for (hypothesis, cutoff), frame in grouped:
        grouped_counts[f"hypothesis={hypothesis},cutoff={float(cutoff):g}"] = {
            "candidates": int(len(frame)),
            "passes_source_auc_gates": int(frame["passes_source_auc_gates"].sum()),
            "passes_safe_all_cells": int(frame["passes_safe_all_cells"].sum()),
            "passes_broad_all_cells": int(frame["passes_broad_all_cells"].sum()),
            "passes_all_discovery_gates": int(
                frame["passes_all_discovery_gates"].sum()
            ),
        }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "candidate_key_sha256": candidate_key_sha256,
        "base_endpoint_reproduced": True,
        "eligible_hypotheses": list(eligible_names),
        "eligible_hypothesis_count": len(eligible_names),
        "certificate_cutoffs": CERTIFICATE_CUTOFFS,
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
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next163_interior_family_attenuation_search.py": Path(n163.__file__).resolve(),
        "src/next194_signed_local_closure_audit.py": Path(n194.__file__).resolve(),
        "src/next195_signed_local_closure_search.py": Path(n195.__file__).resolve(),
        "src/next196_signed_local_closure_broad_residual.py": Path(n196.__file__).resolve(),
        "src/next197_discrete_protected_exception_search.py": Path(__file__).resolve(),
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
                "evaluation_mode": "fixed_discrete_protected_exception_search",
                "base_endpoint_reproduced": True,
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "candidate_count": int(result["candidate_count"]),
                "candidate_key_sha256": candidate_key_sha256,
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_hypothesis_and_cutoff": grouped_counts,
                "selected_record": selected["record"],
                "selected_formula": formula,
                "selected_safe": selected["safe"],
                "selected_safe_diagnostic": selected["safe_diagnostic"],
                "selected_broad": selected["broad"],
                "selected_source_diagnostics": selected["source_diagnostics"],
                "pauling_by_cell": result["pauling_by_cell"],
                "cells": result["cells"],
                "passes_all_cross_source_discovery_gates": passes_any,
                "freeze_authorized": passes_any,
                "requires_unopened_internal_validation_before_claim": True,
            },
        )
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        outputs = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "candidate_key_sha256": candidate_key_sha256,
            "eligible_hypothesis_count": len(eligible_names),
            "search_workers": search_workers,
            "base_endpoint_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes_any,
            "freeze_authorized": passes_any,
            "requires_unopened_internal_validation_before_claim": True,
            "discrete_protected_exception_search_branch_terminated": not passes_any,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "formula_or_threshold_changed_after_search": False,
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
            raise RuntimeError("NEXT197 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT197 source changed before publication")
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
        196,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_discrete_protected_exception_search(
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
    "BROAD_THRESHOLD",
    "CERTIFICATE_CUTOFFS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "SAFE_THRESHOLD",
    "build_candidate_specs",
    "discrete_protected_exception_score",
    "materialize_discrete_exception_candidates",
    "run_discrete_protected_exception_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
