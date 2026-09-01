#!/usr/bin/env python3
"""Search frozen pardon depths for motif-conjunction protected exceptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next204_motif_conjunction_broad_residual as n204
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n203 = n204.n203
PROTOCOL = "2026-08-08-next205-motif-exception-depth-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT205_MOTIF_EXCEPTION_DEPTH_CATALOGUE.json"
EVALUATION_NAME = "NEXT205_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT205_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next205_motif_exception_depth_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "452efd5817c65c9b988d825628aa4c2cc5ad4c6a05f96834920e4b040fb84295"
)
EXPECTED_ELIGIBLE_COUNT = n203.EXPECTED_ELIGIBLE_COUNT
CERTIFICATE_CUTOFFS = n203.CERTIFICATE_CUTOFFS
DEPTH_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_CANDIDATE_COUNT = (
    1 + EXPECTED_ELIGIBLE_COUNT * len(CERTIFICATE_CUTOFFS) * len(DEPTH_LEVELS)
)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "base_score_if_outside_frozen_repair_interval_or_certificate_missing_or_"
    "certificate_below_cutoff_else_base_score*pardon_depth*(broad_threshold/"
    "safe_threshold)"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n204.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next202_design": n204.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next204_manifest": "289b75e2a4a0ee8a2ac28cc2f423253061029d4f5398a8ed021e46fb61883566",
    "next204_diagnostic": "2593ae4975578e23608a215c72dbd22822f16bc6bcd2c476758cd29bb13bd6ca",
    "next204_table": "3298ff863ac7e71767ab8f8c52e601e8cf38e6ea15b769e5922934f83f857ca0",
}


def motif_exception_depth_score(
    *,
    base_score: object,
    base_support: object,
    certificate: object,
    certificate_cutoff: float,
    pardon_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move certified repair-interval rows to a frozen sub-BROAD depth."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(certificate, dtype=float)
    cutoff = float(certificate_cutoff)
    depth = float(pardon_depth)
    if (
        score.ndim != 1
        or support.shape != score.shape
        or values.shape != score.shape
        or cutoff not in {0.0, *CERTIFICATE_CUTOFFS}
        or depth not in DEPTH_LEVELS
        or np.any(~np.isfinite(score[support]))
        or np.any(score[support] < -1.0e-12)
    ):
        raise ValueError("NEXT205 base score, cutoff, or pardon depth differs")
    finite = np.isfinite(values)
    if np.any((values[finite] < -1.0e-12) | (values[finite] > 1.0 + 1.0e-12)):
        raise ValueError("NEXT205 certificate is outside [0,1]")
    active = (
        support
        & finite
        & (score >= n203.BROAD_THRESHOLD)
        & (score < n203.SAFE_THRESHOLD)
        & (values >= cutoff)
        & (cutoff > 0.0)
    )
    corrected = score.copy()
    corrected[active] = (
        score[active] * depth * n203.INTERVAL_FOLD_RATIO
    )
    if np.any(corrected[active] >= n203.BROAD_THRESHOLD):
        raise RuntimeError("NEXT205 pardon depth did not land below BROAD")
    return corrected, support.copy(), active


def build_candidate_specs(
    *, base_candidate_key: str,
    eligible_hypotheses: Sequence[str],
) -> list[dict[str, object]]:
    """Build the unchanged base plus the exact certificate/cutoff/depth grid."""

    if not isinstance(base_candidate_key, str) or not base_candidate_key:
        raise ValueError("NEXT205 base candidate key must be nonempty")
    eligible = tuple(sorted(str(value) for value in eligible_hypotheses))
    if (
        len(eligible) != EXPECTED_ELIGIBLE_COUNT
        or len(set(eligible)) != len(eligible)
        or any(name not in n203.n202.HYPOTHESES for name in eligible)
    ):
        raise ValueError("NEXT205 eligible hypothesis universe differs")
    triples: list[tuple[str | None, float, float | None]] = [(None, 0.0, None)]
    triples.extend(
        (hypothesis, cutoff, depth)
        for hypothesis in eligible
        for cutoff in CERTIFICATE_CUTOFFS
        for depth in DEPTH_LEVELS
    )
    specs = []
    for hypothesis, cutoff, depth in triples:
        definition = (
            None if hypothesis is None else n203.n202.HYPOTHESES[hypothesis]
        )
        multiplier = (
            None if depth is None else float(depth * n203.INTERVAL_FOLD_RATIO)
        )
        payload = {
            "active_score_multiplier": multiplier,
            "base_candidate_key": base_candidate_key,
            "broad_threshold": n203.BROAD_THRESHOLD,
            "certificate_cutoff": cutoff,
            "certificate_hypothesis": hypothesis,
            "conjunction": None if definition is None else definition[2],
            "floor_threshold": None if definition is None else definition[1],
            "missing_policy": "TERM_OFF_KEEP_BASE",
            "pardon_depth": depth,
            "safe_threshold": n203.SAFE_THRESHOLD,
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
        raise RuntimeError("NEXT205 candidate universe differs")
    return specs


def materialize_motif_exception_depth_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    base_support: object,
    certificates: Mapping[str, object],
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every depth candidate as an exactly recoverable virtual term."""

    score = np.asarray(base_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    if score.shape != (len(features),) or support.shape != score.shape:
        raise ValueError("NEXT205 base score shape differs")
    certificate_arrays = {
        str(name): np.asarray(values, dtype=float)
        for name, values in certificates.items()
    }
    if (
        len(certificate_arrays) != EXPECTED_ELIGIBLE_COUNT
        or any(name not in n203.n202.HYPOTHESES for name in certificate_arrays)
        or any(values.shape != score.shape for values in certificate_arrays.values())
    ):
        raise ValueError("NEXT205 certificate table differs")
    expected_specs = build_candidate_specs(
        base_candidate_key=(
            str(specs[0].get("base_candidate_key", "")) if specs else ""
        ),
        eligible_hypotheses=tuple(certificate_arrays),
    )
    if [str(spec.get("candidate_key", "")) for spec in specs] != [
        str(spec["candidate_key"]) for spec in expected_specs
    ]:
        raise ValueError("NEXT205 candidate specs differ")

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
        depth = 0.0 if spec["pardon_depth"] is None else float(spec["pardon_depth"])
        corrected, corrected_support, _ = motif_exception_depth_score(
            base_score=score,
            base_support=support,
            certificate=certificate,
            certificate_cutoff=float(spec["certificate_cutoff"]),
            pardon_depth=depth,
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
        virtual_id = f"next205_virtual_candidate__{digest}"
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
                "group": "next205_motif_exception_depth",
                "encoding": "asinh_sinh_exact_motif_exception_depth_score",
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
    roots: Mapping[str, Path],
    freeze_path: Path,
    next202_design_path: Path,
    design_path: Path,
) -> dict[str, Path]:
    paths = n204._paths(roots, freeze_path, next202_design_path)
    paths["next202_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next204_manifest": roots["next204"] / n204.MANIFEST_NAME,
            "next204_diagnostic": roots["next204"] / n204.DIAGNOSTIC_NAME,
            "next204_table": roots["next204"] / n204.PER_CANDIDATE_NAME,
        }
    )
    return paths


def _verify_prior(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> list[str]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next202_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next202_design"]
    eligible = n204._verify_next203(prior_paths, prior_hashes)
    manifest = json.loads(paths["next204_manifest"].read_text())
    diagnostic = json.loads(paths["next204_diagnostic"].read_text())
    if (
        manifest.get("protocol") != n204.PROTOCOL
        or manifest.get("candidate_count") != n204.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n204.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("next203_records_reproduced") is not True
        or manifest.get("motif_conjunction_broad_residual_diagnosed") is not True
        or manifest.get("motif_conjunction_exception_branch_closed") is not True
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
            n204.DIAGNOSTIC_NAME: input_hashes["next204_diagnostic"],
            n204.PER_CANDIDATE_NAME: input_hashes["next204_table"],
        }
        or manifest.get("executed_source_sha256", {}).get(
            "src/next204_motif_conjunction_broad_residual.py"
        )
        != _sha256_file(Path(n204.__file__).resolve())
        or diagnostic.get("protocol") != n204.PROTOCOL
        or diagnostic.get("candidate_count") != n204.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256")
        != n204.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("new_formula_searched") is not False
        or diagnostic.get("validation_outputs_opened") is not False
    ):
        raise ValueError("NEXT205 NEXT204 provenance differs")
    return eligible


def _reconstruct_discovery(
    *, paths: Mapping[str, Path], eligible_names: Sequence[str]
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    str,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
]:
    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n203.EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT205 base candidate identity differs")

    combined, feature_tables, old_terms, mhcr_terms = (
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
        raise ValueError("NEXT205 motif row accounting differs")

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
        raise ValueError("NEXT205 base reconstruction differs")
    combined, base_terms, base_runtime = (
        n203.n202.n200.n194.n163.materialize_candidates(
            features=combined,
            physical_terms=physical_terms,
            specs=selected_specs,
        )
    )
    if len(base_terms) != 1 or len(base_runtime) != 1:
        raise RuntimeError("NEXT205 base materialization differs")
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
        raise ValueError("NEXT205 endpoint row accounting differs")
    return (
        combined,
        feature_tables,
        base_key,
        base_score,
        base_support,
        certificates,
        endpoint,
    )


def _verify_next203_depth_one(
    records: pd.DataFrame, published203: pd.DataFrame
) -> None:
    metrics = (
        "scigen_pooled_auc", "scigen_macro_auc", "scigen_worst_auc",
        "wyformer_pooled_auc", "wyformer_macro_auc", "wyformer_worst_auc",
        "safe_threshold",
    )
    booleans = (
        "passes_source_auc_gates", "passes_safe_all_cells",
        "passes_broad_all_cells", "passes_all_discovery_gates",
    )
    current = records.loc[
        records["pardon_depth"].isna() | (records["pardon_depth"] == 1.0)
    ].copy()
    current["join_hypothesis"] = current["certificate_hypothesis"].fillna("BASE")
    prior = published203.copy()
    prior["join_hypothesis"] = prior["certificate_hypothesis"].fillna("BASE")
    joined = current.merge(
        prior,
        on=["join_hypothesis", "certificate_cutoff"],
        how="inner",
        suffixes=("_new", "_prior"),
        validate="one_to_one",
    )
    if len(current) != n203.EXPECTED_CANDIDATE_COUNT or len(joined) != len(current):
        raise RuntimeError("NEXT205 depth-one identity differs")
    for name in metrics:
        if not np.allclose(
            joined[f"{name}_new"].to_numpy(float),
            joined[f"{name}_prior"].to_numpy(float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"NEXT205 depth-one metric differs: {name}")
    for name in booleans:
        if not np.array_equal(
            joined[f"{name}_new"].astype(bool).to_numpy(),
            joined[f"{name}_prior"].astype(bool).to_numpy(),
        ):
            raise RuntimeError(f"NEXT205 depth-one gate differs: {name}")


def run_motif_exception_depth_search(
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
    next204_dir: Path,
    next135_freeze_path: Path,
    next202_design_path: Path,
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT205 depth search atomically."""

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
        "next204": Path(next204_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots,
        Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(),
        Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT205 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT205 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT205 formal input identity differs: {differing}")
    eligible_names = _verify_prior(paths, input_hashes)
    (
        combined,
        feature_tables,
        base_key,
        base_score,
        base_support,
        certificates,
        endpoint,
    ) = _reconstruct_discovery(paths=paths, eligible_names=eligible_names)

    specs = build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible_names
    )
    combined, virtual_terms, runtime = materialize_motif_exception_depth_candidates(
        features=combined,
        base_score=base_score,
        base_support=base_support,
        certificates=certificates,
        specs=specs,
    )
    started = time.perf_counter()
    result = n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined,
        endpoint=endpoint,
        old_terms=virtual_terms,
        optional_terms=[],
        candidate_specs=runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT205 evaluator count differs")

    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "active_score_multiplier": spec["active_score_multiplier"],
                "base_candidate_key": base_key,
                "broad_threshold_frozen": n203.BROAD_THRESHOLD,
                "certificate_hypothesis": spec["certificate_hypothesis"],
                "certificate_cutoff": float(spec["certificate_cutoff"]),
                "secondary_feature": spec["secondary_feature"],
                "floor_threshold": spec["floor_threshold"],
                "conjunction": spec["conjunction"],
                "missing_policy": "TERM_OFF_KEEP_BASE",
                "pardon_depth": spec["pardon_depth"],
                "safe_threshold_frozen": n203.SAFE_THRESHOLD,
                "score_composition": SCORE_COMPOSITION,
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "pardon_depth" not in selected["record"]:
        decorate(selected["record"])
    records = pd.DataFrame(result["candidate_records"])
    _verify_next203_depth_one(
        records,
        pd.read_parquet(paths["next203_search"]),
    )

    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    prior = json.loads(paths["next163_evaluation"].read_text())
    formula = {
        "protocol": PROTOCOL,
        "kind": "motif_conjunction_protected_exception_depth_no_dft_score",
        "base_candidate_key": base_key,
        "base_formula": prior["selected_formula"],
        "certificate_hypothesis": selected_spec["certificate_hypothesis"],
        "secondary_feature": selected_spec["secondary_feature"],
        "floor_threshold": selected_spec["floor_threshold"],
        "conjunction": selected_spec["conjunction"],
        "certificate_cutoff": float(selected_spec["certificate_cutoff"]),
        "pardon_depth": selected_spec["pardon_depth"],
        "active_score_multiplier": selected_spec["active_score_multiplier"],
        "broad_threshold": n203.BROAD_THRESHOLD,
        "safe_threshold": n203.SAFE_THRESHOLD,
        "interval_policy": "BROAD_INCLUSIVE_SAFE_EXCLUSIVE_ON_ORIGINAL_BASE_SCORE",
        "missing_policy": "TERM_OFF_KEEP_BASE",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    counts_by_depth = {}
    grouped = records.assign(
        depth_label=records["pardon_depth"].map(
            lambda value: "BASE" if pd.isna(value) else f"{float(value):g}"
        )
    ).groupby("depth_label", sort=True)
    for depth, frame in grouped:
        counts_by_depth[str(depth)] = {
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
        "base_candidate_key_sha256": n203.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "next203_depth_one_reproduced": True,
        "eligible_hypotheses": eligible_names,
        "eligible_hypothesis_count": len(eligible_names),
        "certificate_cutoff_grid": CERTIFICATE_CUTOFFS,
        "pardon_depth_grid": DEPTH_LEVELS,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "broad_threshold": n203.BROAD_THRESHOLD,
        "safe_threshold": n203.SAFE_THRESHOLD,
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
        "src/next203_motif_conjunction_exception_search.py": Path(
            n203.__file__
        ).resolve(),
        "src/next204_motif_conjunction_broad_residual.py": Path(
            n204.__file__
        ).resolve(),
        "src/next205_motif_exception_depth_search.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
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
                "evaluation_mode": "fixed_motif_exception_depth_search",
                "next203_depth_one_reproduced": True,
                "rows": {
                    "scigen": int(len(feature_tables["scigen"])),
                    "wyformer": int(len(feature_tables["wyformer"])),
                    "total": int(len(combined)),
                },
                "candidate_count": int(result["candidate_count"]),
                "elapsed_seconds": elapsed,
                "search_workers": search_workers,
                "counts_by_depth": counts_by_depth,
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
            "next203_depth_one_reproduced": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
            "motif_exception_depth_search_branch_terminated": not passes,
            "next206_residual_diagnostic_authorized": not passes,
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
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT205 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT205 source changed before publication")
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
    for stage in (194, 199, 200, 201, 202, 203, 204):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--next202-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_motif_exception_depth_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        **{
            f"next{stage}_dir": getattr(args, f"next{stage}_dir")
            for stage in (194, 199, 200, 201, 202, 203, 204)
        },
        next135_freeze_path=args.next135_freeze_path,
        next202_design_path=args.next202_design_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEPTH_LEVELS",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "build_candidate_specs",
    "materialize_motif_exception_depth_candidates",
    "motif_exception_depth_score",
    "run_motif_exception_depth_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
