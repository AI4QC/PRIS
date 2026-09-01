"""Exhaustive union follow-up for the NEXT98 cross-source discovery search."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next85_scigen_label_free_features import FEATURE_NAMES as SCIGEN_FEATURE_NAMES
from src.next86_scigen_endpoint_router import ENDPOINT_NAME as SCIGEN_ENDPOINT_NAME
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import ENDPOINT_NAME as WYFORMER_ENDPOINT_NAME
from src.next94_wyformer_label_free_features import FEATURE_NAMES as WYFORMER_FEATURE_NAMES
from src.next95_wyformer_sparse_law_search import DEFAULT_GATES, _endpoint_numeric
from src.next98_cross_source_discovery_search import (
    AUC_GATES,
    BROAD_MIN_PRECISION_LOWER,
    CATALOGUE_NAME as NEXT98_CATALOGUE_NAME,
    PROTOCOL as NEXT98_PROTOCOL,
    _candidate_key,
    _read_json,
    search_cross_source_law,
)


PROTOCOL = "2026-08-04-next98b-cross-source-exhaustive-search-v1"
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT98B_CROSS_SOURCE_EXHAUSTIVE_EVALUATION.json"
SEARCH_NAME = "next98b_cross_source_exhaustive_candidate_search.parquet"
EXPECTED_INPUT_SHA256 = {
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "scigen_search_records": "438c98681ddf7366bccaf88f36221142c1851873d89632c9d04196bffed7dac2",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "wyformer_search_records": "36c18f23e2b7c8d5ad7df16da34205a2dedbd1cf1e5ba544299f501653a87c35",
    "next98_manifest": "5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c",
    "next98_term_catalogue": "f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41",
    "design": "d4e4f27123e96906fb2ac967948ee968ffd19615e285f074c9b667c458e4c994",
}


def _list_value(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError("NEXT98b candidate list differs")


def build_exhaustive_candidate_specs(
    *,
    scigen_records: pd.DataFrame,
    wyformer_records: pd.DataFrame,
    eligible_terms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Union every eligible unique formula/weight tuple from both catalogues."""

    eligible_ids = {str(term["term_id"]) for term in eligible_terms}
    specs: dict[str, dict[str, object]] = {}

    def add(term_ids: list[str], weights: list[float], origin: str) -> None:
        if (
            not 1 <= len(term_ids) <= 3
            or len(weights) != len(term_ids)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in eligible_ids for term_id in term_ids)
            or any(not np.isfinite(weight) or weight <= 0 for weight in weights)
        ):
            return
        key = _candidate_key(term_ids, weights)
        if key not in specs:
            specs[key] = {
                "candidate_key": key,
                "term_ids": term_ids,
                "weights": weights,
                "origins": [origin],
            }
        elif origin not in specs[key]["origins"]:
            specs[key]["origins"].append(origin)

    for _, row in scigen_records.iterrows():
        add(
            [str(value) for value in json.loads(str(row["term_ids_json"]))],
            [float(value) for value in json.loads(str(row["weights_json"]))],
            "next87_complete",
        )
    for _, row in wyformer_records.iterrows():
        add(
            [str(value) for value in _list_value(row["term_ids"])],
            [float(value) for value in _list_value(row["weights"])],
            "next95_complete",
        )
    for term_id in sorted(eligible_ids):
        add([term_id], [1.0], "all_eligible_singles")
    return [specs[key] for key in sorted(specs)]


def run_cross_source_exhaustive_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    scigen_search_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    wyformer_search_dir: Path,
    next98_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the complete prior-catalogue union without validation or replication."""

    scigen_feature_root = Path(scigen_feature_dir).resolve()
    scigen_endpoint_root = Path(scigen_discovery_endpoint_dir).resolve()
    scigen_search_root = Path(scigen_search_dir).resolve()
    wyformer_feature_root = Path(wyformer_feature_dir).resolve()
    wyformer_endpoint_root = Path(wyformer_discovery_endpoint_dir).resolve()
    wyformer_search_root = Path(wyformer_search_dir).resolve()
    next98_root = Path(next98_dir).resolve()
    paths = {
        "scigen_features": scigen_feature_root / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": scigen_endpoint_root / SCIGEN_ENDPOINT_NAME,
        "scigen_search_records": scigen_search_root
        / "next87_complete_candidate_search.parquet",
        "wyformer_features": wyformer_feature_root
        / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": wyformer_endpoint_root / WYFORMER_ENDPOINT_NAME,
        "wyformer_search_records": wyformer_search_root
        / "next95_complete_candidate_search.parquet",
        "next98_manifest": next98_root / MANIFEST_NAME,
        "next98_term_catalogue": next98_root / NEXT98_CATALOGUE_NAME,
        "design": Path(design_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT98b input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT98b formal input identity differs")
    next98_manifest = _read_json(paths["next98_manifest"])
    next98_outputs = next98_manifest.get("outputs_sha256")
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or not isinstance(next98_outputs, Mapping)
        or next98_outputs.get(NEXT98_CATALOGUE_NAME)
        != input_hashes["next98_term_catalogue"]
    ):
        raise ValueError("NEXT98b NEXT98 provenance differs")

    scigen_features = pd.read_parquet(paths["scigen_features"])
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_features = pd.read_parquet(paths["wyformer_features"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    scigen = scigen_features.merge(
        scigen_endpoints.loc[:, ["material_id", "distortion_ratio"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    wyformer = wyformer_features.merge(
        wyformer_endpoints.loc[:, ["material_id", "endpoint_stratum"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(scigen) != len(scigen_features) or len(scigen) != len(scigen_endpoints):
        raise ValueError("NEXT98b SCIGEN row accounting differs")
    if len(wyformer) != len(wyformer_features) or len(wyformer) != len(wyformer_endpoints):
        raise ValueError("NEXT98b WyFormer row accounting differs")
    scigen = scigen.copy()
    wyformer = wyformer.copy()
    scigen["material_id"] = "scigen:" + scigen["material_id"].astype(str)
    wyformer["material_id"] = "wyformer:" + wyformer["material_id"].astype(str)
    scigen["source_dataset"] = "scigen"
    wyformer["source_dataset"] = "wyformer"
    scigen["crystal_system"] = scigen["lattice_class"].astype(str)
    endpoint = np.concatenate(
        [
            pd.to_numeric(scigen["distortion_ratio"], errors="coerce").to_numpy(float),
            _endpoint_numeric(wyformer["endpoint_stratum"]),
        ]
    )
    combined = pd.concat([scigen, wyformer], ignore_index=True, sort=False)
    catalogue = _read_json(paths["next98_term_catalogue"])
    eligible_terms = catalogue.get("eligible_terms")
    if not isinstance(eligible_terms, list):
        raise ValueError("NEXT98b pooled term catalogue differs")
    specs = build_exhaustive_candidate_specs(
        scigen_records=pd.read_parquet(paths["scigen_search_records"]),
        wyformer_records=pd.read_parquet(paths["wyformer_search_records"]),
        eligible_terms=eligible_terms,
    )
    started = time.perf_counter()
    result = search_cross_source_law(
        features=combined,
        endpoint=endpoint,
        eligible_terms=eligible_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    try:
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "exhaustive_prior_catalogue_union_discovery_only",
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "safe_gates": dict(DEFAULT_GATES),
            "source_auc_gates": dict(AUC_GATES),
            "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
            "selected_record": selected["record"],
            "selected_formula": selected["formula"],
            "selected_safe": selected["safe"],
            "selected_broad": selected["broad"],
            "selected_source_diagnostics": selected["source_diagnostics"],
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
        }
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {
                "src/next98b_cross_source_exhaustive_search.py": source_hash
            },
            "outputs_sha256": {
                EVALUATION_NAME: _sha256_file(evaluation_path),
                SEARCH_NAME: _sha256_file(search_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT98b input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT98b source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "build_exhaustive_candidate_specs",
    "run_cross_source_exhaustive_search",
]
