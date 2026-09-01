#!/usr/bin/env python3
"""Frozen +0.1 BVTC/PRLR rescue over the NEXT121 SAFE12 frontier."""

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

import src.next121_bvtbd_frontier_rescue as n121
from src.next103_dobvr_optional_guard_search import search_optional_guard_laws


PROTOCOL = "2026-08-08-next122-safe12-bvtc-prlr-rescue-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT122_SAFE12_BVTC_PRLR_CATALOGUE.json"
EVALUATION_NAME = "NEXT122_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next122_safe12_bvtc_prlr_candidate_search.parquet"
EXPECTED_FREEZE_SHA256 = (
    "65573952e7d4e6c70e63c7d9e39cf0ebae34d43ffd07cccfbf86fc88bb75522e"
)
RESCUE_TERM_IDS = (
    "bvtc_correction_rms__high",
    "prlr_bar_stress_amplification__high",
)
RESCUE_WEIGHT = 0.1
EXPECTED_FRONTIER_BASES = 3_573
EXPECTED_CANDIDATE_COUNT = 14_292
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = (
    "5de720ee249b70606d6e13257048713b860b5cb924a937e953176eebc67e56f8"
)
EXPECTED_BASE_FORMULA_SHA256 = (
    "048052033d91a733ac6afc90858e977070ec7ed78ff40b3f65d93489ddb5c240"
)
EXPECTED_CANDIDATE_KEY_SHA256 = (
    "56d9f4cfcd02806c858e7f26b1ec919290c7d63e48112c0e2a41e7a8aa763dbd"
)
BASE_REPRODUCTION_AUC_TOLERANCE = 2.0e-5

EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n121.EXPECTED_INPUT_SHA256.items() if key != "freeze"},
    "next121_manifest": "029e14f43ac08a6365d33a7830898303ebd394166e7a4cfa9bb1e7cc98308c1b",
    "next121_catalogue": "f7b3a9f843199e127b177a929bd33234d5569eca0b9ffddccbd42fff5dc8679d",
    "next121_search_records": "6cdbecf01ab8d7ca45395a6eef3c2dc6181e72cff0c5234b254c4213e26009db",
    "freeze": EXPECTED_FREEZE_SHA256,
}


def select_safe12_bases(records: pd.DataFrame) -> pd.DataFrame:
    """Flatten every complete NEXT121 formula that passes all SAFE cells."""

    required = {
        "candidate_key",
        "base_term_ids_json",
        "base_weights_json",
        "optional_term_ids_json",
        "optional_weights_json",
        "passes_safe_all_cells",
        "safe_passing_cells",
    }
    if required - set(records.columns):
        raise ValueError("NEXT122 SAFE12 base schema differs")
    keep = records["passes_safe_all_cells"].fillna(False).astype(bool)
    if not pd.to_numeric(records.loc[keep, "safe_passing_cells"], errors="coerce").eq(12).all():
        raise ValueError("NEXT122 SAFE12 base count differs")
    selected: list[dict[str, object]] = []
    for _, row in records.loc[keep].iterrows():
        base_ids = [str(value) for value in json.loads(str(row["base_term_ids_json"]))]
        base_weights = [float(value) for value in json.loads(str(row["base_weights_json"]))]
        optional_ids = [str(value) for value in json.loads(str(row["optional_term_ids_json"]))]
        optional_weights = [float(value) for value in json.loads(str(row["optional_weights_json"]))]
        term_ids = [*base_ids, *optional_ids]
        weights = [*base_weights, *optional_weights]
        if (
            len(term_ids) < 4
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT122 flattened SAFE12 formula differs")
        selected.append(
            {
                "prior_candidate_key": str(row["candidate_key"]),
                "term_ids_json": json.dumps(term_ids, separators=(",", ":")),
                "weights_json": json.dumps(weights, separators=(",", ":")),
            }
        )
    return pd.DataFrame(
        selected,
        columns=("prior_candidate_key", "term_ids_json", "weights_json"),
    ).reset_index(drop=True)


def build_rescue_candidate_specs(
    *, base_records: pd.DataFrame, old_term_ids: set[str]
) -> list[dict[str, object]]:
    """Append neither, either, or both frozen +0.1 analytic rescue terms."""

    required = {"prior_candidate_key", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT122 rescue base columns differ")
    operations = (
        (),
        (RESCUE_TERM_IDS[0],),
        (RESCUE_TERM_IDS[1],),
        RESCUE_TERM_IDS,
    )
    specs: dict[str, dict[str, object]] = {}
    for _, row in base_records.iterrows():
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if any(term_id in term_ids for term_id in RESCUE_TERM_IDS):
            raise ValueError("NEXT122 base already contains rescue term")
        if (
            len(term_ids) < 4
            or len(term_ids) != len(weights)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
            or any(term_id not in old_term_ids for term_id in RESCUE_TERM_IDS)
        ):
            raise ValueError("NEXT122 rescue base formula differs")
        for added in operations:
            candidate_ids = [*term_ids, *added]
            candidate_weights = [*weights, *([RESCUE_WEIGHT] * len(added))]
            payload = {
                "base_term_ids": candidate_ids,
                "base_weights": candidate_weights,
                "optional_configuration_id": None,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {
                "candidate_key": key,
                "base_term_ids": candidate_ids,
                "base_weights": candidate_weights,
                "optional_term_id": None,
                "optional_weight": 0.0,
                "rescue_term_ids": list(added),
            }
    return [specs[key] for key in sorted(specs)]


def _verify_base_reproduction(
    *, result_records: Sequence[Mapping[str, object]], prior: pd.DataFrame
) -> None:
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    observed = {
        n121.prior._formula_identity(
            json.loads(str(record["base_term_ids_json"])),
            json.loads(str(record["base_weights_json"])),
        ): record
        for record in result_records
    }
    if len(observed) != len(result_records):
        raise RuntimeError("NEXT122 evaluated formula identities are duplicated")
    for _, row in prior.iterrows():
        key = n121.prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        record = observed.get(key)
        source = row["_prior_record"]
        if record is None or any(
            not math.isclose(
                float(record[name]),
                float(source[name]),
                rel_tol=0.0,
                abs_tol=BASE_REPRODUCTION_AUC_TOLERANCE,
            )
            for name in metrics
        ) or any(
            bool(record[name]) != bool(source[name])
            for name in ("passes_source_auc_gates", "passes_safe_all_cells")
        ) or int(record["safe_passing_cells"]) != int(source["safe_passing_cells"]):
            raise RuntimeError("NEXT122 base-only diagnostics do not reproduce NEXT121")


def _paths(roots: Mapping[str, Path], freeze_path: Path) -> dict[str, Path]:
    paths = n121._paths(roots, freeze_path)
    paths.update(
        {
            "next121_manifest": roots["next121"] / n121.MANIFEST_NAME,
            "next121_catalogue": roots["next121"] / n121.CATALOGUE_NAME,
            "next121_search_records": roots["next121"] / n121.SEARCH_NAME,
        }
    )
    return paths


def _reconstruct_label_free_table(
    paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, object]]]:
    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    cmvo_tables = {
        "scigen": pd.read_parquet(paths["next110_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next110_wyformer_features"]),
    }
    morphology_tables = {
        "scigen": pd.read_parquet(paths["next113_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next113_wyformer_features"]),
    }
    hcid_tables = {
        "scigen": pd.read_parquet(paths["next116_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next116_wyformer_features"]),
    }
    bvtbd_tables = {
        "scigen": pd.read_parquet(paths["next120_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next120_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        tables = (
            old_tables[source], cmvo_tables[source], morphology_tables[source],
            hcid_tables[source], bvtbd_tables[source],
        )
        if any(table["material_id"].astype(str).duplicated().any() for table in tables):
            raise ValueError(f"NEXT122 {source} feature identities are duplicated")
        merged = old_tables[source].merge(
            cmvo_tables[source].loc[:, ["material_id", *n121.prior.NEXT110_FEATURE_COLUMNS]],
            on="material_id", how="inner", validate="one_to_one",
        ).merge(
            morphology_tables[source].loc[:, ["material_id", *n121.prior.NEXT113_FEATURE_COLUMNS]],
            on="material_id", how="inner", validate="one_to_one",
        ).merge(
            hcid_tables[source].loc[:, ["material_id", *n121.prior.NEXT116_FEATURE_COLUMNS]],
            on="material_id", how="inner", validate="one_to_one",
        ).merge(
            bvtbd_tables[source].loc[:, ["material_id", *n121.NEXT120_FEATURE_COLUMNS]],
            on="material_id", how="inner", validate="one_to_one",
        )
        if any(len(merged) != len(table) for table in tables):
            raise ValueError(f"NEXT122 {source} feature row accounting differs")
        merged = merged.copy()
        merged["source_dataset"] = source
        if source == "scigen":
            merged["crystal_system"] = merged["lattice_class"].astype(str)
        merged["material_id"] = source + ":" + merged["material_id"].astype(str)
        feature_tables[source] = merged
    features = pd.concat(
        [feature_tables["scigen"], feature_tables["wyformer"]],
        ignore_index=True, sort=False,
    )
    cmvo_features, cmvo_terms = n121.prior.materialize_cmvo_tail_terms(features)
    morphology_features, morphology_terms = n121.prior.materialize_cmvom_tail_terms(cmvo_features)
    hcid_features, hcid_terms = n121.prior.materialize_hcid_tail_terms(morphology_features)
    complete, bvtbd_terms = n121.materialize_bvtbd_tail_terms(hcid_features)

    old_terms_raw = n121.prior._read_json(paths["next98_term_catalogue"]).get("eligible_terms")
    if not isinstance(old_terms_raw, list):
        raise ValueError("NEXT122 original term catalogue differs")
    if (
        n121.prior._read_json(paths["next111_catalogue"]).get("eligible_optional_terms") != cmvo_terms
        or n121.prior._read_json(paths["next114_catalogue"]).get("eligible_optional_terms") != morphology_terms
        or n121.prior._read_json(paths["next117_catalogue"]).get("eligible_optional_terms") != hcid_terms
        or n121.prior._read_json(paths["next121_catalogue"]).get("eligible_optional_terms") != bvtbd_terms
    ):
        raise ValueError("NEXT122 prior optional term catalogue differs")
    terms = [*old_terms_raw, *cmvo_terms, *morphology_terms, *hcid_terms, *bvtbd_terms]
    if len({str(term["term_id"]) for term in terms}) != len(terms):
        raise ValueError("NEXT122 term identities are duplicated")
    return complete, feature_tables, terms


def run_safe12_bvtc_prlr_rescue(
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
    freeze_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the fixed SAFE12 rescue after hashing all formula identities."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in (
            (98, next98_dir), (110, next110_dir), (111, next111_dir),
            (113, next113_dir), (114, next114_dir), (116, next116_dir),
            (117, next117_dir), (120, next120_dir), (121, next121_dir),
        )},
    }
    target = Path(output_dir).resolve()
    paths = _paths(roots, freeze_path)
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT122 discovery input is missing")
    input_hashes = {name: n121.prior._sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT122 formal input identity differs")
    manifest121 = n121.prior._read_json(paths["next121_manifest"])
    if (
        manifest121.get("protocol") != n121.PROTOCOL
        or manifest121.get("passes_all_cross_source_discovery_gates") is not False
        or manifest121.get("opened_validation_outputs_used") is not False
        or manifest121.get("scigen_replication_endpoint_opened") is not False
        or manifest121.get("wyformer_replication_endpoint_opened") is not False
        or manifest121.get("dft_values_used_by_executable_formula") is not False
    ):
        raise ValueError("NEXT122 NEXT121 provenance differs")
    outputs121 = manifest121.get("outputs_sha256")
    if not isinstance(outputs121, Mapping) or any(
        outputs121.get(filename) != input_hashes[key]
        for filename, key in {
            n121.CATALOGUE_NAME: "next121_catalogue",
            n121.SEARCH_NAME: "next121_search_records",
        }.items()
    ):
        raise ValueError("NEXT122 NEXT121 output provenance differs")

    features, feature_tables, old_terms = _reconstruct_label_free_table(paths)
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    prior_records = pd.read_parquet(paths["next121_search_records"])
    bases = select_safe12_bases(prior_records)
    prior_by_key = {
        str(row["candidate_key"]): row.to_dict() for _, row in prior_records.iterrows()
    }
    bases["_prior_record"] = [prior_by_key[str(key)] for key in bases["prior_candidate_key"]]
    specs = build_rescue_candidate_specs(base_records=bases, old_term_ids=old_term_ids)
    base_keys = sorted(bases["prior_candidate_key"].astype(str))
    base_formulas = sorted(
        n121.prior._formula_identity(
            json.loads(str(row["term_ids_json"])),
            json.loads(str(row["weights_json"])),
        )
        for _, row in bases.iterrows()
    )
    base_key_sha = hashlib.sha256("\n".join(base_keys).encode()).hexdigest()
    base_formula_sha = hashlib.sha256("\n".join(base_formulas).encode()).hexdigest()
    candidate_key_sha = hashlib.sha256(
        "\n".join(spec["candidate_key"] for spec in specs).encode()
    ).hexdigest()
    label_free_catalogue = {
        "protocol": PROTOCOL,
        "calibration_stage": "no_new_features_fixed_safe12_rescue_before_endpoint_reread",
        "freeze_sha256": input_hashes["freeze"],
        "rescue_term_ids": list(RESCUE_TERM_IDS),
        "rescue_weight": RESCUE_WEIGHT,
        "operations": [[], [RESCUE_TERM_IDS[0]], [RESCUE_TERM_IDS[1]], list(RESCUE_TERM_IDS)],
        "frontier_base_count": len(bases),
        "frontier_base_key_sha256": base_key_sha,
        "frontier_base_formula_sha256": base_formula_sha,
        "candidate_count": len(specs),
        "candidate_key_sha256": candidate_key_sha,
        "new_feature_calibration_performed": False,
        "new_features_joined_to_endpoint_before_freeze": False,
        "discovery_endpoints_previously_opened_by_next121": True,
    }
    label_free_catalogue_sha256 = hashlib.sha256(
        n121.prior._json_bytes(label_free_catalogue)
    ).hexdigest()
    if require_formal_inputs and (
        len(bases) != EXPECTED_FRONTIER_BASES
        or len(specs) != EXPECTED_CANDIDATE_COUNT
        or base_key_sha != EXPECTED_BASE_CANDIDATE_KEY_SHA256
        or base_formula_sha != EXPECTED_BASE_FORMULA_SHA256
        or candidate_key_sha != EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT122 frozen candidate universe differs")

    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame({
                "material_id": "scigen:" + scigen_endpoints["material_id"].astype(str),
                "_endpoint_numeric": pd.to_numeric(scigen_endpoints["distortion_ratio"], errors="coerce"),
            }),
            pd.DataFrame({
                "material_id": "wyformer:" + wyformer_endpoints["material_id"].astype(str),
                "_endpoint_numeric": n121.prior._endpoint_numeric(wyformer_endpoints["endpoint_stratum"]),
            }),
        ], ignore_index=True,
    )
    combined = features.merge(endpoint_frame, on="material_id", how="inner", validate="one_to_one")
    if len(combined) != len(features) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT122 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT122 endpoint conversion differs")

    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=[],
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    _verify_base_reproduction(result_records=result["candidate_records"], prior=bases)
    result["selected"]["formula"]["kind"] = "next121_safe12_base_plus_fixed_bvtc_prlr_weight_increment"
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next117_hcid_frontier_rescue.py": repository_root / "src/next117_hcid_frontier_rescue.py",
        "src/next121_bvtbd_frontier_rescue.py": repository_root / "src/next121_bvtbd_frontier_rescue.py",
        "src/next122_safe12_bvtc_prlr_rescue.py": Path(__file__).resolve(),
    }
    source_hashes = {name: n121.prior._sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        n121.prior._write_json(catalogue_path, {**label_free_catalogue, "label_free_catalogue_sha256": label_free_catalogue_sha256})
        n121.prior._write_json(evaluation_path, {
            "protocol": PROTOCOL,
            "evaluation_mode": "fixed_safe12_bvtc_prlr_increment_rescue",
            "rows": {source: int(len(feature_tables[source])) for source in ("scigen", "wyformer")},
            "frontier_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "base_only_reproduced_next121": True,
            "safe_gates": dict(n121.prior.DEFAULT_GATES),
            "source_auc_gates": dict(n121.prior.AUC_GATES),
            "selected_record": selected["record"],
            "selected_formula": selected["formula"],
            "selected_safe": selected["safe"],
            "selected_safe_diagnostic": selected["safe_diagnostic"],
            "selected_broad": selected["broad"],
            "selected_source_diagnostics": selected["source_diagnostics"],
            "pauling_by_cell": result["pauling_by_cell"],
            "cells": result["cells"],
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
        })
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest = {
            "protocol": PROTOCOL,
            "label_free_catalogue_sha256": label_free_catalogue_sha256,
            "frontier_base_count": len(bases),
            "candidate_count": int(result["candidate_count"]),
            "base_only_reproduced_next121": True,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "requires_unopened_internal_validation_before_claim": True,
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
            "outputs_sha256": {path.name: n121.prior._sha256_file(path) for path in output_paths},
        }
        n121.prior._write_json(staging / MANIFEST_NAME, manifest)
        if any(n121.prior._sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT122 input changed before publication")
        if any(n121.prior._sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT122 source changed before publication")
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
    for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121):
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_safe12_bvtc_prlr_rescue(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in (98, 110, 111, 113, 114, 116, 117, 120, 121)},
        freeze_path=args.freeze_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CANDIDATE_COUNT", "EXPECTED_FREEZE_SHA256", "EXPECTED_INPUT_SHA256",
    "PROTOCOL", "RESCUE_TERM_IDS", "RESCUE_WEIGHT", "build_rescue_candidate_specs",
    "run_safe12_bvtc_prlr_rescue", "select_safe12_bases",
]
