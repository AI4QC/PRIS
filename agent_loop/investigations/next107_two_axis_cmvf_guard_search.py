#!/usr/bin/env python3
"""Cross-source discovery search for up to two same-catalogue CMVF guards."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    MANIFEST_NAME as NEXT98_MANIFEST_NAME,
    PROTOCOL as NEXT98_PROTOCOL,
    _read_json,
)
from src.next98b_cross_source_exhaustive_search import (
    MANIFEST_NAME as NEXT98B_MANIFEST_NAME,
    PROTOCOL as NEXT98B_PROTOCOL,
    SEARCH_NAME as NEXT98B_SEARCH_NAME,
)
from src.next103_dobvr_optional_guard_search import (
    OPTIONAL_WEIGHT_GRID,
    _optional_term_risk,
    calibrate_optional_terms,
    search_optional_guard_laws,
)
from src.next105_cross_source_cmvf_features import (
    FEATURE_COLUMNS as NEXT105_FEATURE_COLUMNS,
    FEATURE_FILES as NEXT105_FEATURE_FILES,
    MANIFEST_NAME as NEXT105_MANIFEST_NAME,
    PROTOCOL as NEXT105_PROTOCOL,
)
from src.next106_cmvf_optional_guard_search import (
    CATALOGUE_NAME as NEXT106_CATALOGUE_NAME,
    EVALUATION_NAME as NEXT106_EVALUATION_NAME,
    EXPECTED_AUC_PASSING_BASES,
    MANIFEST_NAME as NEXT106_MANIFEST_NAME,
    MIN_SOURCE_COVERAGE,
    OPTIONAL_TERM_TEMPLATES,
    PROTOCOL as NEXT106_PROTOCOL,
    SEARCH_NAME as NEXT106_SEARCH_NAME,
)


PROTOCOL = "2026-08-04-next107-two-axis-cmvf-guard-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT107_TWO_AXIS_TERM_CATALOGUE.json"
EVALUATION_NAME = "NEXT107_DISCOVERY_EVALUATION.json"
SEARCH_NAME = "next107_two_axis_candidate_search.parquet"
EXPECTED_CONFIGURATION_COUNT = 180
EXPECTED_CANDIDATE_COUNT = 12_127

EXPECTED_INPUT_SHA256 = {
    "scigen_features": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "scigen_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "wyformer_features": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "wyformer_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "next98_manifest": "5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c",
    "next98_term_catalogue": "f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41",
    "next98b_manifest": "b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d",
    "next98b_search_records": "748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4",
    "next105_manifest": "a2340605d9e8f97165ed8fad10c33f401dc17cdade6c5552e0867923fe5002e3",
    "next105_scigen_features": "d4d7974439ea9a39cf9db0bf458c13253f80e1baf5d9faf31594182473e2a90a",
    "next105_wyformer_features": "299f5ab2060aebaa4c5915aac7543fadc16728ffc055a3bd341373d820aeba99",
    "next106_manifest": "352fd653e9de5425894971a344116ef9ad2e50b71af823a1d855f2b1b8638534",
    "next106_term_catalogue": "d54cb249f56921b56176dc0268a3f1c825f9588653e31b9b76b992fccad19150",
    "next106_evaluation": "c9bc8611f730d43883b3f3c900e4385042a520f8899ceb8a027fe6e5d91fa5ce",
    "next106_search_records": "6c14c99c1db78bfa912e63cd364805d837fd924e1a3226954afad8a57cd57d07",
    "design": "ecb4c937f45d2f27aa606eed4c71c74f8c166645e96293c61b7acbf7c10d6290",
}


def _configuration_id(components: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "group": str(component["group"]),
            "term_id": str(component["term_id"]),
            "weight": float(component["weight"]),
        }
        for component in components
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _composite_term_id(configuration_id: str) -> str:
    digest = hashlib.sha256(configuration_id.encode("utf-8")).hexdigest()[:24]
    return f"next107_composite_guard__{digest}"


def build_two_axis_guard_configurations(
    eligible_terms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Enumerate one guard or two distinct guards within one catalogue mode."""

    by_group: dict[str, list[dict[str, object]]] = {}
    seen: set[str] = set()
    for raw in eligible_terms:
        term = dict(raw)
        term_id = term.get("term_id")
        group = term.get("group")
        if (
            not isinstance(term_id, str)
            or not isinstance(group, str)
            or group not in {"cmvf_core", "cmvf_expanded"}
            or term_id in seen
        ):
            raise ValueError("NEXT107 eligible CMVF term identity differs")
        seen.add(term_id)
        by_group.setdefault(group, []).append(term)

    configurations: dict[str, dict[str, object]] = {}

    def add(terms: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        components = [
            {
                "term_id": str(term["term_id"]),
                "group": str(term["group"]),
                "weight": float(weight),
            }
            for term, weight in zip(terms, weights, strict=True)
        ]
        configuration_id = _configuration_id(components)
        configurations[configuration_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }

    all_terms = sorted(
        (term for terms in by_group.values() for term in terms),
        key=lambda term: str(term["term_id"]),
    )
    for term in all_terms:
        for weight in OPTIONAL_WEIGHT_GRID:
            add([term], [weight])
    for group in sorted(by_group):
        terms = sorted(by_group[group], key=lambda term: str(term["term_id"]))
        for first, second in itertools.combinations(terms, 2):
            for first_weight in OPTIONAL_WEIGHT_GRID:
                for second_weight in OPTIONAL_WEIGHT_GRID:
                    add([first, second], [first_weight, second_weight])
    return [configurations[key] for key in sorted(configurations)]


def materialize_composite_guard_terms(
    *,
    features: pd.DataFrame,
    eligible_terms: Sequence[Mapping[str, object]],
    configurations: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, dict[str, object]]]:
    """Reversibly encode a weighted guard sum for the validated old evaluator."""

    by_id = {str(term["term_id"]): dict(term) for term in eligible_terms}
    if len(by_id) != len(eligible_terms):
        raise ValueError("NEXT107 eligible term IDs are duplicated")
    risks: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for term_id, term in by_id.items():
        risks[term_id], active[term_id] = _optional_term_risk(features, term)

    columns: dict[str, object] = {}
    composite_terms: list[dict[str, object]] = []
    mapping: dict[str, dict[str, object]] = {}
    for raw_configuration in configurations:
        configuration = dict(raw_configuration)
        configuration_id = str(configuration.get("configuration_id"))
        components_raw = configuration.get("components")
        if not isinstance(components_raw, list) or len(components_raw) not in (1, 2):
            raise ValueError("NEXT107 guard component count differs")
        components = [dict(component) for component in components_raw]
        term_ids = [str(component.get("term_id")) for component in components]
        groups = {str(component.get("group")) for component in components}
        weights = [float(component.get("weight")) for component in components]
        if (
            configuration_id != _configuration_id(components)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in by_id for term_id in term_ids)
            or len(groups) != 1
            or any(str(by_id[term_id].get("group")) not in groups for term_id in term_ids)
            or any(weight not in OPTIONAL_WEIGHT_GRID for weight in weights)
        ):
            raise ValueError("NEXT107 guard configuration differs")

        combined = np.zeros(len(features), dtype=float)
        is_active = np.ones(len(features), dtype=bool)
        for term_id, weight in zip(term_ids, weights, strict=True):
            combined += weight * risks[term_id]
            is_active &= active[term_id]
        combined[~is_active] = 0.0
        if not np.isfinite(combined).all() or np.any(combined < 0.0):
            raise ValueError("NEXT107 composite guard risk is not finite")
        maximum = float(np.max(combined)) if len(combined) else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.sinh(combined / divisor)
        if not np.isfinite(encoded).all():
            raise ValueError("NEXT107 reversible guard encoding overflowed")

        term_id = _composite_term_id(configuration_id)
        feature_name = f"_{term_id}_value"
        support_name = f"_{term_id}_active"
        columns[feature_name] = encoded
        columns[support_name] = is_active
        composite_terms.append(
            {
                "term_id": term_id,
                "feature": feature_name,
                "direction": 1,
                "transform": "asinh",
                "group": next(iter(groups)),
                "support_column": support_name,
                "center": 0.0,
                "scale": 1.0 / divisor,
                "configuration_id": configuration_id,
                "components": components,
                "encoding": "asinh_sinh_exact_weighted_risk_sum",
            }
        )
        mapping[term_id] = {
            "configuration_id": configuration_id,
            "components": components,
        }
    extended = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(columns)], axis=1
    )
    return extended, composite_terms, mapping


def build_two_axis_candidate_specs(
    *,
    base_records: pd.DataFrame,
    old_term_ids: set[str],
    configurations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach zero or one frozen composite configuration to each passing base."""

    required = {"passes_source_auc_gates", "term_ids_json", "weights_json"}
    if required - set(base_records.columns):
        raise ValueError("NEXT107 base candidate columns differ")
    specs: dict[str, dict[str, object]] = {}
    configuration_ids = sorted(
        str(configuration["configuration_id"]) for configuration in configurations
    )
    for _, row in base_records.iterrows():
        if not bool(row["passes_source_auc_gates"]):
            continue
        term_ids = [str(value) for value in json.loads(str(row["term_ids_json"]))]
        weights = [float(value) for value in json.loads(str(row["weights_json"]))]
        if (
            not 1 <= len(term_ids) <= 3
            or len(weights) != len(term_ids)
            or len(set(term_ids)) != len(term_ids)
            or any(term_id not in old_term_ids for term_id in term_ids)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise ValueError("NEXT107 base formula differs")

        def add(configuration_id: str | None) -> None:
            payload = {
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_configuration_id": configuration_id,
            }
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            specs[key] = {
                "candidate_key": key,
                "base_term_ids": term_ids,
                "base_weights": weights,
                "optional_term_id": None
                if configuration_id is None
                else _composite_term_id(configuration_id),
                "optional_weight": 0.0 if configuration_id is None else 1.0,
                "optional_configuration_id": configuration_id,
            }

        add(None)
        for configuration_id in configuration_ids:
            add(configuration_id)
    return [specs[key] for key in sorted(specs)]


def _decorate_result(
    *,
    result: dict[str, object],
    mapping: Mapping[str, Mapping[str, object]],
    eligible_terms: Sequence[Mapping[str, object]],
) -> None:
    by_id = {str(term["term_id"]): dict(term) for term in eligible_terms}
    for record in result["candidate_records"]:
        synthetic_id = record.get("optional_term_id")
        configuration = None if synthetic_id is None else mapping[str(synthetic_id)]
        components = [] if configuration is None else list(configuration["components"])
        record["optional_configuration_id"] = (
            None if configuration is None else configuration["configuration_id"]
        )
        record["optional_term_ids_json"] = json.dumps(
            [component["term_id"] for component in components], separators=(",", ":")
        )
        record["optional_weights_json"] = json.dumps(
            [float(component["weight"]) for component in components],
            separators=(",", ":"),
        )
        record["optional_physical_term_count"] = len(components)
        record["physical_term_count"] = len(
            json.loads(str(record["base_term_ids_json"]))
        ) + len(components)

    selected = result["selected"]
    synthetic_id = selected["record"].get("optional_term_id")
    configuration = None if synthetic_id is None else mapping[str(synthetic_id)]
    components = [] if configuration is None else list(configuration["components"])
    optional_formula_terms = [
        {
            **by_id[str(component["term_id"])],
            "weight": float(component["weight"]),
        }
        for component in components
    ]
    selected["formula"].pop("optional_term", None)
    selected["formula"]["kind"] = (
        "base_nonnegative_hinge_sum_plus_up_to_two_same_catalogue_cmvf_guards"
    )
    selected["formula"]["optional_terms"] = optional_formula_terms
    selected["formula"]["optional_configuration_id"] = (
        None if configuration is None else configuration["configuration_id"]
    )


def run_two_axis_cmvf_guard_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next98b_dir: Path,
    next105_dir: Path,
    next106_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only same-catalogue two-axis CMVF search."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        "next98": Path(next98_dir).resolve(),
        "next98b": Path(next98b_dir).resolve(),
        "next105": Path(next105_dir).resolve(),
        "next106": Path(next106_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = {
        "scigen_features": roots["scigen_features"] / SCIGEN_FEATURE_NAMES["discovery"],
        "scigen_endpoint": roots["scigen_endpoint"] / SCIGEN_ENDPOINT_NAME,
        "wyformer_features": roots["wyformer_features"] / WYFORMER_FEATURE_NAMES["discovery"],
        "wyformer_endpoint": roots["wyformer_endpoint"] / WYFORMER_ENDPOINT_NAME,
        "next98_manifest": roots["next98"] / NEXT98_MANIFEST_NAME,
        "next98_term_catalogue": roots["next98"] / NEXT98_CATALOGUE_NAME,
        "next98b_manifest": roots["next98b"] / NEXT98B_MANIFEST_NAME,
        "next98b_search_records": roots["next98b"] / NEXT98B_SEARCH_NAME,
        "next105_manifest": roots["next105"] / NEXT105_MANIFEST_NAME,
        "next105_scigen_features": roots["next105"] / NEXT105_FEATURE_FILES["scigen"],
        "next105_wyformer_features": roots["next105"] / NEXT105_FEATURE_FILES["wyformer"],
        "next106_manifest": roots["next106"] / NEXT106_MANIFEST_NAME,
        "next106_term_catalogue": roots["next106"] / NEXT106_CATALOGUE_NAME,
        "next106_evaluation": roots["next106"] / NEXT106_EVALUATION_NAME,
        "next106_search_records": roots["next106"] / NEXT106_SEARCH_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT107 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT107 formal input identity differs")

    next98_manifest = _read_json(paths["next98_manifest"])
    next98b_manifest = _read_json(paths["next98b_manifest"])
    next105_manifest = _read_json(paths["next105_manifest"])
    next106_manifest = _read_json(paths["next106_manifest"])
    next106_catalogue = _read_json(paths["next106_term_catalogue"])
    if (
        next98_manifest.get("protocol") != NEXT98_PROTOCOL
        or next98_manifest.get("opened_validation_outputs_used") is not False
        or next98_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next98b_manifest.get("protocol") != NEXT98B_PROTOCOL
        or next98b_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next98b_manifest.get("opened_validation_outputs_used") is not False
        or next98b_manifest.get("scigen_replication_endpoint_opened") is not False
        or next98b_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next105_manifest.get("protocol") != NEXT105_PROTOCOL
        or next105_manifest.get("labels_opened") is not False
        or next105_manifest.get("endpoint_payloads_opened") is not False
        or next105_manifest.get("validation_geometry_opened") is not False
        or next105_manifest.get("replication_geometry_opened") is not False
        or next105_manifest.get("dft_values_used_by_features") is not False
        or next105_manifest.get("solver_thread_environment")
        != {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        or next106_manifest.get("protocol") != NEXT106_PROTOCOL
        or next106_manifest.get("candidate_count") != 2077
        or next106_manifest.get("passes_all_cross_source_discovery_gates") is not False
        or next106_manifest.get("freeze_authorized") is not False
        or next106_manifest.get("opened_validation_outputs_used") is not False
        or next106_manifest.get("scigen_replication_endpoint_opened") is not False
        or next106_manifest.get("wyformer_replication_endpoint_opened") is not False
        or next106_catalogue.get("protocol") != NEXT106_PROTOCOL
        or not isinstance(next106_catalogue.get("eligible_optional_terms"), list)
    ):
        raise ValueError("NEXT107 prior provenance differs")
    for manifest, prefix, names in (
        (
            next105_manifest,
            "next105",
            {
                NEXT105_FEATURE_FILES["scigen"]: "next105_scigen_features",
                NEXT105_FEATURE_FILES["wyformer"]: "next105_wyformer_features",
            },
        ),
        (
            next106_manifest,
            "next106",
            {
                NEXT106_CATALOGUE_NAME: "next106_term_catalogue",
                NEXT106_EVALUATION_NAME: "next106_evaluation",
                NEXT106_SEARCH_NAME: "next106_search_records",
            },
        ),
    ):
        outputs = manifest.get("outputs_sha256")
        if not isinstance(outputs, Mapping) or any(
            outputs.get(filename) != input_hashes[key]
            for filename, key in names.items()
        ):
            raise ValueError(f"NEXT107 {prefix} output provenance differs")

    old_tables = {
        "scigen": pd.read_parquet(paths["scigen_features"]),
        "wyformer": pd.read_parquet(paths["wyformer_features"]),
    }
    new_tables = {
        "scigen": pd.read_parquet(paths["next105_scigen_features"]),
        "wyformer": pd.read_parquet(paths["next105_wyformer_features"]),
    }
    feature_tables: dict[str, pd.DataFrame] = {}
    for source in ("scigen", "wyformer"):
        old, new = old_tables[source], new_tables[source]
        if (
            old["material_id"].astype(str).duplicated().any()
            or new["material_id"].astype(str).duplicated().any()
            or set(NEXT105_FEATURE_COLUMNS) - set(new.columns)
        ):
            raise ValueError(f"NEXT107 {source} feature identity differs")
        merged = old.merge(
            new.loc[:, ["material_id", *NEXT105_FEATURE_COLUMNS]],
            on="material_id",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(old) or len(merged) != len(new):
            raise ValueError(f"NEXT107 {source} feature row accounting differs")
        merged = merged.copy()
        merged["source_dataset"] = source
        if source == "scigen":
            merged["crystal_system"] = merged["lattice_class"].astype(str)
        merged["material_id"] = source + ":" + merged["material_id"].astype(str)
        feature_tables[source] = merged
    feature_combined = pd.concat(
        [feature_tables["scigen"], feature_tables["wyformer"]],
        ignore_index=True,
        sort=False,
    )
    eligible_terms, excluded_terms = calibrate_optional_terms(
        feature_combined,
        templates=OPTIONAL_TERM_TEMPLATES,
        min_source_coverage=MIN_SOURCE_COVERAGE,
        min_unique_values=8,
    )
    if eligible_terms != next106_catalogue["eligible_optional_terms"]:
        raise ValueError("NEXT107 label-free CMVF calibration differs from NEXT106")
    configurations = build_two_axis_guard_configurations(eligible_terms)
    extended_features, composite_terms, composite_mapping = (
        materialize_composite_guard_terms(
            features=feature_combined,
            eligible_terms=eligible_terms,
            configurations=configurations,
        )
    )

    # Endpoint payloads are opened only after all label-free terms/configurations exist.
    scigen_endpoints = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoints = pd.read_parquet(paths["wyformer_endpoint"])
    if (
        scigen_endpoints["material_id"].astype(str).duplicated().any()
        or wyformer_endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT107 discovery endpoint identities are duplicated")
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": pd.to_numeric(
                        scigen_endpoints["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoints["material_id"].astype(str),
                    "_endpoint_numeric": _endpoint_numeric(
                        wyformer_endpoints["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended_features.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(extended_features) or len(combined) != len(endpoint_frame):
        raise ValueError("NEXT107 endpoint row accounting differs")
    endpoint = pd.to_numeric(combined.pop("_endpoint_numeric"), errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError("NEXT107 endpoint conversion differs")

    old_catalogue = _read_json(paths["next98_term_catalogue"])
    old_terms = old_catalogue.get("eligible_terms")
    if not isinstance(old_terms, list):
        raise ValueError("NEXT107 old term catalogue differs")
    old_term_ids = {str(term["term_id"]) for term in old_terms}
    base_records = pd.read_parquet(paths["next98b_search_records"])
    passing_bases = int(base_records["passes_source_auc_gates"].eq(True).sum())
    specs = build_two_axis_candidate_specs(
        base_records=base_records,
        old_term_ids=old_term_ids,
        configurations=configurations,
    )
    if require_formal_inputs and (
        passing_bases != EXPECTED_AUC_PASSING_BASES
        or len(configurations) != EXPECTED_CONFIGURATION_COUNT
        or len(specs) != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT107 frozen candidate count differs")

    started = time.perf_counter()
    result = search_optional_guard_laws(
        features=combined,
        endpoint=endpoint,
        old_terms=old_terms,
        optional_terms=composite_terms,
        candidate_specs=specs,
    )
    elapsed = time.perf_counter() - started
    _decorate_result(
        result=result,
        mapping=composite_mapping,
        eligible_terms=eligible_terms,
    )
    selected = result["selected"]
    passes = bool(selected["record"]["passes_all_discovery_gates"])

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next103_dobvr_optional_guard_search.py": repository_root / "src/next103_dobvr_optional_guard_search.py",
        "src/next106_cmvf_optional_guard_search.py": repository_root / "src/next106_cmvf_optional_guard_search.py",
        "src/next107_two_axis_cmvf_guard_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    output_paths: list[Path] = []
    try:
        catalogue = {
            "protocol": PROTOCOL,
            "calibration_stage": "label_free_before_endpoint_join",
            "min_source_coverage": MIN_SOURCE_COVERAGE,
            "optional_weight_grid": list(OPTIONAL_WEIGHT_GRID),
            "expected_auc_passing_bases": EXPECTED_AUC_PASSING_BASES,
            "observed_auc_passing_bases": passing_bases,
            "templates": list(OPTIONAL_TERM_TEMPLATES),
            "eligible_optional_terms": eligible_terms,
            "excluded_optional_terms": excluded_terms,
            "guard_configurations": configurations,
            "configuration_count": len(configurations),
            "candidate_count": len(specs),
            "candidate_grammar": "each AUC-passing NEXT98b base plus zero, one, or two same-catalogue CMVF guards",
            "cross_catalogue_pairs_allowed": False,
            "optional_missing_policy": "OPTIONAL_GUARD_OFF_KEEP_BASE",
            "composite_encoding": "reversible asinh(sinh(weighted-risk-sum/divisor))",
        }
        evaluation = {
            "protocol": PROTOCOL,
            "evaluation_mode": "cross_source_discovery_only_same_catalogue_two_axis_cmvf_guard",
            "rows": {
                "scigen": int(len(feature_tables["scigen"])),
                "wyformer": int(len(feature_tables["wyformer"])),
                "total": int(len(combined)),
            },
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "elapsed_seconds": elapsed,
            "safe_gates": dict(DEFAULT_GATES),
            "source_auc_gates": dict(AUC_GATES),
            "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
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
        }
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        pd.DataFrame(result["candidate_records"]).to_parquet(search_path, index=False)
        output_paths.extend([catalogue_path, evaluation_path, search_path])
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "configuration_count": len(configurations),
            "candidate_count": int(result["candidate_count"]),
            "eligible_optional_term_count": len(eligible_terms),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
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
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT107 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT107 source changed before publication")
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
    parser.add_argument("--next98-dir", type=Path, required=True)
    parser.add_argument("--next98b-dir", type=Path, required=True)
    parser.add_argument("--next105-dir", type=Path, required=True)
    parser.add_argument("--next106-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_two_axis_cmvf_guard_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        next98_dir=args.next98_dir,
        next98b_dir=args.next98b_dir,
        next105_dir=args.next105_dir,
        next106_dir=args.next106_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PROTOCOL",
    "build_two_axis_candidate_specs",
    "build_two_axis_guard_configurations",
    "materialize_composite_guard_terms",
    "run_two_axis_cmvf_guard_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
