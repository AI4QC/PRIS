"""NEXT90 fixed mechanism envelopes for SCIGEN pre-DFT screening."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME as TERM_CATALOGUE_NAME,
    MANIFEST_NAME as TERM_MANIFEST_NAME,
    PROTOCOL as TERM_PROTOCOL,
)

from src.next87_scigen_sparse_law_search import (
    DEFAULT_GATES,
    FOLD_MIN_PRECISION,
    FOLD_MIN_PROTECTED_RECALL,
    GROUP_FOLDS,
    MISSING_POLICY,
    WEIGHT_GRID,
    _operating_pass,
    _pauling_baseline,
    _publish_directory_no_replace,
    _pooled_auc,
    _read_json,
    _search_rank,
    _sha256_file,
    _term_risk,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
    select_threshold,
)


PROTOCOL = "2026-08-03-next90-scigen-coupled-mechanism-envelope-v1"
FORMULA_KIND = "coupled_mechanism_envelope"
ENVELOPE_IDS = ("B", "V", "E", "L")
FIXED_ENVELOPE_TERMS = {
    "B": (
        "scbv_anion_mismatch_rms__high",
        "scbv_mismatch_q95__high",
        "scbv_mismatch_max__high",
    ),
    "V": (
        "sivr_edge_mismatch_rms__high",
        "sivr_edge_mismatch_max__high",
        "sivr_stiffness_min__low",
    ),
    "E": (
        "aefi_residual_rms__high",
        "aefi_residual_q95__high",
        "aefi_residual_max__high",
    ),
    "L": (
        "sscp_load_rms__high",
        "sscp_load_q95__high",
        "sscp_load_fraction__low",
        "prlr_residual_fraction__high",
        "prlr_cell_residual_fraction__high",
        "prlr_risk__high",
    ),
}
MANIFEST_NAME = "MANIFEST.json"
FORMULA_NAME = "NEXT90_FROZEN_FORMULA.json"
EVALUATION_NAME = "NEXT90_DISCOVERY_EVALUATION.json"
FOLD_DIAGNOSTICS_NAME = "NEXT90_FOLD_DIAGNOSTICS.json"
SEARCH_RECORD_NAME = "next90_complete_weight_search.parquet"
PREDICTION_NAMES = {
    role: f"next90_frozen_predictions_{role}.parquet" for role in FEATURE_NAMES
}
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "feature_catalogue": "f34b09a4a9f18b0202b8daf606b7baab7bdae826871bcc60a4be858a8c1cc96a",
    "features_discovery": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "features_internal_validation": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "features_internal_replication": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "term_manifest": "5b80f948a35a40ef79438ea1902b92a40dd07c35a4b541826252eb92cf96f1eb",
    "term_catalogue": "e8f9fe532c15673c0a74737632b0145d43f6494cb1ea7e94e7380198fd4e4dee",
    "discovery_endpoint_manifest": "35792117310f04daa8c383bddb5d4012084d47c7d904706d86cbe33e0a55a6ea",
    "discovery_endpoint": "f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958",
    "design": "a635a80f3f216bf0b7c5cd926960bacbbe3a55f1b89c0e971a5ade6e6f6f4eef",
}


def apply_mechanism_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply four max envelopes; one missing constituent fails open."""

    if formula.get("kind") != FORMULA_KIND:
        raise ValueError("NEXT90 formula kind differs")
    if formula.get("missing_policy") != MISSING_POLICY:
        raise ValueError("NEXT90 formula missing policy differs")
    envelopes = formula.get("envelopes")
    if (
        not isinstance(envelopes, list)
        or tuple(
            envelope.get("envelope_id") if isinstance(envelope, Mapping) else None
            for envelope in envelopes
        )
        != ENVELOPE_IDS
    ):
        raise ValueError("NEXT90 formulas require envelopes B, V, E, L")

    supported = np.ones(len(features), dtype=bool)
    score = np.zeros(len(features), dtype=float)
    for envelope_index, envelope in enumerate(envelopes):
        if envelope.get("aggregation") != "max":
            raise ValueError("NEXT90 envelope aggregation differs")
        weight = envelope.get("weight")
        if (
            not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or (envelope_index == 0 and float(weight) != 1.0)
            or (envelope_index > 0 and float(weight) not in WEIGHT_GRID)
        ):
            raise ValueError("NEXT90 envelope weight differs")
        terms = envelope.get("terms")
        if not isinstance(terms, list) or not terms:
            raise ValueError("NEXT90 envelope term list differs")
        risks: list[np.ndarray] = []
        for term in terms:
            if not isinstance(term, Mapping):
                raise ValueError("NEXT90 envelope term differs")
            risk, term_supported = _term_risk(features, term)
            risks.append(risk)
            supported &= term_supported
        envelope_risk = np.max(np.column_stack(risks), axis=1)
        score += float(weight) * envelope_risk

    threshold = formula.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("NEXT90 formula threshold differs")
    score[~supported] = np.nan
    reject = supported & (score >= float(threshold))
    return score, supported, reject


def _fixed_envelope_arrays(
    features: pd.DataFrame, eligible_terms: list[Mapping[str, object]]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, list[dict[str, object]]]]:
    by_id: dict[str, dict[str, object]] = {}
    for raw_term in eligible_terms:
        term = dict(raw_term)
        term_id = str(term.get("term_id"))
        if not term_id or term_id in by_id:
            raise ValueError("NEXT90 term catalogue identities differ")
        by_id[term_id] = term
    required_ids = {
        term_id for term_ids in FIXED_ENVELOPE_TERMS.values() for term_id in term_ids
    }
    if required_ids - set(by_id):
        raise ValueError("NEXT90 fixed envelope term is missing")

    arrays: dict[str, np.ndarray] = {}
    all_supported = np.ones(len(features), dtype=bool)
    envelope_specs: dict[str, list[dict[str, object]]] = {}
    for envelope_id in ENVELOPE_IDS:
        risks: list[np.ndarray] = []
        specs: list[dict[str, object]] = []
        for term_id in FIXED_ENVELOPE_TERMS[envelope_id]:
            term = by_id[term_id]
            risk, supported = _term_risk(features, term)
            risks.append(risk)
            all_supported &= supported
            specs.append(
                {
                    key: term[key]
                    for key in (
                        "term_id",
                        "feature",
                        "direction",
                        "transform",
                        "center",
                        "scale",
                    )
                }
            )
        arrays[envelope_id] = np.max(np.column_stack(risks), axis=1)
        envelope_specs[envelope_id] = specs
    return arrays, all_supported, envelope_specs


def _formula(
    envelope_specs: Mapping[str, list[dict[str, object]]],
    weights: tuple[float, float, float, float],
    threshold: float,
) -> dict[str, object]:
    return {
        "kind": FORMULA_KIND,
        "missing_policy": MISSING_POLICY,
        "envelopes": [
            {
                "envelope_id": envelope_id,
                "aggregation": "max",
                "weight": float(weight),
                "terms": envelope_specs[envelope_id],
            }
            for envelope_id, weight in zip(ENVELOPE_IDS, weights, strict=True)
        ],
        "threshold": float(threshold),
    }


def search_mechanism_envelope_law(
    *,
    features: pd.DataFrame,
    distortion_ratio: object,
    eligible_terms: list[Mapping[str, object]],
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> dict[str, object]:
    """Search 125 fixed four-envelope weight vectors on discovery only."""

    endpoint = np.asarray(distortion_ratio, dtype=float)
    required = {"material_id", "reduced_formula", "lattice_class", "pauling_p2_p5_decision"}
    if (
        len(features) != len(endpoint)
        or endpoint.ndim != 1
        or not np.isfinite(endpoint).all()
        or required - set(features.columns)
        or features["material_id"].astype(str).duplicated().any()
        or not (endpoint <= 1.0).any()
        or not (endpoint >= 2.0).any()
    ):
        raise ValueError("NEXT90 discovery arrays differ")
    envelopes, supported, envelope_specs = _fixed_envelope_arrays(
        features, eligible_terms
    )
    folds = assign_group_folds(features["reduced_formula"].astype(str).to_numpy())
    if set(np.unique(folds)) != set(range(GROUP_FOLDS)):
        raise ValueError("NEXT90 discovery groups do not populate all folds")
    pauling = _pauling_baseline(features, endpoint)

    candidates: list[dict[str, object]] = []
    fold_winners: list[dict[str, object] | None] = [None] * GROUP_FOLDS
    for variable_weights in product(WEIGHT_GRID, repeat=3):
        weights = (1.0, *(float(value) for value in variable_weights))
        score = sum(
            float(weight) * envelopes[envelope_id]
            for envelope_id, weight in zip(ENVELOPE_IDS, weights, strict=True)
        )
        score = np.asarray(score, dtype=float)
        score[~supported] = np.nan
        selected = select_threshold(
            score=score,
            supported=supported,
            distortion_ratio=endpoint,
            gates=gates,
        )
        pooled_auc = _pooled_auc(score, supported, endpoint)
        if selected is None:
            threshold = None
            metrics = decision_metrics(
                supported=supported,
                reject=np.zeros(len(features), dtype=bool),
                distortion_ratio=endpoint,
            )
        else:
            threshold = float(selected["threshold"])
            metrics = dict(selected["metrics"])
        rank = _search_rank(metrics, pooled_auc, len(ENVELOPE_IDS), gates)
        weight_key = str(tuple(weights))
        fold_train_summaries: list[dict[str, object]] = []
        for held_out in range(GROUP_FOLDS):
            train = folds != held_out
            fold_selected = select_threshold(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                gates=gates,
                row_mask=train,
            )
            fold_auc = _pooled_auc(score, supported, endpoint, train)
            if fold_selected is None:
                fold_threshold = None
                fold_metrics = decision_metrics(
                    supported=supported[train],
                    reject=np.zeros(int(train.sum()), dtype=bool),
                    distortion_ratio=endpoint[train],
                )
            else:
                fold_threshold = float(fold_selected["threshold"])
                fold_metrics = dict(fold_selected["metrics"])
            fold_rank = _search_rank(fold_metrics, fold_auc, len(ENVELOPE_IDS), gates)
            fold_train_summaries.append(
                {
                    "held_out_fold": held_out,
                    "threshold": fold_threshold,
                    "rank": list(fold_rank),
                }
            )
            winner = fold_winners[held_out]
            if (
                winner is None
                or fold_rank > winner["rank_tuple"]
                or (fold_rank == winner["rank_tuple"] and weight_key < winner["weight_key"])
            ):
                fold_winners[held_out] = {
                    "held_out_fold": held_out,
                    "weights": list(weights),
                    "weight_key": weight_key,
                    "threshold": fold_threshold,
                    "rank": list(fold_rank),
                    "rank_tuple": fold_rank,
                }
        candidates.append(
            {
                "weights": weights,
                "weight_key": weight_key,
                "threshold": threshold,
                "metrics": metrics,
                "pooled_extreme_auc": pooled_auc,
                "rank": list(rank),
                "rank_tuple": rank,
                "fold_train_summaries": fold_train_summaries,
                "score": score,
            }
        )

    if any(winner is None for winner in fold_winners):
        raise RuntimeError("NEXT90 fold search produced no winner")
    win_counts: dict[str, int] = {}
    for winner in fold_winners:
        key = str(winner["weight_key"])
        win_counts[key] = win_counts.get(key, 0) + 1
    stable_weights = {key for key, count in win_counts.items() if count >= 4}
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["rank_tuple"],
            tuple(-ord(char) for char in candidate["weight_key"]),
        ),
        reverse=True,
    )
    finalist_pool = [
        candidate for candidate in ranked if candidate["weight_key"] in stable_weights
    ]
    if not finalist_pool:
        finalist_pool = ranked[:1]
    finalists: list[dict[str, object]] = []
    for candidate in finalist_pool:
        score = candidate["score"]
        threshold = candidate["threshold"]
        reject = (
            supported & (score >= float(threshold))
            if threshold is not None
            else np.zeros(len(features), dtype=bool)
        )
        diagnostics = auc_diagnostics(
            score=score,
            supported=supported,
            distortion_ratio=endpoint,
            lattice_class=features["lattice_class"].astype(str).to_numpy(),
        )
        fold_diagnostics: list[dict[str, object]] = []
        all_support = True
        all_raw = True
        for held_out in range(GROUP_FOLDS):
            train = folds != held_out
            test = folds == held_out
            fold_selected = select_threshold(
                score=score,
                supported=supported,
                distortion_ratio=endpoint,
                gates=gates,
                row_mask=train,
            )
            from src.next23_relaxation_rule import wilson_lower_bound

            support_lower = wilson_lower_bound(int(supported[test].sum()), int(test.sum()))
            support_pass = support_lower >= float(gates["coverage_lower"])
            all_support &= support_pass
            if fold_selected is None:
                fold_threshold = None
                fold_metrics = decision_metrics(
                    supported=supported[test],
                    reject=np.zeros(int(test.sum()), dtype=bool),
                    distortion_ratio=endpoint[test],
                )
            else:
                fold_threshold = float(fold_selected["threshold"])
                fold_metrics = decision_metrics(
                    supported=supported[test],
                    reject=supported[test] & (score[test] >= fold_threshold),
                    distortion_ratio=endpoint[test],
                )
            raw_pass = (
                fold_selected is not None
                and float(fold_metrics["severe_rejection_precision"]) >= FOLD_MIN_PRECISION
                and float(fold_metrics["protected_recall"]) >= FOLD_MIN_PROTECTED_RECALL
            )
            all_raw &= raw_pass
            fold_diagnostics.append(
                {
                    "held_out_fold": held_out,
                    "train_threshold": fold_threshold,
                    "support_coverage_lower": support_lower,
                    "passes_support_gate": support_pass,
                    "metrics": fold_metrics,
                    "passes_raw_fold_gates": raw_pass,
                }
            )
        metrics = candidate["metrics"]
        auc_pass = (
            diagnostics["pooled_extreme_auc"] is not None
            and float(diagnostics["pooled_extreme_auc"]) >= float(gates["pooled_extreme_auc"])
            and diagnostics["macro_lattice_auc"] is not None
            and float(diagnostics["macro_lattice_auc"]) >= float(gates["macro_lattice_auc"])
            and diagnostics["worst_lattice_auc"] is not None
            and float(diagnostics["worst_lattice_auc"]) >= float(gates["worst_lattice_auc"])
            and int(diagnostics["evaluable_lattices"]) >= int(gates["evaluable_lattices"])
        )
        beats_pauling = (
            int(metrics["severe_rejected"]) > int(pauling["severe_rejected"])
            and float(metrics["severe_rejection_precision_lower"])
            > float(pauling["severe_rejection_precision_lower"])
        )
        stable_count = win_counts.get(str(candidate["weight_key"]), 0)
        passes = bool(
            threshold is not None
            and _operating_pass(metrics, gates)
            and auc_pass
            and stable_count >= 4
            and all_support
            and all_raw
            and beats_pauling
        )
        finalists.append(
            {
                "candidate": candidate,
                "reject": reject,
                "diagnostics": diagnostics,
                "fold_diagnostics": fold_diagnostics,
                "stable_count": stable_count,
                "beats_pauling": beats_pauling,
                "passes": passes,
            }
        )
    selected_finalist = max(
        finalists,
        key=lambda finalist: (
            1 if finalist["passes"] else 0,
            finalist["candidate"]["rank_tuple"],
            tuple(-ord(char) for char in finalist["candidate"]["weight_key"]),
        ),
    )
    selected = selected_finalist["candidate"]
    threshold = selected["threshold"]
    formula = (
        _formula(envelope_specs, selected["weights"], float(threshold))
        if threshold is not None
        else None
    )
    diagnostics = selected_finalist["diagnostics"]
    public_candidates = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"rank_tuple", "score"}
        }
        for candidate in candidates
    ]
    public_winners = [
        {key: value for key, value in winner.items() if key != "rank_tuple"}
        for winner in fold_winners
    ]
    return {
        "selected_formula": formula,
        "discovery_metrics": {
            **selected["metrics"],
            **{
                key: diagnostics[key]
                for key in (
                    "pooled_extreme_auc",
                    "macro_lattice_auc",
                    "worst_lattice_auc",
                    "evaluable_lattices",
                )
            },
            "beats_pauling_severe_count_and_precision_lower": selected_finalist[
                "beats_pauling"
            ],
        },
        "lattice_diagnostics": diagnostics["lattices"],
        "fold_diagnostics": selected_finalist["fold_diagnostics"],
        "weight_stability": {
            "fold_winners": public_winners,
            "weight_win_counts": win_counts,
            "stable_weights": sorted(stable_weights),
            "selected_weight_win_count": selected_finalist["stable_count"],
        },
        "pauling_baseline": pauling,
        "passes_discovery_gates": bool(selected_finalist["passes"]),
        "candidate_count": len(candidates),
        "search_records": public_candidates,
        "score": selected["score"],
        "supported": supported,
        "reject": selected_finalist["reject"],
        "fold": folds,
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _search_table(records: list[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        metrics = record["metrics"]
        weights = tuple(record["weights"])
        rows.append(
            {
                "weight_B": weights[0],
                "weight_V": weights[1],
                "weight_E": weights[2],
                "weight_L": weights[3],
                "weight_key": record["weight_key"],
                "threshold": record["threshold"],
                "pooled_extreme_auc": record["pooled_extreme_auc"],
                "rank_json": json.dumps(record["rank"], separators=(",", ":")),
                "fold_train_summaries_json": json.dumps(
                    record["fold_train_summaries"], sort_keys=True, separators=(",", ":")
                ),
                **{f"metric_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def run_scigen_mechanism_search(
    *,
    feature_dir: Path,
    term_catalogue_dir: Path,
    discovery_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run NEXT90 with no interface capable of opening locked endpoints."""

    feature_root = Path(feature_dir).resolve()
    term_root = Path(term_catalogue_dir).resolve()
    endpoint_root = Path(discovery_endpoint_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "feature_manifest": feature_root / FEATURE_MANIFEST_NAME,
        "feature_catalogue": feature_root / FEATURE_CATALOGUE_NAME,
        **{
            f"features_{role}": feature_root / FEATURE_NAMES[role]
            for role in FEATURE_NAMES
        },
        "term_manifest": term_root / TERM_MANIFEST_NAME,
        "term_catalogue": term_root / TERM_CATALOGUE_NAME,
        "discovery_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "discovery_endpoint": endpoint_root / ENDPOINT_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT90 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT90 formal input identity differs")

    feature_manifest = _read_json(paths["feature_manifest"], role="NEXT85 manifest")
    feature_outputs = feature_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_payloads_opened") is not False
        or feature_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(FEATURE_CATALOGUE_NAME) != hashes["feature_catalogue"]
        or any(
            feature_outputs.get(FEATURE_NAMES[role]) != hashes[f"features_{role}"]
            for role in FEATURE_NAMES
        )
    ):
        raise ValueError("NEXT90 received invalid NEXT85 provenance")
    term_manifest = _read_json(paths["term_manifest"], role="NEXT86 term manifest")
    term_catalogue = _read_json(paths["term_catalogue"], role="NEXT86 term catalogue")
    term_outputs = term_manifest.get("outputs_sha256")
    if (
        term_manifest.get("protocol") != TERM_PROTOCOL
        or term_manifest.get("labels_opened") is not False
        or term_manifest.get("endpoint_payloads_opened") is not False
        or not isinstance(term_outputs, Mapping)
        or term_outputs.get(TERM_CATALOGUE_NAME) != hashes["term_catalogue"]
        or term_catalogue.get("protocol") != TERM_PROTOCOL
        or term_catalogue.get("labels_opened") is not False
        or not isinstance(term_catalogue.get("eligible_terms"), list)
    ):
        raise ValueError("NEXT90 received invalid NEXT86 term provenance")
    endpoint_manifest = _read_json(
        paths["discovery_endpoint_manifest"], role="NEXT86 discovery endpoint manifest"
    )
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "discovery"
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(ENDPOINT_NAME) != hashes["discovery_endpoint"]
    ):
        raise ValueError("NEXT90 received invalid discovery endpoint provenance")

    feature_tables: dict[str, pd.DataFrame] = {}
    for role in FEATURE_NAMES:
        table = pd.read_parquet(paths[f"features_{role}"])
        if (
            "material_id" not in table
            or "partition_role" not in table
            or table["material_id"].astype(str).duplicated().any()
            or set(table["partition_role"].astype(str)) != {role}
        ):
            raise ValueError(f"NEXT90 {role} feature identity differs")
        feature_tables[role] = table
    endpoints = pd.read_parquet(paths["discovery_endpoint"])
    if (
        {"material_id", "partition_role", "distortion_ratio"} - set(endpoints.columns)
        or endpoints["material_id"].astype(str).duplicated().any()
        or set(endpoints["partition_role"].astype(str)) != {"discovery"}
    ):
        raise ValueError("NEXT90 discovery endpoint table differs")
    discovery = feature_tables["discovery"].merge(
        endpoints.loc[:, ["material_id", "distortion_ratio"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(discovery) != len(feature_tables["discovery"]) or len(discovery) != len(endpoints):
        raise ValueError("NEXT90 discovery identity join differs")
    result = search_mechanism_envelope_law(
        features=discovery,
        distortion_ratio=discovery["distortion_ratio"].to_numpy(float),
        eligible_terms=term_catalogue["eligible_terms"],
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    try:
        search_path = staging / SEARCH_RECORD_NAME
        _search_table(result["search_records"]).to_parquet(search_path, index=False)
        output_paths.append(search_path)
        evaluation = {
            "protocol": PROTOCOL,
            "status": (
                "discovery_gates_passed_predictions_frozen"
                if result["passes_discovery_gates"]
                else "discovery_gates_failed_stop_without_lockbox_opening"
            ),
            "passes_discovery_gates": result["passes_discovery_gates"],
            "candidate_count": result["candidate_count"],
            "selected_formula": result["selected_formula"],
            "discovery_metrics": result["discovery_metrics"],
            "pauling_baseline": result["pauling_baseline"],
            "lattice_diagnostics": result["lattice_diagnostics"],
            "weight_stability": result["weight_stability"],
        }
        evaluation_path = staging / EVALUATION_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        output_paths.append(evaluation_path)
        folds_path = staging / FOLD_DIAGNOSTICS_NAME
        folds_path.write_bytes(_json_bytes(result["fold_diagnostics"]))
        output_paths.append(folds_path)
        if result["passes_discovery_gates"]:
            formula = {
                **result["selected_formula"],
                "protocol": PROTOCOL,
                "training_partition": "SCIGEN discovery only",
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
            }
            formula_path = staging / FORMULA_NAME
            formula_path.write_bytes(_json_bytes(formula))
            formula_sha256 = _sha256_file(formula_path)
            output_paths.append(formula_path)
            for role, table in feature_tables.items():
                score, supported, reject = apply_mechanism_formula(table, formula)
                predictions = pd.DataFrame(
                    {
                        "material_id": table["material_id"].astype(str).to_numpy(),
                        "partition_role": role,
                        "next90_score": score,
                        "next90_supported": supported,
                        "next90_reject": reject,
                        "next90_decision": np.where(reject, "REJECT", "KEEP"),
                        "formula_sha256": formula_sha256,
                    }
                ).sort_values("material_id", kind="stable", ignore_index=True)
                prediction_path = staging / PREDICTION_NAMES[role]
                predictions.to_parquet(prediction_path, index=False)
                output_paths.append(prediction_path)

        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "fixed_mechanism_envelope_discovery_and_conditional_prediction_freeze",
            "passes_discovery_gates": result["passes_discovery_gates"],
            "discovery_endpoint_opened": True,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "candidate_count": result["candidate_count"],
            "prediction_partitions_frozen": (
                list(FEATURE_NAMES) if result["passes_discovery_gates"] else []
            ),
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next90_scigen_mechanism_envelope.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": bool(result["passes_discovery_gates"]),
            "universal_or_dft_equivalence_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT90 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT90 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "ENVELOPE_IDS",
    "FIXED_ENVELOPE_TERMS",
    "FORMULA_KIND",
    "MANIFEST_NAME",
    "PREDICTION_NAMES",
    "PROTOCOL",
    "apply_mechanism_formula",
    "run_scigen_mechanism_search",
    "search_mechanism_envelope_law",
]
