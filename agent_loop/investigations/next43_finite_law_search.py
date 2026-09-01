#!/usr/bin/env python3
"""Finite development-only search for an explicit DFT-free NEXT43 law."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations, product
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next23_evaluate import _continuous_diagnostics, _decision_metrics, _roc_auc
from src.next23_relaxation_rule import ENDPOINT_COLUMN, PRIMARY_GATES, PROTECTED_MAX
from src.next42_alexandria_evaluate import (
    JOINED_NAME as NEXT42_JOINED_NAME,
    PROTOCOL as NEXT42_EVALUATION_PROTOCOL,
)
from src.next43_analytic_feature_bank import (
    CANDIDATE_FEATURE_NAMES,
    FEATURE_NAME as FEATURE_BANK_NAME,
    PROTOCOL as FEATURE_BANK_PROTOCOL,
)


PROTOCOL = "2026-08-03-next43-finite-analytic-law-search-v1"
SPLIT_PROTOCOL = "2026-08-03-next43-material-id-hash-60-40-v1"
FORMULA_NAME = "NEXT43_DEVELOPMENT_CANDIDATE.json"
SEARCH_NAME = "NEXT43_FINITE_SEARCH.json"
PREDICTION_NAME = "next43_development_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
REJECTION_FRACTIONS = tuple(float(value) for value in np.arange(0.10, 0.401, 0.01))
PAIR_WEIGHTS = (0.5, 1.0, 2.0)
PAIR_SHORTLIST = 18
TRIPLE_SHORTLIST = 10
PAIR_CONJUNCTIVE_TAILS = (0.35, 0.45, 0.55, 0.65)
TRIPLE_CONJUNCTIVE_TAILS = (0.50, 0.625, 0.75)


def deterministic_split(material_ids: Sequence[str]) -> np.ndarray:
    """Assign identities to a stable 60/40 split without reading labels."""

    roles: list[str] = []
    for raw in material_ids:
        material_id = str(raw)
        digest = hashlib.sha256(
            f"{SPLIT_PROTOCOL}\0{material_id}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % 5
        roles.append("discovery" if bucket < 3 else "validation")
    return np.asarray(roles, dtype=object)


def _numeric_column(table: pd.DataFrame, name: str) -> np.ndarray:
    if name not in table:
        return np.full(len(table), np.nan, dtype=float)
    return pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)


def apply_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an explicit analytic formula; any missing term fails open."""

    if formula.get("missing_policy") != "KEEP":
        raise ValueError("NEXT43 formulas must fail open on missing values")
    kind = formula.get("kind")
    terms = formula.get("terms")
    if kind not in {"additive", "conjunctive"} or not isinstance(terms, list) or not terms:
        raise ValueError("invalid NEXT43 formula schema")
    supported = np.ones(len(features), dtype=bool)
    risks: list[np.ndarray] = []
    weights: list[float] = []
    cutoffs: list[float] = []
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("NEXT43 formula term must be an object")
        feature = term.get("feature")
        direction = term.get("direction")
        center = term.get("center")
        scale = term.get("scale")
        if (
            not isinstance(feature, str)
            or direction not in {-1, 1}
            or not isinstance(center, (int, float))
            or not math.isfinite(float(center))
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise ValueError("invalid NEXT43 formula term parameters")
        values = _numeric_column(features, feature)
        finite = np.isfinite(values)
        supported &= finite
        risks.append(int(direction) * (values - float(center)) / float(scale))
        if kind == "additive":
            weight = term.get("weight")
            if (
                not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
            ):
                raise ValueError("invalid NEXT43 additive weight")
            weights.append(float(weight))
        else:
            cutoff = term.get("cutoff")
            if not isinstance(cutoff, (int, float)) or not math.isfinite(float(cutoff)):
                raise ValueError("invalid NEXT43 conjunctive cutoff")
            cutoffs.append(float(cutoff))
    if kind == "additive":
        threshold = formula.get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError("invalid NEXT43 additive threshold")
        score = np.sum(
            np.stack([weight * risk for weight, risk in zip(weights, risks, strict=True)]),
            axis=0,
        )
        reject = supported & (score >= float(threshold))
    else:
        margins = np.stack(
            [risk - cutoff for risk, cutoff in zip(risks, cutoffs, strict=True)]
        )
        score = np.min(margins, axis=0)
        reject = supported & (score >= 0.0)
    score = np.asarray(score, dtype=float)
    score[~supported] = np.nan
    reject &= supported
    return score, supported, reject


def _quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("cannot fit a threshold without finite discovery values")
    return float(np.quantile(finite, quantile, method="inverted_cdf"))


def _term(info: Mapping[str, object], *, weight: float | None = None, cutoff: float | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "feature": str(info["feature"]),
        "direction": int(info["direction"]),
        "center": float(info["center"]),
        "scale": float(info["scale"]),
    }
    if weight is not None:
        result["weight"] = float(weight)
    if cutoff is not None:
        result["cutoff"] = float(cutoff)
    return result


def _candidate_rank(metrics: Mapping[str, object], auc: float | None) -> tuple[float, ...]:
    ratios = [float(metrics[name]) / float(cutoff) for name, cutoff in PRIMARY_GATES.items()]
    return (
        1.0 if bool(metrics["passes_primary_gates"]) else 0.0,
        min(ratios),
        float(metrics["protected_recall_lower"]),
        float(metrics["rejection_precision_lower"]),
        float(metrics["savings_lower"]),
        float(metrics["coverage_lower"]),
        float(auc) if auc is not None else -1.0,
    )


def _json_key(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def search_development_candidate(
    *,
    features: pd.DataFrame,
    material_ids: Sequence[str],
    endpoint: Sequence[float],
    split: Sequence[str],
    candidate_features: Sequence[str] = CANDIDATE_FEATURE_NAMES,
) -> dict[str, object]:
    """Select one formula using discovery labels, then evaluate validation once."""

    material_ids = np.asarray(material_ids, dtype=str)
    endpoint = np.asarray(endpoint, dtype=float)
    split = np.asarray(split, dtype=object)
    if (
        len(features) != len(material_ids)
        or endpoint.shape != (len(features),)
        or split.shape != (len(features),)
        or not np.isfinite(endpoint).all()
        or set(np.unique(split)) != {"discovery", "validation"}
        or len(set(material_ids)) != len(material_ids)
    ):
        raise ValueError("NEXT43 development arrays differ")
    discovery = split == "discovery"
    validation = split == "validation"
    changed = endpoint > PROTECTED_MAX
    infos: list[dict[str, object]] = []
    for feature in candidate_features:
        if feature not in features:
            continue
        values = _numeric_column(features, str(feature))
        finite = discovery & np.isfinite(values)
        if int(finite.sum()) < 4:
            continue
        supported_metrics = _decision_metrics(
            supported=np.isfinite(values[discovery]),
            reject=np.zeros(int(discovery.sum()), dtype=bool),
            endpoint=endpoint[discovery],
        )
        if float(supported_metrics["coverage_lower"]) < float(PRIMARY_GATES["coverage_lower"]):
            continue
        discovery_values = values[finite]
        center = float(np.median(discovery_values))
        q25, q75 = np.quantile(discovery_values, (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
            continue
        raw_auc = _roc_auc(values[finite], changed[finite])
        if raw_auc is None:
            continue
        direction = 1 if float(raw_auc) >= 0.5 else -1
        directional_auc = max(float(raw_auc), 1.0 - float(raw_auc))
        risk = direction * (values - center) / scale
        infos.append(
            {
                "feature": str(feature),
                "center": center,
                "scale": scale,
                "direction": direction,
                "directional_auc_changed": directional_auc,
                "discovery_supported": int(finite.sum()),
                "risk": risk,
            }
        )
    if not infos:
        raise ValueError("NEXT43 has no coverage-qualified variable descriptor")
    infos.sort(
        key=lambda item: (-float(item["directional_auc_changed"]), str(item["feature"]))
    )

    best_formula: dict[str, object] | None = None
    best_metrics: dict[str, object] | None = None
    best_rank: tuple[float, ...] | None = None
    best_key: str | None = None
    considered = 0
    leaderboard: list[tuple[tuple[float, ...], str, dict[str, object], dict[str, object]]] = []

    def consider(formula: dict[str, object]) -> None:
        nonlocal best_formula, best_metrics, best_rank, best_key, considered, leaderboard
        score, supported, reject = apply_formula(features, formula)
        metrics = _decision_metrics(
            supported=supported[discovery],
            reject=reject[discovery],
            endpoint=endpoint[discovery],
        )
        auc = _roc_auc(score[discovery & supported], changed[discovery & supported])
        rank = _candidate_rank(metrics, auc)
        key = _json_key(formula)
        considered += 1
        leaderboard.append((rank, key, formula, metrics))
        leaderboard.sort(key=lambda row: (row[0], tuple(-ord(c) for c in row[1])), reverse=True)
        if len(leaderboard) > 20:
            leaderboard.pop()
        if best_rank is None or rank > best_rank or (rank == best_rank and key < str(best_key)):
            best_formula = formula
            best_metrics = metrics
            best_rank = rank
            best_key = key

    def additive_formulas(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        discovery_score = np.zeros(int(discovery.sum()), dtype=float)
        finite = np.ones(int(discovery.sum()), dtype=bool)
        terms: list[dict[str, object]] = []
        for info, weight in zip(selected, weights, strict=True):
            risk = np.asarray(info["risk"], dtype=float)[discovery]
            finite &= np.isfinite(risk)
            discovery_score += float(weight) * risk
            terms.append(_term(info, weight=float(weight)))
        if not finite.any():
            return
        thresholds = {
            _quantile(discovery_score[finite], 1.0 - fraction)
            for fraction in REJECTION_FRACTIONS
        }
        for threshold in sorted(thresholds):
            consider(
                {
                    "kind": "additive",
                    "terms": terms,
                    "threshold": float(threshold),
                    "missing_policy": "KEEP",
                }
            )

    for info in infos:
        additive_formulas((info,), (1.0,))
    pair_infos = infos[: min(PAIR_SHORTLIST, len(infos))]
    for left, right in combinations(pair_infos, 2):
        for ratio in PAIR_WEIGHTS:
            additive_formulas((left, right), (1.0, ratio))
        for left_tail, right_tail in product(
            PAIR_CONJUNCTIVE_TAILS, repeat=2
        ):
            left_cut = _quantile(np.asarray(left["risk"])[discovery], 1.0 - left_tail)
            right_cut = _quantile(np.asarray(right["risk"])[discovery], 1.0 - right_tail)
            consider(
                {
                    "kind": "conjunctive",
                    "terms": [
                        _term(left, cutoff=left_cut),
                        _term(right, cutoff=right_cut),
                    ],
                    "missing_policy": "KEEP",
                }
            )
    triple_infos = infos[: min(TRIPLE_SHORTLIST, len(infos))]
    for selected in combinations(triple_infos, 3):
        additive_formulas(selected, (1.0, 1.0, 1.0))
        for tails in product(TRIPLE_CONJUNCTIVE_TAILS, repeat=3):
            terms = [
                _term(
                    info,
                    cutoff=_quantile(
                        np.asarray(info["risk"])[discovery], 1.0 - tail
                    ),
                )
                for info, tail in zip(selected, tails, strict=True)
            ]
            consider(
                {
                    "kind": "conjunctive",
                    "terms": terms,
                    "missing_policy": "KEEP",
                }
            )
    if best_formula is None or best_metrics is None or best_rank is None:
        raise RuntimeError("NEXT43 finite catalogue produced no formula")
    score, supported, reject = apply_formula(features, best_formula)
    validation_metrics = _decision_metrics(
        supported=supported[validation],
        reject=reject[validation],
        endpoint=endpoint[validation],
    )
    full_metrics = _decision_metrics(
        supported=supported,
        reject=reject,
        endpoint=endpoint,
    )
    diagnostics = {
        role: _continuous_diagnostics(score[mask], supported[mask], endpoint[mask])
        for role, mask in (
            ("discovery", discovery),
            ("validation", validation),
            ("full_development", np.ones(len(features), dtype=bool)),
        )
    }
    serializable_features = [
        {key: value for key, value in info.items() if key != "risk"}
        for info in infos
    ]
    serializable_leaderboard = [
        {"formula": formula, "discovery_metrics": metrics, "rank": list(rank)}
        for rank, _key, formula, metrics in sorted(
            leaderboard, key=lambda row: (row[0], row[1]), reverse=True
        )
    ]
    return {
        "selected_formula": best_formula,
        "discovery_metrics": best_metrics,
        "validation_metrics": validation_metrics,
        "full_development_metrics": full_metrics,
        "continuous_diagnostics": diagnostics,
        "passes_both_internal_splits": bool(
            best_metrics["passes_primary_gates"]
            and validation_metrics["passes_primary_gates"]
        ),
        "candidate_count": considered,
        "coverage_qualified_feature_count": len(infos),
        "feature_diagnostics": serializable_features,
        "discovery_leaderboard": serializable_leaderboard,
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _validate_inputs(
    *,
    feature_path: Path,
    feature_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = (feature_path, feature_manifest_path, evaluation_path, evaluation_manifest_path)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("NEXT43 finite-search input is missing")
    if feature_path.name != FEATURE_BANK_NAME or evaluation_path.name != NEXT42_JOINED_NAME:
        raise ValueError("NEXT43 finite-search input filename differs")
    feature_manifest = _strict_json(feature_manifest_path, role="NEXT43 feature manifest")
    evaluation_manifest = _strict_json(
        evaluation_manifest_path, role="NEXT42 evaluation manifest"
    )
    feature_outputs = feature_manifest.get("outputs_sha256")
    evaluation_outputs = evaluation_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != FEATURE_BANK_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_fields_read") is not False
        or feature_manifest.get("dft_values_used") is not False
        or feature_manifest.get("mlip_or_model_potential_used") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(feature_path.name) != _sha256(feature_path)
    ):
        raise ValueError("NEXT43 feature bank crossed the analytic boundary")
    if (
        evaluation_manifest.get("protocol") != NEXT42_EVALUATION_PROTOCOL
        or evaluation_manifest.get("production_protocol_eligible") is not True
        or evaluation_manifest.get("later_geometry_opened_after_prediction_freeze") is not True
        or evaluation_manifest.get("evaluation_only_dft_energy_read") is not False
        or not isinstance(evaluation_outputs, Mapping)
        or evaluation_outputs.get(evaluation_path.name) != _sha256(evaluation_path)
    ):
        raise ValueError("NEXT43 development endpoint source differs")
    features = pd.read_parquet(feature_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    evaluation = pd.read_parquet(
        evaluation_path,
        columns=[
            "material_id",
            "source_family",
            "natoms",
            "next23_supported",
            "next23_reject",
            "pauling_p2_p5_decision",
            "force_converged",
            "primary_evaluation_supported",
            ENDPOINT_COLUMN,
        ],
    ).sort_values("material_id", kind="stable", ignore_index=True)
    if (
        features.empty
        or features.material_id.astype(str).duplicated().any()
        or evaluation.material_id.astype(str).duplicated().any()
        or features.material_id.astype(str).tolist()
        != evaluation.material_id.astype(str).tolist()
        or not set(CANDIDATE_FEATURE_NAMES).issubset(features.columns)
        or not evaluation.force_converged.map(lambda value: type(value) is bool).all()
        or not evaluation.force_converged.all()
        or not evaluation.primary_evaluation_supported.all()
    ):
        raise ValueError("NEXT43 development identity or endpoint coverage differs")
    endpoint = pd.to_numeric(evaluation[ENDPOINT_COLUMN], errors="coerce")
    if endpoint.isna().any() or not np.isfinite(endpoint.to_numpy(float)).all():
        raise ValueError("NEXT43 development endpoint contains non-finite values")
    return features, evaluation


def _baseline_metrics(evaluation: pd.DataFrame, masks: Mapping[str, np.ndarray]) -> dict[str, object]:
    endpoint = evaluation[ENDPOINT_COLUMN].to_numpy(float)
    next23_supported = evaluation.next23_supported.to_numpy(bool)
    next23_reject = evaluation.next23_reject.to_numpy(bool)
    pauling_decision = evaluation.pauling_p2_p5_decision.astype(str).to_numpy()
    pauling_supported = pauling_decision != "ABSTAIN"
    pauling_reject = pauling_decision == "REJECT"
    result: dict[str, object] = {}
    for role, mask in masks.items():
        result[role] = {
            "next23_b_plus_e": _decision_metrics(
                supported=next23_supported[mask],
                reject=next23_reject[mask],
                endpoint=endpoint[mask],
            ),
            "pauling_p2_p5": _decision_metrics(
                supported=pauling_supported[mask],
                reject=pauling_reject[mask],
                endpoint=endpoint[mask],
            ),
        }
    return result


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_finite_search(
    *,
    feature_path: Path,
    feature_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run the fixed catalogue and publish one development candidate."""

    feature_path = Path(feature_path).resolve()
    feature_manifest_path = Path(feature_manifest_path).resolve()
    evaluation_path = Path(evaluation_path).resolve()
    evaluation_manifest_path = Path(evaluation_manifest_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    features, evaluation = _validate_inputs(
        feature_path=feature_path,
        feature_manifest_path=feature_manifest_path,
        evaluation_path=evaluation_path,
        evaluation_manifest_path=evaluation_manifest_path,
    )
    material_ids = features.material_id.astype(str).to_numpy()
    split = deterministic_split(material_ids)
    endpoint = evaluation[ENDPOINT_COLUMN].to_numpy(float)
    result = search_development_candidate(
        features=features,
        material_ids=material_ids,
        endpoint=endpoint,
        split=split,
        candidate_features=CANDIDATE_FEATURE_NAMES,
    )
    masks = {
        "discovery": split == "discovery",
        "validation": split == "validation",
        "full_development": np.ones(len(split), dtype=bool),
    }
    baselines = _baseline_metrics(evaluation, masks)
    formula_document = {
        "protocol": PROTOCOL,
        "role": "development candidate; not frozen for unseen confirmation",
        "formula": result["selected_formula"],
        "missing_policy": "KEEP",
        "execution_input": "one_raw_pre_dft_pre_mlip_x0_only",
        "execution_uses_dft": False,
        "execution_uses_endpoint_or_later_geometry": False,
        "execution_uses_mlip_or_model_potential": False,
        "execution_runs_physical_relaxation": False,
        "passes_both_internal_splits": result["passes_both_internal_splits"],
        "confirmation_candidate_ready": result["passes_both_internal_splits"],
        "requires_unseen_source_qualified_confirmation": True,
    }
    search_document = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject"}
    }
    search_document.update(
        {
            "protocol": PROTOCOL,
            "data_role": "NEXT42 opened converged endpoints used for development only",
            "split_protocol": SPLIT_PROTOCOL,
            "split_counts": {
                role: int(mask.sum()) for role, mask in masks.items()
            },
            "primary_gates": dict(PRIMARY_GATES),
            "catalogue": {
                "rejection_fractions": list(REJECTION_FRACTIONS),
                "pair_weights": list(PAIR_WEIGHTS),
                "pair_shortlist": PAIR_SHORTLIST,
                "triple_shortlist": TRIPLE_SHORTLIST,
                "pair_conjunctive_tails": list(PAIR_CONJUNCTIVE_TAILS),
                "triple_conjunctive_tails": list(TRIPLE_CONJUNCTIVE_TAILS),
            },
            "frozen_baselines": baselines,
            "scientific_confirmation": False,
        }
    )
    prediction = pd.DataFrame(
        {
            "material_id": material_ids,
            "split_role": split,
            "analytic_supported": result["supported"],
            "analytic_score": result["score"],
            "analytic_reject": result["reject"],
            ENDPOINT_COLUMN: endpoint,
        }
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        prediction_path = staging / PREDICTION_NAME
        formula_path.write_bytes(_json_bytes(formula_document))
        search_path.write_bytes(_json_bytes(search_document))
        prediction.to_parquet(prediction_path, index=False)
        repository = Path(__file__).resolve().parents[1]
        input_paths = {
            "features": feature_path,
            "feature_manifest": feature_manifest_path,
            "development_evaluation": evaluation_path,
            "development_evaluation_manifest": evaluation_manifest_path,
        }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "finite analytic formula development, not confirmation",
            "development_labels_opened": True,
            "offline_dft_final_geometry_label_used": True,
            "offline_dft_force_convergence_filter_used": True,
            "offline_dft_energy_used": False,
            "law_execution_dft_values_read": False,
            "law_execution_endpoint_or_later_geometry_read": False,
            "law_execution_mlip_or_model_potential_used": False,
            "law_execution_learned_energy_force_stress_proxy_used": False,
            "law_execution_physical_relaxation_executed": False,
            "thresholds_fit_on_discovery_only": True,
            "validation_labels_used_for_selection": False,
            "passes_both_internal_splits": result["passes_both_internal_splits"],
            "scientific_confirmation": False,
            "inputs_sha256": {
                role: _sha256(path) for role, path in input_paths.items()
            },
            "executed_source_sha256": {
                "src/next43_analytic_feature_bank.py": _sha256(
                    repository / "src/next43_analytic_feature_bank.py"
                ),
                "src/next43_finite_law_search.py": _sha256(
                    repository / "src/next43_finite_law_search.py"
                ),
            },
            "outputs_sha256": {
                path.name: _sha256(path)
                for path in (formula_path, search_path, prediction_path)
            },
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != manifest["inputs_sha256"][role] for role, path in input_paths.items()):
            raise RuntimeError("NEXT43 input changed during finite search")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--development-evaluation", required=True, type=Path)
    parser.add_argument("--development-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_finite_search(
        feature_path=args.features,
        feature_manifest_path=args.feature_manifest,
        evaluation_path=args.development_evaluation,
        evaluation_manifest_path=args.development_evaluation_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passes_both_internal_splits": manifest["passes_both_internal_splits"],
                "output": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
