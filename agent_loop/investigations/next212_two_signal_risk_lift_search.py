#!/usr/bin/env python3
"""Search one frozen secondary risk lift on the closest NEXT211 anchor."""

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

import src.next211_residual_risk_lift_broad_diagnostic as n211
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n210 = n211.n210
PROTOCOL = "2026-08-08-next212-two-signal-risk-lift-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT212_TWO_SIGNAL_RISK_LIFT_CATALOGUE.json"
EVALUATION_NAME = "NEXT212_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT212_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next212_two_signal_risk_lift_search.parquet"
EXPECTED_DESIGN_SHA256 = (
    "b2565b6e8135beac57ba3e3120692f1b91affb5ef79f57f783d8e355fec0feea"
)
EXPECTED_ANCHOR_KEY_SHA256 = (
    "c66e9a7afc180e7060eb8e4de408c2552af24f45a9b493984d65de5250479ebb"
)
EXPECTED_ANCHOR_HYPOTHESIS = "scbv_mismatch_max__protected_low"
EXPECTED_SECONDARY_HYPOTHESIS_COUNT = n210.EXPECTED_ELIGIBLE_COUNT - 1
EXPECTED_CANDIDATE_COUNT = (
    1 + EXPECTED_SECONDARY_HYPOTHESIS_COUNT * len(n210.AMPLITUDE_FRACTIONS)
)
SEARCH_WORKERS = 4
SCORE_COMPOSITION = (
    "frozen_next211_anchor_plus_one_secondary_frozen_bounded_directional_risk_"
    "lift_if_original_next206_score_at_or_above_residual_threshold"
)
EXPECTED_NEXT211_SOURCE_SHA256 = (
    "a3e8e84b7646c4376d5bce4673faf1a22dbfee488ce1ad698a1b3cbfb36962cb"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n211.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next211_design": n211.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next211_manifest": (
        "fa5287b8099d7b302caf360ddcab617b3ec46b58227b5d09352cdf5f80406de2"
    ),
    "next211_diagnostic": (
        "c48bc36d7a00f3a48e3e6c91248e69b30525acd888adc3890daa1f90d5926f1a"
    ),
    "next211_table": (
        "a860da0c31382aa582b3de158d87bce9e0790707e631ddbf0b979b8fa395809a"
    ),
}


def anchored_two_signal_score(
    *,
    anchor_score: object,
    activation_score: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    q_lo: float | None,
    q_hi: float | None,
    residual_threshold: float,
    amplitude_fraction: float,
    risk_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add one nonnegative risk term to a fixed anchor score."""

    anchor = np.asarray(anchor_score, dtype=float)
    activation = np.asarray(activation_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    threshold = float(residual_threshold)
    amplitude = float(amplitude_fraction)
    scale = float(risk_scale)
    if (
        anchor.ndim != 1
        or activation.shape != anchor.shape
        or support.shape != anchor.shape
        or values.shape != anchor.shape
        or not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(amplitude)
        or amplitude < 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
        or np.any(~np.isfinite(anchor[support]))
        or np.any(~np.isfinite(activation[support]))
    ):
        raise ValueError("NEXT212 anchor arrays or lift parameters differ")
    if direction is None and q_lo is None and q_hi is None and amplitude == 0.0:
        return anchor.copy(), support.copy(), np.zeros(anchor.shape, dtype=bool)
    if direction not in n210.n208.n207.PROTECTION_DIRECTIONS:
        raise ValueError("NEXT212 secondary direction differs")
    if q_lo is None or q_hi is None or amplitude <= 0.0:
        raise ValueError("NEXT212 secondary lift specification differs")
    risk = n210.bounded_directional_risk(values, direction, q_lo, q_hi)
    active = support & (activation >= threshold) & np.isfinite(risk)
    corrected = anchor.copy()
    corrected[active] = anchor[active] + amplitude * scale * risk[active]
    if np.any(corrected[active] + 1.0e-12 < anchor[active]):
        raise RuntimeError("NEXT212 secondary risk lift reduced a score")
    return corrected, support.copy(), active


def _unique_hypothesis_specs(
    next210_specs: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for raw in next210_specs:
        spec = dict(raw)
        hypothesis = spec.get("hypothesis")
        if hypothesis is None:
            continue
        name = str(hypothesis)
        payload = {
            "feature": str(spec["feature"]),
            "hypothesis": name,
            "direction": str(spec["direction"]),
            "q_lo": float(spec["q_lo"]),
            "q_hi": float(spec["q_hi"]),
        }
        if name in unique and any(unique[name][key] != value for key, value in payload.items()):
            raise ValueError("NEXT212 NEXT210 normalization differs across amplitudes")
        unique[name] = payload
    if not unique:
        raise ValueError("NEXT212 secondary hypothesis universe is empty")
    return unique


def build_candidate_specs(
    *,
    anchor_spec: Mapping[str, object],
    next210_specs: Sequence[Mapping[str, object]],
    amplitude_fractions: Sequence[float] = n210.AMPLITUDE_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the exact anchor plus one-secondary-signal candidate grid."""

    anchor = dict(anchor_spec)
    anchor_key = str(anchor.get("candidate_key", ""))
    anchor_hypothesis = str(anchor.get("hypothesis", ""))
    if (
        not anchor_key
        or not anchor_hypothesis
        or anchor.get("feature") is None
        or anchor.get("direction") not in n210.n208.n207.PROTECTION_DIRECTIONS
    ):
        raise ValueError("NEXT212 anchor specification differs")
    unique = _unique_hypothesis_specs(next210_specs)
    if anchor_hypothesis not in unique:
        raise ValueError("NEXT212 anchor is absent from the NEXT210 universe")
    amplitudes = tuple(float(value) for value in amplitude_fractions)
    numerators = tuple(n210._amplitude_numerator(value) for value in amplitudes)
    if not amplitudes or len(numerators) != len(set(numerators)):
        raise ValueError("NEXT212 amplitude grid differs")

    common = {
        "anchor_candidate_key": anchor_key,
        "anchor_hypothesis": anchor_hypothesis,
        "anchor_feature": str(anchor["feature"]),
        "anchor_direction": str(anchor["direction"]),
        "anchor_q_lo": float(anchor["q_lo"]),
        "anchor_q_hi": float(anchor["q_hi"]),
        "anchor_amplitude_fraction": float(anchor["amplitude_fraction"]),
        "anchor_amplitude_numerator": int(anchor["amplitude_numerator"]),
        "amplitude_denominator": n210.AMPLITUDE_DENOMINATOR,
        "risk_scale": float(anchor["risk_scale"]),
        "residual_threshold": float(anchor["residual_threshold"]),
        "missing_policy": "SECONDARY_TERM_OFF_KEEP_ANCHOR",
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "secondary_hypothesis": None,
            "secondary_feature": None,
            "secondary_direction": None,
            "secondary_q_lo": None,
            "secondary_q_hi": None,
            "secondary_amplitude_fraction": 0.0,
            "secondary_amplitude_numerator": 0,
        }
    ]
    for hypothesis in sorted(set(unique) - {anchor_hypothesis}):
        secondary = unique[hypothesis]
        for amplitude, numerator in zip(amplitudes, numerators, strict=True):
            payloads.append(
                {
                    **common,
                    "secondary_hypothesis": hypothesis,
                    "secondary_feature": secondary["feature"],
                    "secondary_direction": secondary["direction"],
                    "secondary_q_lo": secondary["q_lo"],
                    "secondary_q_hi": secondary["q_hi"],
                    "secondary_amplitude_fraction": amplitude,
                    "secondary_amplitude_numerator": numerator,
                }
            )
    specs = [
        {
            **payload,
            "candidate_key": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }
        for payload in payloads
    ]
    if len({str(spec["candidate_key"]) for spec in specs}) != len(specs):
        raise RuntimeError("NEXT212 candidate keys are not unique")
    return specs


def materialize_two_signal_candidates(
    *,
    features: pd.DataFrame,
    anchor_score: object,
    activation_score: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    """Encode every anchored two-signal score as one exact virtual term."""

    anchor = np.asarray(anchor_score, dtype=float)
    activation = np.asarray(activation_score, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    raw_specs = [dict(value) for value in specs]
    if (
        not isinstance(features, pd.DataFrame)
        or anchor.shape != (len(features),)
        or activation.shape != anchor.shape
        or support.shape != anchor.shape
        or not raw_specs
        or len({str(spec.get("candidate_key", "")) for spec in raw_specs})
        != len(raw_specs)
    ):
        raise ValueError("NEXT212 materializer inputs differ")
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for spec in raw_specs:
        feature = spec.get("secondary_feature")
        if feature is None:
            values = np.full(len(features), np.nan)
        elif str(feature) in features.columns:
            values = pd.to_numeric(features[str(feature)], errors="coerce").to_numpy(float)
        else:
            raise ValueError("NEXT212 materializer feature differs")
        corrected, corrected_support, _ = anchored_two_signal_score(
            anchor_score=anchor,
            activation_score=activation,
            base_support=support,
            feature_values=values,
            direction=spec.get("secondary_direction"),
            q_lo=spec.get("secondary_q_lo"),
            q_hi=spec.get("secondary_q_hi"),
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["secondary_amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        maximum = float(np.max(corrected[corrected_support])) if corrected_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[corrected_support] = np.sinh(corrected[corrected_support] / divisor)
        key = str(spec["candidate_key"])
        virtual_id = f"next212_virtual_candidate__{hashlib.sha256(key.encode()).hexdigest()[:24]}"
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
                "group": "next212_two_signal_risk_lift",
                "encoding": "asinh_sinh_exact_two_signal_risk_score",
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
    roots: Mapping[str, Path], freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path, design_path: Path,
) -> dict[str, Path]:
    paths = n211._paths(
        roots, freeze_path, next202_design_path, next205_design_path,
        next207_design_path, next208_design_path, next209_design_path,
        next210_design_path, next211_design_path,
    )
    paths["next211_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next211_manifest": roots["next211"] / n211.MANIFEST_NAME,
            "next211_diagnostic": roots["next211"] / n211.DIAGNOSTIC_NAME,
            "next211_table": roots["next211"] / n211.TABLE_NAME,
        }
    )
    return paths


def _verify_next211(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], str]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next211_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next211_design"]
    eligible, _, _ = n211._verify_next210(prior_paths, prior_hashes)
    manifest = json.loads(paths["next211_manifest"].read_text())
    diagnostic = json.loads(paths["next211_diagnostic"].read_text())
    table = pd.read_parquet(paths["next211_table"])
    expected_outputs = {
        n211.DIAGNOSTIC_NAME: input_hashes["next211_diagnostic"],
        n211.TABLE_NAME: input_hashes["next211_table"],
    }
    closest = diagnostic.get("global_closest", {})
    anchor_key = str(closest.get("candidate_key", ""))
    if (
        manifest.get("protocol") != n211.PROTOCOL
        or manifest.get("candidate_count") != n211.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256") != n211.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("next210_record_population_reproduced") is not True
        or manifest.get("next210_candidate_universe_reproduced") is not True
        or manifest.get("next210_all_gate_candidate_count") != 0
        or manifest.get("continuous_residual_risk_lift_branch_closed") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in n210.n208.BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next211_residual_risk_lift_broad_diagnostic.py"
        )
        != EXPECTED_NEXT211_SOURCE_SHA256
        or _sha256_file(Path(n211.__file__).resolve()) != EXPECTED_NEXT211_SOURCE_SHA256
        or diagnostic.get("protocol") != n211.PROTOCOL
        or diagnostic.get("candidate_count") != n211.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256") != n211.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("new_formula_searched") is not False
        or diagnostic.get("validation_outputs_opened") is not False
        or hashlib.sha256(anchor_key.encode()).hexdigest() != EXPECTED_ANCHOR_KEY_SHA256
        or closest.get("feature") != "scbv_mismatch_max"
        or closest.get("direction") != "protected_low"
        or not math.isclose(float(closest.get("amplitude_fraction")), 1 / 16)
        or len(table) != n211.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT212 NEXT211 provenance differs")
    return eligible, anchor_key


def run_two_signal_risk_lift_search(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path,
    next113_dir: Path, next114_dir: Path, next116_dir: Path,
    next117_dir: Path, next120_dir: Path, next121_dir: Path,
    next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    next134_dir: Path, next163_dir: Path, next164_dir: Path,
    next168_dir: Path, next173_dir: Path, next179_dir: Path,
    next180_dir: Path, next181_dir: Path, next182_dir: Path,
    next183_dir: Path, next184_dir: Path, next185_dir: Path,
    next186_dir: Path, next188_dir: Path, next190_dir: Path,
    next192_dir: Path, next194_dir: Path, next199_dir: Path,
    next200_dir: Path, next201_dir: Path, next202_dir: Path,
    next203_dir: Path, next204_dir: Path, next205_dir: Path,
    next206_dir: Path, next207_dir: Path, next208_dir: Path,
    next209_dir: Path, next210_dir: Path, next211_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT212 two-signal search."""

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
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (194, next194_dir), (199, next199_dir), (200, next200_dir),
                (201, next201_dir), (202, next202_dir), (203, next203_dir),
                (204, next204_dir), (205, next205_dir), (206, next206_dir),
                (207, next207_dir), (208, next208_dir), (209, next209_dir),
                (210, next210_dir), (211, next211_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(), Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(), Path(next208_design_path).resolve(),
        Path(next209_design_path).resolve(), Path(next210_design_path).resolve(),
        Path(next211_design_path).resolve(), Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT212 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT212 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT212 formal input identity differs: {differing}")
    eligible, anchor_key = _verify_next211(paths, input_hashes)
    combined, feature_tables, base_key, base_score, base_support, endpoint = (
        n210.n208._reconstruct_next206(paths=paths)
    )
    next210_specs = n210.build_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=base_support,
        residual_threshold=n210.n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    anchors = [spec for spec in next210_specs if str(spec["candidate_key"]) == anchor_key]
    if len(anchors) != 1 or str(anchors[0]["hypothesis"]) != EXPECTED_ANCHOR_HYPOTHESIS:
        raise ValueError("NEXT212 frozen anchor specification differs")
    anchor_spec = anchors[0]
    anchor_values = pd.to_numeric(
        combined[str(anchor_spec["feature"])], errors="coerce"
    ).to_numpy(float)
    anchor_score, anchor_support, _ = n210.residual_risk_lift_score(
        base_score=base_score,
        base_support=base_support,
        feature_values=anchor_values,
        direction=str(anchor_spec["direction"]),
        q_lo=float(anchor_spec["q_lo"]),
        q_hi=float(anchor_spec["q_hi"]),
        residual_threshold=float(anchor_spec["residual_threshold"]),
        amplitude_fraction=float(anchor_spec["amplitude_fraction"]),
        risk_scale=float(anchor_spec["risk_scale"]),
    )
    specs = build_candidate_specs(anchor_spec=anchor_spec, next210_specs=next210_specs)
    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT212 candidate universe differs")
    combined_virtual, terms, runtime = materialize_two_signal_candidates(
        features=combined, anchor_score=anchor_score, activation_score=base_score,
        base_support=anchor_support, specs=specs,
    )
    started = time.perf_counter()
    result = n210.n208.n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
        features=combined_virtual, endpoint=endpoint, old_terms=terms,
        optional_terms=[], candidate_specs=runtime, workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if int(result["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("NEXT212 evaluator count differs")
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    source = combined["source_dataset"].astype(str).to_numpy()

    def decorate(record: dict[str, object]) -> None:
        spec = spec_by_key[str(record["candidate_key"])]
        feature = spec["secondary_feature"]
        values = (
            np.full(len(combined), np.nan)
            if feature is None
            else pd.to_numeric(combined[str(feature)], errors="coerce").to_numpy(float)
        )
        _, _, active = anchored_two_signal_score(
            anchor_score=anchor_score, activation_score=base_score,
            base_support=anchor_support, feature_values=values,
            direction=spec["secondary_direction"], q_lo=spec["secondary_q_lo"],
            q_hi=spec["secondary_q_hi"],
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["secondary_amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        record.update(
            {
                **{key: value for key, value in spec.items() if key != "candidate_key"},
                "secondary_active_rows": int(active.sum()),
                "secondary_active_scigen": int((active & (source == "scigen")).sum()),
                "secondary_active_wyformer": int((active & (source == "wyformer")).sum()),
            }
        )

    for record in result["candidate_records"]:
        decorate(record)
    selected = result["selected"]
    if "anchor_hypothesis" not in selected["record"]:
        decorate(selected["record"])
    records = pd.DataFrame(result["candidate_records"])
    all_gate = records["passes_all_discovery_gates"].fillna(False).astype(bool)
    diagnostic_mask = (
        records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~records["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    next213_keys = sorted(records.loc[diagnostic_mask, "candidate_key"].astype(str))
    next213_sha = hashlib.sha256("\n".join(next213_keys).encode()).hexdigest()
    selected_spec = spec_by_key[str(selected["record"]["candidate_key"])]
    passes = bool(selected["record"]["passes_all_discovery_gates"])
    formula = {
        "protocol": PROTOCOL,
        "kind": "anchored_two_signal_continuous_x0_risk_lift_no_dft_score",
        **{key: value for key, value in selected_spec.items() if key != "candidate_key"},
        "support_policy": "UNCHANGED_FROM_NEXT206",
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    counts = {
        name: int(records[name].fillna(False).astype(bool).sum())
        for name in (
            "passes_source_auc_gates", "passes_safe_all_cells",
            "passes_broad_all_cells", "passes_all_discovery_gates",
        )
    }
    counts["passes_auc_and_safe_but_not_broad"] = int(diagnostic_mask.sum())
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "anchor_candidate_key_sha256": EXPECTED_ANCHOR_KEY_SHA256,
        "anchor_hypothesis": EXPECTED_ANCHOR_HYPOTHESIS,
        "eligible_hypothesis_count": len(eligible),
        "secondary_hypothesis_count": EXPECTED_SECONDARY_HYPOTHESIS_COUNT,
        "amplitude_fractions": list(n210.AMPLITUDE_FRACTIONS),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "score_composition": SCORE_COMPOSITION,
        "normalization_refit": False,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "fixed_anchored_two_signal_continuous_x0_risk_lift_search",
        "next211_anchor_reproduced": True,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": int(result["candidate_count"]),
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts": counts,
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
        "next213_diagnostic_authorized": bool(not passes and next213_keys),
        "next213_candidate_count": len(next213_keys),
        "next213_candidate_key_sha256": next213_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next210_residual_risk_lift_search.py": Path(n210.__file__).resolve(),
        "src/next211_residual_risk_lift_broad_diagnostic.py": Path(n211.__file__).resolve(),
        "src/next212_two_signal_risk_lift_search.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        outputs = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": int(result["candidate_count"]),
            "secondary_hypothesis_count": EXPECTED_SECONDARY_HYPOTHESIS_COUNT,
            "anchor_candidate_key_sha256": EXPECTED_ANCHOR_KEY_SHA256,
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next213_diagnostic_authorized": bool(not passes and next213_keys),
            "next213_candidate_count": len(next213_keys),
            "next213_candidate_key_sha256": next213_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "two_signal_risk_lift_branch_terminated": bool(not passes and not next213_keys),
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **n210.n208.BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT212 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT212 source changed before publication")
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
    later_stages = (194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211)
    for stage in stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_two_signal_risk_lift_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages + later_stages},
        next135_freeze_path=args.next135_freeze_path,
        **{
            f"next{stage}_design_path": getattr(args, f"next{stage}_design_path")
            for stage in (202, 205, 207, 208, 209, 210, 211)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "EXPECTED_CANDIDATE_COUNT",
    "anchored_two_signal_score",
    "build_candidate_specs",
    "materialize_two_signal_candidates",
    "run_two_signal_risk_lift_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
