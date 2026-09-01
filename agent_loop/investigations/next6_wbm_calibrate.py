#!/usr/bin/env python3
"""Select a sparse x0 physics formula, then calibrate its rejection threshold."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.next6_wbm_build import sha256_file
from src.next6_wbm_protocol import select_rejection_threshold


SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
PACK_LOW_GRID = (0.0, 0.4, 0.6)
PACK_HIGH_GRID = (0.8, 1.0, 1.2, 1.5)
PACK_WEIGHT_GRID = (0.25, 1.0, 4.0)
SCALE_PENALTY_GRID = (0.0, 0.02, 0.1)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    mode: str
    pack_low: float
    pack_high: float
    pack_weight: float
    scale_penalty: float
    complexity: int


def candidate_catalog() -> list[CandidateSpec]:
    """Return the finite, direction-constrained formula catalog."""

    specs = [
        CandidateSpec("min_pair_overlap", "min_pair", "static", 0, 0, 0, 0, 1),
        CandidateSpec("repulsion_static", "repulsion", "static", 0, 0, 0, 0, 1),
    ]
    for high in PACK_HIGH_GRID:
        specs.append(
            CandidateSpec(
                f"packing_high_h{high:g}",
                "packing_high",
                "static",
                0,
                high,
                1,
                0,
                1,
            )
        )
    for low in PACK_LOW_GRID:
        for high in PACK_HIGH_GRID:
            if low >= high:
                continue
            for weight in PACK_WEIGHT_GRID:
                specs.append(
                    CandidateSpec(
                        f"born_pack_static_lo{low:g}_hi{high:g}_w{weight:g}",
                        "born_pack",
                        "static",
                        low,
                        high,
                        weight,
                        0,
                        2,
                    )
                )
                for penalty in SCALE_PENALTY_GRID:
                    specs.append(
                        CandidateSpec(
                            f"born_pack_env_lo{low:g}_hi{high:g}_w{weight:g}_eta{penalty:g}",
                            "born_pack",
                            "envelope",
                            low,
                            high,
                            weight,
                            penalty,
                            3,
                        )
                    )
    return specs


def _packing_penalty(values: np.ndarray, low: float, high: float) -> np.ndarray:
    below = np.maximum(low - values, 0.0)
    above = np.maximum(values - high, 0.0)
    return below**2 + above**2


def score_candidate(features: pd.DataFrame, spec: CandidateSpec) -> np.ndarray:
    """Score one frozen formula; larger values are more rejectable."""

    ok = features["feature_ok"].fillna(False).to_numpy(dtype=bool)
    score = np.full(len(features), np.nan, dtype=float)
    if spec.family == "min_pair":
        ratio = features["min_pair_ratio"].to_numpy(dtype=float)
        values = np.maximum(1.0 / ratio - 1.0, 0.0) ** 2
    elif spec.family == "repulsion":
        values = features["repulsion_p2_l100"].to_numpy(dtype=float)
    elif spec.family == "packing_high":
        packing = features["packing_l100"].to_numpy(dtype=float)
        values = np.maximum(packing - spec.pack_high, 0.0) ** 2
    elif spec.family == "born_pack":
        if spec.mode == "static":
            repulsion = features["repulsion_p2_l100"].to_numpy(dtype=float)
            packing = features["packing_l100"].to_numpy(dtype=float)
            values = repulsion + spec.pack_weight * _packing_penalty(
                packing, spec.pack_low, spec.pack_high
            )
        elif spec.mode == "envelope":
            scaled_scores = []
            for scale in SCALES:
                suffix = f"l{int(round(scale * 100)):03d}"
                repulsion = features[f"repulsion_p2_{suffix}"].to_numpy(dtype=float)
                packing = features[f"packing_{suffix}"].to_numpy(dtype=float)
                scaled_scores.append(
                    repulsion
                    + spec.pack_weight
                    * _packing_penalty(packing, spec.pack_low, spec.pack_high)
                    + spec.scale_penalty * abs(math.log(scale))
                )
            values = np.min(np.vstack(scaled_scores), axis=0)
        else:
            raise ValueError(f"unknown candidate mode: {spec.mode}")
    else:
        raise ValueError(f"unknown candidate family: {spec.family}")
    valid = ok & np.isfinite(values)
    score[valid] = values[valid]
    return score


def _aligned(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if features["material_id"].duplicated().any() or labels["material_id"].duplicated().any():
        raise ValueError("material_id must be unique in features and labels")
    if set(features["material_id"]) != set(labels["material_id"]):
        raise ValueError("feature and label material_id sets differ")
    return features.merge(
        labels[["material_id", "stable"]], on="material_id", how="inner", validate="one_to_one"
    )


def evaluate_catalog(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    specs: Sequence[CandidateSpec] | None = None,
    max_false_negative_ucb: float,
    confidence: float,
) -> pd.DataFrame:
    """Evaluate every formula on the formula-selection compositions."""

    joined = _aligned(features, labels)
    catalog = list(candidate_catalog() if specs is None else specs)
    rows: list[dict[str, object]] = []
    for spec in catalog:
        selected = select_rejection_threshold(
            score_candidate(joined, spec),
            joined["stable"].to_numpy(dtype=bool),
            max_false_negative_ucb=max_false_negative_ucb,
            confidence=confidence,
        )
        rows.append({**asdict(spec), **selected})
    return pd.DataFrame(rows).sort_values("name", kind="stable").reset_index(drop=True)


def choose_formula(frontier: pd.DataFrame) -> pd.Series:
    """Choose deterministically by certified savings, then simplicity and name."""

    eligible = frontier.loc[frontier["certified"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("no formula satisfies the calibration risk bound")
    eligible = eligible.sort_values(
        ["dft_savings", "complexity", "name"],
        ascending=[False, True, True],
        kind="stable",
    )
    return eligible.iloc[0]


def run_calibration(
    artifact_dir: Path,
    output_dir: Path,
    *,
    max_false_negative_ucb: float = 0.01,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Select on one physical partition and calibrate threshold on the other."""

    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_x_path = artifact_dir / "formula_selection_x0_features.parquet"
    selection_y_path = artifact_dir / "formula_selection_labels.parquet"
    calibration_x_path = artifact_dir / "threshold_calibration_x0_features.parquet"
    calibration_y_path = artifact_dir / "threshold_calibration_labels.parquet"

    frontier = evaluate_catalog(
        pd.read_parquet(selection_x_path),
        pd.read_parquet(selection_y_path),
        max_false_negative_ucb=max_false_negative_ucb,
        confidence=confidence,
    )
    frontier_path = output_dir / "formula_selection_frontier.parquet"
    frontier.to_parquet(frontier_path, index=False)
    chosen = choose_formula(frontier)
    spec = CandidateSpec(
        name=str(chosen["name"]),
        family=str(chosen["family"]),
        mode=str(chosen["mode"]),
        pack_low=float(chosen["pack_low"]),
        pack_high=float(chosen["pack_high"]),
        pack_weight=float(chosen["pack_weight"]),
        scale_penalty=float(chosen["scale_penalty"]),
        complexity=int(chosen["complexity"]),
    )

    calibration_joined = _aligned(
        pd.read_parquet(calibration_x_path), pd.read_parquet(calibration_y_path)
    )
    calibrated = select_rejection_threshold(
        score_candidate(calibration_joined, spec),
        calibration_joined["stable"].to_numpy(dtype=bool),
        max_false_negative_ucb=max_false_negative_ucb,
        confidence=confidence,
    )
    frozen: dict[str, object] = {
        "protocol": "2026-08-01-dft-pre-screening-design-v1",
        "input_role": "unrelaxed_x0_only",
        "formula": asdict(spec),
        "threshold": calibrated,
        "selection_metrics": {
            key: chosen[key].item() if hasattr(chosen[key], "item") else chosen[key]
            for key in (
                "n",
                "n_stable",
                "n_reject",
                "n_abstain",
                "stable_recall",
                "false_negative_ucb",
                "dft_savings",
                "threshold",
            )
        },
        "risk": {
            "max_false_negative_ucb": max_false_negative_ucb,
            "confidence": confidence,
        },
        "inputs_sha256": {
            path.name: sha256_file(path)
            for path in (
                selection_x_path,
                selection_y_path,
                calibration_x_path,
                calibration_y_path,
            )
        },
    }
    frozen_path = output_dir / "frozen_rule.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "frozen_rule_sha256": sha256_file(frozen_path),
        "frontier_sha256": sha256_file(frontier_path),
        "n_candidates": len(frontier),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fnr-ucb", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    frozen = run_calibration(
        args.artifacts,
        args.output,
        max_false_negative_ucb=args.max_fnr_ucb,
        confidence=args.confidence,
    )
    print(json.dumps(frozen["formula"], sort_keys=True))
    print(json.dumps(frozen["threshold"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
