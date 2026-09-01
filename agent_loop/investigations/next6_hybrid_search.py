#!/usr/bin/env python3
"""Post-hoc hybrid hypothesis: MatterSim x0 gap plus a sparse physics rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.next6_elementa_search import FormulaSpec, TermSpec, run_prepared_search
from src.next6_mattersim_evaluate import join_mattersim_feature, mattersim_formulas
from src.next6_wbm_build import sha256_file


PHYSICS_WEIGHT_EV_PER_ATOM = 0.05


def _rank_bad(
    values: np.ndarray,
    groups: pd.Series,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(np.where(eligible, values, np.nan), index=groups.index)
    rank = series.groupby(groups, sort=False).rank(method="average") - 1.0
    count = series.groupby(groups, sort=False).transform("count")
    result = (rank / (count - 1.0)).to_numpy(float)
    supported = eligible & (count.to_numpy(float) > 1.0) & np.isfinite(result)
    result[~supported] = np.nan
    return result, supported


def build_hybrid_features(
    prepared: pd.DataFrame,
    mattersim_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Add the fixed Ewald + low-packing + P2 correction without IDs/order."""

    joined = join_mattersim_feature(prepared, mattersim_predictions)
    required = {
        "support_geom",
        "support_ewald",
        "support_p2",
        "geom_packing_fraction",
        "ewald_per_atom",
        "p2_mean_dev",
    }
    missing = required - set(joined.columns)
    if missing:
        raise ValueError(f"missing hybrid physics columns: {sorted(missing)}")
    numeric = {
        column: pd.to_numeric(joined[column], errors="coerce").to_numpy(float)
        for column in (
            "geom_packing_fraction",
            "ewald_per_atom",
            "p2_mean_dev",
            "mattersim_predicted_gap",
        )
    }
    eligible = (
        joined["support_mattersim"].fillna(False).to_numpy(bool)
        & joined["support_geom"].fillna(False).to_numpy(bool)
        & joined["support_ewald"].fillna(False).to_numpy(bool)
        & joined["support_p2"].fillna(False).to_numpy(bool)
        & np.logical_and.reduce([np.isfinite(values) for values in numeric.values()])
    )
    packing_rank, packing_support = _rank_bad(
        -numeric["geom_packing_fraction"], joined["rk"], eligible
    )
    ewald_rank, ewald_support = _rank_bad(
        numeric["ewald_per_atom"], joined["rk"], eligible
    )
    p2_rank, p2_support = _rank_bad(
        numeric["p2_mean_dev"], joined["rk"], eligible
    )
    support = packing_support & ewald_support & p2_support
    physics = (packing_rank + ewald_rank + p2_rank) / 3.0
    hybrid = numeric["mattersim_predicted_gap"] + PHYSICS_WEIGHT_EV_PER_ATOM * physics
    physics[~support] = np.nan
    hybrid[~support] = np.nan
    joined["hybrid_physics_rank"] = physics
    joined["hybrid_gap_p005"] = hybrid
    joined["support_hybrid"] = support
    return joined


def hybrid_formula() -> FormulaSpec:
    return FormulaSpec(
        name="hybrid_mattersim_gap_plus_0p05_physics_rank",
        track="cohort_margin",
        role="candidate",
        terms=(
            TermSpec(
                "hybrid_gap_p005",
                1,
                1,
                "mlip_plus_physics",
                "support_hybrid",
            ),
        ),
    )


def run_hybrid_search(
    elementa_dir: Path,
    prepared_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    alpha_values: Sequence[float] = (0.03, 0.01, 0.05),
) -> dict[str, object]:
    elementa_dir = Path(elementa_dir)
    prepared_path = Path(prepared_path)
    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    labels_path = elementa_dir / "elementa_labels.parquet"
    features = build_hybrid_features(
        pd.read_parquet(prepared_path), pd.read_parquet(prediction_path)
    )
    gap = next(
        formula
        for formula in mattersim_formulas()
        if formula.name == "mattersim_5m_predicted_gap"
    )
    manifest = run_prepared_search(
        features,
        pd.read_parquet(labels_path),
        output_dir,
        specs=[gap, hybrid_formula()],
        alpha_values=alpha_values,
        protected_column="near_min",
        within_group="min",
    )
    manifest["hypothesis_status"] = "post-hoc after ELEMENTA test exposure"
    manifest["inputs_sha256"] = {
        prepared_path.name: sha256_file(prepared_path),
        prediction_path.name: sha256_file(prediction_path),
        labels_path.name: sha256_file(labels_path),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elementa", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.03, 0.01, 0.05))
    args = parser.parse_args(argv)
    manifest = run_hybrid_search(
        args.elementa,
        args.prepared,
        args.predictions,
        args.output,
        alpha_values=tuple(args.alphas),
    )
    print(json.dumps({"selected_formula": manifest["selected_formula"], **manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_hybrid_features", "hybrid_formula", "run_hybrid_search"]
