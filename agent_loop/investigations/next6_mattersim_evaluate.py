#!/usr/bin/env python3
"""Evaluate MatterSim x0 energies through the same group-risk protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.next6_elementa_search import (
    FormulaSpec,
    TermSpec,
    run_prepared_search,
)
from src.next6_wbm_build import sha256_file


def mattersim_formula() -> FormulaSpec:
    return FormulaSpec(
        name="mattersim_5m_relative_x0_energy",
        track="cohort_relative",
        role="baseline",
        terms=(
            TermSpec(
                "mattersim_energy_per_atom",
                1,
                1,
                "mlip_energy",
                "support_mattersim",
            ),
        ),
    )


def mattersim_formulas() -> list[FormulaSpec]:
    """Return discrete-rank and continuous predicted-gap MLIP baselines."""

    return [
        mattersim_formula(),
        FormulaSpec(
            name="mattersim_5m_predicted_gap",
            track="cohort_margin",
            role="baseline",
            terms=(
                TermSpec(
                    "mattersim_predicted_gap",
                    1,
                    1,
                    "mlip_energy",
                    "support_mattersim",
                ),
            ),
        ),
    ]


def join_mattersim_feature(
    prepared: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join only the MLIP per-atom score and its explicit support flag by sid."""

    required_prepared = {"sid", "rk", "strict_x0_ok"}
    required_prediction = {
        "sid",
        "rk",
        "mattersim_feature_ok",
        "mattersim_energy_per_atom",
    }
    if required_prepared - set(prepared.columns) or required_prediction - set(
        predictions.columns
    ):
        raise ValueError("missing prepared or MatterSim columns")
    if prepared["sid"].duplicated().any() or predictions["sid"].duplicated().any():
        raise ValueError("MatterSim join sid must be unique")
    prediction = predictions[
        ["sid", "rk", "mattersim_feature_ok", "mattersim_energy_per_atom"]
    ].copy()
    joined = prepared.merge(
        prediction,
        on="sid",
        how="inner",
        suffixes=("", "_mattersim"),
        validate="one_to_one",
    )
    if len(joined) != len(prepared) or len(joined) != len(prediction):
        raise ValueError("prepared and MatterSim sid sets differ")
    if not joined["rk"].eq(joined["rk_mattersim"]).all():
        raise ValueError("prepared and MatterSim composition keys differ")
    joined = joined.drop(columns="rk_mattersim")
    energy = pd.to_numeric(
        joined["mattersim_energy_per_atom"], errors="coerce"
    ).to_numpy(float)
    joined["support_mattersim"] = (
        joined["strict_x0_ok"].fillna(False).to_numpy(bool)
        & joined["mattersim_feature_ok"].fillna(False).to_numpy(bool)
        & np.isfinite(energy)
    )
    supported_energy = pd.Series(
        np.where(joined["support_mattersim"].to_numpy(bool), energy, np.nan),
        index=joined.index,
    )
    group_min = supported_energy.groupby(joined["rk"], sort=False).transform("min")
    joined["mattersim_predicted_gap"] = supported_energy - group_min
    return joined


def run_mattersim_evaluation(
    elementa_dir: Path,
    prepared_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    alpha_values: Sequence[float] = (0.05, 0.01),
    protected_column: str = "valuable",
    within_group: str = "max",
) -> dict[str, object]:
    elementa_dir = Path(elementa_dir)
    prepared_path = Path(prepared_path)
    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    labels_path = elementa_dir / "elementa_labels.parquet"
    joined = join_mattersim_feature(
        pd.read_parquet(prepared_path), pd.read_parquet(prediction_path)
    )
    manifest = run_prepared_search(
        joined,
        pd.read_parquet(labels_path),
        output_dir,
        specs=mattersim_formulas(),
        alpha_values=alpha_values,
        protected_column=protected_column,
        within_group=within_group,
    )
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
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.05, 0.01))
    parser.add_argument("--protected-column", choices=("valuable", "near_min", "exact_min"), default="valuable")
    parser.add_argument("--within-group", choices=("max", "min"), default="max")
    args = parser.parse_args(argv)
    manifest = run_mattersim_evaluation(
        args.elementa,
        args.prepared,
        args.predictions,
        args.output,
        alpha_values=tuple(args.alphas),
        protected_column=args.protected_column,
        within_group=args.within_group,
    )
    print(json.dumps({"selected_formula": manifest["selected_formula"], **manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "join_mattersim_feature",
    "mattersim_formula",
    "mattersim_formulas",
    "run_mattersim_evaluation",
]
