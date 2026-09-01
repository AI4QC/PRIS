#!/usr/bin/env python3
"""Paired stability diagnostics for the np-next-20260801 formula loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from better_formula_search import (  # noqa: E402
    _pair_scores,
    evaluate_fixed_thresholds,
    fit_sparse_pair_model,
    inner_oof_confidence_thresholds,
)
from better_search import deterministic_group_folds  # noqa: E402
from formula_stability import (  # noqa: E402
    _group_accuracies,
    paired_group_stability,
    summarize_fixed_commitment,
)
from next_formula_search import (  # noqa: E402
    load_isolated_formula_frame,
    next_eligible_features,
)

VARIANTS = ("existing_formula_loop", "additive_p235_formula_loop")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--formula-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    source = json.loads(args.formula_report.read_text(encoding="utf-8"))
    protocol = source["protocol"]
    min_gap = float(protocol["min_gap_eV_per_atom"])
    min_coverage = float(source["counts"].get("min_coverage", 0.90))
    frame = load_isolated_formula_frame(args.isolated_dir, args.real_descriptors)
    discovery = frame[frame["split"].eq("discovery")].reset_index(drop=True)
    outer_folds = len(source["variants"]["existing_formula_loop"]["outer_folds"])
    assignment = deterministic_group_folds(
        discovery["rk"].to_numpy(dtype=object),
        n_splits=outer_folds,
        seed=args.seed,
    )
    all_group_accuracy: dict[str, dict[object, float]] = {name: {} for name in VARIANTS}
    reproduced_fold_scores = {name: [] for name in VARIANTS}
    fixed_commitment = {name: [] for name in VARIANTS}
    for fold in range(outer_folds):
        train = discovery.loc[assignment != fold].reset_index(drop=True)
        test = discovery.loc[assignment == fold].reset_index(drop=True)
        for variant in VARIANTS:
            include_new = variant != "existing_formula_loop"
            features = next_eligible_features(
                discovery,
                include_new=include_new,
                min_coverage=min_coverage,
            )
            fold_features = [
                feature
                for feature in features
                if np.isfinite(train[feature].to_numpy(dtype=float)).mean()
                >= min_coverage
                and np.nanstd(train[feature].to_numpy(dtype=float)) > 1e-12
            ]
            record = source["variants"][variant]["outer_folds"][fold]
            config = record["selected_config"]
            model = fit_sparse_pair_model(
                train,
                feature_columns=fold_features,
                l1_c=float(config["l1_c"]),
                max_terms=int(config["max_terms"]),
                min_gap=min_gap,
            )
            pairs, scores = _pair_scores(model, test, min_gap=min_gap)
            group_accuracy = _group_accuracies(pairs, scores)
            thresholds, _ = inner_oof_confidence_thresholds(
                train,
                feature_columns=fold_features,
                l1_c=float(config["l1_c"]),
                max_terms=int(config["max_terms"]),
                n_folds=args.inner_folds,
                seed=args.seed + 100 + fold,
                min_gap=min_gap,
            )
            fixed_commitment[variant].append(
                evaluate_fixed_thresholds(
                    scores=scores,
                    labels=pairs.y,
                    groups=pairs.groups,
                    thresholds=thresholds,
                )
            )
            overlap = set(all_group_accuracy[variant]).intersection(group_accuracy)
            if overlap:
                raise ValueError("an outer group appeared in more than one fold")
            all_group_accuracy[variant].update(group_accuracy)
            reproduced = float(np.mean(list(group_accuracy.values())))
            expected = float(record["metrics"]["group_equal_accuracy"])
            if not np.isclose(reproduced, expected, atol=1e-12, rtol=0):
                raise ValueError(
                    f"{variant} fold {fold} did not reproduce: "
                    f"{reproduced} != {expected}"
                )
            reproduced_fold_scores[variant].append(reproduced)

    output = {
        "protocol": {
            "experiment": "np-next-20260801",
            "diagnostic": "paired outer-fold per-composition stability",
            "identifier_output": False,
            "selection_refit": False,
            "p1_vocabulary": "frozen",
            **dict(frame.attrs["source_access_audit"]),
        },
        "paired_full_coverage": paired_group_stability(
            all_group_accuracy["existing_formula_loop"],
            all_group_accuracy["additive_p235_formula_loop"],
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        ),
        "outer_fixed_commitment": summarize_fixed_commitment(
            fixed_commitment["existing_formula_loop"],
            fixed_commitment["additive_p235_formula_loop"],
        ),
        "reproduced_outer_fold_group_equal_accuracy": reproduced_fold_scores,
        "provenance": {
            "formula_report_sha256": _hash_file(args.formula_report),
            "descriptor_sha256": _hash_file(args.real_descriptors),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
