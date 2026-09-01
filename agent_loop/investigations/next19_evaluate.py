"""Development search and cross-source evaluation for frozen NEXT19 scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace


PROTOCOL = "2026-08-02-next19-valence-transport-development-v1"
RESULT_NAME = "NEXT19_VALENCE_TRANSPORT_DEVELOPMENT.json"
MANIFEST_NAME = "MANIFEST.json"
PRIVATE_WBM_NAME = "joined_wbm.parquet"
PRIVATE_ELEMENTA_NAME = "joined_elementa.parquet"
DECISIONS = ("ABSTAIN", "KEEP", "REJECT")
MINIMUM_TOLERANCE = 1.0e-8
VALUABLE_THRESHOLD = 0.05
HIGH_THRESHOLD = 0.20
REJECTION_FRACTIONS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
FORMULAS = (
    "overload",
    "reallocation",
    "anion_mismatch",
    "overload_plus_reallocation",
    "overload_plus_mismatch",
    "reallocation_plus_mismatch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validate_feature_manifest(
    manifest: Mapping[str, object], *, feature_path: Path, role: str
) -> None:
    if manifest.get("endpoint_fields_read") is not False:
        raise ValueError(f"{role} feature manifest endpoint boundary failed")
    if manifest.get("model_or_proxy_potential_used") is not False:
        raise ValueError(f"{role} feature manifest model boundary failed")
    outputs = manifest.get("outputs_sha256")
    if (
        not isinstance(outputs, Mapping)
        or outputs.get(feature_path.name) != _sha256(feature_path)
    ):
        raise ValueError(f"{role} feature hash mismatch")


def join_feature_labels(
    features: pd.DataFrame, labels: pd.DataFrame, *, source: str
) -> pd.DataFrame:
    """Join private labels only after exact identifier-domain validation."""

    if "material_id" not in features or "material_id" not in labels:
        raise ValueError(f"{source} tables need material_id")
    left = features.copy()
    right = labels.copy()
    for role, table in (("features", left), ("labels", right)):
        if table["material_id"].isna().any():
            raise ValueError(f"{source} {role} identifiers contain nulls")
        table["material_id"] = table["material_id"].astype(str)
        if table["material_id"].duplicated().any():
            raise ValueError(f"{source} {role} identifiers are not unique")
    if set(left["material_id"]) != set(right["material_id"]):
        raise ValueError(f"{source} identifier coverage mismatch")
    overlap = (set(left) & set(right)) - {"material_id"}
    if overlap:
        raise ValueError(f"{source} tables have overlapping columns: {sorted(overlap)}")
    joined = left.merge(
        right, on="material_id", how="inner", validate="one_to_one", sort=True
    )
    if len(joined) != len(left):
        raise ValueError(f"{source} joined row count mismatch")
    return joined.sort_values("material_id", kind="stable").reset_index(drop=True)


def proportion(successes: int, total: int) -> dict[str, object]:
    """Return a binomial estimate and Wilson 95% interval."""

    if type(successes) is not int or type(total) is not int:
        raise ValueError("proportion counts must be integers")
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("proportion counts are invalid")
    if total == 0:
        return {
            "numerator": successes,
            "denominator": total,
            "estimate": None,
            "wilson_ci95": [None, None],
        }
    value = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (value + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            value * (1.0 - value) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": float(value),
        "wilson_ci95": [max(0.0, center - half), min(1.0, center + half)],
    }


def decisions_from_score(
    score: pd.Series, supported: pd.Series, *, threshold: float
) -> pd.Series:
    """Apply one high-is-worse threshold with fail-open abstention."""

    if len(score) != len(supported):
        raise ValueError("score and support lengths differ")
    if not math.isfinite(float(threshold)):
        raise ValueError("decision threshold must be finite")
    values = pd.to_numeric(score, errors="coerce").to_numpy(float)
    support = supported.astype(bool).to_numpy()
    if not np.isfinite(values[support]).all():
        raise ValueError("supported scores must be finite")
    return pd.Series(
        np.where(~support, "ABSTAIN", np.where(values >= threshold, "REJECT", "KEEP")),
        index=score.index,
        dtype=object,
    )


def candidate_thresholds(
    score: pd.Series,
    supported: pd.Series,
    *,
    rejection_fractions: Sequence[float] = REJECTION_FRACTIONS,
) -> tuple[float, ...]:
    """Derive deterministic WBM-only score quantiles for the fixed catalogue."""

    values = pd.to_numeric(score, errors="coerce").to_numpy(float)
    support = supported.astype(bool).to_numpy()
    finite = values[support & np.isfinite(values)]
    if not len(finite):
        return ()
    thresholds: set[float] = set()
    for fraction in rejection_fractions:
        numeric = float(fraction)
        if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
            raise ValueError("rejection fractions must be within (0, 1)")
        thresholds.add(
            float(np.quantile(finite, 1.0 - numeric, method="higher"))
        )
    return tuple(sorted(thresholds))


def formula_scores(table: pd.DataFrame, *, prefix: str, formula: str) -> pd.Series:
    """Compute one member of the frozen monotone score catalogue."""

    if formula not in FORMULAS:
        raise ValueError(f"unknown NEXT19 formula: {formula}")
    overload = pd.to_numeric(table[f"{prefix}__vt_overload"], errors="coerce")
    reallocation = pd.to_numeric(
        table[f"{prefix}__vt_reallocation"], errors="coerce"
    )
    mismatch = pd.to_numeric(
        table[f"{prefix}__vt_anion_mismatch_max"], errors="coerce"
    )
    values = {
        "overload": overload,
        "reallocation": reallocation,
        "anion_mismatch": mismatch,
        "overload_plus_reallocation": overload + reallocation,
        "overload_plus_mismatch": overload + mismatch,
        "reallocation_plus_mismatch": reallocation + mismatch,
    }
    return values[formula].astype(float)


def _validate_decisions(joined: pd.DataFrame, decisions: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(joined) != len(decisions):
        raise ValueError("joined table and decisions lengths differ")
    decision = decisions.astype(str).to_numpy(object)
    if not set(decision).issubset(DECISIONS):
        raise ValueError("unknown decision")
    return decision, decision == "REJECT"


def elementa_metrics(joined: pd.DataFrame, decisions: pd.Series) -> dict[str, object]:
    """Compute complete-composition safety and utility metrics."""

    decision, reject = _validate_decisions(joined, decisions)
    covered = decision != "ABSTAIN"
    regret = pd.to_numeric(
        joined["dft_group_regret_ev_per_atom"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(regret).all():
        raise ValueError("ELEMENTA regret must be finite")
    minimum = regret <= MINIMUM_TOLERANCE
    valuable = regret <= VALUABLE_THRESHOLD
    high = regret >= HIGH_THRESHOLD
    above_minimum = regret > MINIMUM_TOLERANCE
    grouped = pd.DataFrame(
        {
            "rk": joined["rk"].astype(str),
            "covered": covered,
            "reject": reject,
        }
    ).groupby("rk", sort=True)
    all_rejected = grouped.apply(
        lambda frame: bool(frame["covered"].all() and frame["reject"].all()),
        include_groups=False,
    )
    return {
        "decision_counts": {
            value: int((decision == value).sum()) for value in DECISIONS
        },
        "coverage": proportion(int(covered.sum()), len(joined)),
        "dft_savings": proportion(int(reject.sum()), len(joined)),
        "group_minimum_recall": proportion(
            int((minimum & ~reject).sum()), int(minimum.sum())
        ),
        "valuable_recall": proportion(
            int((valuable & ~reject).sum()), int(valuable.sum())
        ),
        "high_energy_rejection_recall": proportion(
            int((high & reject).sum()), int(high.sum())
        ),
        "reject_precision_above_minimum": proportion(
            int((above_minimum & reject).sum()), int(reject.sum())
        ),
        "all_rejected_groups": int(all_rejected.sum()),
    }


def wbm_metrics(joined: pd.DataFrame, decisions: pd.Series) -> dict[str, object]:
    """Compute WBM absolute and group-safe development metrics."""

    decision, reject = _validate_decisions(joined, decisions)
    covered = decision != "ABSTAIN"
    stable = joined["stable"].astype(bool).to_numpy()
    values = pd.to_numeric(
        joined["e_above_hull_mp2020_corrected_ppd_mp"], errors="raise"
    ).to_numpy(float)
    finite = np.isfinite(values)
    valuable = finite & (values <= VALUABLE_THRESHOLD)
    high = finite & (values >= HIGH_THRESHOLD)
    unstable = finite & (values > 0.0)
    grouped = pd.DataFrame(
        {
            "formula_key": joined["formula_key"].astype(str),
            "covered": covered,
            "reject": reject,
        }
    ).groupby("formula_key", sort=True)
    all_rejected = grouped.apply(
        lambda frame: bool(frame["covered"].all() and frame["reject"].all()),
        include_groups=False,
    )
    return {
        "decision_counts": {
            value: int((decision == value).sum()) for value in DECISIONS
        },
        "coverage": proportion(int(covered.sum()), len(joined)),
        "dft_savings": proportion(int(reject.sum()), len(joined)),
        "stable_recall": proportion(
            int((stable & ~reject).sum()), int(stable.sum())
        ),
        "valuable_recall": proportion(
            int((valuable & ~reject).sum()), int(valuable.sum())
        ),
        "high_energy_rejection_recall": proportion(
            int((high & reject).sum()), int(high.sum())
        ),
        "reject_precision_unstable": proportion(
            int((unstable & reject).sum()), int(reject.sum())
        ),
        "all_rejected_groups": int(all_rejected.sum()),
    }


def _lower(metric: Mapping[str, object]) -> float:
    interval = metric.get("wilson_ci95")
    if not isinstance(interval, Sequence) or len(interval) != 2 or interval[0] is None:
        return math.nan
    return float(interval[0])


def _wbm_eligible(metrics: Mapping[str, object]) -> tuple[bool, dict[str, bool]]:
    clauses = {
        "coverage_lower_at_least_0_90": _lower(metrics["coverage"]) >= 0.90,
        "stable_recall_lower_at_least_0_95": _lower(metrics["stable_recall"]) >= 0.95,
        "valuable_recall_lower_at_least_0_95": _lower(metrics["valuable_recall"]) >= 0.95,
        "reject_precision_lower_at_least_0_90": _lower(
            metrics["reject_precision_unstable"]
        )
        >= 0.90,
        "savings_lower_at_least_0_10": _lower(metrics["dft_savings"]) >= 0.10,
        "no_complete_group_rejection": int(metrics["all_rejected_groups"]) == 0,
    }
    return all(clauses.values()), clauses


def _elementa_eligible(
    metrics: Mapping[str, object]
) -> tuple[bool, dict[str, bool]]:
    clauses = {
        "coverage_lower_at_least_0_90": _lower(metrics["coverage"]) >= 0.90,
        "group_minimum_recall_lower_at_least_0_95": _lower(
            metrics["group_minimum_recall"]
        )
        >= 0.95,
        "valuable_recall_lower_at_least_0_95": _lower(metrics["valuable_recall"]) >= 0.95,
        "reject_precision_lower_at_least_0_90": _lower(
            metrics["reject_precision_above_minimum"]
        )
        >= 0.90,
        "savings_lower_at_least_0_10": _lower(metrics["dft_savings"]) >= 0.10,
        "no_complete_group_rejection": int(metrics["all_rejected_groups"]) == 0,
    }
    return all(clauses.values()), clauses


def _formula_complexity(formula: str) -> int:
    return 2 if "plus" in formula else 1


def search_catalogue(
    wbm_joined: pd.DataFrame,
    elementa_joined: pd.DataFrame,
    *,
    prefixes: Sequence[str],
    formulas: Sequence[str] = FORMULAS,
    rejection_fractions: Sequence[float] = REJECTION_FRACTIONS,
) -> dict[str, object]:
    """Select on WBM and apply unchanged to ELEMENTA with fixed safety gates."""

    if not prefixes or len(set(prefixes)) != len(prefixes):
        raise ValueError("prefix catalogue must be nonempty and unique")
    if not formulas or len(set(formulas)) != len(formulas):
        raise ValueError("formula catalogue must be nonempty and unique")
    if not set(formulas).issubset(FORMULAS):
        raise ValueError("formula catalogue contains an unknown formula")
    pauling = {
        "wbm": wbm_metrics(
            wbm_joined, wbm_joined["pauling_p2_p5_decision"].astype(str)
        ),
        "elementa": elementa_metrics(
            elementa_joined,
            elementa_joined["pauling_p2_p5_decision"].astype(str),
        ),
    }
    scan: dict[str, object] = {}
    eligible: list[tuple[float, int, str, str, float, dict[str, object]]] = []
    for prefix in prefixes:
        support_column = f"{prefix}__supported"
        if support_column not in wbm_joined or support_column not in elementa_joined:
            raise ValueError(f"missing support column for {prefix}")
        wbm_supported = wbm_joined[support_column].astype(bool)
        elementa_supported = elementa_joined[support_column].astype(bool)
        for formula in formulas:
            wbm_score = formula_scores(wbm_joined, prefix=prefix, formula=formula)
            elementa_score = formula_scores(
                elementa_joined, prefix=prefix, formula=formula
            )
            for threshold in candidate_thresholds(
                wbm_score,
                wbm_supported,
                rejection_fractions=rejection_fractions,
            ):
                wbm_decision = decisions_from_score(
                    wbm_score, wbm_supported, threshold=threshold
                )
                wbm_result = wbm_metrics(wbm_joined, wbm_decision)
                wbm_ok, wbm_clauses = _wbm_eligible(wbm_result)
                elementa_result: dict[str, object] | None = None
                elementa_clauses: dict[str, bool] | None = None
                elementa_ok = False
                if wbm_ok:
                    elementa_decision = decisions_from_score(
                        elementa_score, elementa_supported, threshold=threshold
                    )
                    elementa_result = elementa_metrics(
                        elementa_joined, elementa_decision
                    )
                    elementa_ok, elementa_clauses = _elementa_eligible(
                        elementa_result
                    )
                key = f"{prefix}|{formula}|{threshold:.17g}"
                entry: dict[str, object] = {
                    "candidate": {
                        "prefix": prefix,
                        "formula": formula,
                        "threshold": threshold,
                    },
                    "wbm_eligible": wbm_ok,
                    "wbm_clauses": wbm_clauses,
                    "wbm_metrics": wbm_result,
                    "elementa_evaluated": wbm_ok,
                    "elementa_eligible": elementa_ok,
                    "elementa_clauses": elementa_clauses,
                    "elementa_metrics": elementa_result,
                }
                scan[key] = entry
                if elementa_ok and elementa_result is not None:
                    savings = elementa_result["dft_savings"]["estimate"]
                    assert isinstance(savings, float)
                    eligible.append(
                        (
                            savings,
                            -_formula_complexity(formula),
                            prefix,
                            formula,
                            threshold,
                            entry,
                        )
                    )
    selected = max(eligible) if eligible else None
    selected_candidate = None
    selected_metrics = None
    if selected is not None:
        _savings, _complexity, prefix, formula, threshold, entry = selected
        selected_candidate = {
            "prefix": prefix,
            "formula": formula,
            "threshold": threshold,
        }
        selected_metrics = {
            "wbm": entry["wbm_metrics"],
            "elementa": entry["elementa_metrics"],
        }
    return {
        "protocol": PROTOCOL,
        "evidence_role": "historically exposed two-source development",
        "scientific_improvement_claim": False,
        "candidate_catalogue": {
            "prefixes": list(prefixes),
            "formulas": list(formulas),
            "rejection_fractions": [float(value) for value in rejection_fractions],
        },
        "pauling_metrics": pauling,
        "scan": scan,
        "development_promotion": selected_candidate is not None,
        "selected_candidate": selected_candidate,
        "selected_metrics": selected_metrics,
    }


def _selected_private_columns(
    joined: pd.DataFrame, result: Mapping[str, object]
) -> pd.DataFrame:
    output = joined.copy()
    selected = result.get("selected_candidate")
    if not isinstance(selected, Mapping):
        return output
    prefix = selected.get("prefix")
    formula = selected.get("formula")
    threshold = selected.get("threshold")
    if not isinstance(prefix, str) or not isinstance(formula, str) or not isinstance(
        threshold, (int, float)
    ):
        raise ValueError("selected candidate schema is invalid")
    output["next19_selected_score"] = formula_scores(
        output, prefix=prefix, formula=formula
    )
    output["next19_selected_decision"] = decisions_from_score(
        output["next19_selected_score"],
        output[f"{prefix}__supported"],
        threshold=float(threshold),
    )
    return output


def evaluate_development(
    *,
    wbm_features_path: Path,
    wbm_feature_manifest_path: Path,
    wbm_labels_path: Path,
    elementa_features_path: Path,
    elementa_feature_manifest_path: Path,
    elementa_labels_path: Path,
    aggregate_output_dir: Path,
    private_output_dir: Path,
    formulas: Sequence[str] = FORMULAS,
    rejection_fractions: Sequence[float] = REJECTION_FRACTIONS,
) -> None:
    """Run the sealed two-source development search and publish split outputs."""

    aggregate_target = Path(aggregate_output_dir)
    private_target = Path(private_output_dir)
    if aggregate_target.exists() or private_target.exists():
        raise FileExistsError("development output target already exists")
    wbm_feature_path = Path(wbm_features_path)
    elementa_feature_path = Path(elementa_features_path)
    wbm_feature_manifest = _json_object(
        Path(wbm_feature_manifest_path), role="WBM feature manifest"
    )
    elementa_feature_manifest = _json_object(
        Path(elementa_feature_manifest_path), role="ELEMENTA feature manifest"
    )
    _validate_feature_manifest(
        wbm_feature_manifest, feature_path=wbm_feature_path, role="WBM"
    )
    _validate_feature_manifest(
        elementa_feature_manifest,
        feature_path=elementa_feature_path,
        role="ELEMENTA",
    )
    wbm_features = pd.read_parquet(wbm_feature_path)
    elementa_features = pd.read_parquet(elementa_feature_path)
    wbm_labels = pd.read_parquet(
        Path(wbm_labels_path),
        columns=[
            "material_id",
            "formula_key",
            "stable",
            "e_above_hull_mp2020_corrected_ppd_mp",
            "pauling_p2_p5_decision",
        ],
    )
    elementa_labels = pd.read_parquet(
        Path(elementa_labels_path),
        columns=[
            "material_id",
            "dft_group_regret_ev_per_atom",
            "pauling_p2_p5_decision",
        ],
    )
    wbm_joined = join_feature_labels(wbm_features, wbm_labels, source="WBM")
    elementa_joined = join_feature_labels(
        elementa_features, elementa_labels, source="ELEMENTA"
    )
    prefixes = tuple(
        sorted(
            column[: -len("__supported")]
            for column in wbm_features
            if column.endswith("__supported")
            and column[: -len("__supported")] + "__supported"
            in elementa_features
        )
    )
    if not prefixes:
        raise ValueError("no shared NEXT19 feature configurations")
    result = search_catalogue(
        wbm_joined,
        elementa_joined,
        prefixes=prefixes,
        formulas=formulas,
        rejection_fractions=rejection_fractions,
    )
    result["inputs_sha256"] = {
        "wbm_features": _sha256(wbm_feature_path),
        "wbm_feature_manifest": _sha256(Path(wbm_feature_manifest_path)),
        "wbm_labels": _sha256(Path(wbm_labels_path)),
        "elementa_features": _sha256(elementa_feature_path),
        "elementa_feature_manifest": _sha256(
            Path(elementa_feature_manifest_path)
        ),
        "elementa_labels": _sha256(Path(elementa_labels_path)),
    }
    result["executed_source_sha256"] = {
        "src/next19_evaluate.py": _sha256(Path(__file__))
    }

    aggregate_target.parent.mkdir(parents=True, exist_ok=True)
    private_target.parent.mkdir(parents=True, exist_ok=True)
    aggregate_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{aggregate_target.name}.staging-", dir=aggregate_target.parent
        )
    )
    private_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{private_target.name}.staging-", dir=private_target.parent
        )
    )
    private_published = False
    try:
        private_wbm = private_staging / PRIVATE_WBM_NAME
        private_elementa = private_staging / PRIVATE_ELEMENTA_NAME
        _selected_private_columns(wbm_joined, result).to_parquet(
            private_wbm, index=False
        )
        _selected_private_columns(elementa_joined, result).to_parquet(
            private_elementa, index=False
        )
        private_manifest = {
            "protocol": PROTOCOL,
            "identifier_bearing": True,
            "scientific_improvement_claim": False,
            "outputs_sha256": {
                PRIVATE_WBM_NAME: _sha256(private_wbm),
                PRIVATE_ELEMENTA_NAME: _sha256(private_elementa),
            },
        }
        (private_staging / MANIFEST_NAME).write_text(
            json.dumps(private_manifest, indent=2, sort_keys=True) + "\n"
        )
        result_path = aggregate_staging / RESULT_NAME
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        aggregate_manifest = {
            "protocol": PROTOCOL,
            "identifier_bearing": False,
            "scientific_improvement_claim": False,
            "private_manifest_sha256": _sha256(private_staging / MANIFEST_NAME),
            "outputs_sha256": {RESULT_NAME: _sha256(result_path)},
        }
        (aggregate_staging / MANIFEST_NAME).write_text(
            json.dumps(aggregate_manifest, indent=2, sort_keys=True) + "\n"
        )
        _publish_directory_no_replace(private_staging, private_target)
        private_published = True
        _publish_directory_no_replace(aggregate_staging, aggregate_target)
    except Exception:
        if aggregate_staging.exists():
            shutil.rmtree(aggregate_staging)
        if private_staging.exists():
            shutil.rmtree(private_staging)
        if private_published:
            raise RuntimeError(
                f"private output published but aggregate publication failed: {private_target}"
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wbm-features", type=Path, required=True)
    parser.add_argument("--wbm-feature-manifest", type=Path, required=True)
    parser.add_argument("--wbm-labels", type=Path, required=True)
    parser.add_argument("--elementa-features", type=Path, required=True)
    parser.add_argument("--elementa-feature-manifest", type=Path, required=True)
    parser.add_argument("--elementa-labels", type=Path, required=True)
    parser.add_argument("--aggregate-output-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_development(
        wbm_features_path=args.wbm_features,
        wbm_feature_manifest_path=args.wbm_feature_manifest,
        wbm_labels_path=args.wbm_labels,
        elementa_features_path=args.elementa_features,
        elementa_feature_manifest_path=args.elementa_feature_manifest,
        elementa_labels_path=args.elementa_labels,
        aggregate_output_dir=args.aggregate_output_dir,
        private_output_dir=args.private_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
