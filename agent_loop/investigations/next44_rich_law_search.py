#!/usr/bin/env python3
"""Join sealed NEXT43/NEXT44 x0 features and reuse the fixed finite search."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next23_evaluate import _decision_metrics
from src.next23_relaxation_rule import ENDPOINT_COLUMN, PRIMARY_GATES
from src.next43_analytic_feature_bank import (
    CANDIDATE_FEATURE_NAMES as BASE_FEATURE_NAMES,
    FEATURE_NAME as BASE_FEATURE_FILE,
)
from src.next43_finite_law_search import (
    PROTOCOL as NEXT43_SEARCH_PROTOCOL,
    SEARCH_NAME as NEXT43_SEARCH_NAME,
    _baseline_metrics,
    _validate_inputs as _validate_next43_inputs,
    apply_formula,
    deterministic_split,
    search_development_candidate,
)
from src.next44_rich_analytic_features import (
    CANDIDATE_FEATURE_NAMES as RICH_FEATURE_NAMES,
    FEATURE_NAME as RICH_FEATURE_FILE,
    PROTOCOL as RICH_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next44-rich-finite-analytic-law-search-v1"
FORMULA_NAME = "NEXT44_DEVELOPMENT_CANDIDATE.json"
SEARCH_NAME = "NEXT44_FINITE_SEARCH.json"
PREDICTION_NAME = "next44_development_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
ALL_FEATURE_NAMES = tuple(BASE_FEATURE_NAMES) + tuple(RICH_FEATURE_NAMES)


if len(ALL_FEATURE_NAMES) != len(set(ALL_FEATURE_NAMES)):
    raise RuntimeError("NEXT43/NEXT44 candidate feature names overlap")


def combine_feature_tables(
    base: pd.DataFrame,
    rich: pd.DataFrame,
    *,
    base_features: Sequence[str],
    rich_features: Sequence[str],
) -> pd.DataFrame:
    """Return a canonical exact-identity join of two label-free tables."""

    base_features = tuple(str(value) for value in base_features)
    rich_features = tuple(str(value) for value in rich_features)
    if set(base_features) & set(rich_features):
        raise ValueError("NEXT44 feature namespaces overlap")
    required_base = {"material_id", *base_features}
    required_rich = {"material_id", *rich_features}
    if not required_base.issubset(base.columns) or not required_rich.issubset(rich.columns):
        raise ValueError("NEXT44 feature table schema differs")
    for table in (base, rich):
        if (
            table.empty
            or table.material_id.isna().any()
            or table.material_id.astype(str).duplicated().any()
        ):
            raise ValueError("NEXT44 feature identity differs")
    left = base.loc[:, ["material_id", *base_features]].copy()
    right = rich.loc[:, ["material_id", *rich_features]].copy()
    left["material_id"] = left.material_id.astype(str)
    right["material_id"] = right.material_id.astype(str)
    left = left.sort_values("material_id", kind="stable", ignore_index=True)
    right = right.sort_values("material_id", kind="stable", ignore_index=True)
    if left.material_id.tolist() != right.material_id.tolist():
        raise ValueError("NEXT44 feature identity coverage differs")
    return left.merge(right, on="material_id", how="inner", validate="one_to_one")


def search_rich_candidate(
    *,
    features: pd.DataFrame,
    material_ids: Sequence[str],
    endpoint: Sequence[float],
    split: Sequence[str],
    candidate_features: Sequence[str] = ALL_FEATURE_NAMES,
) -> dict[str, object]:
    """Reuse NEXT43's exact discovery-only finite catalogue and gates."""

    return search_development_candidate(
        features=features,
        material_ids=material_ids,
        endpoint=endpoint,
        split=split,
        candidate_features=candidate_features,
    )


def _validate_rich_table(
    rich_feature_path: Path, rich_manifest_path: Path
) -> pd.DataFrame:
    if rich_feature_path.name != RICH_FEATURE_FILE:
        raise ValueError("NEXT44 rich feature filename differs")
    manifest = _strict_json(rich_manifest_path, role="NEXT44 rich feature manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != RICH_FEATURE_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("dft_values_used") is not False
        or manifest.get("mlip_or_model_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(rich_feature_path.name) != _sha256(rich_feature_path)
    ):
        raise ValueError("NEXT44 rich feature table crossed the analytic boundary")
    return pd.read_parquet(rich_feature_path)


def _validate_next43_search(path: Path, manifest_path: Path) -> dict[str, object]:
    if path.name != NEXT43_SEARCH_NAME:
        raise ValueError("NEXT43 search filename differs")
    manifest = _strict_json(manifest_path, role="NEXT43 search manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != NEXT43_SEARCH_PROTOCOL
        or manifest.get("validation_labels_used_for_selection") is not False
        or manifest.get("law_execution_dft_values_read") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(path.name) != _sha256(path)
    ):
        raise ValueError("NEXT43 comparison search contract differs")
    return _strict_json(path, role="NEXT43 search result")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_rich_search(
    *,
    base_feature_path: Path,
    base_manifest_path: Path,
    rich_feature_path: Path,
    rich_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
    next43_search_path: Path,
    next43_search_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish one NEXT44 development candidate without opening new endpoints."""

    input_paths = {
        "next43_features": Path(base_feature_path).resolve(),
        "next43_feature_manifest": Path(base_manifest_path).resolve(),
        "next44_features": Path(rich_feature_path).resolve(),
        "next44_feature_manifest": Path(rich_manifest_path).resolve(),
        "development_evaluation": Path(evaluation_path).resolve(),
        "development_evaluation_manifest": Path(evaluation_manifest_path).resolve(),
        "next43_search": Path(next43_search_path).resolve(),
        "next43_search_manifest": Path(next43_search_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in input_paths.values()):
        raise FileNotFoundError("NEXT44 rich-search input is missing")
    if input_paths["next43_features"].name != BASE_FEATURE_FILE:
        raise ValueError("NEXT43 base feature filename differs")
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    base, evaluation = _validate_next43_inputs(
        feature_path=input_paths["next43_features"],
        feature_manifest_path=input_paths["next43_feature_manifest"],
        evaluation_path=input_paths["development_evaluation"],
        evaluation_manifest_path=input_paths["development_evaluation_manifest"],
    )
    rich = _validate_rich_table(
        input_paths["next44_features"], input_paths["next44_feature_manifest"]
    )
    previous = _validate_next43_search(
        input_paths["next43_search"], input_paths["next43_search_manifest"]
    )
    combined = combine_feature_tables(
        base,
        rich,
        base_features=BASE_FEATURE_NAMES,
        rich_features=RICH_FEATURE_NAMES,
    )
    material_ids = combined.material_id.astype(str).to_numpy()
    if material_ids.tolist() != evaluation.material_id.astype(str).tolist():
        raise ValueError("NEXT44 feature and development endpoint identity differs")
    endpoint = evaluation[ENDPOINT_COLUMN].to_numpy(float)
    split = deterministic_split(material_ids)
    result = search_rich_candidate(
        features=combined,
        material_ids=material_ids,
        endpoint=endpoint,
        split=split,
        candidate_features=ALL_FEATURE_NAMES,
    )
    masks = {
        "discovery": split == "discovery",
        "validation": split == "validation",
        "full_development": np.ones(len(split), dtype=bool),
    }
    previous_formula = previous.get("selected_formula")
    if not isinstance(previous_formula, Mapping):
        raise ValueError("NEXT43 selected formula is missing")
    previous_score, previous_supported, previous_reject = apply_formula(
        combined, previous_formula
    )
    previous_metrics = {
        role: _decision_metrics(
            supported=previous_supported[mask],
            reject=previous_reject[mask],
            endpoint=endpoint[mask],
        )
        for role, mask in masks.items()
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
            "candidate_feature_count": len(ALL_FEATURE_NAMES),
            "next43_candidate_feature_count": len(BASE_FEATURE_NAMES),
            "next44_added_feature_count": len(RICH_FEATURE_NAMES),
            "split_counts": {role: int(mask.sum()) for role, mask in masks.items()},
            "primary_gates": dict(PRIMARY_GATES),
            "next43_selected_formula_recomputed_metrics": previous_metrics,
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
            "next43_score": previous_score,
            "next43_supported": previous_supported,
            "next43_reject": previous_reject,
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
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "rich finite analytic formula development, not confirmation",
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
                "src/next43_finite_law_search.py": _sha256(
                    repository / "src/next43_finite_law_search.py"
                ),
                "src/next44_rich_analytic_features.py": _sha256(
                    repository / "src/next44_rich_analytic_features.py"
                ),
                "src/next44_rich_law_search.py": _sha256(
                    repository / "src/next44_rich_law_search.py"
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
            raise RuntimeError("NEXT44 input changed during rich search")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next43-features", required=True, type=Path)
    parser.add_argument("--next43-feature-manifest", required=True, type=Path)
    parser.add_argument("--next44-features", required=True, type=Path)
    parser.add_argument("--next44-feature-manifest", required=True, type=Path)
    parser.add_argument("--development-evaluation", required=True, type=Path)
    parser.add_argument("--development-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--next43-search", required=True, type=Path)
    parser.add_argument("--next43-search-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_rich_search(
        base_feature_path=args.next43_features,
        base_manifest_path=args.next43_feature_manifest,
        rich_feature_path=args.next44_features,
        rich_manifest_path=args.next44_feature_manifest,
        evaluation_path=args.development_evaluation,
        evaluation_manifest_path=args.development_evaluation_manifest,
        next43_search_path=args.next43_search,
        next43_search_manifest_path=args.next43_search_manifest,
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
