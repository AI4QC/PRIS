#!/usr/bin/env python3
"""Add sealed local-motif descriptors to the fixed NEXT43 finite search."""

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
from src.next43_analytic_feature_bank import CANDIDATE_FEATURE_NAMES as BASE_FEATURE_NAMES
from src.next43_finite_law_search import (
    _baseline_metrics,
    _validate_inputs as _validate_next43_inputs,
    apply_formula,
    deterministic_split,
    search_development_candidate,
)
from src.next44_rich_analytic_features import CANDIDATE_FEATURE_NAMES as RICH_FEATURE_NAMES
from src.next44_rich_law_search import _validate_rich_table, combine_feature_tables
from src.next45_exhaustive_rectangles import _validate_next44_search
from src.next46_motif_coherence_features import (
    FEATURE_NAME as MOTIF_FEATURE_FILE,
    FEATURE_NAMES as MOTIF_FEATURE_NAMES,
    PROTOCOL as MOTIF_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next46-motif-finite-analytic-law-search-v1"
FORMULA_NAME = "NEXT46_DEVELOPMENT_CANDIDATE.json"
SEARCH_NAME = "NEXT46_FINITE_SEARCH.json"
PREDICTION_NAME = "next46_development_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
ALL_FEATURE_NAMES = (
    tuple(BASE_FEATURE_NAMES) + tuple(RICH_FEATURE_NAMES) + tuple(MOTIF_FEATURE_NAMES)
)


if len(ALL_FEATURE_NAMES) != len(set(ALL_FEATURE_NAMES)):
    raise RuntimeError("NEXT46 feature namespaces overlap")


def combine_three_feature_tables(
    base: pd.DataFrame,
    rich: pd.DataFrame,
    motif: pd.DataFrame,
    *,
    base_features: Sequence[str],
    rich_features: Sequence[str],
    motif_features: Sequence[str],
) -> pd.DataFrame:
    """Join three label-free tables with exact identity and disjoint schemas."""

    first = combine_feature_tables(
        base,
        rich,
        base_features=base_features,
        rich_features=rich_features,
    )
    return combine_feature_tables(
        first,
        motif,
        base_features=tuple(base_features) + tuple(rich_features),
        rich_features=motif_features,
    )


def search_motif_candidate(
    *,
    features: pd.DataFrame,
    material_ids: Sequence[str],
    endpoint: Sequence[float],
    split: Sequence[str],
    candidate_features: Sequence[str] = ALL_FEATURE_NAMES,
) -> dict[str, object]:
    return search_development_candidate(
        features=features,
        material_ids=material_ids,
        endpoint=endpoint,
        split=split,
        candidate_features=candidate_features,
    )


def _validate_motif_table(path: Path, manifest_path: Path) -> pd.DataFrame:
    if path.name != MOTIF_FEATURE_FILE:
        raise ValueError("NEXT46 motif feature filename differs")
    manifest = _strict_json(manifest_path, role="NEXT46 motif feature manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != MOTIF_FEATURE_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("dft_values_used") is not False
        or manifest.get("mlip_or_model_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(path.name) != _sha256(path)
    ):
        raise ValueError("NEXT46 motif table crossed the analytic boundary")
    return pd.read_parquet(path)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_motif_search(
    *,
    base_feature_path: Path,
    base_manifest_path: Path,
    rich_feature_path: Path,
    rich_manifest_path: Path,
    motif_feature_path: Path,
    motif_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
    next44_search_path: Path,
    next44_search_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish one motif-augmented development candidate."""

    input_paths = {
        "next43_features": Path(base_feature_path).resolve(),
        "next43_feature_manifest": Path(base_manifest_path).resolve(),
        "next44_features": Path(rich_feature_path).resolve(),
        "next44_feature_manifest": Path(rich_manifest_path).resolve(),
        "next46_motif_features": Path(motif_feature_path).resolve(),
        "next46_motif_manifest": Path(motif_manifest_path).resolve(),
        "development_evaluation": Path(evaluation_path).resolve(),
        "development_evaluation_manifest": Path(evaluation_manifest_path).resolve(),
        "next44_search": Path(next44_search_path).resolve(),
        "next44_search_manifest": Path(next44_search_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in input_paths.values()):
        raise FileNotFoundError("NEXT46 motif-search input is missing")
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
    motif = _validate_motif_table(
        input_paths["next46_motif_features"], input_paths["next46_motif_manifest"]
    )
    previous = _validate_next44_search(
        input_paths["next44_search"], input_paths["next44_search_manifest"]
    )
    combined = combine_three_feature_tables(
        base,
        rich,
        motif,
        base_features=BASE_FEATURE_NAMES,
        rich_features=RICH_FEATURE_NAMES,
        motif_features=MOTIF_FEATURE_NAMES,
    )
    material_ids = combined.material_id.astype(str).to_numpy()
    if material_ids.tolist() != evaluation.material_id.astype(str).tolist():
        raise ValueError("NEXT46 feature and endpoint identity differs")
    endpoint = evaluation[ENDPOINT_COLUMN].to_numpy(float)
    split = deterministic_split(material_ids)
    result = search_motif_candidate(
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
        raise ValueError("NEXT44 selected formula is missing")
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
    formula_document = {
        "protocol": PROTOCOL,
        "role": "motif-augmented development candidate; not frozen for unseen confirmation",
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
            "motif_added_feature_count": len(MOTIF_FEATURE_NAMES),
            "split_counts": {role: int(mask.sum()) for role, mask in masks.items()},
            "primary_gates": dict(PRIMARY_GATES),
            "next44_selected_formula_recomputed_metrics": previous_metrics,
            "frozen_baselines": _baseline_metrics(evaluation, masks),
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
            "next44_score": previous_score,
            "next44_supported": previous_supported,
            "next44_reject": previous_reject,
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
            "evidence_role": "motif finite analytic formula development, not confirmation",
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
            "inputs_sha256": {role: _sha256(path) for role, path in input_paths.items()},
            "executed_source_sha256": {
                "src/next43_finite_law_search.py": _sha256(
                    repository / "src/next43_finite_law_search.py"
                ),
                "src/next46_motif_coherence_features.py": _sha256(
                    repository / "src/next46_motif_coherence_features.py"
                ),
                "src/next46_motif_law_search.py": _sha256(
                    repository / "src/next46_motif_law_search.py"
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
            raise RuntimeError("NEXT46 input changed during motif search")
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
    parser.add_argument("--next46-motif-features", required=True, type=Path)
    parser.add_argument("--next46-motif-manifest", required=True, type=Path)
    parser.add_argument("--development-evaluation", required=True, type=Path)
    parser.add_argument("--development-evaluation-manifest", required=True, type=Path)
    parser.add_argument("--next44-search", required=True, type=Path)
    parser.add_argument("--next44-search-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_motif_search(
        base_feature_path=args.next43_features,
        base_manifest_path=args.next43_feature_manifest,
        rich_feature_path=args.next44_features,
        rich_manifest_path=args.next44_feature_manifest,
        motif_feature_path=args.next46_motif_features,
        motif_manifest_path=args.next46_motif_manifest,
        evaluation_path=args.development_evaluation,
        evaluation_manifest_path=args.development_evaluation_manifest,
        next44_search_path=args.next44_search,
        next44_search_manifest_path=args.next44_search_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passes_both_internal_splits": manifest["passes_both_internal_splits"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
