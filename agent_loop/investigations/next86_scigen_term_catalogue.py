"""Freeze a finite physics-directed SCIGEN term catalogue before labels open."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next12_dft_queue import _json_bytes
from src.next13d_acsc_dft_pairs import _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as UPSTREAM_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as UPSTREAM_MANIFEST_NAME,
    PROTOCOL as UPSTREAM_PROTOCOL,
)


PROTOCOL = "2026-08-03-next86-scigen-physics-directed-term-catalogue-v1"
CATALOGUE_NAME = "SCIGEN_TERM_CATALOGUE.json"
MANIFEST_NAME = "MANIFEST.json"
MIN_COVERAGE = 0.90
MIN_UNIQUE = 16
MIN_SCALE = 1.0e-12
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "feature_catalogue": "f34b09a4a9f18b0202b8daf606b7baab7bdae826871bcc60a4be858a8c1cc96a",
    "features_discovery": "7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16",
    "features_internal_validation": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "features_internal_replication": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "amendment": "06a6f0f7c3c605746907d4e7cfbdc4e5e3e7170e9198483ec81cfc3ec2eee8a4",
}


def _specs(
    group: str,
    direction: int,
    transform: str,
    names: Sequence[str],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "feature": name,
            "group": group,
            "direction": direction,
            "transform": transform,
        }
        for name in names
    )


PRESPECIFIED_TERMS = (
    *_specs("short_contact", -1, "log1p_nonnegative", ("cov_q01", "cov_q05")),
    *_specs(
        "short_contact",
        1,
        "log1p_nonnegative",
        (
            "cov_contact085_pa",
            "cov_overlap2_pa",
            "cov_site_overlap_q95",
            "cov_site_overlap_max",
            "cov_coord100_zero_fraction",
            "cov_coord110_zero_fraction",
        ),
    ),
    *_specs(
        "cell_crowding",
        1,
        "log1p_nonnegative",
        (
            "geom_cell_length_ratio",
            "geom_cell_angle_dev",
            "geom_covalent_packing",
            "geom_vdw_packing",
        ),
    ),
    *_specs(
        "symmetry_recovery",
        1,
        "log1p_nonnegative",
        (
            "sym_recovery_onset_rel",
            "sym_recovery_gain_log2",
            "sym_orbit_collapse",
            "sym_recovery_residual_rms_rel",
            "sym_recovery_residual_q95_rel",
            "sym_recovery_residual_max_rel",
        ),
    ),
    *_specs(
        "directional_sterics",
        1,
        "log1p_nonnegative",
        (
            "steric_rep12_pa",
            "steric_rep12_site_q95",
            "steric_rep12_site_max",
            "steric_rep12_vector_rms",
            "steric_rep12_vector_q95",
            "steric_rep12_vector_max",
            "steric_rep12_tensor_deviator",
            "steric_overlap2_vector_rms",
            "steric_overlap2_vector_q95",
            "steric_overlap2_tensor_deviator",
        ),
    ),
    *_specs(
        "valence_rigidity",
        1,
        "log1p_nonnegative",
        (
            "sivr_edge_mismatch_q95",
            "sivr_edge_mismatch_rms",
            "sivr_edge_mismatch_max",
            "sivr_site_imbalance_rms",
            "sivr_site_imbalance_max",
            "sivr_cell_anisotropy",
            "sivr_cell_hydro_abs",
            "sivr_negative_mode_fraction",
            "sivr_soft_mode_fraction",
        ),
    ),
    *_specs("valence_rigidity", -1, "asinh", ("sivr_stiffness_min",)),
    *_specs(
        "bond_valence_equilibrium",
        1,
        "log1p_nonnegative",
        (
            "scbv_mismatch_q95",
            "scbv_mismatch_rms",
            "scbv_mismatch_max",
            "scbv_cation_mismatch_rms",
            "scbv_anion_mismatch_rms",
            "scbv_vector_asymmetry_rms",
            "scbv_vector_asymmetry_max",
            "scbv_isolated_site_fraction",
            "scbv_parameter_generic_fraction",
        ),
    ),
    *_specs(
        "bond_valence_equilibrium",
        -1,
        "log1p_nonnegative",
        ("scbv_effective_cn_min",),
    ),
    *_specs(
        "valence_transport",
        1,
        "log1p_nonnegative",
        tuple(
            f"vt_a{alpha}_{metric}"
            for alpha in (0, 2, 4, 6)
            for metric in ("overload", "reallocation", "anion_mismatch_max")
        ),
    ),
    *_specs(
        "analytic_electrostatics",
        1,
        "log1p_nonnegative",
        (
            "aefi_residual_rms",
            "aefi_residual_q95",
            "aefi_residual_max",
            "aefi_field_tensor_deviator",
        ),
    ),
    *_specs(
        "coulomb_steric_balance",
        1,
        "log1p_nonnegative",
        (
            "acsb_opposition_deficit",
            "acsb_global_residual",
            "acsb_site_residual_rms",
            "acsb_site_residual_q95",
            "acsb_site_residual_max",
            "acsb_site_direction_deficit_q95",
            "acsb_active_disagreement_fraction",
        ),
    ),
    *_specs("madelung_site_heterogeneity", 1, "log1p_nonnegative", ("nm_site_spread",)),
    *_specs("self_stress", -1, "log1p_nonnegative", ("sscp_load_fraction",)),
    *_specs(
        "self_stress",
        1,
        "log1p_nonnegative",
        ("sscp_load_rms", "sscp_load_q95", "sscp_load_localization"),
    ),
    *_specs(
        "bond_valence_transport_compatibility",
        1,
        "log1p_nonnegative",
        ("bvtc_correction_rms", "bvtc_incompatible_rms", "bvtc_incompatible_fraction"),
    ),
    *_specs(
        "repulsive_load_resolvability",
        1,
        "log1p_nonnegative",
        (
            "prlr_residual_fraction",
            "prlr_atomic_residual_fraction",
            "prlr_cell_residual_fraction",
            "prlr_site_residual_q95",
            "prlr_bar_stress_rms",
            "prlr_bar_stress_amplification",
            "prlr_bar_stress_localization",
            "prlr_contact_weight_rms",
            "prlr_contact_weight_max",
            "prlr_contact_active_site_fraction",
            "prlr_risk",
        ),
    ),
)


def _read_json(path: Path, *, role: str) -> Mapping[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p_nonnegative":
        if np.any(values < -1.0e-12):
            raise ValueError("negative value under nonnegative transform")
        return np.log1p(np.maximum(values, 0.0))
    if transform == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"unknown term transform: {transform}")


def freeze_scigen_term_catalogue(
    *,
    feature_dir: Path,
    amendment_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Freeze eligible terms from discovery x0 support and robust scale only."""

    feature_root = Path(feature_dir).resolve()
    amendment = Path(amendment_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "feature_manifest": feature_root / UPSTREAM_MANIFEST_NAME,
        "feature_catalogue": feature_root / UPSTREAM_CATALOGUE_NAME,
        **{
            f"features_{role}": feature_root / FEATURE_NAMES[role]
            for role in FEATURE_NAMES
        },
        "amendment": amendment,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT86 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT86 formal input identity differs")
    upstream = _read_json(paths["feature_manifest"], role="NEXT85 manifest")
    outputs = upstream.get("outputs_sha256")
    if (
        upstream.get("protocol") != UPSTREAM_PROTOCOL
        or upstream.get("labels_opened") is not False
        or upstream.get("endpoint_payloads_opened") is not False
        or upstream.get("relaxed_structures_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(UPSTREAM_CATALOGUE_NAME) != hashes["feature_catalogue"]
        or any(
            outputs.get(FEATURE_NAMES[role]) != hashes[f"features_{role}"]
            for role in FEATURE_NAMES
        )
    ):
        raise ValueError("NEXT85 label-free feature provenance differs")
    upstream_catalogue = _read_json(
        paths["feature_catalogue"], role="NEXT85 feature catalogue"
    )
    upstream_names = upstream_catalogue.get("feature_names")
    if not isinstance(upstream_names, list):
        raise ValueError("NEXT85 feature catalogue lacks names")
    prespecified_names = [str(spec["feature"]) for spec in PRESPECIFIED_TERMS]
    if len(prespecified_names) != len(set(prespecified_names)):
        raise RuntimeError("NEXT86 prespecified feature names are not unique")
    if set(prespecified_names) - set(map(str, upstream_names)):
        raise ValueError("NEXT86 prespecified feature is absent upstream")

    discovery = pd.read_parquet(paths["features_discovery"])
    if set(discovery.get("partition_role", [])) != {"discovery"}:
        raise ValueError("NEXT86 received a non-discovery feature partition")
    eligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for spec in PRESPECIFIED_TERMS:
        name = str(spec["feature"])
        raw = pd.to_numeric(discovery[name], errors="coerce").to_numpy(float)
        finite = raw[np.isfinite(raw)]
        coverage = len(finite) / len(raw) if len(raw) else 0.0
        record = {**spec, "finite_rows": len(finite), "coverage": coverage}
        if coverage < MIN_COVERAGE:
            excluded.append({**record, "reason": "finite_coverage_below_0.90"})
            continue
        try:
            transformed = _transform(finite, str(spec["transform"]))
        except ValueError as exc:
            excluded.append({**record, "reason": str(exc)})
            continue
        unique = int(len(np.unique(transformed)))
        record["unique_transformed_values"] = unique
        if unique < MIN_UNIQUE:
            excluded.append({**record, "reason": "fewer_than_16_unique_values"})
            continue
        q10, center, q90 = np.quantile(transformed, [0.10, 0.50, 0.90])
        scale = float((q90 - q10) / 2.0)
        if not math.isfinite(scale) or scale <= MIN_SCALE:
            excluded.append({**record, "reason": "robust_scale_not_positive"})
            continue
        eligible.append(
            {
                **record,
                "term_id": f"{name}__{'high' if int(spec['direction']) == 1 else 'low'}",
                "center": float(center),
                "scale": scale,
                "transformed_q10": float(q10),
                "transformed_q90": float(q90),
            }
        )

    catalogue = {
        "protocol": PROTOCOL,
        "status": "term_catalogue_frozen_before_any_scigen_endpoint_opening",
        "formula_family": "nonnegative sum of at most three one-sided robust hinge terms",
        "eligibility": {
            "minimum_discovery_finite_coverage": MIN_COVERAGE,
            "minimum_unique_transformed_values": MIN_UNIQUE,
            "minimum_robust_scale": MIN_SCALE,
            "center": "transformed discovery median",
            "scale": "(transformed discovery q90 - q10) / 2",
        },
        "eligible_terms": eligible,
        "excluded_terms": excluded,
        "counts": {
            "discovery_rows": len(discovery),
            "prespecified_terms": len(PRESPECIFIED_TERMS),
            "eligible_terms": len(eligible),
            "excluded_terms": len(excluded),
        },
        "identity_features_allowed": False,
        "validation_or_replication_values_used_for_term_statistics": False,
        "labels_opened": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
    }
    source = Path(__file__).resolve()
    source_hash = _sha256_file(source)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "label_free_discovery_x0_term_catalogue_freeze",
        "labels_opened": False,
        "endpoint_payloads_opened": False,
        "relaxed_structures_opened": False,
        "validation_or_replication_feature_values_read": False,
        "counts": catalogue["counts"],
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next86_scigen_term_catalogue.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        catalogue_path = staging / CATALOGUE_NAME
        catalogue_path.write_bytes(_json_bytes(catalogue))
        manifest["outputs_sha256"] = {CATALOGUE_NAME: _sha256_file(catalogue_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT86 input changed before publication")
        if _sha256_file(source) != source_hash:
            raise RuntimeError("NEXT86 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    freeze_scigen_term_catalogue(
        feature_dir=args.feature_dir,
        amendment_path=args.amendment,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CATALOGUE_NAME",
    "MANIFEST_NAME",
    "PRESPECIFIED_TERMS",
    "PROTOCOL",
    "freeze_scigen_term_catalogue",
]
