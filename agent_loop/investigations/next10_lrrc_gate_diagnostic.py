"""Fixed posthoc LRRC diagnostic on the exposed next8 development gate.

This module never fits a threshold and never accepts a candidate catalog from
the caller.  It verifies and replays the frozen next8 M5/AGREE995 rules before
evaluating the predeclared LRRC/Quota-CRC catalog.  The result remains an
exploratory diagnostic and cannot carry a scientific-improvement claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.next6_elementa_diagnostics import paired_cluster_bootstrap
from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    evaluate_group_triage,
)
from src.next8_mattersim_committee_protocol import (
    DEVELOPMENT_FREEZE_PROTOCOL,
    FORMULA_NAMES,
    THRESHOLD_SPLIT_SALT,
    TRACKS,
    _validated_final_thresholds,
    construct_committee_scores,
    derive_disagreement_cutoffs,
    passes_safety_gate,
    serialize_formula_catalog,
)
from src.next9_lrrc import (
    STEP_FRACTION,
    Decision,
    LRRCResult,
    LRRCStatus,
    QuotaCRCRow,
    compose_decision,
    lrrc_negative_gate,
    quota_crc,
)
from src.next9_lrrc_synthetic import _rename_noreplace


PROTOCOL = "2026-08-01-next10-lrrc-fixed-gate-diagnostic-v1"
LRRC_FEATURE_PROTOCOL = "2026-08-01-next10-lrrc-mattersim-features-v1"
EVALUATION_ROLE = "posthoc_exploratory_diagnostic"
BOOTSTRAP_SEED = 20260801
PRODUCTION_BOOTSTRAP_RESAMPLES = 20_000

FIXED_FORMULA_ORDER = (
    "M5",
    "AGREE995",
    "M5_LRRC_OR",
    "M5_LRRC_QCRC",
    "AGREE995_LRRC_QCRC",
)
TRACK_ORDER = ("primary", "comparator")
LRRC_FEATURE_COLUMNS = (
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "strict_x0_ok",
    "natoms",
    "lrrc_status",
    "lrrc_negative",
    "d_star_angstrom",
    "h_angstrom",
    "kappa_h_ev_per_a2",
    "kappa_h2_ev_per_a2",
    "kappa_r_ev_per_a2",
    "error_proxy_ev_per_a2",
    "u_num_ev_per_a2",
    "force_call_count",
    "error",
)
OUTPUT_NAMES = (
    "predictions.parquet",
    "metrics.parquet",
    "PAIRED_BOOTSTRAP.json",
    "FROZEN_CATALOG.json",
    "MANIFEST.json",
)
_NEXT8_ARTIFACT_NAMES = frozenset(
    {
        "threshold_role_assignments.parquet",
        "development_frontier.parquet",
        "threshold_fit_rules.parquet",
        "development_gate_metrics.parquet",
        "PAIRED_BOOTSTRAP.json",
        "IMPROVEMENT_GATE.json",
    }
)
_BASE_METHOD = {"M5": "m5_baseline", "AGREE995": "selected_candidate"}
_LRRC_COMPARISONS = (
    ("M5", "M5_LRRC_OR"),
    ("M5", "M5_LRRC_QCRC"),
    ("AGREE995", "AGREE995_LRRC_QCRC"),
)
_NUMERIC_LRRC_COLUMNS = (
    "d_star_angstrom",
    "h_angstrom",
    "kappa_h_ev_per_a2",
    "kappa_h2_ev_per_a2",
    "kappa_r_ev_per_a2",
    "error_proxy_ev_per_a2",
    "u_num_ev_per_a2",
)
_LRRC_EXECUTED_SOURCE_RELATIVE = (
    "src/next10_lrrc_mattersim_features.py",
    "src/next9_lrrc.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
)
_FORMAL_EXECUTED_SOURCE_RELATIVE = (
    "src/next10_lrrc_gate_diagnostic.py",
    "src/next9_lrrc.py",
    "src/next9_lrrc_synthetic.py",
    "src/next8_mattersim_committee_protocol.py",
    "src/next6_elementa_protocol.py",
    "src/next6_elementa_diagnostics.py",
)
_FORMAL_BOOTSTRAP_PARAMETERS = (
    PRODUCTION_BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    1_000,
)
_FORMAL_PRODUCER_REACHABLE_LRRC_STATUSES = frozenset(
    {
        LRRCStatus.OK,
        LRRCStatus.STATIONARY_FALLBACK,
        LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
        LRRCStatus.ABSTAIN_NUMERICAL_FAILURE,
    }
)


class _DuplicateJSONKeyError(ValueError):
    """Internal marker for a repeated key anywhere in a JSON document."""


def validate_catalog_order(order: Sequence[str]) -> tuple[str, ...]:
    """Reject any catalog that differs from the predeclared fixed order."""

    if isinstance(order, (str, bytes)):
        raise ValueError("fixed catalog order must be a sequence of names")
    observed = tuple(order)
    if observed != FIXED_FORMULA_ORDER:
        raise ValueError("candidate catalog differs from the frozen next10 catalog")
    return observed


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_cuda_device(value: object) -> bool:
    if type(value) is not str:
        return False
    normalized = value.lower()
    if not normalized.startswith("cuda:"):
        return False
    index = normalized.removeprefix("cuda:")
    return bool(
        index.isdigit()
        and str(int(index)) == index
        and normalized == value
    )


def _strict_json_bytes(data: bytes, *, role: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKeyError(key)
            result[key] = value
        return result

    try:
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
            object_pairs_hook=unique_object,
        )
    except _DuplicateJSONKeyError as exc:
        raise ValueError(f"{role} contains duplicate key: {exc}") from exc
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON for {role}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object")
    return payload


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    raise TypeError(f"value is not strict-JSON serializable: {type(value)!r}")


def _strict_json_text(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            _json_safe(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _mapping(value: object, *, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _validate_string_key_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, role: str
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"{role} is missing {column}")
        if not frame[column].map(lambda value: type(value) is str and bool(value)).all():
            raise ValueError(f"{role} {column} must contain nonempty exact strings")


def _validate_roles(
    committee_features: pd.DataFrame,
    threshold_roles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(committee_features, pd.DataFrame):
        raise TypeError("committee_features must be a pandas DataFrame")
    if not isinstance(threshold_roles, pd.DataFrame):
        raise TypeError("threshold_roles must be a pandas DataFrame")
    _validate_string_key_columns(
        committee_features, ("sid", "rk", "stage"), role="committee features"
    )
    _validate_string_key_columns(
        threshold_roles,
        ("sid", "rk", "stage", "threshold_role", "split_salt"),
        role="threshold roles",
    )
    if committee_features["sid"].duplicated().any():
        raise ValueError("committee features contain duplicate sid values")
    if threshold_roles["sid"].duplicated().any():
        raise ValueError("threshold roles contain duplicate sid values")
    observed_stages = set(committee_features["stage"])
    expected_stages = {
        "search_calibration",
        "formula_selection",
        "threshold_calibration",
    }
    if observed_stages != expected_stages:
        raise ValueError("committee features must contain exactly the development stages")
    if set(threshold_roles["stage"]) != {"threshold_calibration"}:
        raise ValueError("threshold roles must contain only threshold_calibration rows")
    if set(threshold_roles["threshold_role"]) != {
        "threshold_fit",
        "development_gate",
    }:
        raise ValueError("threshold roles must contain exactly fit and gate assignments")
    if set(threshold_roles["split_salt"]) != {THRESHOLD_SPLIT_SALT}:
        raise ValueError("threshold role split salt differs from the frozen protocol")
    roles_per_group = threshold_roles.groupby("rk", sort=False)[
        "threshold_role"
    ].nunique()
    if (roles_per_group != 1).any():
        raise ValueError("a composition group spans threshold roles")

    threshold_features = committee_features.loc[
        committee_features["stage"].eq("threshold_calibration")
    ].copy()
    feature_keys = set(
        zip(
            threshold_features["sid"],
            threshold_features["rk"],
            threshold_features["stage"],
            strict=True,
        )
    )
    role_keys = set(
        zip(
            threshold_roles["sid"],
            threshold_roles["rk"],
            threshold_roles["stage"],
            strict=True,
        )
    )
    if feature_keys != role_keys:
        raise ValueError("threshold role assignments do not exactly cover feature rows")
    joined = threshold_features.merge(
        threshold_roles[["sid", "rk", "stage", "threshold_role"]],
        on=["sid", "rk", "stage"],
        how="inner",
        validate="one_to_one",
    )
    gate = joined.loc[joined["threshold_role"].eq("development_gate")].copy()
    if gate.empty:
        raise ValueError("development gate is empty")
    return joined, gate


def _validate_frozen_protocol(
    committee_features: pd.DataFrame,
    threshold_roles: pd.DataFrame,
    frozen_protocol: Mapping[str, object],
    *,
    feature_sha256: str | None = None,
    feature_manifest_sha256: str | None = None,
    artifact_sha256: Mapping[str, str] | None = None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    frozen = _mapping(frozen_protocol, role="frozen protocol")
    if frozen.get("protocol") != DEVELOPMENT_FREEZE_PROTOCOL:
        raise ValueError("frozen protocol identifier mismatch")
    if frozen.get("state") != "frozen":
        raise ValueError("next8 protocol is not frozen")

    _, gate = _validate_roles(committee_features, threshold_roles)
    split = _mapping(frozen.get("split"), role="frozen split")
    if split.get("salt") != THRESHOLD_SPLIT_SALT:
        raise ValueError("frozen split salt mismatch")
    if split.get("ordering") != "sha256(salt+'\\0'+rk),rk":
        raise ValueError("frozen split ordering mismatch")
    observed_counts = {
        "threshold_fit_groups": int(
            threshold_roles.loc[
                threshold_roles["threshold_role"].eq("threshold_fit"), "rk"
            ].nunique()
        ),
        "development_gate_groups": int(gate["rk"].nunique()),
    }
    for name, value in observed_counts.items():
        if type(split.get(name)) is not int or split.get(name) != value:
            raise ValueError(f"frozen split {name} mismatch")

    selection = _mapping(frozen.get("selection"), role="frozen selection")
    expected_selection = {
        "state": "selected",
        "name": "AGREE995",
        "catalog_order": FORMULA_NAMES.index("AGREE995"),
        "complexity": 3,
        "cost": 6,
    }
    for name, expected in expected_selection.items():
        if selection.get(name) != expected:
            raise ValueError("frozen selection is not the predeclared AGREE995 rule")

    tracks = _mapping(frozen.get("tracks"), role="frozen tracks")
    if set(tracks) != set(TRACK_ORDER):
        raise ValueError("frozen tracks differ from the exact two-track catalog")
    for name in TRACK_ORDER:
        observed = _mapping(tracks[name], role=f"frozen track {name}")
        spec = TRACKS[name]
        expected = {
            "alpha": spec.alpha,
            "protected": spec.protected_column,
            "protected_ev_per_atom": spec.protected_ev_per_atom,
            "within_group": spec.within_group,
        }
        if dict(observed) != expected:
            raise ValueError(f"frozen {name} track semantics mismatch")

    search = committee_features.loc[
        committee_features["stage"].eq("search_calibration")
    ].copy()
    cutoffs = derive_disagreement_cutoffs(search)
    serialized = serialize_formula_catalog(cutoffs)
    serialized_sha256 = _sha256_bytes(serialized.encode("utf-8"))
    catalog = _mapping(frozen.get("catalog"), role="frozen formula catalog")
    if catalog.get("serialized") != serialized:
        raise ValueError("frozen formula catalog does not reproduce from features")
    if catalog.get("sha256") != serialized_sha256:
        raise ValueError("frozen formula catalog hash mismatch")
    catalog_document = _strict_json_bytes(
        serialized.encode("utf-8"), role="serialized frozen formula catalog"
    )
    formula_documents = catalog_document.get("formulas")
    if not isinstance(formula_documents, list) or [
        item.get("name") if isinstance(item, Mapping) else None
        for item in formula_documents
    ] != list(FORMULA_NAMES):
        raise ValueError("frozen next8 formula catalog order mismatch")

    cutoff_document = _mapping(frozen.get("cutoffs"), role="frozen cutoffs")
    expected_cutoffs = {
        "q99_ev_per_atom": cutoffs.q99_ev_per_atom,
        "q995_ev_per_atom": cutoffs.q995_ev_per_atom,
        "q995_force_ev_per_a": cutoffs.q995_force_ev_per_a,
        "eligible_row_count": cutoffs.eligible_row_count,
        "source_stage": cutoffs.source_stage,
        "quantile_method": cutoffs.quantile_method,
        "calibration_fingerprint_sha256": cutoffs.calibration_fingerprint_sha256,
    }
    if dict(cutoff_document) != expected_cutoffs:
        raise ValueError("frozen disagreement cutoffs do not reproduce exactly")

    provenance = _mapping(
        frozen.get("cutoff_provenance"), role="frozen cutoff provenance"
    )
    if provenance.get("catalog_serialization_sha256") != serialized_sha256:
        raise ValueError("frozen cutoff catalog provenance mismatch")
    repository_root = Path(__file__).resolve().parents[1]
    protocol_source_sha256 = _sha256_file(
        repository_root / "src/next8_mattersim_committee_protocol.py"
    )
    if provenance.get("protocol_code_sha256") != protocol_source_sha256:
        raise ValueError("frozen next8 protocol source hash mismatch")
    for name, supplied in (
        ("feature_sha256", feature_sha256),
        ("feature_manifest_sha256", feature_manifest_sha256),
    ):
        recorded = provenance.get(name)
        if not _is_sha256(recorded):
            raise ValueError(f"invalid frozen provenance hash: {name}")
        if supplied is not None and recorded != supplied:
            raise ValueError(f"frozen provenance hash mismatch: {name}")

    raw_rules = frozen.get("final_rules")
    if not isinstance(raw_rules, list) or len(raw_rules) != 4:
        raise ValueError("frozen final rules must contain exactly four rows")
    rules = _validated_final_thresholds(
        pd.DataFrame(raw_rules), selection=selection
    ).copy()
    expected_pairs = {
        (track, formula) for track in TRACK_ORDER for formula in ("M5", "AGREE995")
    }
    observed_pairs = set(zip(rules["track"], rules["formula"], strict=True))
    if observed_pairs != expected_pairs:
        raise ValueError("frozen threshold formula/track grid mismatch")
    for row in rules.to_dict("records"):
        spec = TRACKS[str(row["track"])]
        threshold = row["threshold"]
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise ValueError("frozen threshold must be a non-boolean real number")
        if not math.isfinite(float(threshold)):
            raise ValueError("next10 requires the finite frozen next8 thresholds")
        if (
            row["operator"] != "score > threshold"
            or row["threshold_source_role"] != "threshold_fit"
            or row["unsupported_decision"] != "ABSTAIN"
            or row.get("protected") != spec.protected_column
        ):
            raise ValueError("frozen threshold deployment semantics mismatch")

    development_hashes = _mapping(
        frozen.get("development_artifacts_sha256"),
        role="frozen development artifact hashes",
    )
    if set(development_hashes) != _NEXT8_ARTIFACT_NAMES:
        raise ValueError("frozen development artifact hash closure mismatch")
    if not all(_is_sha256(value) for value in development_hashes.values()):
        raise ValueError("invalid frozen development artifact hash")
    if artifact_sha256 is not None and dict(development_hashes) != dict(
        artifact_sha256
    ):
        raise ValueError("frozen development artifact hashes mismatch")
    return cutoffs, rules, gate


def _nullable_bool(value: object, *, role: str) -> bool | None:
    if value is None or value is pd.NA or (
        isinstance(value, (float, np.floating)) and np.isnan(float(value))
    ):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{role} must be boolean or missing")


def _is_missing_scalar(value: object) -> bool:
    return bool(
        value is None
        or value is pd.NA
        or (
            isinstance(value, (float, np.floating))
            and np.isnan(float(value))
        )
    )


def _validate_lrrc_features(
    lrrc_features: pd.DataFrame,
    gate_features: pd.DataFrame,
    *,
    formal_producer: bool = False,
) -> pd.DataFrame:
    if not isinstance(lrrc_features, pd.DataFrame):
        raise TypeError("lrrc_features must be a pandas DataFrame")
    if tuple(lrrc_features.columns) != LRRC_FEATURE_COLUMNS:
        raise ValueError("LRRC feature columns/order differ from the frozen schema")
    work = lrrc_features.copy().reset_index(drop=True)
    _validate_string_key_columns(
        work,
        ("sid", "rk", "stage", "threshold_role", "lrrc_status"),
        role="LRRC features",
    )
    if work["sid"].duplicated().any():
        raise ValueError("LRRC features contain duplicate sid values")
    if set(work["threshold_role"]) != {"development_gate"}:
        raise ValueError("LRRC features must contain only development_gate rows")
    if set(work["stage"]) != {"threshold_calibration"}:
        raise ValueError("LRRC feature stage mismatch")
    gate_keys = set(zip(gate_features["sid"], gate_features["rk"], strict=True))
    lrrc_keys = set(zip(work["sid"], work["rk"], strict=True))
    if gate_keys != lrrc_keys:
        raise ValueError("LRRC features do not exactly cover the development gate")
    alignment = work.merge(
        gate_features[["sid", "rk", "stage", "strict_x0_ok", "natoms"]],
        on=["sid", "rk", "stage"],
        how="inner",
        suffixes=("", "_committee"),
        validate="one_to_one",
    )
    if len(alignment) != len(work):
        raise ValueError("LRRC and committee features are not one-to-one")
    for name in ("strict_x0_ok", "natoms"):
        if not alignment[name].equals(alignment[f"{name}_committee"]):
            raise ValueError(f"LRRC feature {name} mismatch")
    if not work["strict_x0_ok"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("LRRC strict_x0_ok values must be exact booleans")
    natoms = pd.to_numeric(work["natoms"], errors="coerce").to_numpy(float)
    if not np.isfinite(natoms).all() or (natoms < 0).any() or not np.equal(
        natoms, np.floor(natoms)
    ).all():
        raise ValueError("LRRC natoms values must be nonnegative integers")
    force_calls = pd.to_numeric(
        work["force_call_count"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(force_calls).all() or (force_calls < 0).any() or (
        force_calls > 5
    ).any() or not np.equal(force_calls, np.floor(force_calls)).all():
        raise ValueError("LRRC force_call_count must be an integer in [0, 5]")

    allowed_statuses = {status.value for status in LRRCStatus}
    if not set(work["lrrc_status"]) <= allowed_statuses:
        raise ValueError("LRRC features contain an unknown status")
    for index, row in work.iterrows():
        status = LRRCStatus(str(row["lrrc_status"]))
        if (
            formal_producer
            and status not in _FORMAL_PRODUCER_REACHABLE_LRRC_STATUSES
        ):
            raise ValueError(
                "formal LRRC row status is not producer-reachable: "
                f"{status.value}"
            )
        strict_x0_ok = bool(row["strict_x0_ok"])
        atom_count = int(natoms[index])
        if strict_x0_ok:
            if atom_count < 1:
                raise ValueError("strict LRRC rows require natoms >= 1")
        elif atom_count != 0:
            raise ValueError("nonstrict LRRC rows require natoms == 0")
        negative = _nullable_bool(row["lrrc_negative"], role="lrrc_negative")
        numerics = {
            name: pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            for name in _NUMERIC_LRRC_COLUMNS
        }
        error = row["error"]
        calls = int(force_calls[index])
        if status is LRRCStatus.OK:
            values = np.asarray(list(numerics.values()), dtype=float)
            if negative is None or not np.isfinite(values).all() or calls != 5:
                raise ValueError("successful LRRC rows must contain five finite diagnostics")
            d_star = float(numerics["d_star_angstrom"])
            h = float(numerics["h_angstrom"])
            kappa_h = float(numerics["kappa_h_ev_per_a2"])
            kappa_h2 = float(numerics["kappa_h2_ev_per_a2"])
            kappa_r = float(numerics["kappa_r_ev_per_a2"])
            error_proxy = float(numerics["error_proxy_ev_per_a2"])
            u_num = float(numerics["u_num_ev_per_a2"])
            if d_star <= 0.0 or not math.isclose(
                h, STEP_FRACTION * d_star, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("successful LRRC step does not match LRRC-v0")
            if error_proxy < 0.0 or not math.isclose(
                kappa_r,
                (4.0 * kappa_h2 - kappa_h) / 3.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                error_proxy,
                abs(kappa_h2 - kappa_h) / 3.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                u_num,
                kappa_r + error_proxy,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("successful LRRC diagnostics violate LRRC-v0")
            if negative != lrrc_negative_gate(kappa_h, kappa_h2, u_num):
                raise ValueError("LRRC negative flag does not match the frozen gate")
            if not (_is_missing_scalar(error) or type(error) is str and error == ""):
                raise ValueError("successful LRRC rows must not contain an error")
            work.at[index, "error"] = ""
        elif status is LRRCStatus.STATIONARY_FALLBACK:
            if negative is not None or calls != 1 or not (
                _is_missing_scalar(error) or type(error) is str and error == ""
            ):
                raise ValueError("stationary LRRC fallback semantics mismatch")
            if any(not _is_missing_scalar(row[name]) for name in _NUMERIC_LRRC_COLUMNS):
                raise ValueError("stationary LRRC rows must not contain diagnostics")
            work.at[index, "error"] = ""
        else:
            if not status.name.startswith("ABSTAIN_") or negative is not None:
                raise ValueError("invalid LRRC abstention semantics")
            if any(not _is_missing_scalar(row[name]) for name in _NUMERIC_LRRC_COLUMNS):
                raise ValueError("LRRC abstention rows must not contain diagnostics")
            if type(error) is not str or not error:
                raise ValueError("LRRC abstention rows require an error diagnostic")
            if formal_producer and not strict_x0_ok and error != "nonstrict_x0":
                raise ValueError(
                    "formal nonstrict LRRC rows require the producer nonstrict_x0 error"
                )
            allowed_calls = {
                LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY: {0, 1},
                LRRCStatus.ABSTAIN_FORCE_FAILURE: {0, 1, 2, 3, 4},
                LRRCStatus.ABSTAIN_INVALID_FORCE: {0, 1, 2, 3, 4},
                LRRCStatus.ABSTAIN_NUMERICAL_FAILURE: {1, 5},
            }[status]
            if calls not in allowed_calls:
                raise ValueError("LRRC abstention force-call count contradicts status")
        if not strict_x0_ok and (
            status is not LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY or calls != 0
        ):
            raise ValueError("nonstrict LRRC rows require the zero-call unsupported sentinel")
    return work.sort_values("sid", kind="stable", ignore_index=True)


def _prepare_labels(labels: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    required = {"sid", "rk", "stage", "e_per_atom"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")
    raw = labels.loc[:, ["sid", "rk", "stage", "e_per_atom"]].copy()
    _validate_string_key_columns(raw, ("sid", "rk", "stage"), role="labels")
    if raw["sid"].duplicated().any():
        raise ValueError("labels contain duplicate sid values")
    gate_sids = set(gate["sid"])
    selected = raw.loc[raw["sid"].isin(gate_sids)].copy()
    if set(selected["sid"]) != gate_sids:
        raise ValueError("labels do not exactly cover the development gate")
    expected_keys = set(zip(gate["sid"], gate["rk"], gate["stage"], strict=True))
    observed_keys = set(
        zip(selected["sid"], selected["rk"], selected["stage"], strict=True)
    )
    if observed_keys != expected_keys:
        raise ValueError("label sid/rk/stage keys differ from gate features")
    energy = pd.to_numeric(selected["e_per_atom"], errors="coerce").to_numpy(float)
    if not np.isfinite(energy).all():
        raise ValueError("gate labels must contain finite energies")
    selected["e_per_atom"] = energy
    return attach_energy_labels(selected).sort_values(
        "sid", kind="stable", ignore_index=True
    )


def _base_decision_rows(
    gate: pd.DataFrame,
    *,
    cutoffs: Any,
    rules: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    scores = construct_committee_scores(
        gate,
        cutoffs=cutoffs,
        expected_stage="threshold_calibration",
    )
    scores = scores.loc[scores["formula"].isin(("M5", "AGREE995"))].copy()
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for track_name in TRACK_ORDER:
        for formula in ("M5", "AGREE995"):
            rule = rules.loc[
                rules["track"].eq(track_name) & rules["formula"].eq(formula)
            ].iloc[0]
            rows = scores.loc[scores["formula"].eq(formula)].copy()
            rows["score"] = rows["score_ev_per_atom"]
            rows["supported"] = rows["state"].eq("KEEP") & np.isfinite(
                rows["score_ev_per_atom"].to_numpy(float)
            )
            rows["decision"] = apply_group_threshold(
                rows["score"].to_numpy(float),
                rows["supported"].to_numpy(bool),
                float(rule["threshold"]),
            )
            rows["baseline_decision"] = rows["decision"]
            rows["source_formula"] = formula
            rows["track"] = track_name
            rows["alpha"] = TRACKS[track_name].alpha
            rows["threshold"] = float(rule["threshold"])
            result[(track_name, formula)] = rows.sort_values(
                "sid", kind="stable", ignore_index=True
            )
    return result


def _base_rows(
    gate: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cutoffs: Any,
    rules: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    decisions = _base_decision_rows(gate, cutoffs=cutoffs, rules=rules)
    label_columns = (
        "sid",
        "rk",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "high_energy",
    )
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for key, rows in decisions.items():
        joined = rows.merge(
            labels.loc[:, label_columns],
            on=["sid", "rk"],
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(gate):
            raise ValueError("next8 score rows and labels are not exactly aligned")
        result[key] = joined.sort_values("sid", kind="stable", ignore_index=True)
    return result


def _baseline_metric_table(
    base_rows: Mapping[tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for track_name in TRACK_ORDER:
        for formula in ("M5", "AGREE995"):
            metrics = evaluate_group_triage(base_rows[(track_name, formula)])
            records.append(
                {
                    "track": track_name,
                    "method": _BASE_METHOD[formula],
                    "source_formula": formula,
                    "evaluation_role": "development_gate",
                    "threshold_source_role": "threshold_fit",
                    "threshold": float(
                        base_rows[(track_name, formula)]["threshold"].iloc[0]
                    ),
                    "alpha": TRACKS[track_name].alpha,
                    **metrics,
                    "passes_safety_gate": passes_safety_gate(metrics, track_name),
                }
            )
    return pd.DataFrame(records)


def _compare_baseline_metrics(
    observed: pd.DataFrame, expected: pd.DataFrame
) -> None:
    if not isinstance(expected, pd.DataFrame):
        raise TypeError("baseline_metrics must be a pandas DataFrame")
    if set(expected.columns) != set(observed.columns):
        raise ValueError("baseline metrics artifact schema mismatch")
    keys = ["track", "method"]
    if expected.duplicated(keys).any() or set(
        zip(expected["track"], expected["method"], strict=True)
    ) != set(zip(observed["track"], observed["method"], strict=True)):
        raise ValueError("baseline metrics artifact row grid mismatch")
    expected_ordered = expected.set_index(keys).loc[
        observed.set_index(keys).index, observed.columns.drop(keys)
    ].reset_index()
    observed_ordered = observed.loc[:, expected_ordered.columns].reset_index(drop=True)
    for column in observed_ordered.columns:
        left = observed_ordered[column]
        right = expected_ordered[column]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
            right.dtype
        ):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
            if not np.array_equal(left_values, right_values, equal_nan=True):
                raise ValueError(f"baseline metrics mismatch in {column}")
        elif not left.astype(object).equals(right.astype(object)):
            raise ValueError(f"baseline metrics mismatch in {column}")


def _baseline_reproduction_record(
    base_rows: Mapping[tuple[str, str], pd.DataFrame],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for track_name in TRACK_ORDER:
        for formula in ("M5", "AGREE995"):
            for row in base_rows[(track_name, formula)].to_dict("records"):
                score = float(row["score_ev_per_atom"])
                records.append(
                    {
                        "sid": str(row["sid"]),
                        "rk": str(row["rk"]),
                        "track": track_name,
                        "formula": formula,
                        "state": str(row["state"]),
                        "score_ev_per_atom": score.hex()
                        if math.isfinite(score)
                        else None,
                        "supported": bool(row["supported"]),
                        "decision": str(row["decision"]),
                        "threshold": float(row["threshold"]).hex(),
                    }
                )
    track_index = {name: index for index, name in enumerate(TRACK_ORDER)}
    formula_index = {name: index for index, name in enumerate(("M5", "AGREE995"))}
    records.sort(
        key=lambda row: (
            track_index[str(row["track"])],
            formula_index[str(row["formula"])],
            str(row["rk"]),
            str(row["sid"]),
        )
    )
    semantic_bytes = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "canonical_constructor": (
            "src.next8_mattersim_committee_protocol.construct_committee_scores"
        ),
        "canonical_threshold_application": (
            "src.next6_elementa_protocol.apply_group_threshold"
        ),
        "previous_per_row_artifact_available": False,
        "per_row_historical_comparison_performed": False,
        "per_row_evidence_role": "deterministic_canonical_reconstruction",
        "per_row_semantic_sha256": _sha256_bytes(semantic_bytes),
        "per_row_semantic_rows": len(records),
        "aggregate_artifact_exact_match": True,
    }


def _lrrc_result(row: Mapping[str, object]) -> LRRCResult:
    status = LRRCStatus(str(row["lrrc_status"]))
    negative = _nullable_bool(row["lrrc_negative"], role="lrrc_negative")
    if status is not LRRCStatus.OK:
        return LRRCResult(status=status, error=str(row["error"]) or None)
    return LRRCResult(
        status=status,
        negative=negative,
        d_star=float(row["d_star_angstrom"]),
        h=float(row["h_angstrom"]),
        kappa_h=float(row["kappa_h_ev_per_a2"]),
        kappa_h2=float(row["kappa_h2_ev_per_a2"]),
        kappa_r=float(row["kappa_r_ev_per_a2"]),
        error_proxy=float(row["error_proxy_ev_per_a2"]),
        u_num=float(row["u_num_ev_per_a2"]),
    )


def _compose_lrrc_rows(
    baseline: pd.DataFrame, lrrc_features: pd.DataFrame
) -> pd.DataFrame:
    rows = baseline.merge(
        lrrc_features,
        on=["sid", "rk", "stage"],
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(baseline):
        raise ValueError("baseline and LRRC rows are not exactly aligned")
    decisions: list[str] = []
    for row in rows.to_dict("records"):
        baseline_decision = Decision(str(row["baseline_decision"]).lower())
        decisions.append(compose_decision(baseline_decision, _lrrc_result(row)).value.upper())
    rows["decision"] = decisions
    return rows.sort_values("sid", kind="stable", ignore_index=True)


def _quota_rows(rows: pd.DataFrame) -> pd.DataFrame:
    inputs = tuple(
        QuotaCRCRow(
            row_id=str(row.sid),
            group_id=str(row.rk),
            score=float(row.score),
            decision=Decision(str(row.decision).lower()),
            supported=bool(row.supported),
        )
        for row in rows.itertuples(index=False)
    )
    outputs = quota_crc(inputs)
    decisions = [row.decision.value.upper() for row in outputs]
    before_reject = rows["decision"].eq("REJECT").to_numpy(bool)
    after_reject = np.asarray(decisions, dtype=object) == "REJECT"
    if np.any(after_reject & ~before_reject):
        raise RuntimeError("Quota-CRC introduced a rejection")
    result = rows.copy()
    result["decision"] = decisions
    return result


def _catalog_rows(
    base_rows: Mapping[tuple[str, str], pd.DataFrame],
    lrrc_features: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for track_name in TRACK_ORDER:
        m5 = base_rows[(track_name, "M5")].copy()
        agree = base_rows[(track_name, "AGREE995")].copy()
        m5_or = _compose_lrrc_rows(m5, lrrc_features)
        m5_or["prequota_decision"] = m5_or["decision"]
        m5_quota = _quota_rows(m5_or)
        agree_or = _compose_lrrc_rows(agree, lrrc_features)
        agree_or["prequota_decision"] = agree_or["decision"]
        agree_quota = _quota_rows(agree_or)
        m5["prequota_decision"] = m5["decision"]
        agree["prequota_decision"] = agree["decision"]
        by_formula = {
            "M5": (m5, False),
            "AGREE995": (agree, False),
            "M5_LRRC_OR": (m5_or, False),
            "M5_LRRC_QCRC": (m5_quota, True),
            "AGREE995_LRRC_QCRC": (agree_quota, True),
        }
        for formula in FIXED_FORMULA_ORDER:
            rows, quota_applied = by_formula[formula]
            rows = rows.copy()
            if "lrrc_status" not in rows.columns:
                rows = rows.merge(
                    lrrc_features,
                    on=["sid", "rk", "stage"],
                    how="inner",
                    validate="one_to_one",
                )
            rows["formula"] = formula
            rows["quota_applied"] = quota_applied
            rows["threshold_role"] = "development_gate"
            parts.append(rows)
    predictions = pd.concat(parts, ignore_index=True)
    expected_rows = len(lrrc_features) * len(FIXED_FORMULA_ORDER) * len(TRACK_ORDER)
    if len(predictions) != expected_rows:
        raise RuntimeError("fixed prediction catalog row count mismatch")
    return predictions


def _metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for track_name in TRACK_ORDER:
        for formula in FIXED_FORMULA_ORDER:
            rows = predictions.loc[
                predictions["track"].eq(track_name)
                & predictions["formula"].eq(formula)
            ]
            metrics = evaluate_group_triage(rows)
            base_formula = "AGREE995" if formula.startswith("AGREE995") else "M5"
            records.append(
                {
                    "track": track_name,
                    "formula": formula,
                    "baseline_formula": base_formula,
                    "source_formula": str(rows["source_formula"].iloc[0]),
                    "evaluation_role": EVALUATION_ROLE,
                    "threshold_source_role": "threshold_fit",
                    "threshold": float(rows["threshold"].iloc[0]),
                    "alpha": TRACKS[track_name].alpha,
                    **metrics,
                    "passes_safety_gate": passes_safety_gate(metrics, track_name),
                }
            )
    return pd.DataFrame(records)


def _bootstrap_catalog(
    predictions: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    for track_name in TRACK_ORDER:
        track_predictions = predictions.loc[predictions["track"].eq(track_name)]
        for baseline, candidate in _LRRC_COMPARISONS:
            result = paired_cluster_bootstrap(
                track_predictions,
                baseline_formula=baseline,
                candidate_formula=candidate,
                alpha=TRACKS[track_name].alpha,
                n_resamples=n_resamples,
                seed=seed,
                batch_size=batch_size,
            )
            result["batch_size"] = batch_size
            comparisons.append(
                {
                    "track": track_name,
                    "baseline_formula": baseline,
                    "candidate_formula": candidate,
                    "result": result,
                }
            )
    return {
        "method": "paired percentile bootstrap over rk composition clusters",
        "difference_direction": "candidate_minus_baseline",
        "n_resamples": n_resamples,
        "seed": seed,
        "batch_size": batch_size,
        "comparison_order": [
            {
                "track": track,
                "baseline_formula": baseline,
                "candidate_formula": candidate,
            }
            for track in TRACK_ORDER
            for baseline, candidate in _LRRC_COMPARISONS
        ],
        "comparisons": comparisons,
        "gate_reused_after_exposure": True,
        "scientific_improvement_claim": False,
    }


def _catalog_document(
    frozen_protocol: Mapping[str, object],
    rules: pd.DataFrame,
    baseline_reproduction: Mapping[str, object],
) -> dict[str, object]:
    validate_catalog_order(FIXED_FORMULA_ORDER)
    threshold_records = []
    for track_name in TRACK_ORDER:
        for formula in ("M5", "AGREE995"):
            row = rules.loc[
                rules["track"].eq(track_name) & rules["formula"].eq(formula)
            ].iloc[0]
            threshold_records.append(
                {
                    "track": track_name,
                    "formula": formula,
                    "threshold": float(row["threshold"]),
                    "operator": "score > threshold",
                    "threshold_source_role": "threshold_fit",
                }
            )
    return {
        "protocol": PROTOCOL,
        "evaluation_role": EVALUATION_ROLE,
        "formula_order": list(FIXED_FORMULA_ORDER),
        "tracks": list(TRACK_ORDER),
        "thresholds": threshold_records,
        "baseline_reproduction": dict(baseline_reproduction),
        "next8_catalog_sha256": frozen_protocol["catalog"]["sha256"],
        "next8_cutoffs": dict(frozen_protocol["cutoffs"]),
        "decision_contract": {
            "baseline": "finite supported score > frozen threshold is REJECT",
            "lrrc": "OK negative is OR-rejected; OK nonnegative/stationary uses baseline; ABSTAIN_* abstains",
            "quota_crc": "applied last; ceil(sqrt(n)) score boundary and all ties become KEEP",
        },
        "accepts_refit_parameters": False,
        "gate_reused_after_exposure": True,
        "scientific_improvement_claim": False,
    }


def evaluate_fixed_catalog(
    committee_features: pd.DataFrame,
    threshold_roles: pd.DataFrame,
    lrrc_features: pd.DataFrame,
    labels: pd.DataFrame,
    frozen_protocol: Mapping[str, object],
    baseline_metrics: pd.DataFrame,
    *,
    bootstrap_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_batch_size: int = 1_000,
) -> dict[str, object]:
    """Replay next8 baselines, then evaluate only the fixed next10 catalog."""

    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive built-in int")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be a built-in int")
    if type(bootstrap_batch_size) is not int or bootstrap_batch_size <= 0:
        raise ValueError("bootstrap_batch_size must be a positive built-in int")
    validate_catalog_order(FIXED_FORMULA_ORDER)
    cutoffs, rules, gate = _validate_frozen_protocol(
        committee_features, threshold_roles, frozen_protocol
    )
    validated_lrrc = _validate_lrrc_features(lrrc_features, gate)
    prepared_labels = _prepare_labels(labels, gate)
    base_rows = _base_rows(gate, prepared_labels, cutoffs=cutoffs, rules=rules)
    reproduced = _baseline_metric_table(base_rows)
    _compare_baseline_metrics(reproduced, baseline_metrics)
    baseline_reproduction = _baseline_reproduction_record(base_rows)

    predictions = _catalog_rows(base_rows, validated_lrrc)
    metrics = _metric_table(predictions)
    bootstrap = _bootstrap_catalog(
        predictions,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        batch_size=bootstrap_batch_size,
    )
    frozen_catalog = _catalog_document(
        frozen_protocol,
        rules,
        baseline_reproduction,
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "frozen_catalog": frozen_catalog,
        "reproduced_baseline_metrics": reproduced,
        "baseline_reproduction": baseline_reproduction,
        "scientific_improvement_claim": False,
        "gate_reused_after_exposure": True,
    }


def _snapshot(path: Path, *, role: str) -> tuple[bytes, str]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise OSError(f"failed to snapshot {role}") from exc
    return data, _sha256_bytes(data)


def _parquet_snapshot(data: bytes, *, role: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"invalid parquet for {role}") from exc


def _parse_sealed_label_bytes(data: bytes) -> pd.DataFrame:
    if type(data) is not bytes:
        raise TypeError("sealed label snapshot must be exact bytes")
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("invalid parquet for frozen labels") from exc


def _validate_hash_record(
    record: object,
    *,
    role: str,
    expected_path: Path | None = None,
    verify_file: bool = True,
) -> tuple[Path, str]:
    mapping = _mapping(record, role=role)
    path_value = mapping.get("path")
    digest = mapping.get("sha256")
    if type(path_value) is not str or not _is_sha256(digest):
        raise ValueError(f"invalid hash record for {role}")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"hash-record path for {role} must be absolute")
    path = path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"hash-record path mismatch for {role}")
    if verify_file and _sha256_file(path) != digest:
        raise ValueError(f"hash mismatch for {role}")
    return path, str(digest)


def _validate_source_hashes(
    source_hashes: object,
    *,
    role: str,
) -> list[tuple[str, Path, str]]:
    mapping = _mapping(source_hashes, role=f"{role} source hashes")
    if not mapping:
        raise ValueError(f"{role} source hash closure is empty")
    repository_root = Path(__file__).resolve().parents[1]
    records: list[tuple[str, Path, str]] = []
    for relative, expected in mapping.items():
        if type(relative) is not str or not relative or not _is_sha256(expected):
            raise ValueError(f"invalid {role} source hash record")
        path = (repository_root / relative).resolve()
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"{role} source path leaves repository") from exc
        if _sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch for {role}: {relative}")
        records.append((f"{role}_source:{relative}", path, str(expected)))
    return records


def _validate_dependency_hashes(
    inputs: object, *, role: str
) -> list[tuple[str, Path, str]]:
    mapping = _mapping(inputs, role=f"{role} input hashes")
    if not mapping:
        raise ValueError(f"{role} input hash closure is empty")
    records: list[tuple[str, Path, str]] = []
    for name, record in mapping.items():
        if type(name) is not str or not name:
            raise ValueError(f"invalid {role} input role")
        path, digest = _validate_hash_record(record, role=f"{role}:{name}")
        records.append((f"{role}_input:{name}", path, digest))
    return records


def _validate_committee_manifest(
    manifest: Mapping[str, object],
    *,
    features_path: Path,
    features_sha256: str,
) -> list[tuple[str, Path, str]]:
    if manifest.get("protocol") != "2026-08-01-mattersim-dual-checkpoint-x0-v1":
        raise ValueError("committee feature manifest protocol mismatch")
    if (
        manifest.get("mode") != "development"
        or manifest.get("production_protocol_eligible") is not True
        or manifest.get("evidence_role") != "protocol_feature_generation"
        or manifest.get("integrity") != {"prepublish_rehash": "passed"}
    ):
        raise ValueError("committee feature manifest state mismatch")
    outputs = _mapping(manifest.get("outputs_sha256"), role="committee outputs")
    if dict(outputs) != {features_path.name: features_sha256}:
        raise ValueError("committee feature output hash closure mismatch")
    records = _validate_dependency_hashes(
        manifest.get("inputs_sha256"), role="committee manifest"
    )
    records.extend(
        _validate_source_hashes(
            manifest.get("executed_source_sha256"), role="committee manifest"
        )
    )
    checkpoints = _mapping(manifest.get("checkpoints"), role="committee checkpoints")
    loaded = _mapping(
        manifest.get("predictor_loaded_checkpoint_sha256"),
        role="committee loaded checkpoints",
    )
    if set(checkpoints) != {"m1", "m5"} or set(loaded) != {"m1", "m5"}:
        raise ValueError("committee checkpoint closure mismatch")
    for name in ("m1", "m5"):
        path, digest = _validate_hash_record(
            checkpoints[name], role=f"committee checkpoint {name}"
        )
        if loaded[name] != digest:
            raise ValueError("committee loaded checkpoint hash mismatch")
        records.append((f"committee_checkpoint:{name}", path, digest))
    return records


def _validate_lrrc_manifest(
    manifest: Mapping[str, object],
    *,
    features_path: Path,
    features_sha256: str,
    lrrc_features: pd.DataFrame,
    committee_features: pd.DataFrame,
    threshold_roles: pd.DataFrame,
    committee_features_path: Path,
    committee_features_sha256: str,
    threshold_roles_path: Path,
    threshold_roles_sha256: str,
    committee_manifest_path: Path,
    committee_manifest_sha256: str,
    committee_manifest: Mapping[str, object],
) -> list[tuple[str, Path, str]]:
    expected_top_level = {
        "protocol",
        "mode",
        "labels_opened",
        "selection",
        "adapter",
        "predictor_loaded_checkpoint_sha256",
        "production_protocol_eligible",
        "evidence_role",
        "runtime",
        "inputs_sha256",
        "executed_source_sha256",
        "integrity",
        "feature_columns",
        "counts",
        "execution",
        "scientific_improvement_claim",
        "outputs_sha256",
    }
    if set(manifest) != expected_top_level:
        raise ValueError("LRRC manifest top-level schema mismatch")
    if manifest.get("protocol") != LRRC_FEATURE_PROTOCOL:
        raise ValueError("LRRC feature manifest protocol mismatch")
    if (
        manifest.get("mode") != "development_gate"
        or manifest.get("production_protocol_eligible") is not True
        or manifest.get("evidence_role") != "label_free_lrrc_feature_generation"
        or manifest.get("labels_opened") is not False
        or manifest.get("scientific_improvement_claim") is not False
        or manifest.get("integrity") != {"prepublish_rehash": "passed"}
    ):
        raise ValueError("LRRC feature manifest state mismatch")
    if manifest.get("feature_columns") != list(LRRC_FEATURE_COLUMNS):
        raise ValueError("LRRC manifest feature schema mismatch")
    if manifest.get("selection") != {
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
    }:
        raise ValueError("LRRC manifest selection mismatch")

    adapter = _mapping(manifest.get("adapter"), role="LRRC adapter")
    adapter_keys = {
        "mode",
        "device",
        "batch_size",
        "index_alignment",
        "index_alignment_verified",
        "model_parameter_device",
        "result_tensor_devices",
        "evaluations",
    }
    if set(adapter) != adapter_keys:
        raise ValueError("LRRC adapter schema mismatch")
    device = adapter.get("device")
    model_device = adapter.get("model_parameter_device")
    result_devices = adapter.get("result_tensor_devices")
    if (
        adapter.get("mode") != "builtin_indexed_mattersim"
        or not _is_canonical_cuda_device(device)
        or type(adapter.get("batch_size")) is not int
        or int(adapter["batch_size"]) <= 0
        or adapter.get("index_alignment")
        != "sid_indexed_exact_one_to_one"
        or adapter.get("index_alignment_verified") is not True
        or model_device != device
        or not isinstance(result_devices, list)
        or result_devices != [device]
        or type(adapter.get("evaluations")) is not int
        or int(adapter["evaluations"]) <= 0
    ):
        raise ValueError("LRRC adapter does not prove production CUDA execution")

    outputs = _mapping(manifest.get("outputs_sha256"), role="LRRC outputs")
    if dict(outputs) != {features_path.name: features_sha256}:
        raise ValueError("LRRC feature output hash closure mismatch")
    inputs = _mapping(manifest.get("inputs_sha256"), role="LRRC inputs")
    expected_input_names = {
        "committee_features",
        "threshold_roles",
        "frames",
        "feature_manifest",
        "checkpoint",
    }
    if set(inputs) != expected_input_names:
        raise ValueError("LRRC input hash closure mismatch")
    records: list[tuple[str, Path, str]] = []
    for name, expected_path, expected_hash in (
        (
            "committee_features",
            committee_features_path,
            committee_features_sha256,
        ),
        ("threshold_roles", threshold_roles_path, threshold_roles_sha256),
        (
            "feature_manifest",
            committee_manifest_path,
            committee_manifest_sha256,
        ),
    ):
        path, digest = _validate_hash_record(
            inputs[name], role=f"LRRC {name}", expected_path=expected_path
        )
        if digest != expected_hash:
            raise ValueError(f"LRRC {name} hash mismatch")
        records.append((f"LRRC manifest_input:{name}", path, digest))

    committee_inputs = _mapping(
        committee_manifest.get("inputs_sha256"), role="committee inputs"
    )
    committee_frames = _mapping(
        committee_inputs.get("frames"), role="committee frames"
    )
    frames_path, frames_hash = _validate_hash_record(
        inputs["frames"], role="LRRC frames"
    )
    if (
        type(committee_frames.get("path")) is not str
        or Path(str(committee_frames["path"])).resolve() != frames_path.resolve()
        or committee_frames.get("sha256") != frames_hash
    ):
        raise ValueError("LRRC frames are not the frozen committee frames")
    records.append(("LRRC manifest_input:frames", frames_path, frames_hash))

    checkpoint_path, checkpoint_hash = _validate_hash_record(
        inputs["checkpoint"], role="LRRC checkpoint"
    )
    committee_checkpoints = _mapping(
        committee_manifest.get("checkpoints"), role="committee checkpoints"
    )
    m5_record = _mapping(committee_checkpoints.get("m5"), role="committee m5")
    if (
        m5_record.get("path") != str(checkpoint_path.resolve())
        or m5_record.get("sha256") != checkpoint_hash
    ):
        raise ValueError("LRRC checkpoint is not the frozen MatterSim 5M checkpoint")
    if manifest.get("predictor_loaded_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("LRRC loaded checkpoint hash mismatch")
    records.append(("LRRC manifest_input:checkpoint", checkpoint_path, checkpoint_hash))

    source_hashes = _mapping(
        manifest.get("executed_source_sha256"), role="LRRC source hashes"
    )
    if set(source_hashes) != set(_LRRC_EXECUTED_SOURCE_RELATIVE):
        raise ValueError("LRRC executed-source closure mismatch")
    records.extend(
        _validate_source_hashes(
            source_hashes, role="LRRC manifest"
        )
    )

    runtime = _mapping(manifest.get("runtime"), role="LRRC runtime")
    runtime_keys = {
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "pandas_version",
        "ase_version",
        "mattersim_version",
        "device",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
    }
    if set(runtime) != runtime_keys:
        raise ValueError("LRRC runtime schema mismatch")
    string_runtime_keys = runtime_keys - {"cuda_available"}
    if (
        runtime.get("cuda_available") is not True
        or runtime.get("device") != device
        or runtime.get("mattersim_version") != "1.2.3"
        or any(
            type(runtime.get(name)) is not str
            or not str(runtime[name])
            or str(runtime[name]).lower() == "unavailable"
            for name in string_runtime_keys
        )
    ):
        raise ValueError("LRRC runtime does not contain complete CUDA identity")

    execution = _mapping(manifest.get("execution"), role="LRRC execution")
    execution_keys = {
        "batch_predictor_calls",
        "forward_calls",
        "peak_cuda_memory_bytes",
        "wall_time_seconds",
    }
    if set(execution) != execution_keys:
        raise ValueError("LRRC execution telemetry schema mismatch")
    batch_predictor_calls = execution.get("batch_predictor_calls")
    forward_calls = execution.get("forward_calls")
    peak_memory = execution.get("peak_cuda_memory_bytes")
    wall_time = execution.get("wall_time_seconds")
    force_calls = pd.to_numeric(
        lrrc_features["force_call_count"], errors="raise"
    ).to_numpy(dtype=int)
    base_evaluations = int(np.count_nonzero(force_calls >= 1))
    five_step_evaluations = int(np.count_nonzero(force_calls == 5))
    batch_size = int(adapter["batch_size"])
    expected_batch_predictor_calls = int(base_evaluations > 0) + 4 * int(
        five_step_evaluations > 0
    )
    expected_forward_calls = math.ceil(base_evaluations / batch_size) + 4 * math.ceil(
        five_step_evaluations / batch_size
    )
    expected_force_evaluations = base_evaluations + 4 * five_step_evaluations
    if (
        type(batch_predictor_calls) is not int
        or batch_predictor_calls != expected_batch_predictor_calls
        or type(forward_calls) is not int
        or forward_calls != expected_forward_calls
        or type(peak_memory) is not int
        or peak_memory <= 0
        or isinstance(wall_time, (bool, np.bool_))
        or not isinstance(wall_time, (int, float, np.integer, np.floating))
        or not math.isfinite(float(wall_time))
        or float(wall_time) <= 0.0
    ):
        raise ValueError("LRRC execution telemetry is incomplete")

    counts = _mapping(manifest.get("counts"), role="LRRC counts")
    expected_count_keys = {
        "feature_rows",
        "role_assignment_rows",
        "selected_rows",
        "strict_rows",
        "nonstrict_rows",
        "ok_rows",
        "stationary_rows",
        "abstained_rows",
        "batch_predictor_calls",
        "force_evaluations",
    }
    if set(counts) != expected_count_keys or any(
        type(counts.get(name)) is not int for name in expected_count_keys
    ):
        raise ValueError("LRRC count schema mismatch")
    statuses = lrrc_features["lrrc_status"].astype(str)
    strict = lrrc_features["strict_x0_ok"].astype(bool)
    expected_counts = {
        "feature_rows": len(committee_features),
        "role_assignment_rows": len(threshold_roles),
        "selected_rows": len(lrrc_features),
        "strict_rows": int(strict.sum()),
        "nonstrict_rows": int((~strict).sum()),
        "ok_rows": int(statuses.eq(LRRCStatus.OK.value).sum()),
        "stationary_rows": int(
            statuses.eq(LRRCStatus.STATIONARY_FALLBACK.value).sum()
        ),
        "abstained_rows": int(statuses.str.startswith("abstain_").sum()),
        "batch_predictor_calls": expected_batch_predictor_calls,
        "force_evaluations": expected_force_evaluations,
    }
    if dict(counts) != expected_counts:
        raise ValueError("LRRC manifest counts do not match the feature table")
    if (
        adapter.get("evaluations") != expected_force_evaluations
        or counts.get("batch_predictor_calls") != expected_batch_predictor_calls
    ):
        raise ValueError("LRRC adapter/execution telemetry counts mismatch")
    return records


def _validate_next8_manifest(
    manifest: Mapping[str, object],
    *,
    manifest_dir: Path,
    frozen_protocol_path: Path,
    frozen_protocol_sha256: str,
    committee_features_path: Path,
    committee_features_sha256: str,
    committee_manifest_path: Path,
    committee_manifest_sha256: str,
    labels_path: Path,
    threshold_roles_path: Path,
    threshold_roles_sha256: str,
    baseline_metrics_path: Path,
    baseline_metrics_sha256: str,
) -> tuple[
    dict[str, str],
    list[tuple[str, Path, str]],
    Path,
    str,
]:
    if (
        manifest.get("protocol") != DEVELOPMENT_FREEZE_PROTOCOL
        or manifest.get("state") != "frozen"
        or manifest.get("integrity") != {"prepublish_rehash": "passed"}
    ):
        raise ValueError("next8 development manifest state mismatch")
    inputs = _mapping(manifest.get("inputs_sha256"), role="next8 inputs")
    if set(inputs) != {"features", "feature_manifest", "labels"}:
        raise ValueError("next8 input hash closure mismatch")
    expected_nonlabel_inputs = {
        "features": (committee_features_path, committee_features_sha256),
        "feature_manifest": (committee_manifest_path, committee_manifest_sha256),
    }
    records: list[tuple[str, Path, str]] = []
    for name, (path, digest) in expected_nonlabel_inputs.items():
        observed_path, observed_hash = _validate_hash_record(
            inputs[name], role=f"next8 {name}", expected_path=path
        )
        if observed_hash != digest:
            raise ValueError(f"next8 {name} hash mismatch")
        records.append((f"next8_input:{name}", observed_path, observed_hash))
    label_path, expected_label_sha256 = _validate_hash_record(
        inputs["labels"],
        role="next8 labels",
        expected_path=labels_path,
        verify_file=False,
    )

    outputs = _mapping(manifest.get("outputs_sha256"), role="next8 outputs")
    if set(outputs) != _NEXT8_ARTIFACT_NAMES | {"FROZEN_PROTOCOL.json"}:
        raise ValueError("next8 output hash closure mismatch")
    expected_direct = {
        "FROZEN_PROTOCOL.json": (frozen_protocol_path, frozen_protocol_sha256),
        "threshold_role_assignments.parquet": (
            threshold_roles_path,
            threshold_roles_sha256,
        ),
        "development_gate_metrics.parquet": (
            baseline_metrics_path,
            baseline_metrics_sha256,
        ),
    }
    artifact_hashes: dict[str, str] = {}
    for name, expected in outputs.items():
        if not _is_sha256(expected):
            raise ValueError("invalid next8 output hash")
        path = manifest_dir / name
        if name in expected_direct:
            expected_path, expected_hash = expected_direct[name]
            if path.resolve() != expected_path.resolve() or expected != expected_hash:
                raise ValueError(f"next8 direct artifact hash mismatch: {name}")
        if _sha256_file(path) != expected:
            raise ValueError(f"next8 artifact hash mismatch: {name}")
        if name != "FROZEN_PROTOCOL.json":
            artifact_hashes[name] = str(expected)
        records.append((f"next8_output:{name}", path, str(expected)))
    records.extend(
        _validate_source_hashes(
            manifest.get("executed_source_sha256"), role="next8 manifest"
        )
    )
    return artifact_hashes, records, label_path, expected_label_sha256


def _deduplicate_records(
    records: Sequence[tuple[str, Path, str]],
) -> list[tuple[str, Path, str]]:
    observed: dict[Path, str] = {}
    result: list[tuple[str, Path, str]] = []
    for role, path, digest in records:
        resolved = path.resolve()
        previous = observed.get(resolved)
        if previous is not None and previous != digest:
            raise ValueError(f"conflicting hashes for shared input: {resolved}")
        if previous is None:
            observed[resolved] = digest
            result.append((role, resolved, digest))
    return result


def _verify_unchanged(records: Sequence[tuple[str, Path, str]]) -> None:
    for role, path, expected in records:
        if _sha256_file(path) != expected:
            raise RuntimeError(f"sealed dependency changed after label opening: {role}")


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_result(
    output_dir: Path,
    *,
    result: Mapping[str, object],
    catalog_bytes: bytes,
    input_records: Sequence[tuple[str, Path, str]],
    source_records: Sequence[tuple[str, Path, str]],
    executed_sources: Mapping[str, str],
    parameters: Mapping[str, object],
) -> Path:
    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(f"refusing to replace existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=os.fspath(target.parent))
    )
    sealed_records = _deduplicate_records([*input_records, *source_records])
    try:
        predictions_path = staging / "predictions.parquet"
        metrics_path = staging / "metrics.parquet"
        result["predictions"].to_parquet(predictions_path, index=False)
        result["metrics"].to_parquet(metrics_path, index=False)
        _write_exclusive(
            staging / "PAIRED_BOOTSTRAP.json",
            _strict_json_text(result["paired_bootstrap"]).encode("utf-8"),
        )
        _write_exclusive(staging / "FROZEN_CATALOG.json", catalog_bytes)

        output_hashes = {
            name: _sha256_file(staging / name)
            for name in OUTPUT_NAMES
            if name != "MANIFEST.json"
        }
        manifest = {
            "protocol": PROTOCOL,
            "evaluation_role": EVALUATION_ROLE,
            "catalog_frozen_before_label_read": True,
            "gate_reused_after_exposure": True,
            "scientific_improvement_claim": False,
            "parameters": dict(parameters),
            "counts": {
                "prediction_rows": int(len(result["predictions"])),
                "metric_rows": int(len(result["metrics"])),
                "formulas": len(FIXED_FORMULA_ORDER),
                "tracks": len(TRACK_ORDER),
            },
            "inputs_sha256": {
                role: {"path": str(path), "sha256": digest}
                for role, path, digest in input_records
            },
            "executed_source_sha256": dict(executed_sources),
            "baseline_reproduction": dict(result["baseline_reproduction"]),
            "frozen_catalog_sha256": _sha256_bytes(catalog_bytes),
            "outputs_sha256": output_hashes,
            "integrity": {"prepublish_rehash": "passed"},
        }
        _write_exclusive(
            staging / "MANIFEST.json",
            _strict_json_text(manifest).encode("utf-8"),
        )

        if {path.name for path in staging.iterdir()} != set(OUTPUT_NAMES):
            raise RuntimeError("staging directory does not contain exact outputs")
        pd.read_parquet(predictions_path)
        pd.read_parquet(metrics_path)
        for name in ("PAIRED_BOOTSTRAP.json", "FROZEN_CATALOG.json", "MANIFEST.json"):
            _strict_json_bytes((staging / name).read_bytes(), role=name)
        for name, expected in output_hashes.items():
            if _sha256_file(staging / name) != expected:
                raise RuntimeError(f"staged output hash mismatch: {name}")
        _verify_unchanged(sealed_records)
        _rename_noreplace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def run_gate_diagnostic(
    *,
    committee_features_path: Path,
    committee_manifest_path: Path,
    threshold_roles_path: Path,
    lrrc_features_path: Path,
    lrrc_manifest_path: Path,
    frozen_protocol_path: Path,
    next8_manifest_path: Path,
    baseline_metrics_path: Path,
    labels_path: Path,
    output_dir: Path,
    bootstrap_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_batch_size: int = 1_000,
) -> Path:
    """Validate all sealed inputs before the first label-parquet read and publish."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(f"refusing to replace existing output: {target}")
    formal_parameters = (
        bootstrap_resamples,
        bootstrap_seed,
        bootstrap_batch_size,
    )
    if (
        any(type(value) is not int for value in formal_parameters)
        or formal_parameters != _FORMAL_BOOTSTRAP_PARAMETERS
    ):
        raise ValueError(
            "formal bootstrap parameters must be exactly "
            "resamples=20000, seed=20260801, batch_size=1000"
        )

    path_roles = {
        "committee_features": Path(committee_features_path),
        "committee_manifest": Path(committee_manifest_path),
        "threshold_roles": Path(threshold_roles_path),
        "lrrc_features": Path(lrrc_features_path),
        "lrrc_manifest": Path(lrrc_manifest_path),
        "frozen_protocol": Path(frozen_protocol_path),
        "next8_manifest": Path(next8_manifest_path),
        "baseline_metrics": Path(baseline_metrics_path),
    }
    snapshots: dict[str, tuple[bytes, str]] = {
        role: _snapshot(path, role=role) for role, path in path_roles.items()
    }
    committee_manifest = _strict_json_bytes(
        snapshots["committee_manifest"][0], role="committee manifest"
    )
    lrrc_manifest = _strict_json_bytes(
        snapshots["lrrc_manifest"][0], role="LRRC manifest"
    )
    frozen_protocol = _strict_json_bytes(
        snapshots["frozen_protocol"][0], role="frozen protocol"
    )
    next8_manifest = _strict_json_bytes(
        snapshots["next8_manifest"][0], role="next8 manifest"
    )
    committee_features = _parquet_snapshot(
        snapshots["committee_features"][0], role="committee features"
    )
    threshold_roles = _parquet_snapshot(
        snapshots["threshold_roles"][0], role="threshold roles"
    )
    lrrc_features = _parquet_snapshot(
        snapshots["lrrc_features"][0], role="LRRC features"
    )
    baseline_metrics = _parquet_snapshot(
        snapshots["baseline_metrics"][0], role="baseline metrics"
    )

    input_records: list[tuple[str, Path, str]] = [
        (role, path_roles[role].resolve(), digest)
        for role, (_, digest) in snapshots.items()
    ]
    input_records.extend(
        _validate_committee_manifest(
            committee_manifest,
            features_path=path_roles["committee_features"],
            features_sha256=snapshots["committee_features"][1],
        )
    )
    (
        artifact_hashes,
        next8_records,
        resolved_labels_path,
        expected_label_sha256,
    ) = _validate_next8_manifest(
        next8_manifest,
        manifest_dir=path_roles["next8_manifest"].parent,
        frozen_protocol_path=path_roles["frozen_protocol"],
        frozen_protocol_sha256=snapshots["frozen_protocol"][1],
        committee_features_path=path_roles["committee_features"],
        committee_features_sha256=snapshots["committee_features"][1],
        committee_manifest_path=path_roles["committee_manifest"],
        committee_manifest_sha256=snapshots["committee_manifest"][1],
        labels_path=Path(labels_path),
        threshold_roles_path=path_roles["threshold_roles"],
        threshold_roles_sha256=snapshots["threshold_roles"][1],
        baseline_metrics_path=path_roles["baseline_metrics"],
        baseline_metrics_sha256=snapshots["baseline_metrics"][1],
    )
    input_records.extend(next8_records)
    cutoffs, rules, gate = _validate_frozen_protocol(
        committee_features,
        threshold_roles,
        frozen_protocol,
        feature_sha256=snapshots["committee_features"][1],
        feature_manifest_sha256=snapshots["committee_manifest"][1],
        artifact_sha256=artifact_hashes,
    )
    validated_lrrc = _validate_lrrc_features(
        lrrc_features,
        gate,
        formal_producer=True,
    )
    input_records.extend(
        _validate_lrrc_manifest(
            lrrc_manifest,
            features_path=path_roles["lrrc_features"],
            features_sha256=snapshots["lrrc_features"][1],
            lrrc_features=validated_lrrc,
            committee_features=committee_features,
            threshold_roles=threshold_roles,
            committee_features_path=path_roles["committee_features"],
            committee_features_sha256=snapshots["committee_features"][1],
            threshold_roles_path=path_roles["threshold_roles"],
            threshold_roles_sha256=snapshots["threshold_roles"][1],
            committee_manifest_path=path_roles["committee_manifest"],
            committee_manifest_sha256=snapshots["committee_manifest"][1],
            committee_manifest=committee_manifest,
        )
    )
    expected_baseline_reproduction = _baseline_reproduction_record(
        _base_decision_rows(gate, cutoffs=cutoffs, rules=rules)
    )
    catalog = _catalog_document(
        frozen_protocol,
        rules,
        expected_baseline_reproduction,
    )
    catalog_bytes = _strict_json_text(catalog).encode("utf-8")
    _strict_json_bytes(catalog_bytes, role="frozen next10 catalog")
    input_records = _deduplicate_records(input_records)

    repository_root = Path(__file__).resolve().parents[1]
    executed_sources: dict[str, str] = {}
    source_records: list[tuple[str, Path, str]] = []
    for relative in _FORMAL_EXECUTED_SOURCE_RELATIVE:
        source_path = (repository_root / relative).resolve()
        source_sha256 = _sha256_file(source_path)
        executed_sources[relative] = source_sha256
        source_records.append(
            (f"evaluator_source:{relative}", source_path, source_sha256)
        )
    source_records = _deduplicate_records(source_records)
    _deduplicate_records([*input_records, *source_records])

    try:
        label_bytes = resolved_labels_path.read_bytes()
    except OSError as exc:
        raise OSError("failed to read frozen label bytes") from exc
    observed_label_sha256 = _sha256_bytes(label_bytes)
    if observed_label_sha256 != expected_label_sha256:
        raise ValueError("label bytes hash mismatch")
    labels = _parse_sealed_label_bytes(label_bytes)
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("sealed label parser must return a pandas DataFrame")
    input_records = _deduplicate_records(
        [
            *input_records,
            (
                "next8_input:labels",
                resolved_labels_path,
                observed_label_sha256,
            ),
        ]
    )
    result = evaluate_fixed_catalog(
        committee_features,
        threshold_roles,
        validated_lrrc,
        labels,
        frozen_protocol,
        baseline_metrics,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_batch_size=bootstrap_batch_size,
    )
    if result["baseline_reproduction"] != expected_baseline_reproduction:
        raise RuntimeError(
            "baseline semantic reconstruction changed after label opening"
        )
    if result["frozen_catalog"] != catalog:
        raise RuntimeError("frozen catalog changed after label opening")
    return _publish_result(
        target,
        result=result,
        catalog_bytes=catalog_bytes,
        input_records=input_records,
        source_records=source_records,
        executed_sources=executed_sources,
        parameters={
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_batch_size": bootstrap_batch_size,
            "threshold_refit": False,
            "catalog_refit": False,
        },
    )


__all__ = [
    "BOOTSTRAP_SEED",
    "EVALUATION_ROLE",
    "FIXED_FORMULA_ORDER",
    "LRRC_FEATURE_COLUMNS",
    "LRRC_FEATURE_PROTOCOL",
    "OUTPUT_NAMES",
    "PRODUCTION_BOOTSTRAP_RESAMPLES",
    "PROTOCOL",
    "evaluate_fixed_catalog",
    "run_gate_diagnostic",
    "validate_catalog_order",
]
