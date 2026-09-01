from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.next7_fewstep_protocol import (
    DEVELOPMENT_FREEZE_PROTOCOL,
    EVIDENCE_ROLE,
    FEATURE_PROTOCOL,
    FROZEN_CATALOG,
    TRACKS,
)
from src.next7_fewstep_evaluate import run_frozen_evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rule(
    *,
    track: str,
    role: str,
    name: str,
    threshold: float,
) -> dict[str, object]:
    formula = {item.name: item for item in FROZEN_CATALOG}[name]
    semantics = TRACKS[track]
    return {
        "state": "selected" if role == "selected" else "baseline",
        "name": name,
        "protected": semantics.protected,
        "protected_ev_per_atom": semantics.protected_ev_per_atom,
        "within_group": semantics.within_group,
        "alpha": semantics.alpha,
        "max_step": formula.max_step,
        "cost": formula.cost,
        "threshold": threshold,
        "threshold_state": "finite",
        "operator": "score > threshold",
        "unsupported_decision": "ABSTAIN",
        "calibration_n_groups": 9,
        "calibration_order_index": 8,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"frozen-checkpoint")

    rows = []
    values = {
        "a0": ("A", 0.0, 0.0, 0.0, True),
        "a1": ("A", 0.30, 0.15, 0.20, True),
        "b0": ("B", 0.0, 0.0, 0.0, True),
        "b1": ("B", 0.40, 0.25, 0.50, True),
        "c0": ("C", 0.0, 0.0, 0.0, True),
        "c1": ("C", 0.60, 0.30, 0.40, False),
    }
    for sid, (rk, e0, e2, e8, support8) in values.items():
        rows.append(
            {
                "sid": sid,
                "rk": rk,
                "stage": "test",
                "evidence_role": EVIDENCE_ROLE,
                "k0_energy_ev_per_atom": e0,
                "k2_energy_ev_per_atom": e2,
                "k4_energy_ev_per_atom": (e2 + e8) / 2,
                "k8_energy_ev_per_atom": e8,
                "k0_supported": True,
                "k2_supported": True,
                "k4_supported": True,
                "k8_supported": support8,
            }
        )
    features = pd.DataFrame(rows)
    features_path = tmp_path / "test_features.parquet"
    features.to_parquet(features_path, index=False)

    labels = pd.DataFrame(
        {
            "sid": list(values),
            "rk": [values[sid][0] for sid in values],
            "stage": ["test"] * len(values),
            "material": [f"X_{index:02d}" for index in range(len(values))],
            "e_per_atom": [0.0, 0.04, 0.0, 0.20, 0.0, 0.30],
        }
    )
    labels_path = tmp_path / "test_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    feature_inputs = {
        "elementa_initial_frames.zip": "1" * 64,
        "elementa_x0_features.parquet": "2" * 64,
        "elementa_x0_p9_features.parquet": "3" * 64,
        "stage_assignments.parquet": "4" * 64,
    }
    source_root = Path(__file__).resolve().parents[1]
    frozen = {
        "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
        "state": "frozen",
        "frozen_at_utc": "2026-08-01T00:00:00+00:00",
        "evidence_role": EVIDENCE_ROLE,
        "checkpoint_sha256": _sha256(checkpoint),
        "feature_inputs_sha256": feature_inputs,
        "code_sha256": {
            "next7_mattersim_prerelax.py": _sha256(
                source_root / "src" / "next7_mattersim_prerelax.py"
            ),
            "next7_mattersim_features.py": _sha256(
                source_root / "src" / "next7_mattersim_features.py"
            ),
        },
        "selection_code_sha256": _sha256(
            source_root / "src" / "next7_fewstep_protocol.py"
        ),
        "catalog": [item.as_record() for item in FROZEN_CATALOG],
        "tracks": {name: item.as_record() for name, item in TRACKS.items()},
        "rules": {
            "primary": {
                **_rule(
                    track="primary", role="selected", name="S8", threshold=0.20
                ),
                "s0_baseline": _rule(
                    track="primary", role="s0_baseline", name="S0", threshold=0.30
                ),
            },
            "comparator": {
                **_rule(
                    track="comparator", role="selected", name="S2", threshold=0.15
                ),
                "s0_baseline": _rule(
                    track="comparator", role="s0_baseline", name="S0", threshold=0.30
                ),
            },
        },
        "development_artifacts_sha256": {},
    }
    frozen_path = tmp_path / "FROZEN_PROTOCOL.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    manifest = {
        "protocol": FEATURE_PROTOCOL,
        "stages": ["test"],
        "evidence_role": EVIDENCE_ROLE,
        "inputs_sha256": {
            **feature_inputs,
            checkpoint.name: _sha256(checkpoint),
            frozen_path.name: _sha256(frozen_path),
        },
        "outputs_sha256": {features_path.name: _sha256(features_path)},
        "frozen_protocol": {
            "path": str(frozen_path.resolve()),
            "sha256": _sha256(frozen_path),
        },
        "model": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "counts": {
            "selected_rows": len(features),
            "force_evaluations": 54,
            "optimizer_updates": 48,
        },
        "execution": {
            "predictor_forward_calls": 9,
            "total_elapsed_seconds": 1.25,
            "peak_cuda_memory_bytes": 1234,
        },
    }
    manifest_path = tmp_path / "FEATURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "features": features_path,
        "labels": labels_path,
        "feature_manifest": manifest_path,
        "frozen": frozen_path,
        "checkpoint": checkpoint,
    }


def _run(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return run_frozen_evaluation(
        paths["features"],
        paths["labels"],
        paths["feature_manifest"],
        paths["frozen"],
        output,
        checkpoint=paths["checkpoint"],
        bootstrap_resamples=200,
        seed=7,
    )


def test_frozen_evaluation_applies_only_fixed_rules_and_writes_hashed_artifacts(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    summary = _run(paths, output)

    expected = {
        "test_predictions.parquet",
        "test_metrics.parquet",
        "paired_bootstrap.parquet",
        "TEST_OPENING.json",
        "MANIFEST.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    predictions = pd.read_parquet(output / "test_predictions.parquet")
    assert len(predictions) == 4 * 6
    assert set(predictions["method_id"]) == {
        "primary:selected",
        "primary:s0_baseline",
        "comparator:selected",
        "comparator:s0_baseline",
    }
    boundary = predictions[
        predictions["method_id"].eq("primary:selected")
        & predictions["sid"].eq("a1")
    ].iloc[0]
    assert boundary["score"] == pytest.approx(0.20)
    assert boundary["decision"] == "KEEP"
    unsupported = predictions[
        predictions["method_id"].eq("primary:selected")
        & predictions["rk"].eq("C")
    ]
    assert set(unsupported["decision"]) == {"ABSTAIN"}

    metrics = pd.read_parquet(output / "test_metrics.parquet")
    assert len(metrics) == 4
    assert set(metrics["track"]) == {"primary", "comparator"}
    assert set(metrics["role"]) == {"selected", "s0_baseline"}
    assert set(metrics["observed_force_evaluations"]) == {54}
    bootstrap = pd.read_parquet(output / "paired_bootstrap.parquet")
    assert set(bootstrap["track"]) == {"primary", "comparator"}
    assert {"dft_savings", "valuable_item_recall", "abstention_rate"}.issubset(
        set(bootstrap["metric"])
    )

    opening = json.loads((output / "TEST_OPENING.json").read_text())
    assert opening["evidence_role"] == EVIDENCE_ROLE
    assert opening["blind_or_confirmatory"] is False
    assert opening["test_tuning_permitted"] is False
    assert opening["frozen_protocol_sha256"] == _sha256(paths["frozen"])
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        assert _sha256(output / name) == digest
    assert summary["n_rows"] == 6
    with pytest.raises(FileExistsError):
        _run(paths, output)


@pytest.mark.parametrize("bad_stage", ["formula_selection", "search_calibration"])
def test_evaluator_rejects_any_stage_other_than_test(
    tmp_path: Path, bad_stage: str
) -> None:
    paths = _fixture(tmp_path)
    data = pd.read_parquet(paths["features"])
    data["stage"] = bad_stage
    data.to_parquet(paths["features"], index=False)
    manifest = json.loads(paths["feature_manifest"].read_text())
    manifest["outputs_sha256"][paths["features"].name] = _sha256(paths["features"])
    manifest["stages"] = [bad_stage]
    paths["feature_manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="test stage"):
        _run(paths, tmp_path / "out")


def test_evaluator_rejects_label_key_or_frozen_hash_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    labels = pd.read_parquet(paths["labels"])
    labels.loc[0, "sid"] = "different"
    labels.to_parquet(paths["labels"], index=False)
    with pytest.raises(ValueError, match="keys differ"):
        _run(paths, tmp_path / "bad-keys")

    paths = _fixture(tmp_path / "second")
    frozen = json.loads(paths["frozen"].read_text())
    frozen["rules"]["primary"]["threshold"] = 999.0
    paths["frozen"].write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="frozen protocol hash mismatch"):
        _run(paths, tmp_path / "bad-freeze")


def test_evaluator_requires_test_stage_in_label_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    labels = pd.read_parquet(paths["labels"]).drop(columns="stage")
    labels.to_parquet(paths["labels"], index=False)
    with pytest.raises(ValueError, match="missing columns.*stage"):
        _run(paths, tmp_path / "missing-label-stage")


def test_evaluator_rejects_false_cost_on_null_keep_all_rule(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    frozen = json.loads(paths["frozen"].read_text())
    selected = frozen["rules"]["primary"]
    selected.update(
        {
            "state": "null_keep_all",
            "name": "null_keep_all",
            "max_step": 8,
            "cost": 999,
            "threshold": None,
            "threshold_state": "keep_all",
            "operator": "KEEP_ALL",
            "unsupported_decision": "KEEP",
        }
    )
    paths["frozen"].write_text(json.dumps(frozen))
    digest = _sha256(paths["frozen"])
    manifest = json.loads(paths["feature_manifest"].read_text())
    manifest["inputs_sha256"][paths["frozen"].name] = digest
    manifest["frozen_protocol"]["sha256"] = digest
    paths["feature_manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="keep-all cost/step"):
        _run(paths, tmp_path / "bad-null-cost")


def test_evaluator_cleans_staging_when_atomic_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "publish-failure"

    def fail_publish(source: Path, target: Path) -> None:
        raise OSError("injected publish failure")

    monkeypatch.setattr(
        "src.next7_fewstep_evaluate._atomic_publish_directory_no_replace",
        fail_publish,
    )
    with pytest.raises(OSError, match="injected"):
        _run(paths, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".publish-failure.staging-*"))
