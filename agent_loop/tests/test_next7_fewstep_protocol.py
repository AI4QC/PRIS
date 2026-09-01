import ctypes
import errno
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from src import next7_fewstep_protocol as protocol


DEV_STAGES = (
    "search_calibration",
    "formula_selection",
    "threshold_calibration",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _feature_rows():
    rows = []
    for stage in DEV_STAGES:
        for group_index in range(60):
            rk = f"{stage}-g{group_index:02d}"
            for candidate_index, energy in enumerate((0.0, 0.20)):
                row = {
                    "sid": f"{rk}-c{candidate_index}",
                    "rk": rk,
                    "stage": stage,
                    "evidence_role": (
                        "historically seen discovery; not confirmatory"
                    ),
                }
                for step in (0, 2, 4, 8):
                    row[f"k{step}_energy_ev_per_atom"] = energy
                    row[f"k{step}_supported"] = True
                rows.append(row)
    return pd.DataFrame(rows)


def _write_toy_inputs(tmp_path):
    features_path = tmp_path / "mattersim_fewstep_features.parquet"
    labels_path = tmp_path / "labels.parquet"
    feature_manifest_path = tmp_path / "feature-MANIFEST.json"
    checkpoint = tmp_path / "model.pth"
    features = _feature_rows()
    features.to_parquet(features_path, index=False)
    features[["sid", "rk"]].assign(
        e_per_atom=np.tile((0.0, 0.20), len(features) // 2)
    ).to_parquet(labels_path, index=False)
    checkpoint.write_bytes(b"frozen-mattersim-checkpoint")
    checkpoint_sha256 = _digest(checkpoint)
    feature_inputs_sha256 = {
        "elementa_initial_frames.zip": "1" * 64,
        "elementa_x0_p9_features.parquet": "2" * 64,
        "elementa_x0_features.parquet": "3" * 64,
        "stage_assignments.parquet": "4" * 64,
    }
    manifest = {
        "protocol": "2026-08-01-mattersim-fewstep-prerelax-v1",
        "evidence_role": (
            "historically seen discovery; not confirmatory"
        ),
        "stages": list(DEV_STAGES),
        "model": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "inputs_sha256": {
            **feature_inputs_sha256,
            checkpoint.name: checkpoint_sha256,
        },
        "outputs_sha256": {features_path.name: _digest(features_path)},
    }
    feature_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "features_path": features_path,
        "labels_path": labels_path,
        "feature_manifest_path": feature_manifest_path,
        "checkpoint": checkpoint,
        "feature_inputs_sha256": feature_inputs_sha256,
    }


def _refresh_feature_hash(paths):
    manifest_path = paths["feature_manifest_path"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    features_path = paths["features_path"]
    payload["outputs_sha256"][features_path.name] = _digest(features_path)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(paths, output_dir):
    return protocol.run_development_freeze(
        paths["features_path"],
        paths["labels_path"],
        paths["feature_manifest_path"],
        output_dir,
        checkpoint=paths["checkpoint"],
    )


def test_catalog_contains_exactly_the_six_unweighted_energy_formulas():
    catalog = protocol.candidate_catalog()

    assert [item.name for item in catalog] == [
        "S0",
        "S2",
        "S4",
        "S8",
        "Sbest4",
        "Sbest8",
    ]
    assert [(item.max_step, item.cost) for item in catalog] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (8, 9),
        (4, 5),
        (8, 9),
    ]
    assert catalog[0].energy_source == ("k0_energy_ev_per_atom",)
    assert catalog[4].energy_source == (
        "k0_energy_ev_per_atom",
        "k2_energy_ev_per_atom",
        "k4_energy_ev_per_atom",
    )
    assert catalog[5].energy_source[-1] == "k8_energy_ev_per_atom"
    serialized = json.dumps([item.as_record() for item in catalog]).lower()
    for forbidden in ("weight", "force", "stress", "suffix", "material", "sid"):
        assert forbidden not in serialized


def test_prepare_scores_uses_supported_group_min_and_fails_open_below_two():
    features = pd.DataFrame(
        {
            "sid": ["a", "b", "c", "d", "e"],
            "rk": ["g", "g", "g", "h", "h"],
            "stage": ["formula_selection"] * 5,
            "k2_energy_ev_per_atom": [1.0, 1.3, -20.0, 2.0, 3.0],
            "k2_supported": [True, True, False, True, False],
        }
    )

    got = protocol.prepare_fewstep_scores(features, "S2")

    assert got[["sid", "rk", "stage"]].to_dict("records") == features[
        ["sid", "rk", "stage"]
    ].to_dict("records")
    assert got["sid"].is_unique
    assert got.loc[got.sid.eq("a"), "score"].item() == pytest.approx(0.0)
    assert got.loc[got.sid.eq("b"), "score"].item() == pytest.approx(0.3)
    assert not bool(got.loc[got.sid.eq("c"), "supported"].item())
    assert np.isnan(got.loc[got.sid.eq("c"), "score"].item())
    assert not got.loc[got.rk.eq("h"), "supported"].any()
    assert got.loc[got.rk.eq("h"), "score"].isna().all()


def test_best_formulas_use_row_minimum_but_only_the_max_prefix_support():
    features = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["g"] * 3,
            "stage": ["search_calibration"] * 3,
            "k0_energy_ev_per_atom": [5.0, 4.0, 3.0],
            "k2_energy_ev_per_atom": [1.0, 4.0, 3.0],
            "k4_energy_ev_per_atom": [3.0, 2.0, 3.0],
            "k8_energy_ev_per_atom": [10.0, 0.0, -5.0],
            "k0_supported": [False, False, True],
            "k2_supported": [False, False, True],
            "k4_supported": [True, True, False],
            "k8_supported": [True, True, False],
        }
    )

    best4 = protocol.prepare_fewstep_scores(features, "Sbest4")
    best8 = protocol.prepare_fewstep_scores(features, "Sbest8")

    assert best4["supported"].tolist() == [True, True, False]
    assert best4["score"].iloc[:2].tolist() == pytest.approx([0.0, 1.0])
    assert best8["supported"].tolist() == [True, True, False]
    assert best8["score"].iloc[:2].tolist() == pytest.approx([1.0, 0.0])


def test_prepare_scores_rejects_duplicate_keys_and_test_rows():
    duplicate = pd.DataFrame(
        {
            "sid": ["same", "same"],
            "rk": ["g", "g"],
            "stage": ["search_calibration", "search_calibration"],
            "k0_energy_ev_per_atom": [0.0, 1.0],
            "k0_supported": [True, True],
        }
    )
    with pytest.raises(ValueError, match="sid.*unique"):
        protocol.prepare_fewstep_scores(duplicate, "S0")

    duplicate.loc[1, "sid"] = "other"
    duplicate.loc[1, "stage"] = "test"
    with pytest.raises(ValueError, match="development|test"):
        protocol.prepare_fewstep_scores(duplicate, "S0")


def test_tracks_are_the_two_and_only_two_frozen_semantics():
    assert set(protocol.TRACKS) == {"primary", "comparator"}
    primary = protocol.TRACKS["primary"]
    comparator = protocol.TRACKS["comparator"]
    assert (primary.protected, primary.within_group, primary.alpha) == (
        "valuable",
        "max",
        0.01,
    )
    assert (comparator.protected, comparator.within_group, comparator.alpha) == (
        "near_min",
        "min",
        0.035,
    )
    assert primary.protected_ev_per_atom == 0.05
    assert comparator.protected_ev_per_atom == 0.001


def _safe_frontier_row(name, track, savings, max_step, cost=None):
    return {
        "name": name,
        "track": track,
        "dft_savings": savings,
        "max_step": max_step,
        "cost": max_step + 1 if cost is None else cost,
        "exact_min_retention_lower": 0.96,
        "near_min_retention_lower": 0.96,
        "valuable_group_retention_lower": 0.96,
        "regret_p95": 0.05,
        "all_rejected_groups": 0,
    }


def test_safety_gate_and_tie_break_are_frozen_and_allow_s0_to_win():
    primary = _safe_frontier_row("S0", "primary", 0.4, 0)
    assert protocol.passes_safety_gate(primary, "primary")
    for field, unsafe in (
        ("exact_min_retention_lower", 0.949),
        ("valuable_group_retention_lower", 0.949),
        ("regret_p95", 0.051),
        ("all_rejected_groups", 1),
    ):
        candidate = dict(primary)
        candidate[field] = unsafe
        assert not protocol.passes_safety_gate(candidate, "primary")

    comparator = _safe_frontier_row("S0", "comparator", 0.4, 0)
    comparator["valuable_group_retention_lower"] = 0.0
    assert protocol.passes_safety_gate(comparator, "comparator")
    comparator["near_min_retention_lower"] = 0.949
    assert not protocol.passes_safety_gate(comparator, "comparator")

    frontier = pd.DataFrame(
        [
            _safe_frontier_row("S8", "primary", 0.40, 8),
            _safe_frontier_row("S2", "primary", 0.40, 2),
            _safe_frontier_row("S0", "primary", 0.41, 0),
        ]
    )
    selected = protocol.select_frozen_rule(frontier, "primary")
    assert selected["state"] == "selected"
    assert selected["name"] == "S0"

    frontier.loc[frontier.name.eq("S0"), "dft_savings"] = 0.40
    assert protocol.select_frozen_rule(frontier, "primary")["name"] == "S0"

    frontier["exact_min_retention_lower"] = 0.0
    assert protocol.select_frozen_rule(frontier, "primary") == {
        "state": "null_keep_all",
        "name": "null_keep_all",
    }


def test_selector_rejects_formula_names_outside_the_frozen_catalog():
    frontier = pd.DataFrame(
        [_safe_frontier_row("S9", "primary", 0.99, 9, cost=1)]
    )

    with pytest.raises(ValueError, match="frozen catalog"):
        protocol.select_frozen_rule(frontier, "primary")


def test_selector_breaks_savings_ties_by_cost_before_max_step():
    frontier = pd.DataFrame(
        [
            _safe_frontier_row("S0", "primary", 0.40, 0, cost=9),
            _safe_frontier_row("S8", "primary", 0.40, 8, cost=1),
        ]
    )

    selected = protocol.select_frozen_rule(frontier, "primary")

    assert selected["name"] == "S8"


@pytest.mark.parametrize("bad_stage", ["test", "missing_threshold_stage"])
def test_freeze_rejects_test_or_missing_development_stage(tmp_path, bad_stage):
    paths = _write_toy_inputs(tmp_path)
    features = pd.read_parquet(paths["features_path"])
    if bad_stage == "test":
        features.loc[0, "stage"] = "test"
    else:
        features = features.loc[
            ~features.stage.eq("threshold_calibration")
        ].copy()
    features.to_parquet(paths["features_path"], index=False)
    _refresh_feature_hash(paths)

    with pytest.raises(ValueError, match="development stages|test"):
        _run(paths, tmp_path / "freeze")


def test_freeze_requires_exact_sid_rk_key_match(tmp_path):
    paths = _write_toy_inputs(tmp_path)
    labels = pd.read_parquet(paths["labels_path"])
    labels.loc[0, "rk"] = "wrong-rk"
    labels.to_parquet(paths["labels_path"], index=False)

    with pytest.raises(ValueError, match="sid/rk.*differ"):
        _run(paths, tmp_path / "freeze")


@pytest.mark.parametrize(
    "mismatch",
    ["feature_sha", "checkpoint_sha", "protocol", "stages", "evidence"],
)
def test_freeze_rejects_manifest_checkpoint_and_hash_mismatches(
    tmp_path, mismatch
):
    paths = _write_toy_inputs(tmp_path)
    manifest_path = paths["feature_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mismatch == "feature_sha":
        manifest["outputs_sha256"][paths["features_path"].name] = "0" * 64
    elif mismatch == "checkpoint_sha":
        manifest["model"]["checkpoint_sha256"] = "0" * 64
    elif mismatch == "protocol":
        manifest["protocol"] = "stale-feature-protocol"
    elif mismatch == "stages":
        manifest["stages"] = ["search_calibration", "formula_selection"]
    else:
        manifest["evidence_role"] = "confirmatory"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|checkpoint|protocol|stages|evidence"):
        _run(paths, tmp_path / "freeze")


@pytest.mark.parametrize(
    "existing_name",
    [
        "development_frontier.parquet",
        "threshold_calibration_rules.parquet",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    ],
)
def test_freeze_never_overwrites_any_artifact(tmp_path, existing_name):
    paths = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    existing = output_dir / existing_name
    existing.write_bytes(b"preserve-me")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output_dir)

    assert existing.read_bytes() == b"preserve-me"
    assert [path.name for path in output_dir.iterdir()] == [existing_name]


def test_freeze_rejects_an_existing_empty_target_directory_without_deleting_it(
    tmp_path,
):
    paths = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_staging_failure_leaves_no_target_and_the_same_path_can_be_retried(
    tmp_path, monkeypatch
):
    paths = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "freeze"
    original_entries = set(tmp_path.iterdir())
    original_write_text = Path.write_text

    def fail_while_writing_frozen_protocol(path, *args, **kwargs):
        if path.name == "FROZEN_PROTOCOL.json":
            raise RuntimeError("injected frozen-protocol write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_while_writing_frozen_protocol)
    with pytest.raises(RuntimeError, match="injected"):
        _run(paths, output_dir)

    assert not output_dir.exists()
    assert set(tmp_path.iterdir()) == original_entries

    monkeypatch.setattr(Path, "write_text", original_write_text)
    manifest = _run(paths, output_dir)

    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == {
        "development_frontier.parquet",
        "threshold_calibration_rules.parquet",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    }
    assert manifest == json.loads(
        (output_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )


def test_atomic_publish_preserves_target_created_after_the_last_exists_check(
    tmp_path, monkeypatch
):
    paths = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "freeze"
    original_exists = Path.exists
    target_checks = 0

    def create_target_after_reporting_it_absent(path):
        nonlocal target_checks
        observed = original_exists(path)
        if path == output_dir:
            target_checks += 1
            if target_checks == 2:
                assert not observed
                path.mkdir()
        return observed

    monkeypatch.setattr(Path, "exists", create_target_after_reporting_it_absent)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output_dir)

    assert target_checks == 2
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize("unsupported", ["platform", "libc", "kernel"])
def test_atomic_no_replace_publisher_fails_closed_when_unsupported(
    tmp_path, monkeypatch, unsupported
):
    source = tmp_path / "staging"
    target = tmp_path / "published"
    source.mkdir()
    (source / "artifact").write_bytes(b"preserve-staging")

    if unsupported == "platform":
        monkeypatch.setattr(sys, "platform", "unsupported-os")
    elif unsupported == "libc":
        monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: object())
    else:
        class UnsupportedRename:
            argtypes = None
            restype = None

            def __call__(self, *_args):
                ctypes.set_errno(errno.ENOSYS)
                return -1

        libc = type("UnsupportedLibc", (), {"renameat2": UnsupportedRename()})()
        monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: libc)

    with pytest.raises(OSError, match="unsupported"):
        protocol._atomic_publish_directory_no_replace(source, target)

    assert source.is_dir()
    assert (source / "artifact").read_bytes() == b"preserve-staging"
    assert not target.exists()


def test_complete_toy_freeze_is_gate_compatible_and_hash_closed(tmp_path):
    paths = _write_toy_inputs(tmp_path)
    output_dir = tmp_path / "freeze"

    returned = _run(paths, output_dir)

    artifact_names = {
        "development_frontier.parquet",
        "threshold_calibration_rules.parquet",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    }
    assert {path.name for path in output_dir.iterdir()} == artifact_names
    frontier = pd.read_parquet(output_dir / "development_frontier.parquet")
    rules = pd.read_parquet(
        output_dir / "threshold_calibration_rules.parquet"
    )
    frozen = json.loads(
        (output_dir / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )

    assert len(frontier) == 12
    assert set(frontier["name"]) == {
        "S0",
        "S2",
        "S4",
        "S8",
        "Sbest4",
        "Sbest8",
    }
    assert set(frontier["track"]) == {"primary", "comparator"}
    assert frontier["passes_safety_gate"].all()
    assert len(rules) == 4
    assert set(rules["role"]) == {"selected", "s0_baseline"}
    assert set(rules["name"]) == {"S0"}

    assert frozen["protocol"] == (
        "2026-08-01-mattersim-fewstep-development-freeze-v1"
    )
    assert frozen["state"] == "frozen"
    assert frozen["frozen_at_utc"].endswith("+00:00")
    assert frozen["checkpoint_sha256"] == _digest(paths["checkpoint"])
    assert frozen["feature_inputs_sha256"] == paths[
        "feature_inputs_sha256"
    ]
    assert frozen["code_sha256"] == {
        "next7_mattersim_prerelax.py": _digest(
            Path(protocol.__file__).with_name("next7_mattersim_prerelax.py")
        ),
        "next7_mattersim_features.py": _digest(
            Path(protocol.__file__).with_name("next7_mattersim_features.py")
        ),
    }
    assert frozen["selection_code_sha256"] == _digest(protocol.__file__)
    assert [item["name"] for item in frozen["catalog"]] == [
        "S0",
        "S2",
        "S4",
        "S8",
        "Sbest4",
        "Sbest8",
    ]
    assert set(frozen["tracks"]) == {"primary", "comparator"}
    assert set(frozen["rules"]) == {"primary", "comparator"}
    assert all(rule["name"] == "S0" for rule in frozen["rules"].values())
    assert frozen["evidence_role"] == (
        "historically seen discovery; not confirmatory"
    )
    assert frozen["development_artifacts_sha256"] == {
        name: _digest(output_dir / name)
        for name in (
            "development_frontier.parquet",
            "threshold_calibration_rules.parquet",
        )
    }

    expected_inputs = {
        paths[key].name: _digest(paths[key])
        for key in (
            "features_path",
            "labels_path",
            "feature_manifest_path",
            "checkpoint",
        )
    }
    assert manifest["inputs_sha256"] == expected_inputs
    assert manifest["outputs_sha256"] == {
        name: _digest(output_dir / name)
        for name in (
            "development_frontier.parquet",
            "threshold_calibration_rules.parquet",
            "FROZEN_PROTOCOL.json",
        )
    }
    assert returned == manifest
    assert not (output_dir / "TEST_OPENING.json").exists()
    assert "test_metrics" not in json.dumps(frozen).lower()


def test_cli_forwards_only_the_development_freeze_inputs(
    tmp_path, monkeypatch, capsys
):
    captured = {}

    def fake_run(features, labels, feature_manifest, output, *, checkpoint):
        captured.update(
            {
                "features": features,
                "labels": labels,
                "feature_manifest": feature_manifest,
                "output": output,
                "checkpoint": checkpoint,
            }
        )
        return {"outputs_sha256": {"FROZEN_PROTOCOL.json": "f" * 64}}

    monkeypatch.setattr(protocol, "run_development_freeze", fake_run)
    arguments = [
        "--features",
        str(tmp_path / "features.parquet"),
        "--labels",
        str(tmp_path / "labels.parquet"),
        "--feature-manifest",
        str(tmp_path / "feature-manifest.json"),
        "--output",
        str(tmp_path / "freeze"),
        "--checkpoint",
        str(tmp_path / "model.pth"),
    ]

    assert protocol.main(arguments) == 0
    assert captured == {
        "features": tmp_path / "features.parquet",
        "labels": tmp_path / "labels.parquet",
        "feature_manifest": tmp_path / "feature-manifest.json",
        "output": tmp_path / "freeze",
        "checkpoint": tmp_path / "model.pth",
    }
    assert json.loads(capsys.readouterr().out) == {
        "FROZEN_PROTOCOL.json": "f" * 64
    }
