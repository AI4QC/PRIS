"""Pauling controls on the exact NEXT25 OMatG generated-x0 cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _holdout(tmp_path: Path) -> Path:
    from src.next25_omatg_holdout import freeze_omatg_x0
    from tests.test_next25_omatg_holdout import _source

    cohort, composition_manifest, generated, run_manifest, _frames = _source(tmp_path)
    output = tmp_path / "holdout"
    freeze_omatg_x0(
        composition_cohort_path=cohort,
        composition_manifest_path=composition_manifest,
        generated_xyz_path=generated,
        generation_manifest_path=run_manifest,
        output_dir=output,
        require_formal_inputs=False,
    )
    return output


def test_pauling_is_identical_cohort_label_free_and_keeps_abstentions(tmp_path: Path) -> None:
    from src import next25_pauling_controls as module

    holdout = _holdout(tmp_path)

    def fake(atoms):
        first = atoms.get_chemical_symbols()[0]
        if first == "Li":
            return {
                "p2_mean_dev": 0.0,
                "p3_frac_edge_face": 0.0,
                "p4_violate": 0.0,
                "p5_ok": 1.0,
            }, None
        return None, "unsupported"

    output = tmp_path / "pauling"
    manifest = module.run_next25_pauling_controls(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=output,
        feature_calculator=fake,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table.material_id.tolist() == sorted(table.material_id)
    assert set(table.pauling_p2_p5_decision) == {"KEEP", "ABSTAIN"}
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["rules_changed"] is False
    assert sum(manifest["counts"]["combined"].values()) == 2
    assert manifest["production_protocol_eligible"] is False


def test_pauling_validates_hash_no_replace_and_endpoint_cli(tmp_path: Path) -> None:
    from src.next25_pauling_controls import main, run_next25_pauling_controls

    holdout = _holdout(tmp_path)
    kwargs = dict(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=tmp_path / "pauling",
        feature_calculator=lambda _atoms: (None, "unsupported"),
    )
    run_next25_pauling_controls(**kwargs)
    with pytest.raises(FileExistsError):
        run_next25_pauling_controls(**kwargs)
    bad = json.loads((holdout / "MANIFEST.json").read_text())
    bad["outputs_sha256"]["holdout_metadata.parquet"] = "0" * 64
    (holdout / "MANIFEST.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        run_next25_pauling_controls(**{**kwargs, "output_dir": tmp_path / "bad"})
    for forbidden in ("--labels", "--threshold", "--reference", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
