"""Pauling controls on the exact NEXT24 generated x0 cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _holdout(tmp_path: Path) -> Path:
    from src.next24_ssagen_holdout import freeze_ssagen_x0
    from tests.test_next24_ssagen_holdout import _source

    cohort, archive, manifest, _payloads = _source(tmp_path)
    output = tmp_path / "holdout"
    freeze_ssagen_x0(
        cohort_path=cohort,
        frames_zip_path=archive,
        source_manifest_path=manifest,
        output_dir=output,
        require_formal_inputs=False,
    )
    return output


def test_pauling_controls_are_identical_cohort_label_free_and_keep_abstentions(
    tmp_path: Path,
) -> None:
    from src import next24_pauling_controls as module

    holdout = _holdout(tmp_path)

    def fake(atoms):
        symbol = atoms.get_chemical_symbols()[0]
        if symbol == "Li":
            return {
                "p2_mean_dev": 0.0,
                "p3_frac_edge_face": 0.0,
                "p4_violate": 0.0,
                "p5_ok": 1.0,
            }, None
        if symbol == "Na":
            return {
                "p2_mean_dev": 1.0,
                "p3_frac_edge_face": 0.0,
                "p4_violate": 0.0,
                "p5_ok": 1.0,
            }, None
        return None, "oxidation failed"

    output = tmp_path / "pauling"
    manifest = module.run_next24_pauling_controls(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=output,
        feature_calculator=fake,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table.material_id.tolist() == sorted(table.material_id)
    assert set(table.pauling_p2_p5_decision) == {"KEEP", "REJECT", "ABSTAIN"}
    assert len(table) == 3
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["relaxed_structures_opened"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["rules_changed"] is False
    assert sum(manifest["counts"]["combined"].values()) == 3
    assert manifest["production_protocol_eligible"] is False


def test_pauling_validates_cohort_hash_no_replace_and_endpoint_cli(tmp_path: Path) -> None:
    from src.next24_pauling_controls import main, run_next24_pauling_controls

    holdout = _holdout(tmp_path)
    output = tmp_path / "pauling"
    kwargs = dict(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=output,
        feature_calculator=lambda _atoms: (None, "unsupported"),
    )
    run_next24_pauling_controls(**kwargs)
    with pytest.raises(FileExistsError):
        run_next24_pauling_controls(**kwargs)

    bad_manifest = json.loads((holdout / "MANIFEST.json").read_text())
    bad_manifest["outputs_sha256"]["holdout_metadata.parquet"] = "0" * 64
    (holdout / "MANIFEST.json").write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="hash"):
        run_next24_pauling_controls(
            **{**kwargs, "output_dir": tmp_path / "bad"}
        )

    for forbidden in ("--labels", "--threshold", "--dft-results", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2

