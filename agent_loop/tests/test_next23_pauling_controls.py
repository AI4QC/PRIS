"""Pauling controls on the exact NEXT23 blind geometry cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_next23_pauling_controls_are_label_free_and_keep_abstentions(
    tmp_path: Path,
) -> None:
    from src import next23_pauling_controls as module
    from src.next23_wbm_holdout import freeze_disjoint_wbm_holdout
    from tests.test_next23_wbm_holdout import _inputs

    features, archive, upstream, _ids = _inputs(tmp_path)
    exclusion = tmp_path / "exclude.parquet"
    pd.DataFrame({"material_id": ["wbm-f"]}).to_parquet(exclusion, index=False)
    holdout = tmp_path / "holdout"
    freeze_disjoint_wbm_holdout(
        test_features_path=features,
        wbm_manifest_path=upstream,
        initial_zip_path=archive,
        exclusion_metadata_path=exclusion,
        output_dir=holdout,
        sample_size=3,
        min_atoms=2,
        max_atoms=4,
        require_formal_inputs=False,
    )

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
    manifest = module.run_next23_pauling_controls(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=output,
        feature_calculator=fake,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert set(table.pauling_p2_p5_decision) <= {"KEEP", "REJECT", "ABSTAIN"}
    assert len(table) == 3
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["thresholds_refit"] is False
    assert sum(manifest["counts"]["combined"].values()) == 3


def test_next23_pauling_cli_has_no_label_threshold_or_endpoint_argument() -> None:
    from src.next23_pauling_controls import main

    for forbidden in ("--labels", "--threshold", "--dft-results", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2

