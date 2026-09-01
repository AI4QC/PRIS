"""Contracts for label-free Pauling controls on the frozen WBM holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_pauling_controls_keep_reject_and_abstain_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next14_wbm_pauling as module
    from tests.test_next14_wbm_holdout import _inputs
    from src.next14_wbm_holdout import freeze_wbm_holdout

    features, archive, upstream = _inputs(tmp_path)
    holdout = tmp_path / "holdout"
    freeze_wbm_holdout(
        test_features_path=features,
        wbm_manifest_path=upstream,
        initial_zip_path=archive,
        output_dir=holdout,
        sample_size=3,
        min_atoms=2,
        max_atoms=4,
        require_formal_inputs=False,
    )

    def fake(atoms):
        symbol = atoms.get_chemical_symbols()[0]
        if symbol == "Li":
            return {"p2_mean_dev": 0.0, "p3_frac_edge_face": 0.0, "p4_violate": 0.0, "p5_ok": 1.0}, None
        if symbol == "Na":
            return {"p2_mean_dev": 1.0, "p3_frac_edge_face": 0.0, "p4_violate": 0.0, "p5_ok": 1.0}, None
        return None, "oxidation failed"

    output = tmp_path / "pauling"
    manifest = module.run_wbm_pauling_controls(
        metadata_path=holdout / "holdout_metadata.parquet",
        frames_zip_path=holdout / "geometry_only_frames.zip",
        holdout_manifest_path=holdout / "MANIFEST.json",
        output_dir=output,
        feature_calculator=fake,
        require_formal_inputs=False,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    expected = {"wbm-a": "KEEP", "wbm-b": "REJECT"}
    for row in table.itertuples(index=False):
        if row.material_id in expected:
            assert row.pauling_p2_p5_decision == expected[row.material_id]
        else:
            assert row.pauling_p2_p5_decision == "ABSTAIN"
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert sum(manifest["counts"]["combined"].values()) == 3


def test_pauling_cli_has_no_label_or_threshold_argument() -> None:
    from src.next14_wbm_pauling import main

    for forbidden in ("--labels", "--threshold", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
