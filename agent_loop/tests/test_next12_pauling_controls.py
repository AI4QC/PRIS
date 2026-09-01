"""Tests for frozen Pauling 2--5 controls on the NEXT12 cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms


class _Adapter:
    model_class = "fake.RealGenerator"
    model_parameter_count = 5
    latent_dim = 2
    device = "cpu"

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        del latent
        return Atoms(
            "Li2O",
            scaled_positions=[[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]],
            cell=np.eye(3) * (6.0 + attempt_index),
            pbc=True,
        )


def _cohort(tmp_path: Path) -> Path:
    from src.next12_prospective_cohort import freeze_prospective_cohort

    inputs = []
    for name in ("m", "l", "p"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs.append(path)
    output = tmp_path / "cohort"
    freeze_prospective_cohort(
        checkpoint_path=inputs[0],
        lattice_scaler_path=inputs[1],
        prop_scaler_path=inputs[2],
        output_dir=output,
        adapter=_Adapter(),
        attempt_count=3,
        seed=3,
    )
    return output


def test_rule_decision_and_conjunction_are_fail_closed() -> None:
    from src.next12_pauling_controls import _combined_decision, _rule_decision

    assert _rule_decision(0.02, operator="<=", threshold=0.01) == "REJECT"
    assert _rule_decision(np.nan, operator="<=", threshold=0.01) == "ABSTAIN"
    assert _combined_decision(["KEEP", "ABSTAIN", "REJECT", "KEEP"]) == "REJECT"
    assert _combined_decision(["KEEP", "ABSTAIN", "KEEP", "KEEP"]) == "ABSTAIN"
    assert _combined_decision(["KEEP"] * 4) == "KEEP"


def test_injected_control_run_retains_rows_and_never_opens_labels(tmp_path: Path) -> None:
    from src import next12_pauling_controls as module

    cohort = _cohort(tmp_path)
    calls = 0

    def calculator(atoms: Atoms):
        nonlocal calls
        del atoms
        index = calls
        calls += 1
        if index == 2:
            return None, "no oxidation state"
        return {
            "p2_mean_dev": 0.0,
            "p3_frac_edge_face": 0.2 if index == 0 else 0.0,
            "p4_violate": 0.0,
            "p5_ok": 1.0,
        }, None

    output = tmp_path / "controls"
    manifest = module.run_pauling_controls(
        cohort_path=cohort / "cohort.parquet",
        frames_zip_path=cohort / "geometry_only_frames.zip",
        cohort_manifest_path=cohort / "MANIFEST.json",
        output_dir=output,
        feature_calculator=calculator,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table["pauling_p2_p5_decision"].tolist() == [
        "REJECT",
        "KEEP",
        "ABSTAIN",
    ]
    assert table.loc[2, "pauling_p2_decision"] == "ABSTAIN"
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["production_protocol_eligible"] is False
    assert manifest["counts"]["rows"] == 3
    assert manifest["counts"]["combined"] == {
        "KEEP": 1,
        "REJECT": 1,
        "ABSTAIN": 1,
    }
    with pytest.raises(FileExistsError):
        module.run_pauling_controls(
            cohort_path=cohort / "cohort.parquet",
            frames_zip_path=cohort / "geometry_only_frames.zip",
            cohort_manifest_path=cohort / "MANIFEST.json",
            output_dir=output,
            feature_calculator=calculator,
        )


def test_cli_has_no_label_endpoint_or_threshold_arguments() -> None:
    from src.next12_pauling_controls import main

    for forbidden in ("--labels", "--endpoint", "--threshold", "--dft"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
