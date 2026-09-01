"""Contract tests for the NEXT12 prospective M5+PHSC+CHSC gate run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms


def _frozen_protocol(path: Path) -> Path:
    payload = {
        "protocol": "2026-08-01-mattersim-committee-development-freeze-v1",
        "final_rules": [
            {
                "track": "primary",
                "formula": "M5",
                "operator": "score > threshold",
                "threshold": 0.12119269371032715,
                "threshold_state": "finite",
                "threshold_source_role": "threshold_fit",
                "unsupported_decision": "ABSTAIN",
                "within_group": "max",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_primary_m5_rule_is_exact_and_fail_closed(tmp_path: Path) -> None:
    from src.next12_prospective_gates import (
        FROZEN_PRIMARY_M5_THRESHOLD,
        _extract_frozen_primary_m5_rule,
    )

    protocol = json.loads(_frozen_protocol(tmp_path / "protocol.json").read_text())
    rule = _extract_frozen_primary_m5_rule(protocol)
    assert rule == {
        "operator": "score > threshold",
        "threshold": FROZEN_PRIMARY_M5_THRESHOLD,
        "unsupported_decision": "ABSTAIN",
        "within_group": "max",
    }
    protocol["final_rules"][0]["threshold"] += 1e-12
    with pytest.raises(ValueError, match="frozen primary M5"):
        _extract_frozen_primary_m5_rule(protocol)


def test_m5_gap_is_within_exact_composition_and_singletons_are_explicit() -> None:
    from src.next12_prospective_gates import FROZEN_PRIMARY_M5_THRESHOLD, _m5_table

    result = _m5_table(
        sids=["a", "b", "c"],
        formulas=["H2", "H2", "He"],
        natoms=[2, 2, 1],
        total_energies=[-2.0, -1.0, -0.25],
        threshold=FROZEN_PRIMARY_M5_THRESHOLD,
    )
    assert result["m5_group_size"].tolist() == [2, 2, 1]
    assert result["m5_has_competitor"].tolist() == [True, True, False]
    assert result["m5_gap_ev_per_atom"].tolist() == [0.0, 0.5, 0.0]
    assert result["m5_decision"].tolist() == ["KEEP", "REJECT", "KEEP"]


@pytest.mark.parametrize(
    ("baseline", "phsc", "chsc", "expected"),
    [
        ("KEEP", "resolved_nonnegative", "resolved_nonnegative", "KEEP"),
        ("KEEP", "resolved_negative", "resolved_nonnegative", "REJECT"),
        ("KEEP", "resolved_nonnegative", "resolved_negative", "REJECT"),
        ("REJECT", "near_zero_or_inconsistent", "resolved_nonnegative", "REJECT"),
        ("ABSTAIN", "resolved_negative", "resolved_negative", "ABSTAIN"),
        ("KEEP", "abstain_numerical_failure", "resolved_negative", "ABSTAIN"),
        ("REJECT", "resolved_negative", "abstain_unsupported_geometry", "ABSTAIN"),
    ],
)
def test_composition_rule_is_frozen_before_prospective_results(
    baseline: str, phsc: str, chsc: str, expected: str
) -> None:
    from src.next12_prospective_gates import _compose_decision

    assert _compose_decision(baseline, phsc, chsc) == expected


class _TwoStructureAdapter:
    model_class = "fake.RealProspectiveGenerator"
    model_parameter_count = 17
    latent_dim = 3
    device = "cpu"

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        del latent
        side = 4.0 + attempt_index
        return Atoms(
            "H2",
            scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
            cell=np.eye(3) * side,
            pbc=True,
        )


class _BaseEnergyPredictor:
    device = "cpu"

    def __call__(self, structures: list[Atoms]):
        from src.next10_lrrc_mattersim_features import BatchPrediction

        if len(structures) != 2:
            raise AssertionError("the patched diagnostics must leave only one base call")
        return BatchPrediction(
            total_energies_ev=[-2.0, -1.0],
            forces_ev_per_a=[np.zeros((len(atoms), 3)) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )


def test_full_injected_run_is_label_free_aligned_and_nonoverwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next12_prospective_gates as module
    from src.next11_phsc import PHSCResult, PHSCStatus
    from src.next11_phsc_mattersim_features import BatchPHSCResult
    from src.next12_chsc import CHSCResult, CHSCStatus
    from src.next12_chsc_mattersim_features import BatchCHSCResult
    from src.next12_prospective_cohort import freeze_prospective_cohort

    inputs = {}
    for name in ("model.ckpt", "lattice.pt", "prop.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = path
    cohort_dir = tmp_path / "cohort"
    freeze_prospective_cohort(
        checkpoint_path=inputs["model.ckpt"],
        lattice_scaler_path=inputs["lattice.pt"],
        prop_scaler_path=inputs["prop.pt"],
        output_dir=cohort_dir,
        adapter=_TwoStructureAdapter(),
        attempt_count=2,
        seed=31,
    )

    def fake_phsc(sids, structures, predictor, *, groups_per_call):
        del structures, predictor, groups_per_call
        return tuple(
            BatchPHSCResult(
                sid,
                PHSCResult(status=PHSCStatus.RESOLVED_NONNEGATIVE, negative=False),
            )
            for sid in sorted(sids)
        )

    def fake_chsc(sids, structures, predictor, *, structures_per_call):
        del structures, predictor, structures_per_call
        rows = []
        for index, sid in enumerate(sorted(sids)):
            negative = index == 0
            rows.append(
                BatchCHSCResult(
                    sid=sid,
                    result=CHSCResult(
                        status=(
                            CHSCStatus.RESOLVED_NEGATIVE
                            if negative
                            else CHSCStatus.RESOLVED_NONNEGATIVE
                        ),
                        negative=negative,
                    ),
                    hessian_h=np.zeros((6, 6)),
                    hessian_h2=np.zeros((6, 6)),
                )
            )
        return rows

    monkeypatch.setattr(module, "evaluate_phsc_batch", fake_phsc)
    monkeypatch.setattr(module, "evaluate_chsc_batch", fake_chsc)
    checkpoint = tmp_path / "m5.pth"
    checkpoint.write_bytes(b"m5")
    output = tmp_path / "gates"
    manifest = module.run_prospective_gates(
        cohort_path=cohort_dir / "cohort.parquet",
        frames_zip_path=cohort_dir / "geometry_only_frames.zip",
        cohort_manifest_path=cohort_dir / "MANIFEST.json",
        frozen_protocol_path=_frozen_protocol(tmp_path / "frozen.json"),
        checkpoint_path=checkpoint,
        output_dir=output,
        predictor=_BaseEnergyPredictor(),
        device="cpu",
        model_batch_size=2,
        phsc_groups_per_call=3,
        chsc_structures_per_call=2,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table["sid"].is_unique
    assert table["m5_decision"].tolist() == ["KEEP", "REJECT"]
    assert table["chsc_status"].tolist() == [
        "resolved_negative",
        "resolved_nonnegative",
    ]
    assert table["composed_decision"].tolist() == ["REJECT", "REJECT"]
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["production_protocol_eligible"] is False
    assert manifest["counts"]["rows"] == 2
    assert manifest["counts"]["composed_reject"] == 2
    assert manifest["outputs_sha256"][module.OUTPUT_NAME]
    with pytest.raises(FileExistsError):
        module.run_prospective_gates(
            cohort_path=cohort_dir / "cohort.parquet",
            frames_zip_path=cohort_dir / "geometry_only_frames.zip",
            cohort_manifest_path=cohort_dir / "MANIFEST.json",
            frozen_protocol_path=tmp_path / "frozen.json",
            checkpoint_path=checkpoint,
            output_dir=output,
            predictor=_BaseEnergyPredictor(),
        )


def test_cli_exposes_no_label_endpoint_or_threshold_override() -> None:
    from src.next12_prospective_gates import main

    for forbidden in ("--labels", "--endpoint", "--threshold", "--dft-energy"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
