"""Contracts for freezing NEXT23 predictions before NEXT42 endpoints open."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
import pytest

from src.next11_geometry_only_frames import _canonical_frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort(tmp_path: Path, *, opened: bool = False) -> tuple[Path, Path, Path]:
    from src.next42_alexandria_cohort import (
        COHORT_NAME,
        GEOMETRY_NAME,
        INPUT_ROLE,
        PROTOCOL,
    )

    root = tmp_path / ("cohort-opened" if opened else "cohort")
    root.mkdir()
    metadata = pd.DataFrame(
        [
            {
                "material_id": "a",
                "source_family": "cgat_comp/ternaries",
                "source_shard": "pbe_0000",
                "formula": "LiO",
                "reduced_formula": "LiO",
                "natoms": 2,
                "input_role": INPUT_ROLE,
            },
            {
                "material_id": "b",
                "source_family": "cgat_comp/binaries",
                "source_shard": "pbe_0001",
                "formula": "NaCl",
                "reduced_formula": "NaCl",
                "natoms": 2,
                "input_role": INPUT_ROLE,
            },
        ]
    )
    metadata_path = root / COHORT_NAME
    metadata.to_parquet(metadata_path, index=False)
    geometry_path = root / GEOMETRY_NAME
    atoms = {
        "a": Atoms("LiO", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True),
        "b": Atoms("NaCl", positions=[[0, 0, 0], [2, 2, 2]], cell=[5, 5, 5], pbc=True),
    }
    with zipfile.ZipFile(geometry_path, "w") as archive:
        for material_id in sorted(atoms):
            info = zipfile.ZipInfo(
                f"{material_id}.extxyz", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _canonical_frame(atoms[material_id]))
    manifest = {
        "protocol": PROTOCOL,
        "input_role": INPUT_ROLE,
        "later_geometry_accessed": opened,
        "dft_values_read": False,
        "mlip_prerelaxation_used": False,
        "physical_relaxation_executed": False,
        "selection": {"endpoint_fields_used": False, "sampled": False},
        "outputs_sha256": {
            COHORT_NAME: _sha(metadata_path),
            GEOMETRY_NAME: _sha(geometry_path),
        },
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return metadata_path, geometry_path, manifest_path


def _rule(tmp_path: Path) -> tuple[Path, Path]:
    from src.next39_next23_predictions import RULE_PROTOCOL, TERM_COLUMNS

    rule = {
        "protocol": RULE_PROTOCOL,
        "selected_candidate": "B+E",
        "selected_terms": ["B", "E"],
        "reject_when": "supported and score >= threshold",
        "missing_policy": "fail_open_do_not_reject",
        "dft_or_relaxed_input_used": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "threshold": 1.0,
        "base_parameters": {
            "B": {
                "column": TERM_COLUMNS[0],
                "direction": 1,
                "median": 0.0,
                "scale_iqr": 1.0,
            },
            "E": {
                "column": TERM_COLUMNS[1],
                "direction": 1,
                "median": 0.0,
                "scale_iqr": 1.0,
            },
        },
    }
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps(rule), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    sources = (
        "src/next20_valence_rigidity.py",
        "src/next21_normalized_madelung.py",
        "src/next22_bond_valence_equilibrium.py",
        "src/next23_relaxation_rule.py",
    )
    manifest = {
        "protocol": RULE_PROTOCOL,
        "blind_labels_opened": False,
        "outputs_sha256": {rule_path.name: _sha(rule_path)},
        "executed_source_sha256": {
            name: _sha(repository / name) for name in sources
        },
    }
    manifest_path = tmp_path / "rule-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return rule_path, manifest_path


def test_predictions_freeze_without_opening_endpoint_or_refitting(tmp_path: Path) -> None:
    from src import next42_next23_predictions as module

    metadata, geometry, cohort_manifest = _cohort(tmp_path)
    rule, rule_manifest = _rule(tmp_path)

    def terms(atoms: Atoms):
        value = 0.75 if atoms[0].symbol == "Li" else 0.25
        return {module.TERM_COLUMNS[0]: value, module.TERM_COLUMNS[1]: value}, None

    target = tmp_path / "predictions"
    manifest = module.run_next42_predictions(
        metadata_path=metadata,
        frames_zip_path=geometry,
        cohort_manifest_path=cohort_manifest,
        frozen_rule_path=rule,
        frozen_rule_manifest_path=rule_manifest,
        output_dir=target,
        term_calculator=terms,
        pauling_feature_calculator=lambda _atoms: ({}, None),
    )
    table = pd.read_parquet(target / module.PREDICTIONS_NAME)
    assert table.material_id.tolist() == ["a", "b"]
    assert table.next23_supported.tolist() == [True, True]
    assert table.next23_reject.tolist() == [True, False]
    assert manifest["later_geometry_opened"] is False
    assert manifest["dft_values_read"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["model_or_proxy_potential_used"] is False
    assert manifest["input_role"] == "one_raw_pre_dft_pre_mlip_x0_only"
    assert isinstance(manifest["frozen_at_utc"], str)
    with pytest.raises(FileExistsError):
        module.run_next42_predictions(
            metadata_path=metadata,
            frames_zip_path=geometry,
            cohort_manifest_path=cohort_manifest,
            frozen_rule_path=rule,
            frozen_rule_manifest_path=rule_manifest,
            output_dir=target,
            term_calculator=terms,
            pauling_feature_calculator=lambda _atoms: ({}, None),
        )


def test_predictions_refuse_a_cohort_whose_later_geometry_was_opened(tmp_path: Path) -> None:
    from src.next42_next23_predictions import run_next42_predictions

    metadata, geometry, cohort_manifest = _cohort(tmp_path, opened=True)
    rule, rule_manifest = _rule(tmp_path)
    with pytest.raises(ValueError, match="crossed the prediction boundary"):
        run_next42_predictions(
            metadata_path=metadata,
            frames_zip_path=geometry,
            cohort_manifest_path=cohort_manifest,
            frozen_rule_path=rule,
            frozen_rule_manifest_path=rule_manifest,
            output_dir=tmp_path / "bad",
            term_calculator=lambda _atoms: ({}, None),
            pauling_feature_calculator=lambda _atoms: ({}, None),
        )


def test_prediction_cli_cannot_accept_endpoints_or_refit_controls() -> None:
    from src.next42_next23_predictions import main

    for forbidden in ("--labels", "--final", "--energy", "--refit", "--threshold"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
