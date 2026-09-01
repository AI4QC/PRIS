from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

from ase import Atoms
from ase.io import write
import numpy as np
import pandas as pd
import pytest

from src.next11_geometry_only_frames import _load_archive_only
from src.next48_qmof_external_validation import (
    evaluate_qmof_relaxation,
    freeze_qmof_predictions,
    freeze_qmof_protocol,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_zip(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return payload.getvalue()


def _source_archive(path: Path) -> None:
    csv_text = (
        "qmof_id,name,info.formula,info.formula_reduced,info.natoms,info.source,"
        "outputs.pbe.energy_total\n"
        "qmof-a,alpha,C2 H2,CH,4,BoydWoo,-10.0\n"
        "qmof-b,beta,Zn O,ZnO,2,GMOF,-20.0\n"
    ).encode()
    initial = _nested_zip(
        {
            "unrelaxed_structures/other/alpha.cif": b"initial payload must stay unopened",
            "unrelaxed_structures/other/beta.cif": b"initial payload must stay unopened",
            "unrelaxed_structures/other/orphan.cif": b"unmapped payload",
            "unrelaxed_structures/README.md": b"fixture",
        }
    )
    relaxed = _nested_zip(
        {
            "relaxed_structures/qmof-a.cif": b"relaxed payload must stay unopened",
            "relaxed_structures/qmof-b.cif": b"relaxed payload must stay unopened",
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("qmof_database/qmof.csv", csv_text)
        archive.writestr("qmof_database/unrelaxed_structures.zip", initial)
        archive.writestr("qmof_database/relaxed_structures.zip", relaxed)


def _cif(atoms: Atoms) -> bytes:
    payload = io.BytesIO()
    write(payload, atoms, format="cif")
    return payload.getvalue()


def _prediction_source_archive(path: Path) -> None:
    csv_text = (
        "qmof_id,name,info.formula,info.formula_reduced,info.natoms,info.source,"
        "outputs.pbe.energy_total\n"
        "qmof-a,alpha,H2,H,2,BoydWoo,-10.0\n"
        "qmof-b,beta,He2,He,2,GMOF,-20.0\n"
    ).encode()
    initial = _nested_zip(
        {
            "unrelaxed_structures/other/alpha.cif": _cif(
                Atoms("H2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
            ),
            "unrelaxed_structures/other/beta.cif": _cif(
                Atoms("He2", positions=[[0, 0, 0], [2, 2, 2]], cell=[6, 6, 6], pbc=True)
            ),
        }
    )
    relaxed = _nested_zip(
        {
            "relaxed_structures/qmof-a.cif": b"this relaxed payload must not be read",
            "relaxed_structures/qmof-b.cif": b"nor this one",
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("qmof_database/qmof.csv", csv_text)
        archive.writestr("qmof_database/unrelaxed_structures.zip", initial)
        archive.writestr("qmof_database/relaxed_structures.zip", relaxed)


def _evaluation_source_archive(path: Path) -> None:
    csv_text = (
        "qmof_id,name,info.formula,info.formula_reduced,info.natoms,info.source,"
        "outputs.pbe.energy_total\n"
        "qmof-a,alpha,H He,HHe,2,BoydWoo,-10.0\n"
        "qmof-b,beta,Li F,LiF,2,GMOF,-20.0\n"
    ).encode()
    alpha_initial = Atoms(
        ["H", "He"], positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True
    )
    beta_initial = Atoms(
        ["Li", "F"], positions=[[0, 0, 0], [2, 2, 2]], cell=[6, 6, 6], pbc=True
    )
    alpha_final = Atoms(
        ["He", "H"],
        positions=[[1, 1, 1], [0, 0, 0]],
        cell=[155.0 ** (1.0 / 3.0)] * 3,
        pbc=True,
    )
    beta_final = Atoms(
        ["F", "Li"],
        positions=[[2, 2, 2], [0, 0, 0]],
        cell=[221.0 ** (1.0 / 3.0)] * 3,
        pbc=True,
    )
    initial = _nested_zip(
        {
            "unrelaxed_structures/other/alpha.cif": _cif(alpha_initial),
            "unrelaxed_structures/other/beta.cif": _cif(beta_initial),
        }
    )
    relaxed = _nested_zip(
        {
            "relaxed_structures/qmof-a.cif": _cif(alpha_final),
            "relaxed_structures/qmof-b.cif": _cif(beta_final),
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("qmof_database/qmof.csv", csv_text)
        archive.writestr("qmof_database/unrelaxed_structures.zip", initial)
        archive.writestr("qmof_database/relaxed_structures.zip", relaxed)


def _frozen_rule(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "eligible": True,
                "q05_median": 0.9831717659022418,
                "q05_iqr": 0.0955608356181108,
                "coord105_median": 2.152680652680653,
                "coord105_iqr": 3.418499623150786,
                "threshold": 2.4463648618269622,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_freeze_protocol_maps_without_opening_coordinate_payloads(tmp_path: Path) -> None:
    source = tmp_path / "qmof_database.zip"
    _source_archive(source)
    rule = tmp_path / "NEXT31_FROZEN_ENERGY_RULE.json"
    rule.write_text('{"eligible": true}\n', encoding="utf-8")
    output = tmp_path / "protocol"

    manifest = freeze_qmof_protocol(
        source_archive_path=source,
        frozen_rule_path=rule,
        output_dir=output,
        expected_archive_sha256=_sha256(source),
        expected_rule_sha256=_sha256(rule),
    )

    cohort = pd.read_parquet(output / "next48_qmof_cohort.parquet")
    protocol = json.loads((output / "NEXT48_QMOF_PROTOCOL.json").read_text())
    unmatched = json.loads((output / "NEXT48_QMOF_UNMATCHED.json").read_text())
    assert cohort[["material_id", "initial_name", "source_family"]].to_dict(
        "records"
    ) == [
        {"material_id": "qmof-a", "initial_name": "alpha", "source_family": "BoydWoo"},
        {"material_id": "qmof-b", "initial_name": "beta", "source_family": "GMOF"},
    ]
    assert unmatched == {"count": 1, "initial_names": ["orphan"]}
    assert protocol["initial_coordinate_payloads_opened"] is False
    assert protocol["relaxed_coordinate_payloads_opened"] is False
    assert protocol["metadata_columns_selected"] == [
        "qmof_id",
        "name",
        "info.formula",
        "info.formula_reduced",
        "info.natoms",
        "info.source",
    ]
    assert protocol["endpoint_columns_selected"] is False
    assert protocol["thresholds_refit"] is False
    assert protocol["endpoint"]["protected_max"] == 0.10
    assert protocol["endpoint"]["substantial_min"] == 0.20
    assert protocol["gates"]["auc_substantial_at_least"] == 0.85
    assert manifest["counts"] == {"initial_members": 3, "cohort": 2, "unmatched": 1}
    with pytest.raises(FileExistsError):
        freeze_qmof_protocol(
            source_archive_path=source,
            frozen_rule_path=rule,
            output_dir=output,
            expected_archive_sha256=_sha256(source),
            expected_rule_sha256=_sha256(rule),
        )


def test_freeze_protocol_rejects_archive_or_rule_drift(tmp_path: Path) -> None:
    source = tmp_path / "qmof_database.zip"
    _source_archive(source)
    rule = tmp_path / "rule.json"
    rule.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="archive hash"):
        freeze_qmof_protocol(
            source_archive_path=source,
            frozen_rule_path=rule,
            output_dir=tmp_path / "bad-archive",
            expected_archive_sha256="0" * 64,
            expected_rule_sha256=_sha256(rule),
        )
    with pytest.raises(ValueError, match="rule hash"):
        freeze_qmof_protocol(
            source_archive_path=source,
            frozen_rule_path=rule,
            output_dir=tmp_path / "bad-rule",
            expected_archive_sha256=_sha256(source),
            expected_rule_sha256="0" * 64,
        )


def test_prediction_freeze_reads_only_x0_and_fails_open(tmp_path: Path) -> None:
    source = tmp_path / "qmof_database.zip"
    _prediction_source_archive(source)
    rule = tmp_path / "NEXT31_FROZEN_ENERGY_RULE.json"
    _frozen_rule(rule)
    protocol_dir = tmp_path / "protocol"
    freeze_qmof_protocol(
        source_archive_path=source,
        frozen_rule_path=rule,
        output_dir=protocol_dir,
        expected_archive_sha256=_sha256(source),
        expected_rule_sha256=_sha256(rule),
    )

    def analytic(atoms: Atoms) -> dict[str, float]:
        if atoms[0].symbol == "He":
            raise ValueError("fixture analytic abstention")
        return {
            "periodic_nonbond_vdw_q05": 0.50,
            "periodic_contact_coord105": 10.0,
        }

    def pauling(atoms: Atoms) -> tuple[dict[str, float] | None, str | None]:
        del atoms
        return None, "fixture Pauling abstention"

    output = tmp_path / "predictions"
    manifest = freeze_qmof_predictions(
        source_archive_path=source,
        frozen_rule_path=rule,
        cohort_path=protocol_dir / "next48_qmof_cohort.parquet",
        protocol_path=protocol_dir / "NEXT48_QMOF_PROTOCOL.json",
        protocol_manifest_path=protocol_dir / "MANIFEST.json",
        output_dir=output,
        analytic_feature_calculator=analytic,
        pauling_feature_calculator=pauling,
    )

    predictions = pd.read_parquet(output / "next48_qmof_predictions.parquet")
    assert predictions[["material_id", "analytic_supported", "reject"]].to_dict(
        "records"
    ) == [
        {"material_id": "qmof-a", "analytic_supported": True, "reject": True},
        {"material_id": "qmof-b", "analytic_supported": False, "reject": False},
    ]
    assert predictions["pauling_p2_p5_decision"].tolist() == ["ABSTAIN", "ABSTAIN"]
    assert manifest["relaxed_coordinate_payloads_opened"] is False
    assert manifest["counts"]["analytic_supported"] == 1
    assert manifest["counts"]["rejected"] == 1
    ids, structures = _load_archive_only(
        output / "geometry_only_frames.zip", ["qmof-a", "qmof-b"]
    )
    assert ids == ["qmof-a", "qmof-b"]
    assert all(atoms.info == {} and set(atoms.arrays) == {"numbers", "positions"} for atoms in structures)


def test_evaluation_opens_relaxed_only_after_prediction_and_is_order_invariant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qmof_database.zip"
    _evaluation_source_archive(source)
    rule = tmp_path / "NEXT31_FROZEN_ENERGY_RULE.json"
    _frozen_rule(rule)
    protocol_dir = tmp_path / "protocol"
    freeze_qmof_protocol(
        source_archive_path=source,
        frozen_rule_path=rule,
        output_dir=protocol_dir,
        expected_archive_sha256=_sha256(source),
        expected_rule_sha256=_sha256(rule),
    )

    def analytic(atoms: Atoms) -> dict[str, float]:
        if atoms.get_volume() < 200.0:
            return {
                "periodic_nonbond_vdw_q05": 0.50,
                "periodic_contact_coord105": 10.0,
            }
        return {
            "periodic_nonbond_vdw_q05": 1.20,
            "periodic_contact_coord105": 0.0,
        }

    def pauling(atoms: Atoms) -> tuple[dict[str, float], None]:
        reject = atoms.get_volume() < 200.0
        return {
            "p2_mean_dev": 1.0 if reject else 0.0,
            "p3_frac_edge_face": 0.0,
            "p4_violate": 0.0,
            "p5_ok": 1.0,
        }, None

    prediction_dir = tmp_path / "predictions"
    freeze_qmof_predictions(
        source_archive_path=source,
        frozen_rule_path=rule,
        cohort_path=protocol_dir / "next48_qmof_cohort.parquet",
        protocol_path=protocol_dir / "NEXT48_QMOF_PROTOCOL.json",
        protocol_manifest_path=protocol_dir / "MANIFEST.json",
        output_dir=prediction_dir,
        analytic_feature_calculator=analytic,
        pauling_feature_calculator=pauling,
    )

    def fingerprint(atoms: Atoms) -> np.ndarray:
        return np.asarray([atoms.get_volume() / 100.0])

    output = tmp_path / "evaluation"
    manifest = evaluate_qmof_relaxation(
        source_archive_path=source,
        cohort_path=protocol_dir / "next48_qmof_cohort.parquet",
        protocol_path=protocol_dir / "NEXT48_QMOF_PROTOCOL.json",
        protocol_manifest_path=protocol_dir / "MANIFEST.json",
        predictions_path=prediction_dir / "next48_qmof_predictions.parquet",
        prediction_manifest_path=prediction_dir / "MANIFEST.json",
        geometry_path=prediction_dir / "geometry_only_frames.zip",
        output_dir=output,
        fingerprint_calculator=fingerprint,
    )

    joined = pd.read_parquet(output / "next48_qmof_joined_evaluation.parquet")
    result = json.loads((output / "NEXT48_QMOF_EVALUATION.json").read_text())
    assert joined["endpoint_supported"].tolist() == [True, True]
    assert joined["fingerprint_change"].to_numpy() == pytest.approx([0.30, 0.05])
    assert result["next31"]["counts"]["rejected_substantial"] == 1
    assert result["next31"]["counts"]["rejected_protected"] == 0
    assert result["next31"]["reject_precision"]["estimate"] == 1.0
    assert result["pauling_p2_p5"]["counts"]["rejected_substantial"] == 1
    assert result["endpoint_counts"] == {
        "protected": 1,
        "severe": 0,
        "substantial": 1,
        "supported": 2,
    }
    assert manifest["relaxed_coordinate_payloads_opened"] is True
    assert manifest["thresholds_refit"] is False


def test_evaluation_rejects_prediction_hash_drift_before_endpoint_opening(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qmof_database.zip"
    _prediction_source_archive(source)
    rule = tmp_path / "NEXT31_FROZEN_ENERGY_RULE.json"
    _frozen_rule(rule)
    protocol_dir = tmp_path / "protocol"
    freeze_qmof_protocol(
        source_archive_path=source,
        frozen_rule_path=rule,
        output_dir=protocol_dir,
        expected_archive_sha256=_sha256(source),
        expected_rule_sha256=_sha256(rule),
    )
    prediction_dir = tmp_path / "predictions"
    freeze_qmof_predictions(
        source_archive_path=source,
        frozen_rule_path=rule,
        cohort_path=protocol_dir / "next48_qmof_cohort.parquet",
        protocol_path=protocol_dir / "NEXT48_QMOF_PROTOCOL.json",
        protocol_manifest_path=protocol_dir / "MANIFEST.json",
        output_dir=prediction_dir,
        analytic_feature_calculator=lambda atoms: {
            "periodic_nonbond_vdw_q05": 1.0,
            "periodic_contact_coord105": 1.0,
        },
        pauling_feature_calculator=lambda atoms: (None, "abstain"),
    )
    predictions = pd.read_parquet(prediction_dir / "next48_qmof_predictions.parquet")
    predictions.loc[0, "reject"] = not bool(predictions.loc[0, "reject"])
    predictions.to_parquet(prediction_dir / "next48_qmof_predictions.parquet", index=False)

    with pytest.raises(ValueError, match="prediction output hash"):
        evaluate_qmof_relaxation(
            source_archive_path=source,
            cohort_path=protocol_dir / "next48_qmof_cohort.parquet",
            protocol_path=protocol_dir / "NEXT48_QMOF_PROTOCOL.json",
            protocol_manifest_path=protocol_dir / "MANIFEST.json",
            predictions_path=prediction_dir / "next48_qmof_predictions.parquet",
            prediction_manifest_path=prediction_dir / "MANIFEST.json",
            geometry_path=prediction_dir / "geometry_only_frames.zip",
            output_dir=tmp_path / "must-not-exist",
            fingerprint_calculator=lambda atoms: np.zeros(1),
        )
