"""Contract tests for the license-safe NEXT12 VASP queue builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    model_class = "fake.RealGenerator"
    model_parameter_count = 9
    latent_dim = 2
    device = "cpu"

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        del latent
        return Atoms(
            "Li2O",
            scaled_positions=[[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]],
            cell=np.eye(3) * (5.0 + attempt_index),
            pbc=True,
        )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next12_prospective_cohort import freeze_prospective_cohort
    from src.next12_prospective_gates import PROTOCOL as GATE_PROTOCOL
    from src.next12_pauling_controls import PROTOCOL as PAULING_PROTOCOL

    model_inputs = []
    for name in ("m", "l", "p"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        model_inputs.append(path)
    cohort = tmp_path / "cohort"
    freeze_prospective_cohort(
        checkpoint_path=model_inputs[0],
        lattice_scaler_path=model_inputs[1],
        prop_scaler_path=model_inputs[2],
        output_dir=cohort,
        adapter=_Adapter(),
        attempt_count=2,
        seed=5,
    )
    c = pd.read_parquet(cohort / "cohort.parquet")
    gates = tmp_path / "gates.parquet"
    pd.DataFrame(
        {
            "sid": c.sid,
            "m5_decision": ["KEEP", "REJECT"],
            "m5_phsc_decision": ["REJECT", "REJECT"],
            "composed_decision": ["REJECT", "REJECT"],
        }
    ).to_parquet(gates, index=False)
    gate_manifest = tmp_path / "gate-manifest.json"
    gate_manifest.write_text(
        json.dumps(
            {
                "protocol": GATE_PROTOCOL,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "thresholds_refit": False,
                "outputs_sha256": {gates.name: _sha(gates)},
            }
        ),
        encoding="utf-8",
    )
    pauling = tmp_path / "pauling.parquet"
    pd.DataFrame(
        {
            "sid": c.sid,
            "pauling_p2_decision": ["REJECT", "KEEP"],
            "pauling_p3_decision": ["KEEP", "KEEP"],
            "pauling_p4_decision": ["KEEP", "KEEP"],
            "pauling_p5_decision": ["REJECT", "ABSTAIN"],
            "pauling_p2_p5_decision": ["REJECT", "ABSTAIN"],
        }
    ).to_parquet(pauling, index=False)
    pauling_manifest = tmp_path / "pauling-manifest.json"
    pauling_manifest.write_text(
        json.dumps(
            {
                "protocol": PAULING_PROTOCOL,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "thresholds_refit": False,
                "outputs_sha256": {pauling.name: _sha(pauling)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "cohort": cohort / "cohort.parquet",
        "frames": cohort / "geometry_only_frames.zip",
        "cohort_manifest": cohort / "MANIFEST.json",
        "gates": gates,
        "gate_manifest": gate_manifest,
        "pauling": pauling,
        "pauling_manifest": pauling_manifest,
    }


def _potcars(tmp_path: Path) -> Path:
    root = tmp_path / "potcars"
    for symbol, enmax, marker in (("Li_sv", 300.0, "LICENSED-LI"), ("O", 400.0, "LICENSED-O")):
        directory = root / symbol
        directory.mkdir(parents=True)
        (directory / "POTCAR").write_text(
            f"TITEL = PAW_PBE {symbol} TEST\nENMAX = {enmax}; ENMIN = 1\n{marker}\n",
            encoding="utf-8",
        )
    return root


def test_queue_contains_every_sid_and_no_licensed_potcar_payload(tmp_path: Path) -> None:
    from src import next12_dft_queue as module

    inputs = _inputs(tmp_path)
    output = tmp_path / "queue"
    manifest = module.build_dft_queue(
        cohort_path=inputs["cohort"],
        frames_zip_path=inputs["frames"],
        cohort_manifest_path=inputs["cohort_manifest"],
        gate_features_path=inputs["gates"],
        gate_manifest_path=inputs["gate_manifest"],
        pauling_features_path=inputs["pauling"],
        pauling_manifest_path=inputs["pauling_manifest"],
        potcar_root=_potcars(tmp_path),
        output_dir=output,
        potential_map={"Li": "Li_sv", "O": "O"},
    )
    queue = pd.read_parquet(output / module.QUEUE_NAME)
    assert len(queue) == 2
    assert queue.sid.is_unique
    assert queue["task_prefix"].tolist() == [
        f"tasks/{sid}" for sid in queue.sid
    ]
    assert queue["encut_ev"].tolist() == [520, 520]
    assert queue["m5_phsc_chsc_decision"].tolist() == ["REJECT", "REJECT"]
    assert manifest["all_attempts_queued"] is True
    assert manifest["selection_by_gate"] is False
    assert manifest["licensed_potcar_contents_included"] is False
    assert manifest["production_protocol_eligible"] is False
    assert manifest["counts"] == {"attempts": 2, "tasks": 2, "total_atoms": 6}

    payload = (output / module.TASKS_NAME).read_bytes()
    assert b"LICENSED-LI" not in payload
    assert b"LICENSED-O" not in payload
    with zipfile.ZipFile(output / module.TASKS_NAME) as archive:
        names = archive.namelist()
        for sid in queue.sid:
            prefix = f"tasks/{sid}/"
            assert prefix + "POSCAR.x0" in names
            assert prefix + "INCAR.static" in names
            assert prefix + "INCAR.relax" in names
            assert prefix + "POTCAR.spec.json" in names
            assert prefix + "TASK.json" in names
            assert not any(name == prefix + "POTCAR" for name in names)
        incar = archive.read(f"tasks/{queue.sid.iloc[0]}/INCAR.relax").decode()
        assert "ENCUT = 520" in incar
        assert "KSPACING = 0.22" in incar
        assert "KGAMMA = .TRUE." in incar
        assert "IBRION = 2" in incar
        assert "ISIF = 3" in incar
        assert "EDIFFG = -0.03" in incar
        assert "ISPIN = 2" in incar
        assert not any(name.endswith("/KPOINTS") for name in names)
        spec = json.loads(
            archive.read(f"tasks/{queue.sid.iloc[0]}/POTCAR.spec.json")
        )
        assert [row["potential"] for row in spec["sequence"]] == ["Li_sv", "O"]
        assert all(len(row["content_sha256"]) == 64 for row in spec["sequence"])


def test_queue_refuses_overwrite(tmp_path: Path) -> None:
    from src.next12_dft_queue import build_dft_queue

    inputs = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_dft_queue(
            cohort_path=inputs["cohort"],
            frames_zip_path=inputs["frames"],
            cohort_manifest_path=inputs["cohort_manifest"],
            gate_features_path=inputs["gates"],
            gate_manifest_path=inputs["gate_manifest"],
            pauling_features_path=inputs["pauling"],
            pauling_manifest_path=inputs["pauling_manifest"],
            potcar_root=_potcars(tmp_path),
            output_dir=output,
            potential_map={"Li": "Li_sv", "O": "O"},
        )


def test_cli_has_no_subset_or_gate_selection_argument() -> None:
    from src.next12_dft_queue import main

    for forbidden in ("--subset", "--only-rejected", "--sample", "--labels"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
