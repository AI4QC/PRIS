from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME,
    MANIFEST_NAME,
    PARTITIONS,
    SPLIT_SALT,
    _blind_partition_for_reduced_formula,
    build_wyformer_blind_lockbox,
)


def _payload(species: list[str], scale: float) -> str:
    n = len(species)
    structure = Structure(
        Lattice.cubic(5.0 * scale),
        species,
        [[i / n, (i + 1) / (n + 1), (2 * i + 1) / (2 * n + 1)] for i in range(n)],
    )
    return json.dumps(structure.as_dict(), sort_keys=True)


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    raw = pd.DataFrame(
        {
            "material_id": [0, 1, 2, 3],
            "structure": [
                _payload(["Li", "Li", "O"], 1.0),
                _payload(["Na", "Cl"], 1.1),
                _payload(["Mg", "O"], 1.2),
                _payload(["K", "Br"], 1.3),
            ],
            "group": [225, 225, 225, 221],
        }
    )
    dft = pd.DataFrame(
        {
            "material_id": [500, 499, 498],
            "structure": [raw.loc[1, "structure"], raw.loc[0, "structure"], raw.loc[3, "structure"]],
            "dft_e_above_hull_corrected": [0.8, 0.02, 0.3],
        }
    )
    raw_path = root / "raw.csv.gz"
    dft_path = root / "dft.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    dft.to_csv(dft_path, index=False, compression="gzip")
    article = root / "article.json"
    article.write_text(json.dumps({"id": 29094701, "files": []}), encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("indices unfortunately got permuted", encoding="utf-8")
    design = root / "design.md"
    design.write_text("blind reroute frozen", encoding="utf-8")
    return raw_path, dft_path, article, readme, design


def test_blind_split_uses_new_fixed_salt_and_is_deterministic() -> None:
    assert SPLIT_SALT == "NEXT93B_WYFORMER_BLIND_REDUCED_FORMULA_SPLIT_V1"
    assert _blind_partition_for_reduced_formula("Li2O") in PARTITIONS
    assert _blind_partition_for_reduced_formula("Li2O") == _blind_partition_for_reduced_formula(
        "Li2O"
    )


def test_formal_return_and_manifests_do_not_disclose_partition_label_counts(
    tmp_path: Path,
) -> None:
    raw, dft, article, readme, design = _inputs(tmp_path)
    cohort = tmp_path / "cohort"
    endpoints = {role: tmp_path / f"endpoint-{role}" for role in PARTITIONS}
    manifest = build_wyformer_blind_lockbox(
        raw_x0_path=raw,
        dft_success_path=dft,
        figshare_metadata_path=article,
        readme_path=readme,
        design_path=design,
        cohort_output_dir=cohort,
        endpoint_output_dirs=endpoints,
        require_formal_inputs=False,
    )

    assert set(manifest["partition_rows"]) == set(PARTITIONS)
    assert all(type(value) is int for value in manifest["partition_rows"].values())
    assert "partition_counts" not in manifest
    assert manifest["prior_v1_validation_and_replication_invalidated"] is True
    assert manifest["validation_endpoint_opened"] is False
    assert manifest["replication_endpoint_opened"] is False
    rendered = json.dumps(manifest, sort_keys=True)
    for forbidden in (
        '"dft_success"',
        '"dft_failure"',
        '"protected"',
        '"middle"',
        '"severe"',
        '"energy"',
    ):
        assert forbidden not in rendered

    for role in PARTITIONS:
        endpoint_manifest = json.loads(
            (endpoints[role] / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert endpoint_manifest["partition_role"] == role
        assert endpoint_manifest["endpoint_payload_opened"] is False
        assert "observed_label_counts" not in endpoint_manifest
        assert (endpoints[role] / ENDPOINT_NAME).is_file()
