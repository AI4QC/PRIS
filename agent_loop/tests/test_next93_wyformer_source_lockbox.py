from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from src.next93_wyformer_source_lockbox import (
    ENDPOINT_NAME,
    GEOMETRY_NAMES,
    MANIFEST_NAME,
    METADATA_NAME,
    PARTITIONS,
    _composition_key_from_structure_payload,
    _endpoint_stratum,
    _partition_for_reduced_formula,
    build_wyformer_source_lockbox,
)


def _structure(species: list[str], *, scale: float = 1.0) -> Structure:
    n = len(species)
    coords = [[i / n, (2 * i + 1) / (2 * n), (3 * i + 1) / (3 * n)] for i in range(n)]
    return Structure(Lattice.cubic(5.0 * scale), species, coords)


def _payload(structure: Structure) -> str:
    return json.dumps(structure.as_dict(), sort_keys=True)


def _write_source_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    li2o = _structure(["Li", "Li", "O"])
    nacl = _structure(["Na", "Cl"], scale=1.1)
    kbr_a = _structure(["K", "Br"], scale=1.2)
    kbr_b = _structure(["K", "Br"], scale=1.3)
    mgo = _structure(["Mg", "O"], scale=1.4)
    raw = pd.DataFrame(
        {
            "material_id": [0, 1, 2, 3, 4],
            "structure": [_payload(x) for x in (li2o, nacl, kbr_a, kbr_b, mgo)],
            "group": [225, 225, 221, 221, 225],
        }
    )
    # The identifiers are deliberately wrong/permuted.  Pairing must use the
    # unique full-cell composition, not material_id.
    dft = pd.DataFrame(
        {
            "material_id": [91, 90, 89],
            "structure": [_payload(nacl), _payload(li2o), _payload(kbr_a)],
            "dft_e_above_hull_corrected": [0.60, 0.05, 0.20],
            "dft_e_uncorrected": [-4.0, -5.0, -6.0],
            "dft_e_corrected": [-4.1, -5.1, -6.1],
        }
    )
    raw_path = root / "raw.csv.gz"
    dft_path = root / "dft.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    dft.to_csv(dft_path, index=False, compression="gzip")
    article_path = root / "article.json"
    article_path.write_text(json.dumps({"id": 29094701, "files": []}), encoding="utf-8")
    readme_path = root / "README.md"
    readme_path.write_text("indices unfortunately got permuted", encoding="utf-8")
    design_path = root / "design.md"
    design_path.write_text("frozen design", encoding="utf-8")
    return raw_path, dft_path, article_path, readme_path, design_path


def test_exact_composition_key_ignores_site_order_but_not_cell_counts() -> None:
    a = _structure(["Li", "Li", "O"])
    b = _structure(["O", "Li", "Li"])
    reduced_cell = _structure(["Li", "O"])
    assert _composition_key_from_structure_payload(_payload(a)) == _composition_key_from_structure_payload(
        _payload(b)
    )
    assert _composition_key_from_structure_payload(
        _payload(a)
    ) != _composition_key_from_structure_payload(_payload(reduced_cell))


def test_split_is_deterministic_and_keeps_reduced_formula_groups_together() -> None:
    first = _partition_for_reduced_formula("Li2O")
    assert first in PARTITIONS
    assert _partition_for_reduced_formula("Li2O") == first
    assert inspect.signature(_partition_for_reduced_formula).parameters.keys() == {"reduced_formula"}


@pytest.mark.parametrize(
    ("dft_succeeded", "energy", "expected"),
    [
        (True, 0.10, "protected"),
        (True, 0.100001, "middle"),
        (True, 0.499999, "middle"),
        (True, 0.50, "severe"),
        (False, None, "severe"),
    ],
)
def test_endpoint_strata_are_frozen(
    dft_succeeded: bool, energy: float | None, expected: str
) -> None:
    assert _endpoint_stratum(dft_succeeded=dft_succeeded, dft_e_above_hull=energy) == expected


def test_builder_pairs_by_unique_composition_and_physically_separates_labels(
    tmp_path: Path,
) -> None:
    raw, dft, article, readme, design = _write_source_inputs(tmp_path)
    cohort_dir = tmp_path / "cohort"
    endpoint_dirs = {role: tmp_path / f"endpoint-{role}" for role in PARTITIONS}
    manifest = build_wyformer_source_lockbox(
        raw_x0_path=raw,
        dft_success_path=dft,
        figshare_metadata_path=article,
        readme_path=readme,
        design_path=design,
        cohort_output_dir=cohort_dir,
        endpoint_output_dirs=endpoint_dirs,
        require_formal_inputs=False,
    )

    assert manifest["pairing"]["raw_rows"] == 5
    assert manifest["pairing"]["formal_unique_x0_rows"] == 3
    assert manifest["pairing"]["excluded_ambiguous_raw_rows"] == 2
    assert manifest["pairing"]["matched_dft_success_rows"] == 2
    assert manifest["pairing"]["unmatched_dft_failure_rows"] == 1
    assert manifest["labels_opened_by_feature_builder"] is False
    assert manifest["validation_endpoint_opened"] is False
    assert manifest["replication_endpoint_opened"] is False

    metadata = pd.read_parquet(cohort_dir / METADATA_NAME)
    assert len(metadata) == 3
    assert set(metadata["raw_material_id"]) == {0, 1, 4}
    assert metadata["input_role"].eq(
        "raw_generated_pre_chgnet_pre_dft_unrelaxed_x0"
    ).all()
    assert not any("dft" in col.lower() or "energy" in col.lower() for col in metadata.columns)

    endpoint_frames: list[pd.DataFrame] = []
    geometry_ids: set[str] = set()
    for role in PARTITIONS:
        geometry = pd.read_parquet(cohort_dir / GEOMETRY_NAMES[role])
        endpoint = pd.read_parquet(endpoint_dirs[role] / ENDPOINT_NAME)
        geometry_ids.update(geometry["material_id"].astype(str))
        endpoint_frames.append(endpoint)
        assert set(geometry.columns) == {"material_id", "structure_json"}
        assert "structure_json" not in endpoint.columns
        endpoint_manifest = json.loads(
            (endpoint_dirs[role] / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert endpoint_manifest["endpoint_payload_opened"] is False

    endpoint = pd.concat(endpoint_frames, ignore_index=True)
    assert set(endpoint["endpoint_stratum"]) == {"protected", "severe"}
    assert geometry_ids == set(endpoint["material_id"].astype(str))
    failed = endpoint[endpoint["dft_succeeded"].eq(False)]
    assert len(failed) == 1
    assert failed["dft_e_above_hull_corrected"].isna().all()
    assert failed["endpoint_stratum"].eq("severe").all()

    with pytest.raises(FileExistsError):
        build_wyformer_source_lockbox(
            raw_x0_path=raw,
            dft_success_path=dft,
            figshare_metadata_path=article,
            readme_path=readme,
            design_path=design,
            cohort_output_dir=cohort_dir,
            endpoint_output_dirs=endpoint_dirs,
            require_formal_inputs=False,
        )
