from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next13d_acsc_dft_pairs import _sha256_file
from src.next85_scigen_label_free_features import (
    CATALOGUE_NAME as UPSTREAM_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as UPSTREAM_MANIFEST_NAME,
    PROTOCOL as UPSTREAM_PROTOCOL,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME,
    MANIFEST_NAME,
    PRESPECIFIED_TERMS,
    freeze_scigen_term_catalogue,
)


def _features(tmp_path: Path) -> Path:
    root = tmp_path / "features"
    root.mkdir()
    n = 24
    base = {
        "material_id": [f"m-{index:03d}" for index in range(n)],
        "partition_role": ["discovery"] * n,
        "lattice_class": ["tri"] * n,
        "reduced_formula": [f"Na{index + 1}Cl" for index in range(n)],
        "chemical_system": ["Cl-Na"] * n,
    }
    for index, spec in enumerate(PRESPECIFIED_TERMS):
        base[spec["feature"]] = np.linspace(0.01 + index, 2.0 + index, n)
    base["aefi_residual_q95"] = [np.nan] * 22 + [1.0, 2.0]
    discovery = pd.DataFrame(base)
    discovery.to_parquet(root / FEATURE_NAMES["discovery"], index=False)
    for role in ("internal_validation", "internal_replication"):
        pd.DataFrame(
            {
                "material_id": pd.Series(dtype=str),
                "partition_role": pd.Series(dtype=str),
            }
        ).to_parquet(root / FEATURE_NAMES[role], index=False)
    (root / UPSTREAM_CATALOGUE_NAME).write_text(
        json.dumps({"feature_names": sorted({spec["feature"] for spec in PRESPECIFIED_TERMS})}),
        encoding="utf-8",
    )
    outputs = {
        path.name: _sha256_file(path)
        for path in [
            root / UPSTREAM_CATALOGUE_NAME,
            *(root / FEATURE_NAMES[role] for role in FEATURE_NAMES),
        ]
    }
    (root / UPSTREAM_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": UPSTREAM_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": outputs,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_freeze_uses_only_discovery_x0_statistics(tmp_path: Path) -> None:
    features = _features(tmp_path)
    amendment = tmp_path / "amendment.md"
    amendment.write_text("frozen before labels\n", encoding="utf-8")
    target = tmp_path / "catalogue"
    manifest = freeze_scigen_term_catalogue(
        feature_dir=features,
        amendment_path=amendment,
        output_dir=target,
        require_formal_inputs=False,
    )
    catalogue = json.loads((target / CATALOGUE_NAME).read_text(encoding="utf-8"))
    eligible = {term["feature"]: term for term in catalogue["eligible_terms"]}
    excluded = {term["feature"]: term for term in catalogue["excluded_terms"]}
    assert eligible["steric_rep12_pa"]["direction"] == 1
    assert eligible["cov_q01"]["direction"] == -1
    assert eligible["sivr_stiffness_min"]["transform"] == "asinh"
    assert excluded["aefi_residual_q95"]["reason"] == "finite_coverage_below_0.90"
    assert catalogue["labels_opened"] is False
    assert manifest["labels_opened"] is False
    assert (target / MANIFEST_NAME).is_file()


def test_freeze_is_no_replace_and_cli_has_no_endpoint_argument(tmp_path: Path) -> None:
    features = _features(tmp_path)
    amendment = tmp_path / "amendment.md"
    amendment.write_text("frozen\n", encoding="utf-8")
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        freeze_scigen_term_catalogue(
            feature_dir=features,
            amendment_path=amendment,
            output_dir=target,
            require_formal_inputs=False,
        )
    source = Path("src/next86_scigen_term_catalogue.py").read_text(encoding="utf-8")
    assert "--endpoint" not in source
    assert "output.dat" not in source
