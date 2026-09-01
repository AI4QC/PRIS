"""Contracts for the finite NEXT23 development search and frozen law."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _development_tables(n_each: int = 100) -> tuple[pd.DataFrame, ...]:
    from src.next23_relaxation_rule import BASE_TERMS

    n_rows = 2 * n_each
    ids = [f"wbm-{index}" for index in range(n_rows)]
    base = {
        "material_id": ids,
        "rk": ["LiO"] * n_rows,
        "formula": ["LiO"] * n_rows,
        "natoms": [2] * n_rows,
        "input_role": ["unrelaxed_x0_geometry_only"] * n_rows,
    }
    by_source: dict[str, dict[str, object]] = {
        "sivr": dict(base),
        "madelung": dict(base),
        "scbve": dict(base),
    }
    for term in BASE_TERMS.values():
        by_source[term.source][term.column] = np.ones(n_rows)
    by_source["sivr"][BASE_TERMS["A"].column] = np.r_[
        np.zeros(n_each), np.full(n_each, 10.0)
    ]
    labels = pd.DataFrame(
        {
            "material_id": ids,
            "site_stats_fingerprint_init_final_norm_diff": np.r_[
                np.full(n_each, 0.05), np.full(n_each, 0.30)
            ],
        }
    )
    metadata = pd.DataFrame(base)
    return (
        pd.DataFrame(by_source["sivr"]),
        pd.DataFrame(by_source["madelung"]),
        pd.DataFrame(by_source["scbve"]),
        labels,
        metadata,
    )


def test_catalogue_and_risk_directions_are_frozen() -> None:
    from src.next23_relaxation_rule import BASE_TERMS, CANDIDATES

    assert tuple(BASE_TERMS) == ("A", "B", "C", "D", "E", "F", "G", "H")
    assert {key: term.direction for key, term in BASE_TERMS.items()} == {
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
        "E": 1,
        "F": 1,
        "G": -1,
        "H": -1,
    }
    assert CANDIDATES == (
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        ("H",),
        ("A", "C"),
        ("A", "E"),
        ("A", "G"),
        ("A", "H"),
        ("B", "C"),
        ("B", "E"),
        ("A", "C", "E"),
        ("A", "E", "G"),
        ("A", "C", "H"),
    )


def test_wilson_bounds_and_candidate_selection_are_deterministic() -> None:
    from src.next23_relaxation_rule import (
        fit_robust_parameters,
        score_candidate,
        select_candidate,
        wilson_lower_bound,
    )

    assert wilson_lower_bound(9, 10) == pytest.approx(0.6523, abs=5e-4)
    sivr, madelung, scbve, labels, metadata = _development_tables()
    joined = metadata[["material_id"]].merge(sivr, on="material_id").merge(
        madelung[["material_id", "nm_point_reduced"]], on="material_id"
    ).merge(
        scbve[
            [
                "material_id",
                "scbv_vector_asymmetry_rms",
                "scbv_vector_asymmetry_max",
            ]
        ],
        on="material_id",
    )
    parameters = fit_robust_parameters(joined)
    score, support = score_candidate(joined, ("A",), parameters)
    assert support.all()
    assert np.unique(score).tolist() == pytest.approx([-0.5, 0.5])

    result = select_candidate(joined, labels)
    assert result["selected_candidate"] == "A"
    assert result["eligible"] is True
    assert result["selected_metrics"]["coverage_lower"] >= 0.90
    assert result["selected_metrics"]["protected_recall_lower"] >= 0.95
    assert result["selected_metrics"]["rejection_precision_lower"] >= 0.90
    assert result["selected_metrics"]["savings_lower"] >= 0.10


def test_missing_required_feature_fails_open() -> None:
    from src.next23_relaxation_rule import fit_robust_parameters, score_candidate

    sivr, _madelung, _scbve, _labels, _metadata = _development_tables()
    sivr.loc[0, "voronoi_q05__sivr_cell_anisotropy"] = np.nan
    parameters = fit_robust_parameters(sivr)
    score, support = score_candidate(sivr, ("A",), parameters)
    assert not support[0]
    assert np.isnan(score[0])


def test_freeze_development_rule_is_no_replace_and_records_provenance(
    tmp_path: Path,
) -> None:
    from src import next23_relaxation_rule as module

    sivr, madelung, scbve, labels, metadata = _development_tables()
    paths = {}
    for name, frame in {
        "sivr": sivr,
        "madelung": madelung,
        "scbve": scbve,
        "labels": labels,
        "metadata": metadata,
    }.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    output = tmp_path / "freeze"
    manifest = module.freeze_development_rule(
        sivr_features_path=paths["sivr"],
        madelung_features_path=paths["madelung"],
        scbve_features_path=paths["scbve"],
        labels_path=paths["labels"],
        development_metadata_path=paths["metadata"],
        output_dir=output,
    )
    law = json.loads((output / module.FROZEN_RULE_NAME).read_text())
    scan = json.loads((output / module.SCAN_NAME).read_text())
    assert law["selected_candidate"] == "A"
    assert law["blind_labels_opened"] is False
    assert law["executable_inputs"] == "one_unrelaxed_structure_plus_element_tables"
    assert scan["eligible"] is True
    assert manifest["development_labels_opened"] is True
    assert manifest["blind_labels_opened"] is False
    assert set(manifest["inputs_sha256"]) == {
        "sivr_features",
        "madelung_features",
        "scbve_features",
        "development_labels",
        "development_metadata",
    }
    with pytest.raises(FileExistsError):
        module.freeze_development_rule(
            sivr_features_path=paths["sivr"],
            madelung_features_path=paths["madelung"],
            scbve_features_path=paths["scbve"],
            labels_path=paths["labels"],
            development_metadata_path=paths["metadata"],
            output_dir=output,
        )


def test_freeze_rejects_nonfinite_endpoint_or_label_like_feature_column(
    tmp_path: Path,
) -> None:
    from src.next23_relaxation_rule import freeze_development_rule

    sivr, madelung, scbve, labels, metadata = _development_tables()
    sivr["dft_energy"] = 0.0
    labels.loc[0, "site_stats_fingerprint_init_final_norm_diff"] = np.nan
    frames = {
        "sivr": sivr,
        "madelung": madelung,
        "scbve": scbve,
        "labels": labels,
        "metadata": metadata,
    }
    paths = {}
    for name, frame in frames.items():
        path = tmp_path / f"bad-{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    with pytest.raises(ValueError):
        freeze_development_rule(
            sivr_features_path=paths["sivr"],
            madelung_features_path=paths["madelung"],
            scbve_features_path=paths["scbve"],
            labels_path=paths["labels"],
            development_metadata_path=paths["metadata"],
            output_dir=tmp_path / "bad-freeze",
        )

