from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next545_lisi_random_relaxation_confirmation import (
    _extract_last_structure_dict,
    apply_dft_waste_endpoint,
    evaluate_confirmation,
    parse_summary_rows,
)


def test_last_structure_extractor_skips_all_dft_arrays(tmp_path: Path) -> None:
    first = Structure(Lattice.cubic(4), ["Li"], [[0, 0, 0]]).as_dict()
    last = Structure(Lattice.cubic(5), ["Li"], [[0.1, 0.2, 0.3]]).as_dict()
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "forces": [[[999.1]], [[888.2]]],
                "stress": [[[777.3]]],
                "e_fr_energy": [666.4, 555.5],
                "structure": [first, last],
            }
        )
    )

    value, audit = _extract_last_structure_dict(path)

    assert Structure.from_dict(value) == Structure.from_dict(last)
    assert audit["structure_objects_scanned"] == 2
    assert audit["structure_objects_decoded"] == 1
    assert "999.1" not in json.dumps(value)
    assert "555.5" not in json.dumps(value)


def test_summary_parser_and_endpoint_are_prefix_relative(tmp_path: Path) -> None:
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "Li1Si1_02_1, , -10.0 eV, P1, P1, 10, 9, 1, 20\n"
        "Li1Si1_02_2, ionic convergence failed, -8.0 eV, P1, P1, 10, 8, 1, 100\n"
        "Li1Si1_02_3, , -6.0 eV, P1, P1, 10, 7, 1, 30\n"
        "Li1Si1_02_4, , -4.0 eV, P1, P1, 10, 6, 1, 40\n"
    )
    parsed = parse_summary_rows(summary, prefix="Li1Si1_02")
    table = pd.DataFrame(parsed)
    table["n_sites"] = 2

    result = apply_dft_waste_endpoint(table)

    assert result["failed"].tolist() == [False, True, False, False]
    assert result.loc[1, "dft_waste"]
    assert result.loc[1, "waste_severity"] == 1.25
    successful = result.loc[~result["failed"]]
    assert successful["energy_percentile"].min() > 0.0
    assert successful["energy_percentile"].max() < 1.0


def test_confirmation_metrics_respect_fixed_operating_sets() -> None:
    rows = 40
    table = pd.DataFrame(
        {
            "trajectory_id": [f"id-{i}" for i in range(rows)],
            "prefix": ["A"] * 20 + ["B"] * 20,
            "mupr_risk": np.linspace(1, 0, rows),
            "dft_waste": [True] * 12 + [False] * 28,
            "waste_severity": np.linspace(1.25, 0.01, rows),
        }
    )

    metrics = evaluate_confirmation(table, bootstrap_draws=200, seed=29)

    assert metrics["roc_auc"] == 1.0
    assert metrics["spearman"] == 1.0
    assert metrics["top_15_percent"]["rows"] == 6
    assert metrics["top_15_percent"]["precision"] == 1.0
    assert metrics["bottom_50_percent"]["nonwaste_fraction"] == 1.0
