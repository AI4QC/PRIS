from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next542_jarvis_cdvae_dft_response import (
    endpoint_response,
    evaluate_screen,
    match_composition_group,
)


def _lif(a: float, displacement: float = 0.0) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["Li", "F"],
        [[0.0, 0.0, 0.0], [0.5 + displacement, 0.5, 0.5]],
    )


def test_global_match_chooses_distinct_nearest_initial_candidates() -> None:
    initials = [
        {"initial_index": 3, "structure": _lif(4.0, 0.00)},
        {"initial_index": 7, "structure": _lif(4.0, 0.10)},
    ]
    finals = [
        {"endpoint_filename": "POSCAR-LiF.vasp", "structure": _lif(4.0, 0.095)},
        {"endpoint_filename": "POSCAR-LiF-v2.vasp", "structure": _lif(4.0, 0.005)},
    ]

    result = match_composition_group(initials, finals)

    mapped = {row["endpoint_filename"]: row["initial_index"] for row in result}
    assert mapped == {"POSCAR-LiF.vasp": 7, "POSCAR-LiF-v2.vasp": 3}
    assert all(row["mapped"] for row in result)
    assert all(row["match_tier"] == 0 for row in result)


def test_endpoint_response_uses_frozen_dimensionless_thresholds() -> None:
    mild = endpoint_response(
        normalized_rms=0.10,
        normalized_max=0.20,
        initial_volume_per_atom=10.0,
        final_volume_per_atom=11.0,
        match_tier=0,
    )
    severe = endpoint_response(
        normalized_rms=0.10,
        normalized_max=0.20,
        initial_volume_per_atom=10.0,
        final_volume_per_atom=13.0,
        match_tier=0,
    )
    fallback = endpoint_response(
        normalized_rms=0.01,
        normalized_max=0.01,
        initial_volume_per_atom=10.0,
        final_volume_per_atom=10.0,
        match_tier=1,
    )

    assert mild["severe_response"] is False
    assert mild["response_severity"] < 1.0
    assert severe["severe_response"] is True
    assert severe["volume_log_response"] > np.log(1.25)
    assert fallback["severe_response"] is True


def test_screen_metrics_use_fixed_top_and_bottom_operating_sets() -> None:
    table = pd.DataFrame(
        {
            "initial_index": np.arange(20),
            "composition_key": [f"X{i // 2}" for i in range(20)],
            "mupr_risk": np.linspace(1.0, 0.0, 20),
            "severe_response": [True] * 6 + [False] * 14,
            "response_severity": np.linspace(2.0, 0.1, 20),
        }
    )

    metrics = evaluate_screen(table, bootstrap_draws=200, seed=17)

    assert metrics["rows"] == 20
    assert metrics["top_15_percent"]["rows"] == 3
    assert metrics["top_15_percent"]["positives"] == 3
    assert metrics["top_15_percent"]["precision"] == 1.0
    assert metrics["bottom_50_percent"]["rows"] == 10
    assert metrics["bottom_50_percent"]["nonsevere_fraction"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["spearman"] == 1.0
