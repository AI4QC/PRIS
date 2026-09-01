import io

import pandas as pd
import pytest

from src.next6_elementa_initial import (
    ionic_initial_features,
    iter_selected_initial_frames,
    select_ranked_endpoints,
)


def _frame(material: str, step: int, distance: float = 1.0) -> str:
    return f"""3
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 material={material} formula=Li2O structure=structure_01 ionic_step={step} energy=-99 pbc="T T T"
Li 0 0 0
Li {distance} 0 0
O 0 {distance} 0
"""


def test_rank_sid_maps_to_one_indexed_endpoint_row_and_validates_energy():
    # Break caught: treating elem-N as zero-indexed pairs a rank label with the
    # neighboring trajectory.
    rank = pd.DataFrame(
        {"sid": ["elem-2"], "rk": ["Li2|O1"], "e_per_atom": [-2.0], "nat": [3]}
    )
    endpoints = pd.DataFrame(
        {
            "material": ["other", "Li2O_01"],
            "formula": ["NaCl", "Li2O"],
            "structure": ["structure_01", "structure_01"],
            "spin": [None, None],
            "ionic_step": [4, 8],
            "n_sites": [2, 3],
            "energy": [-2.0, -6.0],
            "max_force": [0.04, 0.03],
        }
    )

    got = select_ranked_endpoints(rank, endpoints)

    assert got.material.tolist() == ["Li2O_01"]
    assert got.formula.tolist() == ["Li2O"]
    assert got.final_ionic_step.tolist() == [8]
    assert got.final_max_force.tolist() == [0.03]


def test_rank_endpoint_energy_mismatch_fails_closed():
    # Break caught: a stale rank artifact must not silently join to changed labels.
    rank = pd.DataFrame(
        {"sid": ["elem-1"], "rk": ["Li2|O1"], "e_per_atom": [-3.0], "nat": [3]}
    )
    endpoints = pd.DataFrame(
        {
            "material": ["Li2O_01"],
            "formula": ["Li2O"],
            "structure": ["structure_01"],
            "spin": [None],
            "ionic_step": [8],
            "n_sites": [3],
            "energy": [-6.0],
            "max_force": [0.03],
        }
    )
    with pytest.raises(ValueError, match="energy mismatch"):
        select_ranked_endpoints(rank, endpoints)


def test_stream_keeps_only_first_frame_of_each_selected_trajectory():
    # Break caught: selecting the last frame would recreate the relaxed-geometry leak.
    stream = io.StringIO(
        _frame("Li2O_01", 0, 1.0)
        + _frame("Li2O_01", 1, 1.2)
        + _frame("unselected", 0, 2.0)
    )
    targets = {"Li2O_01": {"sid": "elem-2", "rk": "Li2|O1", "formula": "Li2O"}}

    got = list(iter_selected_initial_frames(stream, targets))

    assert len(got) == 1
    assert got[0]["sid"] == "elem-2"
    assert got[0]["initial_ionic_step"] == 0
    assert "Li 1.0 0 0" in got[0]["text"]
    assert "Li 1.2 0 0" not in got[0]["text"]


def test_initial_feature_worker_never_exports_dft_energy_or_forces():
    # Break caught: passing through extxyz energy or final e_per_atom would turn a
    # pre-DFT feature table into a leaked target table.
    row = {
        "sid": "elem-1",
        "rk": "Li2|O1",
        "formula": "Li2O",
        "initial_ionic_step": 0,
        "text": _frame("Li2O_01", 0, 1.0),
    }
    got = ionic_initial_features(row)

    assert got["sid"] == "elem-1"
    assert got["input_role"] == "unrelaxed_x0_only"
    assert "energy" not in got
    assert "e_per_atom" not in got
    assert "forces" not in got
    assert got["geom_feature_ok"] is True
