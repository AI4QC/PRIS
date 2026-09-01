"""Contract tests for the np-next-20260802 guard columns and drivers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next2_guards import guard_features_from_structure  # noqa: E402
from next2_law_search import (  # noqa: E402
    NEXT2_GUARD_COLUMNS,
    build_next2_candidate_sets,
)


def _structure(formula):
    from pymatgen.core import Composition, Lattice, Structure

    composition = Composition(formula)
    lattice = Lattice.cubic(8.0)
    species = []
    coords = []
    for index, (element, amount) in enumerate(
        composition.get_el_amt_dict().items()
    ):
        for copy in range(int(round(amount))):
            species.append(element)
            coords.append(
                [
                    (0.13 * (index + copy)) % 0.9,
                    (0.29 * (index + copy)) % 0.9,
                    (0.47 * (index + copy)) % 0.9,
                ]
            )
    return Structure(lattice, species, coords)


def test_guard_azide_has_fractional_anion_valence():
    out = guard_features_from_structure(_structure("NaN3"))
    assert out["z_an_abs"] == pytest.approx(1.0 / 3.0)
    assert out["oxi_unique"] == 1.0


def test_guard_rock_salt_integer_and_unique():
    out = guard_features_from_structure(_structure("NaCl"))
    assert out["z_an_abs"] == pytest.approx(1.0)
    assert out["oxi_unique"] == 1.0


def test_guard_nitride_is_trivalent():
    out = guard_features_from_structure(_structure("Ca3N2"))
    assert out["z_an_abs"] == pytest.approx(3.0)


def test_guard_returns_empty_on_impossible_assignment():
    # K3(IrO3)2 admits no charge-balanced assignment under the frozen
    # oxi_state_guesses(max_sites=-10) call; the guard must abstain.
    out = guard_features_from_structure(_structure("K3Ir2O6"))
    assert out == {}


def test_next2_pool_excludes_guard_only_columns_from_thresholds():
    rng = np.random.default_rng(0)
    n_real, n_bad = 300, 250
    real = pd.DataFrame(
        {
            "source_id": [f"r{i}" for i in range(n_real)],
            "split": "discovery",
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_real),
            "p3haw_nnls_relres": np.abs(rng.normal(0.01, 0.005, n_real)),
            "z_an_abs": np.full(n_real, 2.0),
            "oxi_n_guesses": np.ones(n_real),
            "oxi_unique": np.ones(n_real),
            "fi": rng.uniform(0.2, 0.9, n_real),
        }
    )
    bad = pd.DataFrame(
        {
            "sid": [f"b{i}" for i in range(n_bad)],
            "psplit": "discovery",
            "parent": [f"p{i % 40}" for i in range(n_bad)],
            "kind": ["S1", "S2", "S3", "S4", "S5"] * (n_bad // 5),
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_bad),
            "p3haw_nnls_relres": np.abs(rng.normal(0.3, 0.1, n_bad)),
            "z_an_abs": np.full(n_bad, 2.0),
            "oxi_n_guesses": np.ones(n_bad),
            "oxi_unique": np.ones(n_bad),
            "fi": rng.uniform(0.2, 0.9, n_bad),
        }
    )
    pools, counts = build_next2_candidate_sets(
        real, bad, min_coverage=0.5, max_guard_targets=20
    )
    for pool in pools.values():
        for candidate in pool:
            assert candidate.feature not in (
                "z_an_abs",
                "oxi_n_guesses",
                "oxi_unique",
            )
    assert counts["guard_columns"] == len(NEXT2_GUARD_COLUMNS) == 10
    assert "z_an_abs" in NEXT2_GUARD_COLUMNS
