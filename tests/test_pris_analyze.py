"""The public entry point must reproduce the manuscript's Fig. 3a readings.

Fig. 3a reports, for experimental MgAl2O4 and its five damaged variants:
uniaxial compression lowers rho from 0.99 to 0.76, isotropic expansion raises it
to 1.28 (beyond the 1.05 ceiling of Law 2), and a cation-anion exchange leaves
rho at 0.99 while creating like-charge bonds.  If pris_analyze disagrees with
those numbers, either the entry point or the figure is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pymatgen")

import pris_analyze as pa  # noqa: E402
from fig3_anatomy import corrupt, spinel  # noqa: E402


@pytest.fixture(scope="module")
def parent():
    return spinel()


@pytest.fixture(scope="module")
def readings(parent):
    out = {"real": pa.measure(parent)[0]}
    for kind in ("S1", "S2", "S3", "S4", "S5"):
        out[kind] = pa.measure(corrupt(parent, kind, np.random.default_rng(0)))[0]
    return out


def test_thresholds_match_the_manuscript():
    assert pa.LAWS[1]["threshold"] == 0.804
    assert pa.TAU_PERMISSIVE == 0.735
    assert pa.LAWS[2]["threshold"] == 1.05
    assert pa.LAWS[3]["threshold"] == 1.081
    assert pa.LAWS[4]["threshold"] == 31.45
    assert pa.LAWS[5]["threshold"] == 15.17
    assert pa.LAWS[7]["threshold"] == pytest.approx(2 / 3)
    assert pa.LAWS[8]["threshold"] == 0.7143


def test_set_membership_follows_the_nested_ladder():
    # Set 1' carries Law 2 and stands beside the chain, so Set 4 applies seven
    # of the eight laws rather than all eight.
    assert pa.SETS["Set 1"]["laws"] == (1,)
    assert pa.SETS["Set 1'"]["laws"] == (1, 2)
    assert 2 not in pa.SETS["Set 4"]["laws"]
    assert len(pa.SETS["Set 4"]["laws"]) == 7
    for smaller, larger in (("Set 2", "Set 3"), ("Set 3", "Set 4")):
        assert set(pa.SETS[smaller]["laws"]) < set(pa.SETS[larger]["laws"])


def test_rho_matches_figure_3a(readings):
    assert readings["real"]["bl_min"] == pytest.approx(0.99, abs=0.01)
    assert readings["S1"]["bl_min"] == pytest.approx(0.76, abs=0.01)
    assert readings["S4"]["bl_min"] == pytest.approx(1.28, abs=0.01)
    # the cation-anion exchange moves no atom, so rho is unchanged
    assert readings["S5"]["bl_min"] == pytest.approx(readings["real"]["bl_min"],
                                                    abs=0.01)


def test_expansion_crosses_only_the_law_2_ceiling(readings):
    assert pa.judge_law(1, readings["S4"])[0] == "pass"
    assert pa.judge_law(2, readings["S4"])[0] == "fail"


def test_cation_anion_exchange_creates_like_charge_bonds(readings):
    assert readings["real"]["frac_like_bonds"] == pytest.approx(0.0, abs=1e-9)
    assert readings["S5"]["frac_like_bonds"] > 0


def test_the_experimental_parent_survives_every_set(parent, tmp_path):
    path = tmp_path / "mgal2o4.cif"
    parent.to(filename=str(path))
    a = pa.analyse(str(path))
    assert a["verdict"] == "plausible"
    for name in pa.SET_ORDER:
        assert a["sets"][name]["verdict"] == "plausible", name


def test_damage_is_caught_and_names_a_mechanism(parent, tmp_path):
    # compression, displacement and the cation-anion exchange must all fail
    # Set 4 and say which mechanism to review
    for kind in ("S1", "S3", "S5"):
        path = tmp_path / f"{kind}.cif"
        corrupt(parent, kind, np.random.default_rng(0)).to(filename=str(path))
        a = pa.analyse(str(path))
        assert a["verdict"] == "implausible", kind
        assert a["mechanisms_to_review"], kind


def test_pss_weights_are_the_frozen_ones():
    frozen = pa.load_pss()
    if frozen is None:
        pytest.skip("outputs/20260814_f3_synth/F3_frozen.json not present")
    assert frozen["features"] == ["madz_mean", "wyckoff_econ_001", "bv_rel_mean",
                                  "vol_per_atom", "poly_deg_max", "frac_isolated"]
    # equation (1) of the manuscript, to the printed two decimals
    printed = [-1.24, -0.84, -1.18, -4.90, -0.22, 0.59]
    for got, want in zip(frozen["beta"], printed):
        assert got == pytest.approx(want, abs=0.005)


def test_pss_ranks_the_parent_above_its_damaged_variants(parent, tmp_path):
    if pa.load_pss() is None:
        pytest.skip("frozen PSS artefact not present")
    scores = {}
    for label, st in [("real", parent)] + [
            (k, corrupt(parent, k, np.random.default_rng(0))) for k in ("S1", "S4")]:
        path = tmp_path / f"{label}.cif"
        st.to(filename=str(path))
        scores[label] = pa.analyse(str(path))["pss"]
    assert scores["real"] > scores["S1"]
    assert scores["real"] > scores["S4"]


def test_unreadable_input_returns_no_verdict(tmp_path):
    bad = tmp_path / "not-a-structure.cif"
    bad.write_text("this is not a CIF\n")
    a = pa.analyse(str(bad))
    assert a["verdict"] == "no verdict"
    assert "reason" in a
