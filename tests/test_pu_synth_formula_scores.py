from __future__ import annotations
import os

import gzip
import hashlib
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821 import formula_scores
from experiments.pu_synthesizability_20260821.formula_scores import (
    FROZEN_FORMULA_SHA256,
    STAB_FORMULA_FEATURE_COLUMNS,
    SYN_FORMULA_FEATURE_COLUMNS,
    extract_formula_features,
    load_frozen_formula,
    score_frozen_formula,
)
from experiments.pu_synthesizability_20260821.runner import flatten_pris_result


pytestmark = pytest.mark.filterwarnings(
    "ignore:Set OLD_ERROR_HANDLING.*:DeprecationWarning"
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path(os.environ.get("PRIS_FEATURES", "features/"))
MP_PAGE = Path(
    Path(os.environ.get("PRIS_MP_SNAPSHOT", "mp-summary/")) + "/"
    "summary-2026-07-26/page-000105000.json.gz"
)
STRUCTURE_BLOB = Path(
    Path(os.environ.get("PRIS_MATDATA_BLOB", "structures.blob"))
)
SYN_FROZEN = ROOT / "outputs/20260814_f3_synth/F3_frozen.json"
STAB_FROZEN = ROOT / "outputs/20260814_f2r_stability/F2R_frozen.json"


def _require_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"local regression artifact unavailable: {', '.join(missing)}")


def _frozen(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps(
            {
                "features": ["a", "b"],
                "beta": [2.0, -3.0],
                "impute_median": {"a": 10.0, "b": 20.0},
                "mu": {"a": 8.0, "b": 18.0},
                "sd": {"a": 2.0, "b": 4.0},
            }
        )
    )
    return load_frozen_formula(path, name="demo")


def test_frozen_formula_uses_frozen_standardization_and_reports_observation(tmp_path):
    frozen = _frozen(tmp_path)
    frame = pd.DataFrame({"col_a": [12.0, np.nan], "col_b": [22.0, 14.0]})

    got = score_frozen_formula(
        frame,
        frozen,
        feature_columns={"a": "col_a", "b": "col_b"},
        include_components=True,
    )

    # row 0: 2 * ((12 - 8) / 2) - 3 * ((22 - 18) / 4) = 1
    # row 1 imputes a=10: 2 * 1 - 3 * (-1) = 5
    assert got.score.tolist() == pytest.approx([1.0, 5.0])
    assert got.n_observed.tolist() == [2, 1]
    assert got.all_observed.tolist() == [True, False]
    assert got.imputed_fraction.tolist() == pytest.approx([0.0, 0.5])
    assert got.component_a.tolist() == pytest.approx([4.0, 2.0])
    assert got.component_b.tolist() == pytest.approx([-3.0, 3.0])


def test_excluding_shared_terms_preserves_frozen_coefficients_without_refitting(tmp_path):
    frozen = _frozen(tmp_path)
    frame = pd.DataFrame({"a": [12.0], "b": [22.0]})

    got = score_frozen_formula(frame, frozen, exclude_features={"b"})

    assert got.score.iloc[0] == pytest.approx(4.0)
    assert got.n_terms.iloc[0] == 1
    assert got.n_observed.iloc[0] == 1


def test_formula_feature_column_map_distinguishes_two_wyckoff_tolerances():
    assert (
        SYN_FORMULA_FEATURE_COLUMNS["wyckoff_econ_001"]
        == "formula_syn_wyckoff_econ_001"
    )
    assert (
        STAB_FORMULA_FEATURE_COLUMNS["wyckoff_econ_01"]
        == "formula_stab_wyckoff_econ_01"
    )
    assert SYN_FORMULA_FEATURE_COLUMNS["madz_mean"] == "formula_syn_madz_mean"
    assert STAB_FORMULA_FEATURE_COLUMNS["gii"] == "formula_stab_gii"


def test_formula_extractor_uses_balance_for_synthesis_and_pris_guess_for_stability(
    monkeypatch,
):
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.6),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    monkeypatch.setitem(
        sys.modules,
        "polymorph_rank2",
        SimpleNamespace(balance=lambda formula: {"Na": 2, "Cl": -2}),
    )
    guess_calls = []
    monkeypatch.setitem(
        sys.modules,
        "discriminate",
        SimpleNamespace(
            guess_oxi=lambda candidate: (guess_calls.append(candidate) or [1.0, -1.0], True)
        ),
    )
    seen_valences = []
    monkeypatch.setattr(
        formula_scores,
        "_neighbor_table",
        lambda candidate: ([[], []], False),
    )
    monkeypatch.setattr(formula_scores, "_econ_features", lambda candidate, neighbors: {})
    monkeypatch.setattr(
        formula_scores,
        "_madz_mean",
        lambda candidate, values: float(values[0] * 10),
    )

    def bond_valence(candidate, values, neighbors):
        seen_valences.append(tuple(values))
        return {
            "bv_rel_mean": float(abs(values[1]) / 10),
            "gii": float(abs(values[0]) / 10),
        }

    monkeypatch.setattr(formula_scores, "_bond_valence_features", bond_valence)
    monkeypatch.setattr(
        formula_scores,
        "_criteria_selected",
        lambda candidate, values, neighbors, *, neighbor_failure: {
            "poly_deg_max": float(values[0] + 3),
            "frac_isolated": 0.25,
            "cn_cat_max": float(values[0] + 5),
            "p2_max_dev": float(abs(values[1]) / 10),
        },
    )
    pris_result = {
        "charge_assignment_route": "integer",
        # Deliberately conflict with guess_oxi: the strict stability route must
        # recompute the historical primary guess rather than trust PRIS payloads.
        "charge_assignment_values": [9.0, -9.0],
        "features": {
            "econ_max": 6.0,
            "gii": 0.2,
            "cn_cat_max": 6.0,
            "p2_max_dev": 0.1,
            "econ_min": 2.0,
            "wyckoff_econ": 0.25,
        },
        "wyckoff_econ_symprec_0p1": 0.5,
    }

    got = extract_formula_features(structure, pris_result=pris_result, src_dir="src")

    assert got["synthesis"]["valence_route"] == "balance"
    assert got["synthesis"]["features"]["madz_mean"] == pytest.approx(20.0)
    assert got["synthesis"]["features"]["bv_rel_mean"] == pytest.approx(0.2)
    assert got["synthesis"]["features"]["poly_deg_max"] == pytest.approx(5.0)
    assert got["stability"]["valence_route"] == "guess_oxi"
    assert got["stability"]["features"]["gii"] == pytest.approx(0.1)
    assert got["stability"]["features"]["cn_cat_max"] == pytest.approx(6.0)
    assert got["stability"]["features"]["wyckoff_econ_01"] == pytest.approx(0.5)
    assert len(guess_calls) == 1
    assert seen_valences == [(2.0, -2.0), (1.0, -1.0)]


def test_published_frozen_json_hashes_are_enforced(tmp_path):
    _require_paths(SYN_FROZEN, STAB_FROZEN)

    for name, path in (("S_syn", SYN_FROZEN), ("S_stab", STAB_FROZEN)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == FROZEN_FORMULA_SHA256[name]
        assert load_frozen_formula(path, name=name).name == name

        tampered = tmp_path / f"{name}.json"
        tampered.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="SHA-256"):
            load_frozen_formula(tampered, name=name)


def test_mp_aaaaaaib_synthesis_extractor_and_score_regression():
    _require_paths(MP_PAGE, SYN_FROZEN, ROOT / "data/bvparm2020.cif")
    from pymatgen.core import Structure

    with gzip.open(MP_PAGE, "rt") as handle:
        payload = json.load(handle)
    record = next(
        row for row in payload.get("data", []) if row.get("material_id") == "mp-aaaaaaib"
    )
    structure = Structure.from_dict(record["structure"])

    extracted = extract_formula_features(structure, pris_result=None, src_dir=ROOT / "src")
    features = extracted["synthesis"]["features"]
    expected = {
        "madz_mean": -10.008559607142779,
        "wyckoff_econ_001": 0.2,
        "bv_rel_mean": 0.016547728910082396,
        "vol_per_atom": 27.983266980704645,
        "poly_deg_max": 6.0,
        "frac_isolated": 0.0,
    }

    assert extracted["synthesis"]["valence_route"] == "balance"
    assert features == pytest.approx(expected, rel=1e-12, abs=1e-12)
    frozen = load_frozen_formula(SYN_FROZEN, name="S_syn")
    scored = score_frozen_formula(pd.DataFrame([features]), frozen)
    assert scored.score.iloc[0] == pytest.approx(-1.1769693261185197, abs=1e-12)
    assert scored.n_observed.iloc[0] == 6


def test_exp001_stability_extractor_and_score_regression():
    provenance = FEATURES / "provenance.parquet"
    _require_paths(provenance, STRUCTURE_BLOB, STAB_FROZEN, ROOT / "data/bvparm2020.cif")
    from pymatgen.core import Structure

    frame = pd.read_parquet(
        provenance, columns=["source_id", "blob_offset", "blob_length"]
    )
    row = frame.loc[frame.source_id.eq("exp001")].iloc[0]
    with STRUCTURE_BLOB.open("rb") as handle:
        handle.seek(int(row.blob_offset))
        raw = handle.read(int(row.blob_length))
    try:
        cif = zlib.decompress(raw).decode("utf-8", "ignore")
    except zlib.error:
        cif = raw.decode("utf-8", "ignore")
    structure = Structure.from_str(cif, fmt="cif")

    extracted = extract_formula_features(structure, pris_result=None, src_dir=ROOT / "src")
    features = extracted["stability"]["features"]
    expected = {
        "econ_max": 3.867687404244971,
        "gii": 0.10872146951834968,
        "cn_cat_max": 4.0,
        "p2_max_dev": 0.0,
        "wyckoff_econ_01": 0.2916666666666667,
        "econ_min": 1.974876414529969,
    }

    assert extracted["stability"]["valence_route"] == "guess_oxi"
    assert features == pytest.approx(expected, rel=1e-12, abs=1e-12)
    frozen = load_frozen_formula(STAB_FROZEN, name="S_stab")
    scored = score_frozen_formula(pd.DataFrame([features]), frozen)
    assert scored.score.iloc[0] == pytest.approx(1.299223191636605, abs=1e-12)
    assert scored.n_observed.iloc[0] == 6


def test_mp_aaaaaaiq_uses_frozen_bv_median_for_synthesis_score():
    rank = FEATURES / "synth_rank.parquet"
    aug = FEATURES / "synth_rank_aug.parquet"
    _require_paths(rank, aug, SYN_FROZEN)

    base = pd.read_parquet(
        rank, columns=["mp_id", "madz_mean", "bv_rel_mean", "vol_per_atom", "poly_deg_max", "frac_isolated"]
    )
    symmetry = pd.read_parquet(aug, columns=["mp_id", "wyckoff_econ_001"])
    row = base.loc[base.mp_id.eq("mp-aaaaaaiq")].merge(
        symmetry.loc[symmetry.mp_id.eq("mp-aaaaaaiq")], on="mp_id", validate="one_to_one"
    )
    assert row.bv_rel_mean.isna().all()

    frozen = load_frozen_formula(SYN_FROZEN, name="S_syn")
    scored = score_frozen_formula(row, frozen)
    assert scored.score.iloc[0] == pytest.approx(2.5201032433066244, abs=1e-12)
    assert scored.n_observed.iloc[0] == 5


def test_exp002_uses_frozen_gii_median_for_stability_score():
    rank = FEATURES / "real_rank.parquet"
    aug = FEATURES / "real_rank_aug.parquet"
    elec = FEATURES / "elec_real.parquet"
    _require_paths(rank, aug, elec, STAB_FROZEN)

    base = pd.read_parquet(rank, columns=["source_id", "cn_cat_max", "p2_max_dev"])
    symmetry = pd.read_parquet(aug, columns=["source_id", "wyckoff_econ_01"])
    electronic = pd.read_parquet(
        elec, columns=["source_id", "econ_max", "gii", "econ_min"]
    )
    row = (
        base.loc[base.source_id.eq("exp002")]
        .merge(
            symmetry.loc[symmetry.source_id.eq("exp002")],
            on="source_id",
            validate="one_to_one",
        )
        .merge(
            electronic.loc[electronic.source_id.eq("exp002")],
            on="source_id",
            validate="one_to_one",
        )
    )
    assert row.gii.isna().all()

    frozen = load_frozen_formula(STAB_FROZEN, name="S_stab")
    scored = score_frozen_formula(row, frozen)
    assert scored.score.iloc[0] == pytest.approx(-0.25094279546400833, abs=1e-12)
    assert scored.n_observed.iloc[0] == 5


def test_flatten_keeps_formula_routes_and_separate_feature_names():
    result = {
        "charge_assignment_route": "integer",
        "features": {},
        "predicates": {},
        "rungs": {},
        "formula_features": {
            "synthesis": {
                "valence_route": "balance",
                "feature_error": None,
                "features": {"madz_mean": -12.0, "wyckoff_econ_001": 0.25},
            },
            "stability": {
                "valence_route": "guess_oxi",
                "feature_error": None,
                "features": {"gii": 0.2, "wyckoff_econ_01": 0.5},
            },
        },
    }

    got = flatten_pris_result(
        {"cohort": "demo", "record_index": 1},
        result,
        structure_formula="NaCl",
        chemical_system="Cl-Na",
        n_elements=2,
        n_sites=2,
        cif_sha256="abc",
        elapsed_seconds=0.1,
    )

    assert got["formula_syn_valence_route"] == "balance"
    assert got["formula_stab_valence_route"] == "guess_oxi"
    assert got["formula_syn_madz_mean"] == pytest.approx(-12.0)
    assert got["formula_syn_wyckoff_econ_001"] == pytest.approx(0.25)
    assert got["formula_stab_gii"] == pytest.approx(0.2)
    assert got["formula_stab_wyckoff_econ_01"] == pytest.approx(0.5)
