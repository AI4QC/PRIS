"""Regression tests for the revised deployment figure and its frozen evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "fig6_deployment.py"
MATTERGEN = ROOT / "paper" / "data" / "fig7_mattergen_ladder_energy.csv"
WRONG_SITE = ROOT / "paper" / "data" / "fig7_wrong_site.json"
MANIFEST = ROOT / "tex" / "figure_scripts" / "figure_manifest.json"


def _module():
    spec = importlib.util.spec_from_file_location("fig6_deployment_revision", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mattergen_energy_ladder_is_one_to_one_and_preserves_coverage() -> None:
    data = pd.read_csv(MATTERGEN)
    assert len(data) == data["material_id"].nunique() == 500
    assert list(data.columns) == [
        "material_id",
        "e_released_per_atom",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    expected = {
        "L1": {"satisfies": 162, "fails": 0, "not evaluated": 338},
        "L2": {"satisfies": 154, "fails": 8, "not evaluated": 338},
        "L3": {"satisfies": 134, "fails": 11, "not evaluated": 355},
        "L4": {"satisfies": 4, "fails": 155, "not evaluated": 341},
    }
    module = _module()
    assert module.ladder_verdict_counts(data) == expected
    assert (data["e_released_per_atom"] >= 0).all()


def test_wrong_site_summary_uses_parent_controlled_paired_cohort_only() -> None:
    frozen = json.loads(WRONG_SITE.read_text(encoding="utf-8"))
    assert frozen["cohort"] == "A"
    assert frozen["parent_control"] == (
        "parent unflagged by both coordinate checks and PRIS; "
        "damaged structure determinate for both"
    )
    assert frozen["classes"]["S2"] == {
        "label": "cation-cation",
        "n": 69,
        "pris_only": 43,
        "both": 19,
        "coordinate_only": 0,
        "neither": 7,
    }
    assert frozen["classes"]["S5"] == {
        "label": "cation-anion",
        "n": 83,
        "pris_only": 43,
        "both": 39,
        "coordinate_only": 1,
        "neither": 0,
    }
    module = _module()
    rates = module.wrong_site_detection_rates(frozen)
    assert rates == {
        "S2": {"coordinate checks": 19 / 69, "PRIS": 62 / 69},
        "S5": {"coordinate checks": 40 / 83, "PRIS": 82 / 83},
    }


def test_wrong_site_exemplar_changes_only_two_species_labels() -> None:
    exemplar = json.loads(WRONG_SITE.read_text(encoding="utf-8"))["exemplar"]
    assert exemplar["source_archive"].endswith("gnome_d7/by_id.zip")
    assert exemplar["zip_name"] == "by_id/1bac537e26.CIF"
    assert exemplar["parent_id"] == "1bac537e26"
    assert exemplar["klass"] == "S2"
    assert exemplar["formula"] == "KErBr4"
    assert exemplar["swapped_sites"] == [0, 1]
    assert exemplar["parent_species"] == ["K", "Er", "Br", "Br", "Br", "Br"]
    assert exemplar["damaged_species"] == ["Er", "K", "Br", "Br", "Br", "Br"]
    assert len(exemplar["fractional_coordinates"]) == 6
    assert len(exemplar["lattice_matrix_angstrom"]) == 3


def test_charge_coverage_is_demoted_to_new_supplementary_figure() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    si_source = (ROOT / "src" / "figS17_charge_coverage.py").read_text(
        encoding="utf-8"
    )
    assert "def panel_charge_coverage" not in source
    assert "charge-dependent laws not evaluated" in si_source
    assert "mean-valence fallback" in si_source
    assert "no charge assignment" not in si_source

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    s19 = next(entry for entry in entries if entry["id"] == "Fig. S19")
    assert s19["final"] == "figS19_charge_coverage.pdf"
    assert s19["source"] == "paper/figs/figS17_charge_coverage.pdf"


def test_revised_main_figure_has_the_required_panel_order() -> None:
    module = _module()
    assert module.PANEL_ORDER == (
        "generator_screening",
        "controlled_relaxation",
        "mattergen_ladder",
        "gnome_site_complexity",
        "wrong_site_identity",
        "cost",
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert ".set_title(" not in source
    assert ".grid(" not in source
