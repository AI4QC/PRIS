from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "tex" / "front_body.tex"
BODY = ROOT / "tex" / "body.tex"
SI = ROOT / "tex" / "si_body.tex"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_introduction_motivates_all_five_contradictions_with_real_context():
    intro = _text(FRONT)
    lower = intro.lower()

    assert "pauling's rules and the modern distance filter err in opposite directions" in lower
    assert "whether a structure could exist" in lower
    assert "which polymorph" in lower and "try to make" in lower
    assert all(token in intro for token in ("sun2016thermodynamic", "aykol2018thermodynamic"))
    assert "381{,}000" in intro and "421{,}000" in intro
    assert intro.index("merchant2023scaling") < intro.index("cheetham2024artificial")
    for source in (
        "szymanski2023autonomous",
        "leeman2024challenges",
        "szymanski2026correction",
        "yamazaki2026navigating",
    ):
        assert source in intro
    assert "harrison2010falsified" in intro
    assert "iucr2012retraction" in intro


def test_section24_restores_the_tie_refutation_and_three_axis_counts():
    body = _text(BODY)
    flat = re.sub(r"\s+", " ", body)
    assert "Counting ties as errors" in body
    assert re.search(r"median space-group numbers? (?:were|of) 87 and 62", body)
    for result in ("35.6\\%", "41.7\\%", "4{,}271"):
        assert result in body
    assert "At L3, 134 satisfied the rules, 11 failed and 355 received no verdict" in flat
    assert "At L4, 4 satisfied the rules, 155 failed and 341 received no verdict" in flat


def test_wrong_site_story_runs_from_historical_problem_to_controlled_test_and_back():
    body = _text(BODY)
    start = body.index("This chemical-identity failure")
    end = body.index("PRIS is inexpensive enough", start)
    story = body[start:end]

    historical = story.index("at least 70")
    controlled = story.index("every coordinate")
    quantitative = story.index(r"\ref{fig:deploy}e")
    archive = story.index("recovered retraction archive")
    closure = story.index("Cu, Ni, Mn or Fe")
    assert historical < controlled < quantitative < archive < closure
    assert "at least 70" in story
    assert "Hirshfeld" in story
    assert "anomalous distances" in story
    assert "harrison2010falsified" in story
    assert "iucr2012retraction" in story
    assert "19 of 69" in story and "62 of 69" in story
    assert "40 of 83" in story and "82 of 83" in story
    assert "one genuine diffraction data set" not in (body + _text(SI))


def test_external_validation_si_has_parent_label_and_charge_coverage_figure():
    si = _text(SI)
    assert si.count(r"\label{si-note:s17}") == 1
    assert r"figs/figS19_charge_coverage.pdf" in si
    assert r"\label{sifig:charge-coverage}" in si
    caption_start = si.rfind(r"\caption", 0, si.index(r"\label{sifig:charge-coverage}"))
    caption = si[caption_start : si.index(r"\label{sifig:charge-coverage}")]
    assert "charge-dependent" in caption.lower()
    assert "not classified as" in caption.lower()
