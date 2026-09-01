"""Every figure is cited in the main text, and the numbering follows that order.

Supplementary figures are numbered by order of appearance in ``si_body.tex``, so this
also pins the physical order of the float blocks.  Both manuscript trees are checked:
``tex/`` (full version) and ``tex-submission/`` (condensed submission).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREES = ("tex", "tex-submission")


def _main_text(tree: Path) -> str:
    return "\n".join((tree / name).read_text(encoding="utf-8")
                     for name in ("body.tex", "methods.tex")
                     if (tree / name).is_file())


def _first_citations(text: str, pattern: str) -> list[str]:
    seen, order = set(), []
    for match in re.finditer(pattern, text):
        label = match.group(1)
        if label not in seen:
            seen.add(label)
            order.append(label)
    return order


@pytest.mark.parametrize("name", TREES)
def test_main_figures_are_cited_in_numerical_order(name: str) -> None:
    tree = ROOT / name
    body = _main_text(tree)
    defined = re.findall(r"\\label\{(fig:[a-z-]+)\}", body)
    cited = [label for label in _first_citations(body, r"\\ref\{(fig:[a-z-]+)\}")
             if label in defined]
    assert len(defined) == len(set(defined)) == 5
    assert cited == defined


@pytest.mark.parametrize("name", TREES)
def test_supplementary_figures_are_cited_in_numerical_order(name: str) -> None:
    tree = ROOT / name
    si = (tree / "si_body.tex").read_text(encoding="utf-8")
    body = _main_text(tree)

    # supplementary numbering is the order the float blocks appear in
    defined = re.findall(r"\\label\{(sifig:[a-z0-9-]+)\}", si)
    assert len(defined) == len(set(defined))

    cited = _first_citations(body, r"\\ref\{si-(sifig:[a-z0-9-]+)\}")
    uncited = [label for label in defined if label not in set(cited)]
    assert not uncited, f"supplementary figures never cited in the main text: {uncited}"

    rank = {label: i for i, label in enumerate(defined)}
    out_of_order = [(a, b) for a, b in zip(cited, cited[1:]) if rank[a] > rank[b]]
    assert not out_of_order, (
        "supplementary figures cited out of numerical order: "
        + ", ".join(f"S{rank[a] + 1} before S{rank[b] + 1}" for a, b in out_of_order)
    )


@pytest.mark.parametrize("name", TREES)
def test_supplementary_figure_files_are_numbered_by_appearance(name: str) -> None:
    tree = ROOT / name
    si = (tree / "si_body.tex").read_text(encoding="utf-8")
    stems = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figs/(figS\d+_[a-z0-9_]+)\.pdf\}", si)
    assert [int(re.match(r"figS(\d+)", stem).group(1)) for stem in stems] == \
        list(range(1, len(stems) + 1))
    for stem in stems:
        assert (tree / "figs" / f"{stem}.pdf").is_file(), stem
