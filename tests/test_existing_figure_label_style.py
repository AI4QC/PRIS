"""Regression checks for the established main- and Supplementary-figure style.

These checks intentionally cover only the legacy plotting scripts.  The new
merged Fig. 4 and its newly demoted Supplementary panels have their own tests.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "src" / "paper_figs.py",
    ROOT / "src" / "fig3_anatomy.py",
    ROOT / "src" / "fig6_deployment.py",
    ROOT / "src" / "si_figs.py",
    ROOT / "src" / "figS7_amplitude_response.py",
]


def _literal_axis_labels(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    labels: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"set_xlabel", "set_ylabel", "set_label"}:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                labels.append((node.lineno, value))
    return labels


def _starts_with_display_capital(label: str) -> bool:
    stripped = label.lstrip()
    # A mathematical symbol, for example rho or sigma, has no sentence-case
    # initial in the rendered figure.
    if stripped.startswith("$"):
        return True
    first_alpha = next((char for char in stripped if char.isalpha()), "")
    return bool(first_alpha) and first_alpha.isupper()


def test_literal_axis_and_colourbar_labels_start_with_capitals() -> None:
    failures: list[str] = []
    for source in SOURCES:
        for line, label in _literal_axis_labels(source):
            if not _starts_with_display_capital(label):
                failures.append(f"{source.relative_to(ROOT)}:{line}: {label!r}")
    assert not failures, "Lowercase display labels:\n" + "\n".join(failures)


def test_legacy_figure_sources_do_not_create_titles_or_grids() -> None:
    failures: list[str] = []
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        for forbidden in (".set_title(", ".grid("):
            if forbidden in text:
                failures.append(f"{source.relative_to(ROOT)} contains {forbidden}")
    assert not failures, "\n".join(failures)


def test_dynamic_axis_labels_follow_the_same_display_style() -> None:
    deployment = (ROOT / "src" / "fig6_deployment.py").read_text(encoding="utf-8")
    amplitude = (ROOT / "src" / "figS7_amplitude_response.py").read_text(
        encoding="utf-8"
    )
    si = (ROOT / "src" / "si_figs.py").read_text(encoding="utf-8")

    assert 'E_LABEL = "Energy released on relaxation' in deployment
    assert '"Linear strain (%)"' in amplitude
    assert '(axes[0], "sat", "Satisfaction", "a")' in si
    assert '(axes[1], "excl", "Damage detection", "b")' in si


def test_fig1_rulespace_is_named_below_panel_b() -> None:
    source = (ROOT / "src" / "paper_figs.py").read_text(encoding="utf-8")
    # the name now carries the order of magnitude of the searched space
    assert r'r"Explored $2\times10^{6}$ law space", labelpad=2.0,' in source
    assert 'fontsize=plt.rcParams["axes.labelsize"]' in source


def test_fig1_trajectory_uses_a_concise_y_axis_label() -> None:
    source = (ROOT / "src" / "paper_figs.py").read_text(encoding="utf-8")
    assert 'ax.set_ylabel("Best performance\\n(held-out data)")' in source
    assert "Best performance reached so far" not in source


def test_amplitude_figure_keeps_text_editable_in_vector_exports() -> None:
    amplitude = (ROOT / "src" / "figS7_amplitude_response.py").read_text(
        encoding="utf-8"
    )
    assert '"svg.fonttype": "none"' in amplitude
    assert 'for _ext in ("pdf", "svg", "png")' in amplitude
