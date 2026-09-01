import importlib.util
from pathlib import Path
import warnings

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments"
    / "pu_synthesizability_20260821"
    / "render_moved_si_panels.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("render_moved_si_panels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_script_declares_editable_vector_text_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"svg.fonttype": "none"' in source
    assert '"pdf.fonttype": 42' in source


def _has_visible_grid(ax) -> bool:
    return any(line.get_visible() for line in (*ax.get_xgridlines(), *ax.get_ygridlines()))


def _assert_axis_labels_start_with_capital(fig) -> None:
    labels = [
        label
        for ax in fig.axes
        for label in (ax.get_xlabel(), ax.get_ylabel())
        if label.strip()
    ]
    assert labels
    for label in labels:
        if label.lstrip().startswith("$"):
            continue
        first_cased = next(
            (char for char in label if char.isalpha() and char.upper() != char.lower()),
            None,
        )
        if first_cased is not None:
            assert first_cased.isupper(), label


def test_validation_boundary_figure_reuses_old_fig4_cd_without_dashboard_chrome():
    mod = _load_module()

    fig, meta = mod.build_validation_boundary_figure()
    try:
        assert len(fig.axes) == 2
        assert [text.get_text() for text in fig.texts] == ["a", "b"]
        assert fig._suptitle is None
        assert all(ax.get_title() == "" for ax in fig.axes)
        assert not any(_has_visible_grid(ax) for ax in fig.axes)
        _assert_axis_labels_start_with_capital(fig)
        assert meta["panel_map"] == {"a": "old Fig. 4c", "b": "old Fig. 4d"}
        assert meta["rho_hist_rows"] == 34
        assert meta["omitted_classes"] == ["S1", "S2", "S3", "S4", "S5"]
        legend_sizes = [
            text.get_fontsize()
            for ax in fig.axes
            if ax.get_legend() is not None
            for text in ax.get_legend().get_texts()
        ]
        assert min(legend_sizes) >= 7.2
    finally:
        plt.close(fig)


def test_polymorph_ranking_figure_reuses_old_fig5_abc_without_dashboard_chrome():
    mod = _load_module()

    fig, meta = mod.build_polymorph_ranking_figure()
    try:
        assert len(fig.axes) == 3
        assert [text.get_text() for text in fig.texts] == ["a", "b", "c"]
        assert fig._suptitle is None
        assert all(ax.get_title() == "" for ax in fig.axes)
        assert not any(_has_visible_grid(ax) for ax in fig.axes)
        _assert_axis_labels_start_with_capital(fig)
        assert meta["panel_map"] == {
            "a": "old Fig. 5a",
            "b": "old Fig. 5b",
            "c": "old Fig. 5c",
        }
        assert meta["ranking_criteria"] == 13
        assert meta["rule_sets"] == ["L1", "L1'", "L2", "L3", "L4"]
    finally:
        plt.close(fig)


def test_render_all_writes_exact_review_bundle(tmp_path: Path, caplog):
    mod = _load_module()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mod.render_all(tmp_path)
    assert not [
        warning
        for warning in caught
        if "fontTools" in str(warning.filename) or "py23 module" in str(warning.message)
    ]
    assert "Font family ['cursive'] not found" not in caplog.text

    expected_stems = {
        "2026-08-23-SI-validation-boundary-and-omission-robustness",
        "2026-08-23-SI-polymorph-ranking-boundaries",
        "2026-08-23-SI-energy-phonon-record",
    }
    assert set(result) == expected_stems
    for stem in expected_stems:
        for suffix in (".pdf", ".png", ".svg"):
            path = tmp_path / f"{stem}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 10_000
