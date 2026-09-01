from __future__ import annotations

from pathlib import Path

from experiments.pu_synthesizability_20260821.plot_draft_fig5 import build_draft_fig5


def test_draft_fig5_writes_both_raster_and_vector_outputs(tmp_path: Path) -> None:
    png, pdf = build_draft_fig5(tmp_path)
    assert png.is_file() and png.stat().st_size > 10_000
    assert pdf.is_file() and pdf.stat().st_size > 1_000
    assert (tmp_path / "SHA256SUMS_fig5").is_file()
