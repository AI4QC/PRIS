"""Mathematical contracts for corrected np-next-20260801d descriptors."""

from __future__ import annotations

import sys
from pathlib import Path
import hashlib

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import next4_features  # noqa: E402
from next4_features import (  # noqa: E402
    P7_ATOMIC_RADII,
    P7_SHORT_CONTACT_RATIO,
    _sha256,
    infer_formal_valences,
    main,
    next4_local_features,
    p2c_voronoi_site_stats,
    p6c_shell_gap_site_stats,
    p7c_polyanion_features,
    p9c_lewis_features,
)


def _single_site(symbol: str, lattice_constant: float):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(lattice_constant),
        [symbol],
        [[0.0, 0.0, 0.0]],
    )


def test_p6c_retains_nonzero_periodic_self_images():
    structure = _single_site("Na", 2.0)
    sphere = structure.get_all_neighbors(8.0)
    stats = p6c_shell_gap_site_stats(structure, [1.0], sphere=sphere)
    assert stats[0] is not None
    assert stats[0]["gap_pos"] >= 1
    assert stats[0]["gap_ratio"] >= 1.0


def test_p2c_retains_periodic_self_image_faces():
    from next_features import _voronoi_polyhedra

    structure = _single_site("Na", 2.0)
    stats = p2c_voronoi_site_stats(
        structure,
        [1.0],
        polyhedra=_voronoi_polyhedra(structure),
    )
    assert stats[0] is not None
    assert stats[0]["sa_effective_cn"] == pytest.approx(6.0)
    assert stats[0]["sa_like_fraction"] == pytest.approx(1.0)
    assert stats[0]["sa_max_fraction"] == pytest.approx(1.0 / 6.0)


def test_p7c_retains_periodic_self_images_and_uses_all_anions_in_denominator():
    structure = _single_site("Cl", 4.0)
    sphere = structure.get_all_neighbors(8.0)
    out = p7c_polyanion_features(structure, [-1.0], sphere=sphere)
    assert out["p7c_an_contact_min"] == pytest.approx(
        4.0 / (2.0 * P7_ATOMIC_RADII["Cl"])
    )
    assert out["p7c_an_short_contact_frac"] == pytest.approx(0.0)
    assert out["p7c_an_within8_fraction"] == pytest.approx(1.0)
    assert out["p7c_an_censored_fraction"] == pytest.approx(0.0)


def test_p7c_no_within8_neighbor_is_right_censored_but_not_short():
    structure = _single_site("Cl", 20.0)
    sphere = structure.get_all_neighbors(8.0)
    out = p7c_polyanion_features(structure, [-1.0], sphere=sphere)
    censored = 8.0 / (2.0 * P7_ATOMIC_RADII["Cl"])
    assert censored > P7_SHORT_CONTACT_RATIO
    assert out["p7c_an_contact_min"] == pytest.approx(censored)
    assert out["p7c_an_short_contact_frac"] == pytest.approx(0.0)
    assert out["p7c_an_within8_fraction"] == pytest.approx(0.0)
    assert out["p7c_an_censored_fraction"] == pytest.approx(1.0)


def test_p9c_uses_opposite_sign_graph_degree_not_all_neighbor_degree():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(10.0),
        ["Ca", "F", "F", "Na"],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
    )

    def edge(other: int):
        return {"site_index": other, "image": (0.0, 0.0, 0.0)}

    neighbors = [
        [edge(1), edge(2), edge(3)],
        [edge(0)],
        [edge(0)],
        [edge(0)],
    ]
    out = p9c_lewis_features(
        structure,
        [2.0, -1.0, -1.0, 1.0],
        neighbors=neighbors,
    )
    assert out["p9c_bond_mismatch_max"] == pytest.approx(0.0)
    assert out["p9c_cat_site_mismatch_max"] == pytest.approx(0.0)


def test_valence_policy_uses_one_ordered_dataset_independent_cascade(monkeypatch):
    structure = _single_site("Na", 4.0)
    calls: list[str] = []

    def no_integer(_structure):
        calls.append("integer")
        return None, False

    def fractional(_structure):
        calls.append("fractional")
        return [1.25]

    def forbidden_balance(_formula):
        calls.append("balance")
        raise AssertionError("balance must not run after fractional success")

    monkeypatch.setattr(next4_features, "guess_oxi", no_integer)
    monkeypatch.setattr(next4_features, "frac_oxi", fractional)
    monkeypatch.setattr(next4_features, "balance", forbidden_balance)
    values, source = infer_formal_valences(structure)
    np.testing.assert_allclose(values, [1.25])
    assert source == "frac_oxi"
    assert calls == ["integer", "fractional"]


def test_next4_local_features_emits_all_corrected_families():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0.5],
        ],
    )
    out, failures = next4_local_features(structure, [1.0] * 4 + [-1.0] * 4)
    assert failures == {}
    for prefix in ("p2c_", "p6c_", "p7c_", "p9c_"):
        assert any(name.startswith(prefix) for name in out), prefix
    assert out["p9c_bond_mismatch_max"] == pytest.approx(0.0)


def test_sha256_hashes_each_input_byte_once(tmp_path):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abc")
    assert _sha256(source) == hashlib.sha256(b"abc").hexdigest()


def test_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "already-there.parquet"
    output.write_bytes(b"keep")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(
            [
                "real",
                "--isolated-dir",
                str(tmp_path / "missing"),
                "--out",
                str(output),
                "--workers",
                "1",
            ]
        )
    assert output.read_bytes() == b"keep"
