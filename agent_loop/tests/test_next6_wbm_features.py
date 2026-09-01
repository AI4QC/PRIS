import math
import warnings

import numpy as np

from src.next6_wbm_features import geometry_features, parse_extxyz


def _two_atom_xyz(second_x: float) -> str:
    return f"""2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 material_id=toy-1 pbc="T T T"
H 0 0 0
He {second_x} 0 0
"""


def test_parse_extxyz_preserves_lattice_species_coordinates_and_id():
    # Break caught: swapping lattice rows or parsing the quoted comment as tokens
    # would silently corrupt every geometry feature.
    parsed = parse_extxyz(_two_atom_xyz(1.0))

    assert parsed.material_id == "toy-1"
    assert parsed.species == ("H", "He")
    np.testing.assert_allclose(parsed.lattice, np.eye(3) * 10.0)
    np.testing.assert_allclose(parsed.cart_coords, [[0, 0, 0], [1, 0, 0]])


def test_geometry_features_use_periodic_minimum_image_distances():
    # Break caught: direct Cartesian distances report 9.5 A instead of the 0.5 A
    # periodic contact and fail to detect a severe overlap.
    parsed = parse_extxyz(_two_atom_xyz(9.5))
    got = geometry_features(parsed, radii={"H": 1.0, "He": 1.0})

    assert got["feature_ok"] is True
    assert math.isclose(got["min_pair_distance"], 0.5)
    assert math.isclose(got["min_pair_ratio"], 0.25)
    assert math.isclose(got["repulsion_p2_per_atom"], 4.5)


def test_geometry_features_have_hand_checked_packing_and_scale_values():
    # Break caught: normalizing the sphere sum by atom count instead of cell volume
    # changes the physical meaning of packing across cell sizes.
    parsed = parse_extxyz(_two_atom_xyz(1.0))
    got = geometry_features(parsed, radii={"H": 1.0, "He": 1.0})

    expected_packing = 2 * (4 * math.pi / 3) / 1000
    assert math.isclose(got["volume_per_atom"], 500.0)
    assert math.isclose(got["packing_fraction"], expected_packing)
    assert math.isclose(got["min_pair_ratio"], 0.5)
    assert math.isclose(got["repulsion_p2_per_atom"], 0.5)
    assert math.isclose(got["repulsion_p2_l120"], (1 / 0.6 - 1) ** 2 / 2)


def test_missing_radius_abstains_instead_of_becoming_a_rejection():
    # Break caught: a missing element radius represented as zero would make an
    # unsupported structure look safely non-overlapping.
    parsed = parse_extxyz(_two_atom_xyz(1.0))
    got = geometry_features(parsed, radii={"H": 1.0})

    assert got["feature_ok"] is False
    assert got["feature_error"] == "missing_radius:He"
    assert math.isnan(got["min_pair_ratio"])


def test_default_radius_fallback_does_not_emit_missing_data_warnings():
    # Break caught: pymatgen warns for actinide calculated radii even when its
    # ordinary atomic-radius fallback is available, flooding full-run logs.
    text = """2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 material_id=act pbc="T T T"
Ac 0 0 0
U 2 0 0
"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = geometry_features(parse_extxyz(text))

    assert got["feature_ok"] is True
    assert caught == []
