from __future__ import annotations

import numpy as np
import pytest

from src.next118_metallic_applicability_protection import (
    compose_bounded_protection_score,
    metallic_packing_protection,
)


def test_minimum_conjunction_requires_both_analytic_legs() -> None:
    protection = metallic_packing_protection(
        electronegativity_mean=[1.4, 1.8, 1.4, 1.7],
        covalent_ratio_q05=[1.0, 1.0, 0.85, 0.95],
        electronegativity_upper=1.8,
        electronegativity_width=0.4,
        covalent_ratio_lower=0.9,
        covalent_ratio_width=0.1,
        combination="minimum",
    )
    np.testing.assert_allclose(protection, [1.0, 0.0, 0.0, 0.25])


def test_product_conjunction_is_bounded_and_continuous() -> None:
    protection = metallic_packing_protection(
        electronegativity_mean=[1.6, 1.4, 1.9],
        covalent_ratio_q05=[0.95, 1.1, 1.1],
        electronegativity_upper=1.8,
        electronegativity_width=0.4,
        covalent_ratio_lower=0.9,
        covalent_ratio_width=0.1,
        combination="product",
    )
    np.testing.assert_allclose(protection, [0.25, 1.0, 0.0])
    assert np.all((0.0 <= protection) & (protection <= 1.0))


def test_nonfinite_operands_turn_optional_protection_off() -> None:
    protection = metallic_packing_protection(
        electronegativity_mean=[np.nan, 1.4, np.inf],
        covalent_ratio_q05=[1.0, np.nan, 1.0],
        electronegativity_upper=1.8,
        electronegativity_width=0.4,
        covalent_ratio_lower=0.9,
        covalent_ratio_width=0.1,
    )
    np.testing.assert_array_equal(protection, np.zeros(3))


def test_composition_preserves_support_and_nonnegative_risk() -> None:
    base = np.array([3.0, 0.2, 7.0, 4.0])
    supported = np.array([True, True, False, True])
    protection = np.array([0.5, 1.0, 1.0, 0.0])
    base_before = base.copy()
    protection_before = protection.copy()
    score, observed_support = compose_bounded_protection_score(
        base_score=base,
        base_supported=supported,
        protection=protection,
        protection_weight=2.0,
    )
    np.testing.assert_allclose(score[supported], [2.0, 0.0, 4.0])
    assert np.isnan(score[~supported]).all()
    np.testing.assert_array_equal(observed_support, supported)
    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(protection, protection_before)


@pytest.mark.parametrize("combination", ["sum", "maximum", ""])
def test_unknown_conjunction_is_rejected(combination: str) -> None:
    with pytest.raises(ValueError, match="NEXT118"):
        metallic_packing_protection(
            electronegativity_mean=[1.4],
            covalent_ratio_q05=[1.0],
            electronegativity_upper=1.8,
            electronegativity_width=0.4,
            covalent_ratio_lower=0.9,
            covalent_ratio_width=0.1,
            combination=combination,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electronegativity_width": 0.0},
        {"covalent_ratio_width": -0.1},
        {"electronegativity_upper": np.nan},
        {"covalent_ratio_lower": np.inf},
    ],
)
def test_invalid_calibration_is_rejected(kwargs: dict[str, float]) -> None:
    parameters = {
        "electronegativity_mean": [1.4],
        "covalent_ratio_q05": [1.0],
        "electronegativity_upper": 1.8,
        "electronegativity_width": 0.4,
        "covalent_ratio_lower": 0.9,
        "covalent_ratio_width": 0.1,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match="NEXT118"):
        metallic_packing_protection(**parameters)


def test_shape_and_protection_domain_errors_are_rejected() -> None:
    with pytest.raises(ValueError, match="NEXT118"):
        metallic_packing_protection(
            electronegativity_mean=[1.4, 1.5],
            covalent_ratio_q05=[1.0],
            electronegativity_upper=1.8,
            electronegativity_width=0.4,
            covalent_ratio_lower=0.9,
            covalent_ratio_width=0.1,
        )
    with pytest.raises(ValueError, match="NEXT118"):
        compose_bounded_protection_score(
            base_score=[1.0, 2.0],
            base_supported=[True, True],
            protection=[0.0, 1.1],
            protection_weight=1.0,
        )


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_invalid_protection_weight_is_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="NEXT118"):
        compose_bounded_protection_score(
            base_score=[1.0],
            base_supported=[True],
            protection=[0.5],
            protection_weight=weight,
        )
