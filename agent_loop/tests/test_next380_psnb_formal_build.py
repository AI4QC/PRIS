from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next380_psnb_formal_build as b


def _probe() -> dict[str, object]:
    return {
        "protocol": "2026-08-13-next379-psnb-label-blind-probe-v1",
        "design_sha256": b.n379.DESIGN_SHA256,
        "minimum_novelty_joint_finite": 40,
        "probe_passed": True,
        "next380_formal_build_authorized": True,
        "gates": {
            "support": True,
            "closed_domain": True,
            "nondegenerate": True,
            "invariant": True,
            "novel": True,
        },
        **{name: False for name in b.PROBE_BOUNDARY_NAMES},
    }


def test_frozen_formal_schema_and_coverage_are_exact() -> None:
    assert b.PROTOCOL == "2026-08-13-next380-psnb-formal-build-v1"
    assert b.EXPECTED_ROWS == {"scigen": 13_470, "wyformer": 5_232}
    assert b.MINIMUM_FORMAL_COVERAGE == 0.90
    assert set(b.FEATURE_FILES) == {"scigen", "wyformer"}
    assert len(b.EXPECTED_PROBE_SHA256) == 64


def test_probe_authorization_rejects_every_gate_or_boundary_change() -> None:
    b.validate_probe_authorization(_probe())
    changed = _probe()
    changed["gates"] = {**changed["gates"], "novel": False}
    with pytest.raises(ValueError, match="authorization"):
        b.validate_probe_authorization(changed)
    changed = _probe()
    changed["dft_values_used"] = True
    with pytest.raises(ValueError, match="authorization"):
        b.validate_probe_authorization(changed)


def test_label_free_statistics_require_finite_feature() -> None:
    table = pd.DataFrame({b.n379.FEATURE_NAMES[0]: np.linspace(0.0, 1.0, 40)})
    result = b.label_free_statistics(table)
    assert result[b.n379.FEATURE_NAMES[0]]["minimum"] == 0.0
    assert result[b.n379.FEATURE_NAMES[0]]["maximum"] == 1.0
    assert result[b.n379.FEATURE_NAMES[0]]["unique_rounded_10"] == 40
    with pytest.raises(RuntimeError, match="empty"):
        b.label_free_statistics(
            pd.DataFrame({b.n379.FEATURE_NAMES[0]: [np.nan]})
        )


def test_builder_interface_contains_no_endpoint_or_label_input() -> None:
    parameters = tuple(
        inspect.signature(b.build_cross_source_psnb_features).parameters
    )
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "probe_result_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any("endpoint" in name or "label" in name for name in parameters)


def test_no_dft_boundary_flags_are_false() -> None:
    assert all(value is False for value in b.n379.BOUNDARY_FLAGS.values())
