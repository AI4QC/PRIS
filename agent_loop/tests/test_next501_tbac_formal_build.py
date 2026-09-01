from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next501_tbac_formal_build as b


def _probe() -> dict[str, object]:
    return {
        "protocol": "2026-08-13-next500-tbac-label-blind-novelty-probe-v1",
        "design_sha256": b.n500.DESIGN_SHA256,
        "minimum_novelty_joint_finite": 40,
        "probe_passed": True,
        "next501_formal_build_authorized": True,
        "gates": {
            key: True
            for key in ("support", "closed_domain", "nondegenerate", "invariant", "novel")
        },
        **{name: False for name in b.PROBE_BOUNDARY_NAMES},
    }


def test_schema_rows_and_coverage_gate_are_frozen() -> None:
    assert b.PROTOCOL == "2026-08-13-next501-tbac-formal-build-v1"
    assert b.EXPECTED_ROWS == {"scigen": 13_470, "wyformer": 5_232}
    assert b.MINIMUM_FORMAL_COVERAGE == 0.95
    assert b.EXPECTED_PROBE_SHA256 == "643fa5973f26658ee7968733ab7e2102ef2bb7816ef0293b9e8a63a4c69ac5ec"


def test_probe_authorization_rejects_gate_or_boundary_change() -> None:
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
    table = pd.DataFrame({b.n500.FEATURE_NAMES[0]: np.linspace(0, 0.5, 40)})
    result = b.label_free_statistics(table)[b.n500.FEATURE_NAMES[0]]
    assert result["minimum"] == 0 and result["maximum"] == 0.5
    assert result["unique_rounded_10"] == 40
    with pytest.raises(RuntimeError, match="empty"):
        b.label_free_statistics(
            pd.DataFrame({b.n500.FEATURE_NAMES[0]: [np.nan]})
        )


def test_builder_has_no_endpoint_label_or_asset_input() -> None:
    parameters = tuple(
        inspect.signature(b.build_cross_source_tbac_features).parameters
    )
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "engineering_probe_result_path",
        "novelty_probe_result_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "asset", "validation", "replication")
    )
