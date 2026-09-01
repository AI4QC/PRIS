from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.next271_joint_radical_packing_features as n


def test_joint_values_have_exact_schema_and_analytic_cases() -> None:
    zero = n.joint_radical_values(
        volume_cv=n.VOLUME_Q_LO,
        chebyshev_cv=n.CHEBYSHEV_Q_LO,
    )
    assert tuple(zero) == n.FEATURE_NAMES
    assert set(zero.values()) == {0.0}

    values = n.joint_radical_values(
        volume_cv=n.VOLUME_Q_HI,
        chebyshev_cv=n.CHEBYSHEV_Q_HI,
    )
    assert values == {
        "prvj_joint_min": 1.0,
        "prvj_joint_harmonic": 1.0,
        "prvj_joint_geometric": 1.0,
        "prvj_joint_product": 1.0,
        "prvj_joint_mean": 1.0,
        "prvj_joint_max": 1.0,
        "prvj_joint_l1_gap": 0.0,
        "prvj_volume_minus_chebyshev": 0.0,
        "prvj_chebyshev_minus_volume": 0.0,
        "prvj_volume_excess": 0.0,
        "prvj_chebyshev_excess": 0.0,
        "prvj_balance_weighted_joint": 1.0,
    }

    unbalanced = n.joint_radical_values(
        volume_cv=n.VOLUME_Q_LO + 2.0 * n.VOLUME_SPAN,
        chebyshev_cv=n.CHEBYSHEV_Q_LO + 0.5 * n.CHEBYSHEV_SPAN,
    )
    assert unbalanced["prvj_joint_min"] == 0.5
    assert unbalanced["prvj_joint_harmonic"] == 0.8
    assert unbalanced["prvj_joint_geometric"] == 1.0
    assert unbalanced["prvj_joint_product"] == 1.0
    assert unbalanced["prvj_joint_mean"] == 1.25
    assert unbalanced["prvj_joint_max"] == 2.0
    assert unbalanced["prvj_joint_l1_gap"] == 1.5
    assert unbalanced["prvj_volume_minus_chebyshev"] == 1.5
    assert unbalanced["prvj_chebyshev_minus_volume"] == -1.5
    assert unbalanced["prvj_volume_excess"] == 1.5
    assert unbalanced["prvj_chebyshev_excess"] == 0.0
    assert unbalanced["prvj_balance_weighted_joint"] == 0.125


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1])
def test_joint_values_refuse_nonfinite_or_negative_cv(bad: float) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        n.joint_radical_values(volume_cv=bad, chebyshev_cv=0.2)


def _table(source: str) -> pd.DataFrame:
    common = {
        "material_id": ["b", "a", "c"],
        "reduced_formula": ["NaCl", "CsCl", "C"],
        "partition_role": ["discovery"] * 3,
        "input_role": ["raw_generated_pre_dft_unrelaxed_x0"] * 3,
        "prv_volume_ratio_cv": [n.VOLUME_Q_HI, np.nan, n.VOLUME_Q_LO],
        "prv_chebyshev_ratio_cv": [n.CHEBYSHEV_Q_HI, np.nan, n.CHEBYSHEV_Q_LO],
        "prv_supported": [True, False, True],
        "prv_failure": [None, "guard", None],
    }
    if source == "wyformer":
        common["raw_material_id"] = [2, 1, 3]
    else:
        common["lattice_class"] = ["x", "x", "x"]
    return pd.DataFrame(common)


def test_transform_preserves_row_order_support_and_missingness() -> None:
    source = _table("scigen")
    frozen = source.copy(deep=True)
    output = n.transform_prv_table(source, source="scigen")
    pd.testing.assert_frame_equal(source, frozen)
    assert output["material_id"].tolist() == ["b", "a", "c"]
    assert output["prvj_supported"].tolist() == [True, False, True]
    assert output["prvj_failure"].tolist() == [None, "guard", None]
    assert not bool(output.loc[0, list(n.FEATURE_NAMES)].eq(1.0).all())
    assert output.loc[0, "prvj_joint_min"] == 1.0
    assert output.loc[1, list(n.FEATURE_NAMES)].isna().all()
    assert output.loc[2, list(n.FEATURE_NAMES)].eq(0.0).all()


def _write_inputs(root: Path) -> tuple[Path, Path]:
    prv = root / "prv"
    prv.mkdir()
    _table("scigen").to_parquet(
        prv / n.INPUT_FEATURE_FILES["scigen"], index=False
    )
    _table("wyformer").to_parquet(
        prv / n.INPUT_FEATURE_FILES["wyformer"], index=False
    )
    (prv / n.INPUT_MANIFEST_NAME).write_text("{}\n")
    (prv / n.INPUT_CATALOGUE_NAME).write_text("{}\n")
    design = root / "design.md"
    design.write_text("frozen\n")
    return prv, design


def test_builder_is_additive_atomic_and_discloses_boundaries(tmp_path: Path) -> None:
    prv, design = _write_inputs(tmp_path)
    output = tmp_path / "output"
    manifest = n.build_joint_radical_features(
        next267_dir=prv,
        design_path=design,
        output_dir=output,
        require_formal_inputs=False,
    )
    assert manifest["protocol"] == n.PROTOCOL
    assert manifest["next272_audit_authorized"] is True
    assert manifest["counts"]["scigen"] == {
        "rows": 3,
        "supported": 2,
        "failures": 1,
        "finite_feature_counts": {name: 2 for name in n.FEATURE_NAMES},
    }
    for key, expected in n.BOUNDARY_FLAGS.items():
        assert manifest[key] is expected
    assert (output / n.MANIFEST_NAME).is_file()
    assert (output / n.CATALOGUE_NAME).is_file()
    assert all((output / name).is_file() for name in n.FEATURE_FILES.values())
    assert not list(tmp_path.glob(".output.staging-*"))
    with pytest.raises(FileExistsError):
        n.build_joint_radical_features(
            next267_dir=prv,
            design_path=design,
            output_dir=output,
            require_formal_inputs=False,
        )


def test_builder_refuses_nonformal_input_identity(tmp_path: Path) -> None:
    prv, design = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="formal input identity differs"):
        n.build_joint_radical_features(
            next267_dir=prv,
            design_path=design,
            output_dir=tmp_path / "output",
            require_formal_inputs=True,
        )
