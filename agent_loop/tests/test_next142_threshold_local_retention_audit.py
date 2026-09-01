import numpy as np

import src.next142_threshold_local_retention_audit as n142


def test_feature_audit_fixes_sign_from_scigen_and_reports_folds() -> None:
    values = np.array([9, 8, 7, 6, 1, 2, 3, 4, 10, 9, 2, 1], dtype=float)
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    folds = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4], dtype=int)

    result = n142.audit_one_source(
        values=values,
        protected=protected,
        folds=folds,
        direction=None,
        minimum_coverage=0.8,
        minimum_unique=4,
    )

    assert result is not None
    assert result["direction"] == 1
    assert result["pooled_auc"] > 0.9
    assert result["evaluable_folds"] == 5


def test_feature_audit_rejects_low_class_coverage() -> None:
    result = n142.audit_one_source(
        values=np.array([1.0, np.nan, 2.0, 3.0]),
        protected=np.array([True, True, False, False]),
        folds=np.array([0, 1, 0, 1]),
        direction=None,
        minimum_coverage=0.8,
        minimum_unique=2,
    )
    assert result is None


def test_blocked_feature_name_excludes_outcomes_and_tested_protections() -> None:
    assert n142.blocked_feature_name("dft_energy")
    assert n142.blocked_feature_name("coordination_low_volume_product")
    assert n142.blocked_feature_name("analytic_field_balance_protection")
    assert not n142.blocked_feature_name("sivr_site_imbalance_max")
