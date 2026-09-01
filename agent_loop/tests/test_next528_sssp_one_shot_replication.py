import pytest

import src.next528_sssp_one_shot_replication as n


def _manifest(passes=True):
    return {
        "protocol": n.VALIDATION_PROTOCOL,
        "passes_all_validation_gates": passes,
        "next528_internal_replication_authorized": passes,
        "internal_validation_endpoint_values_opened": True,
        "internal_replication_endpoint_values_opened": False,
        "formula_or_threshold_modified": False,
    }


def _evaluation(passes=True):
    return {
        "protocol": n.VALIDATION_PROTOCOL,
        "partition_role": "internal_validation",
        "passes_all_validation_gates": passes,
        "scientific_status": (
            "advance_to_internal_replication" if passes
            else "internal_validation_failure_stop"
        ),
        "formula_or_threshold_modified": False,
    }


def test_replication_authorization_requires_the_complete_validation_pass():
    n.validate_replication_authorization(_manifest(), _evaluation())
    with pytest.raises(PermissionError, match="not authorized"):
        n.validate_replication_authorization(_manifest(False), _evaluation(False))


def test_replication_role_and_gates_are_identical_to_validation():
    assert n.ROLE == "internal_replication"
    assert n.GATES == n.n527.GATES
    assert n.BOOTSTRAP_DRAWS == n.n527.BOOTSTRAP_DRAWS
    assert n.BOOTSTRAP_SEED == n.n527.BOOTSTRAP_SEED
