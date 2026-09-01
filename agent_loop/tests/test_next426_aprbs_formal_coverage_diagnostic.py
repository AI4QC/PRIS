from __future__ import annotations

import inspect

import experiments.next426_aprbs_formal_coverage_diagnostic as d


def test_failure_categories_are_stable_and_exhaustive() -> None:
    cases = {
        "RuntimeError: APRBS maximum-entropy field lost positivity": "strict_positive_transport_boundary",
        "ValueError: APRBS maximum-entropy field did not converge: residual=2.5e-05": "strict_positive_transport_boundary",
        "ValueError: APRBS connected charge marginals are infeasible": "infeasible_component_marginals",
        "ValueError: APRBS periodic graph has an isolated charged site": "isolated_charged_site",
        "ValueError: cation has no opposite-sign periodic neighbor": "opposite_sign_graph_failure",
        "ValueError: formal valence inference returned no assignment": "formal_valence_failure",
        "ValueError: APRBS formal charges must be neutral and nonzero": "formal_valence_failure",
        "NEXT425 features require exact periodic geometry-only Atoms": "geometry_firewall_failure",
        "anything else": "other",
    }
    assert {key: d.failure_category(key) for key in cases} == cases


def test_diagnostic_interface_has_no_endpoint_or_label_input() -> None:
    names = tuple(inspect.signature(d.run_coverage_diagnostic).parameters)
    assert names == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "design_path",
        "probe_result_path", "workers", "require_formal_inputs",
    )
    assert not any(
        token in name for name in names
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_boundary_flags_remain_false() -> None:
    assert all(value is False for value in d.n425.BOUNDARY_FLAGS.values())
