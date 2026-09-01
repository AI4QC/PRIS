from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next359_rfnc_formal_build as n


def test_formal_schema_rows_and_gates_are_frozen() -> None:
    assert n.PROTOCOL == "2026-08-13-next359-rfnc-formal-build-v1"
    assert n.MANIFEST_NAME == "MANIFEST.json"
    assert n.CATALOGUE_NAME == "NEXT359_RFNC_FEATURE_CATALOGUE.json"
    assert n.FEATURE_FILES == {
        "scigen": "next359_scigen_radical_facet_normal_covering.parquet",
        "wyformer": "next359_wyformer_radical_facet_normal_covering.parquet",
    }
    assert n.EXPECTED_ROWS == {"scigen": 13_470, "wyformer": 5_232}
    assert n.MINIMUM_FORMAL_COVERAGE == 0.90
    assert n.EXPECTED_PROBE_SHA256 == (
        "d73041f90930c6b27083edd948950b6bb1d4eae49ecde971d033a0055915e4fe"
    )


def test_payload_workers_produce_real_rows_and_fail_closed() -> None:
    atoms = __import__("ase.build", fromlist=["bulk"]).bulk(
        "NaCl", "rocksalt", a=5.64, cubic=True
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    structure = n.n267.AseAtomsAdaptor.get_structure(atoms)
    material_id, row = n._compute_wyformer_payload(
        ("wy-real", __import__("json").dumps(structure.as_dict()))
    )
    assert material_id == "wy-real"
    assert row["rfnc_supported"] is True
    assert np.isfinite(row[n.n359.FEATURE_NAMES[0]])
    bad_id, bad = n._compute_wyformer_payload(("wy-bad", "not-json"))
    assert bad_id == "wy-bad"
    assert bad["rfnc_supported"] is False
    assert np.isnan(bad[n.n359.FEATURE_NAMES[0]])


def test_label_free_statistics_are_exact_and_ignore_abstentions() -> None:
    table = pd.DataFrame({n.n359.FEATURE_NAMES[0]: [0.0, 0.2, np.nan, 0.8, 1.0]})
    statistics = n._label_free_statistics(table)[n.n359.FEATURE_NAMES[0]]
    assert statistics == {
        "minimum": 0.0,
        "q10": 0.0,
        "median": pytest.approx(0.5),
        "q90": 1.0,
        "maximum": 1.0,
        "unique_rounded_10": 4,
    }


def test_builder_interface_exposes_no_outcome_path_and_fails_closed(tmp_path) -> None:
    parameters = tuple(inspect.signature(n.build_cross_source_rfnc_features).parameters)
    assert parameters == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "design_path",
        "probe_result_path", "output_dir", "workers", "require_formal_inputs",
    )
    assert not any(
        token in name for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
    with pytest.raises(FileNotFoundError, match="NEXT359 input is missing"):
        n.build_cross_source_rfnc_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            probe_result_path=tmp_path / "probe",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
