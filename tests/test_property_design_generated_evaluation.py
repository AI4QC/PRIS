from __future__ import annotations

import hashlib
import json
import sys
import warnings
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from experiments.property_design_20260821 import evaluate_generated as generated
from experiments.property_design_20260821.evaluate_generated import (
    evaluate_rungs_from_features,
    standalone_predicate_verdict,
    summarize_inverse_queue,
)
from experiments.property_design_20260821.uma_bulk import EV_PER_A3_TO_GPA


SATISFYING_FEATURES = {
    "bl_min": 0.90,
    "bl_mean": 1.00,
    "cn_an_mean": 3.0,
    "madz_range": 10.0,
    "mad_max": 5.0,
    "frac_like_bonds": 0.0,
    "fi": 0.70,
    "wyckoff_econ": 0.50,
    "bv_rel_mean": 0.20,
}


def test_minimum_pair_distance_preserves_exact_overlap():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    assert generated._minimum_pair_distance(structure) == 0.0


def test_minimum_pair_distance_finds_shortest_periodic_self_image():
    structure = Structure(
        Lattice([[5.0, 0.0, 0.0], [4.9, 0.1, 0.0], [0.0, 0.0, 5.0]]),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    assert generated._minimum_pair_distance(structure) == pytest.approx(2**0.5 / 10.0)


def _single_site_structure(species: str) -> Structure:
    return Structure(Lattice.cubic(3.0), [species], [[0.0, 0.0, 0.0]])


def test_load_archive_preserves_member_and_uses_natural_numeric_order(tmp_path):
    archive = tmp_path / "generated.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("attempt_100_00.cif", _single_site_structure("Ge").to(fmt="cif"))
        handle.writestr("attempt_10_00.cif", _single_site_structure("Si").to(fmt="cif"))
        handle.writestr("attempt_2_00.cif", _single_site_structure("C").to(fmt="cif"))

    records = generated._load_archive(archive)

    assert [
        record.get("archive_member") if isinstance(record, dict) else None
        for record in records
    ] == ["attempt_2_00.cif", "attempt_10_00.cif", "attempt_100_00.cif"]
    assert [record["candidate_id"] for record in records] == [
        "candidate_0000",
        "candidate_0001",
        "candidate_0002",
    ]


def test_load_archive_rejects_duplicate_member_names(tmp_path):
    archive = tmp_path / "generated.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(archive, "w") as handle:
            handle.writestr("same.cif", _single_site_structure("C").to(fmt="cif"))
            handle.writestr("same.cif", _single_site_structure("Si").to(fmt="cif"))

    with pytest.raises(ValueError, match="duplicate archive member"):
        generated._load_archive(archive)


def test_load_archive_rejects_structurematcher_equivalent_duplicates(tmp_path):
    original = Structure(
        Lattice.cubic(5.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    translated = original.copy()
    translated.translate_sites(range(len(translated)), [0.25, 0.25, 0.25])
    archive = tmp_path / "generated.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("candidate_1.cif", original.to(fmt="cif"))
        handle.writestr("candidate_2.cif", translated.to(fmt="cif"))

    with pytest.raises(ValueError, match="StructureMatcher-equivalent duplicate"):
        generated._load_archive(archive)


def _write_distinct_archive(tmp_path, species=("C", "Si", "Ge")):
    archive = tmp_path / "generated.zip"
    with ZipFile(archive, "w") as handle:
        for index, symbol in enumerate(species):
            handle.writestr(
                f"crystal_{index:06d}.cif",
                _single_site_structure(symbol).to(fmt="cif"),
            )
    return archive


def _aggregate_manifest(archive, records):
    source_protocol = {
        "condition": {"ml_bulk_modulus": 400},
        "generator": {"name": "MatterGen"},
        "sampling": {"guidance_factor": 2.0},
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            source_protocol, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "source": "mattergen",
        "protocol_version": "2026-08-21-mattergen-shard-aggregate-v3",
        "protocol_sha256": protocol_sha256,
        "source_protocol": source_protocol,
        "inputs": [
            {
                "shard_dir": "/fixed/source/shard",
                "manifest_sha256": "b" * 64,
                "source_seed": 20260821,
            }
        ],
        "outputs": {
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "structure_count": len(records),
            "candidates": [
                {
                    "aggregate_member": record["archive_member"],
                    "source_shard": "/fixed/source/shard",
                    "source_member": f"source_{index}.cif",
                    "source_manifest_sha256": "b" * 64,
                    "source_seed": 20260821,
                    "cif_sha256": record["cif_sha256"],
                    "structure_sha256": record["structure_sha256"],
                }
                for index, record in enumerate(records)
            ],
        }
    }


def test_archive_contract_verifies_valid_aggregate_manifest(tmp_path):
    archive = _write_distinct_archive(tmp_path)
    records = generated._load_archive(archive)
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(_aggregate_manifest(archive, records)))
    validator = getattr(generated, "_validate_archive_contract", None)

    assert validator is not None
    validated = validator(
        archive_path=archive,
        records=records,
        source="mattergen",
        input_manifest=manifest_path,
        min_unique_structures=3,
    )

    assert validated["outputs"]["structure_count"] == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["outputs"].__setitem__(
                "archive_sha256", "0" * 64
            ),
            "archive SHA256 mismatch",
        ),
        (
            lambda payload: payload["outputs"].__setitem__("structure_count", 2),
            "structure_count mismatch",
        ),
        (
            lambda payload: payload["outputs"]["candidates"][0].__setitem__(
                "aggregate_member", "missing.cif"
            ),
            "candidate member mapping mismatch",
        ),
        (
            lambda payload: payload["outputs"]["candidates"][0].__setitem__(
                "structure_sha256", "0" * 64
            ),
            "structure_sha256 mismatch",
        ),
        (
            lambda payload: payload["outputs"]["candidates"][0].__setitem__(
                "cif_sha256", "0" * 64
            ),
            "cif_sha256 mismatch",
        ),
        (
            lambda payload: payload.__setitem__("protocol_sha256", "0" * 64),
            "protocol_sha256 mismatch",
        ),
        (
            lambda payload: payload["outputs"]["candidates"][0].__setitem__(
                "source_manifest_sha256", "0" * 64
            ),
            "source manifest mapping mismatch",
        ),
        (
            lambda payload: payload["outputs"]["candidates"][0].__setitem__(
                "source_seed", 1
            ),
            "source seed mapping mismatch",
        ),
    ],
)
def test_archive_contract_rejects_manifest_mismatch(tmp_path, mutation, message):
    archive = _write_distinct_archive(tmp_path)
    records = generated._load_archive(archive)
    manifest = _aggregate_manifest(archive, records)
    mutation(manifest)
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    validator = getattr(generated, "_validate_archive_contract", None)

    assert validator is not None
    with pytest.raises(ValueError, match=message):
        validator(
            archive_path=archive,
            records=records,
            source="mattergen",
            input_manifest=manifest_path,
            min_unique_structures=3,
        )


def test_archive_contract_requires_manifest_for_mattergen_and_minimum_count(tmp_path):
    archive = _write_distinct_archive(tmp_path, species=("C",))
    records = generated._load_archive(archive)
    validator = getattr(generated, "_validate_archive_contract", None)

    assert validator is not None
    with pytest.raises(ValueError, match="--input-manifest is required for MatterGen"):
        validator(
            archive_path=archive,
            records=records,
            source="mattergen",
            input_manifest=None,
            min_unique_structures=1,
        )
    with pytest.raises(ValueError, match="--input-manifest is required"):
        validator(
            archive_path=archive,
            records=records,
            source="llm",
            input_manifest=None,
            min_unique_structures=1,
        )
    with pytest.raises(ValueError, match="requires at least 2 unique structures"):
        validator(
            archive_path=archive,
            records=records,
            source="llm",
            input_manifest=tmp_path / "not-read-before-count-check.json",
            min_unique_structures=2,
        )


def test_archive_contract_binds_cli_source_to_manifest_source(tmp_path):
    archive = _write_distinct_archive(tmp_path, species=("C",))
    records = generated._load_archive(archive)
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(
        json.dumps(_aggregate_manifest(archive, records), sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="source mismatch"):
        generated._validate_archive_contract(
            archive_path=archive,
            records=records,
            source="llm",
            input_manifest=manifest_path,
            min_unique_structures=1,
        )


def test_generated_evaluation_cli_defaults_to_1024_unique_structures():
    parser = getattr(generated, "build_parser", lambda: None)()

    assert parser is not None
    defaults = parser.parse_args(
        ["--input-archive", "generated.zip", "--output-dir", "out", "--source", "mattergen"]
    )
    lowered = parser.parse_args(
        [
            "--input-archive",
            "generated.zip",
            "--input-manifest",
            "aggregate_manifest.json",
            "--output-dir",
            "out",
            "--source",
            "mattergen",
            "--min-unique-structures",
            "3",
        ]
    )

    assert defaults.min_unique_structures == 1024
    assert defaults.input_manifest is None
    assert lowered.min_unique_structures == 3
    assert lowered.input_manifest == "aggregate_manifest.json"


def test_run_evaluation_checks_mattergen_manifest_before_pris(tmp_path, monkeypatch):
    archive = _write_distinct_archive(tmp_path, species=("C",))
    monkeypatch.setattr(
        generated,
        "evaluate_structure_pris",
        lambda *args, **kwargs: pytest.fail("PRIS must not run before manifest validation"),
    )
    args = SimpleNamespace(
        input_archive=str(archive),
        input_manifest=None,
        min_unique_structures=1,
        output_dir=str(tmp_path / "out"),
        source="mattergen",
        src_dir="src",
        checkpoint="unused.pt",
        energy_batch_size=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="--input-manifest is required for MatterGen"):
        generated.run_evaluation(args)


def test_rung_evaluation_passes_all_satisfying_features():
    result = evaluate_rungs_from_features(SATISFYING_FEATURES)

    assert result["rungs"] == {
        "L1": "pass",
        "L1_prime": "pass",
        "L2": "pass",
        "L3": "pass",
        "L4": "pass",
    }


def test_rung_evaluation_preserves_unknown_and_reject_dominance():
    unknown = dict(SATISFYING_FEATURES, mad_max=np.nan)
    rejected = dict(unknown, bl_min=0.70)

    unknown_result = evaluate_rungs_from_features(unknown)
    rejected_result = evaluate_rungs_from_features(rejected)

    assert unknown_result["rungs"]["L2"] == "no verdict"
    assert rejected_result["rungs"]["L2"] == "reject"


def test_guarded_predicate_is_satisfied_when_its_condition_does_not_apply():
    features = dict(
        SATISFYING_FEATURES,
        fi=0.40,
        frac_like_bonds=0.9,
        cn_an_mean=5.0,
        bl_mean=2.0,
    )

    result = evaluate_rungs_from_features(features)

    assert result["predicates"]["D3"] == "not applicable"
    assert result["predicates"]["D6"] == "not applicable"
    assert result["rungs"]["L3"] == "pass"


def test_standalone_predicate_keeps_unknown_but_treats_inapplicable_as_satisfied():
    assert standalone_predicate_verdict("violated") == "reject"
    assert standalone_predicate_verdict("unknown") == "no verdict"
    assert standalone_predicate_verdict("not applicable") == "pass"


def test_charge_independent_d7_remains_decisive_when_charge_features_fail(
    monkeypatch,
):
    fake_rules = SimpleNamespace(
        guess_oxi=lambda structure: (structure, False),
        frac_oxi=lambda structure: None,
        features=lambda structure: (None, "charge assignment unavailable"),
    )
    monkeypatch.setitem(sys.modules, "apply_rules", fake_rules)
    monkeypatch.setattr(
        generated,
        "_wyckoff_economy",
        lambda structure, *, symprec: 1.0 if symprec == 0.01 else 0.5,
    )

    result = generated.evaluate_structure_pris(object(), src_dir="src")

    assert result["predicates"]["D7"] == "violated"
    assert result["rungs"]["L1"] == "no verdict"
    assert result["rungs"]["L4"] == "reject"


def test_pris_evaluation_preserves_the_actual_site_charge_assignment(monkeypatch):
    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    fake_rules = SimpleNamespace(
        guess_oxi=lambda candidate: ([1.0, -1.0], True),
        frac_oxi=lambda candidate: pytest.fail("integer route must be retained"),
        features=lambda candidate: (dict(SATISFYING_FEATURES), None),
    )
    monkeypatch.setitem(sys.modules, "apply_rules", fake_rules)
    monkeypatch.setattr(generated, "_wyckoff_economy", lambda *args, **kwargs: 0.5)

    result = generated.evaluate_structure_pris(structure, src_dir="src")

    assert result["charge_assignment_route"] == "integer"
    assert result["charge_assignment_values"] == [1.0, -1.0]


def test_pris_features_are_computed_once_per_generated_structure(monkeypatch):
    calls = 0

    def features(candidate):
        nonlocal calls
        calls += 1
        return dict(SATISFYING_FEATURES), None

    fake_rules = SimpleNamespace(
        guess_oxi=lambda candidate: ([1.0, -1.0], True),
        frac_oxi=lambda candidate: None,
        features=features,
    )
    monkeypatch.setitem(sys.modules, "apply_rules", fake_rules)
    monkeypatch.setattr(generated, "_wyckoff_economy", lambda *args, **kwargs: 0.5)
    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    generated.evaluate_structure_pris(structure, src_dir="src")

    assert calls == 1


def test_inverse_summary_uses_all_generated_structures_as_primary_queue():
    frame = pd.DataFrame(
        {
            "fit_valid": [True, True, True, False],
            "clamped_bulk_modulus_proxy_gpa": [350.0, 250.0, 100.0, np.nan],
            "rung_L4_verdict": ["pass", "reject", "reject", "no verdict"],
        }
    )

    summary = summarize_inverse_queue(
        frame,
        verdict_column="rung_L4_verdict",
        proxy_threshold_gpa=200.0,
    )

    assert summary["generated_queue_count"] == 4
    assert summary["queue_removed_count"] == 2
    assert summary["queue_reduction"] == 0.5
    assert summary["high_property_candidate_count"] == 2
    assert summary["high_property_removed_count"] == 1
    assert summary["high_property_retained_count"] == 1
    assert summary["high_property_retention"] == 0.5
    assert summary["clamped_bulk_modulus_proxy_threshold_gpa"] == 200.0
    assert (
        summary["high_property_candidate_definition"]
        == "fit_valid and clamped_bulk_modulus_proxy_gpa >= threshold"
    )
    assert "bulk_modulus_gpa" not in str(summary)
    assert "good_material" not in str(summary)


def test_run_evaluation_writes_only_clamped_bulk_modulus_proxy_schema(
    tmp_path, monkeypatch
):
    archive = _write_distinct_archive(tmp_path, species=("C",))
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(
        json.dumps(_aggregate_manifest(archive, generated._load_archive(archive)))
        + "\n"
    )
    evaluated = evaluate_rungs_from_features(SATISFYING_FEATURES)
    monkeypatch.setattr(
        generated,
        "evaluate_structure_pris",
        lambda structure, *, src_dir: {
            "charge_assignment_route": "integer",
            "feature_error": None,
            "minimum_pair_distance_a": 1.0,
            "features": dict(SATISFYING_FEATURES),
            "wyckoff_econ_symprec_0p1": 0.5,
            **evaluated,
        },
    )

    def fake_predict(atoms_batch, **kwargs):
        volumes = np.asarray([atoms.get_volume() for atoms in atoms_batch])
        equilibrium = float(volumes[len(volumes) // 2])
        curvature = 240.0 / (EV_PER_A3_TO_GPA * equilibrium)
        energies = 0.5 * curvature * (volumes - equilibrium) ** 2
        return energies.tolist(), {"backend": "cpu-test-double"}

    monkeypatch.setattr(generated, "predict_uma_energies", fake_predict)
    captured = {}
    real_to_parquet = pd.DataFrame.to_parquet

    def capture_parquet(self, path, *args, **kwargs):
        captured.setdefault("frame", self.copy())
        return real_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        capture_parquet,
    )
    args = SimpleNamespace(
        input_archive=str(archive),
        input_manifest=str(manifest_path),
        min_unique_structures=1,
        output_dir=str(tmp_path / "out"),
        source="mattergen",
        src_dir="src",
        checkpoint="unused.pt",
        energy_batch_size=1,
        device="cpu",
    )

    summary = generated.run_evaluation(args)
    frame = captured["frame"]

    assert "clamped_bulk_modulus_proxy_gpa" in frame.columns
    assert "bulk_modulus_gpa" not in frame.columns
    assert summary["protocol_version"] == "2026-08-21-property-design-v3"
    assert summary["cohort_status"] == "diagnostic"
    predictions_path = tmp_path / "out" / "predictions.parquet"
    assert summary["outputs"]["predictions_sha256"] == hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    metric = summary["threshold_metrics"]["200"]["L4"]
    assert metric["high_property_candidate_count"] == 1
    assert "clamped_bulk_modulus_proxy_gpa" in metric["high_property_candidate_definition"]


def test_mattergen_evaluation_carries_manifest_and_candidate_provenance(
    tmp_path, monkeypatch
):
    archive = _write_distinct_archive(tmp_path, species=("C",))
    loaded = generated._load_archive(archive)
    manifest = _aggregate_manifest(archive, loaded)
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    evaluated = evaluate_rungs_from_features(SATISFYING_FEATURES)
    monkeypatch.setattr(
        generated,
        "evaluate_structure_pris",
        lambda structure, *, src_dir: {
            "charge_assignment_route": "integer",
            "charge_assignment_values": [4.0],
            "feature_error": None,
            "minimum_pair_distance_a": 1.0,
            "features": dict(SATISFYING_FEATURES),
            "wyckoff_econ_symprec_0p1": 0.5,
            **evaluated,
        },
    )

    def fake_predict(atoms_batch, **kwargs):
        volumes = np.asarray([atoms.get_volume() for atoms in atoms_batch])
        equilibrium = float(volumes[len(volumes) // 2])
        curvature = 240.0 / (EV_PER_A3_TO_GPA * equilibrium)
        energies = 0.5 * curvature * (volumes - equilibrium) ** 2
        return energies.tolist(), {"backend": "cpu-test-double"}

    monkeypatch.setattr(generated, "predict_uma_energies", fake_predict)
    captured = {}
    real_to_parquet = pd.DataFrame.to_parquet

    def capture_parquet(self, path, *args, **kwargs):
        captured.setdefault("frame", self.copy())
        return real_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        capture_parquet,
    )
    args = SimpleNamespace(
        input_archive=str(archive),
        input_manifest=str(manifest_path),
        min_unique_structures=1,
        output_dir=str(tmp_path / "out"),
        source="mattergen",
        src_dir="src",
        checkpoint="unused.pt",
        energy_batch_size=1,
        device="cpu",
    )

    summary = generated.run_evaluation(args)
    row = captured["frame"].iloc[0]

    assert row["source_shard"] == "/fixed/source/shard"
    assert row["source_member"] == "source_0.cif"
    assert row["source_manifest_sha256"] == "b" * 64
    assert row["source_seed"] == 20260821
    assert json.loads(row["charge_assignment_values_json"]) == [4.0]
    assert row["geometry_state"] == "raw_unrelaxed"
    assert row["symmetry_relaxation_applied"] == False
    assert summary["input_manifest"] == str(manifest_path.resolve())
    assert summary["input_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert summary["aggregate_protocol_version"].endswith("aggregate-v3")
    assert summary["aggregate_protocol_sha256"] == manifest["protocol_sha256"]
    assert summary["source_protocol"] == manifest["source_protocol"]
    assert summary["outputs"]["predictions_sha256"] == hashlib.sha256(
        (tmp_path / "out" / "predictions.parquet").read_bytes()
    ).hexdigest()
