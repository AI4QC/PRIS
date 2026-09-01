from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pymatgen.core import Lattice, Structure

from experiments.property_design_20260821 import llm_feasibility


admission_summary = llm_feasibility.admission_summary
parse_llm_response = llm_feasibility.parse_llm_response
build_parser = getattr(llm_feasibility, "build_parser", None)
parse_llm_response_detailed = getattr(
    llm_feasibility, "parse_llm_response_detailed", None
)
unique_structure_count = getattr(llm_feasibility, "unique_structure_count", None)
validate_basic_structure = getattr(llm_feasibility, "validate_basic_structure", None)


SILICON_POSCAR = """Si2
1.0
0.000000 2.715000 2.715000
2.715000 0.000000 2.715000
2.715000 2.715000 0.000000
Si
2
Direct
0.000000 0.000000 0.000000
0.250000 0.250000 0.250000
"""

MODEL_SHOW = {
    "capabilities": ["completion", "tools"],
    "details": {
        "parent_model": "/var/lib/ollama/blobs/sha256-" + "1" * 64,
        "format": "gguf",
        "family": "qwen35moe",
        "families": ["qwen35moe"],
        "parameter_size": "35.1B",
        "quantization_level": "Q4_K_M",
    },
    "model_info": {"general.architecture": "qwen35moe"},
}


def _run_admitted_pilot_gate(tmp_path, monkeypatch):
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

    pairs = [
        ("Si", "Si"),
        ("Ge", "Ge"),
        ("C", "C"),
        ("B", "N"),
        ("Al", "N"),
        ("Ga", "N"),
        ("B", "P"),
        ("Si", "C"),
        ("Be", "O"),
        ("Mg", "O"),
    ]
    responses = []
    for species in pairs:
        radius_sum = sum(float(CovalentRadius.radius[value]) for value in species)
        lattice_a = 2.0 * radius_sum / math.sqrt(3.0)
        structure = Structure(
            Lattice.cubic(lattice_a),
            list(species),
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        responses.append(
            json.dumps(
                {
                    "0": {
                        "formula": str(structure.composition.reduced_formula),
                        "poscar": structure.to(fmt="poscar"),
                    }
                }
            )
        )
    iterator = iter(responses)
    monkeypatch.setattr(
        llm_feasibility, "_ollama_show", lambda **_kwargs: MODEL_SHOW
    )
    monkeypatch.setattr(
        llm_feasibility,
        "_ollama_generate",
        lambda **_kwargs: {"response": next(iterator)},
    )
    output_dir = tmp_path / "pilot-10"
    payload = llm_feasibility.run_feasibility(
        SimpleNamespace(
            output_dir=str(output_dir),
            attempts=10,
            seed=1,
            target_bulk_modulus_gpa=400.0,
            endpoint="http://127.0.0.1:11434",
            model="test-model",
            timeout_seconds=1,
            prior_gate=None,
        )
    )
    assert payload["admission"]["admitted"] is True
    return output_dir / "feasibility.json"


def test_parse_llm_response_accepts_clean_json_poscar():
    response = json.dumps({"0": {"formula": "Si", "poscar": SILICON_POSCAR}})

    parsed = parse_llm_response(response, fmt="poscar")

    assert len(parsed) == 1
    assert parsed[0]["formula_reported"] == "Si"
    assert parsed[0]["formula_parsed"] == "Si"
    assert parsed[0]["structure"].num_sites == 2


def test_ollama_thinking_channel_is_used_when_response_is_blank():
    thinking = json.dumps({"0": {"formula": "Si", "poscar": SILICON_POSCAR}})

    text, field, metadata = llm_feasibility._select_ollama_generation_text(
        {"response": "", "thinking": thinking}
    )

    assert text == thinking
    assert field == "thinking"
    assert metadata["response"]["length"] == 0
    assert metadata["thinking"]["length"] == len(thinking)
    assert len(metadata["thinking"]["sha256"]) == 64


def test_ollama_response_channel_has_priority_over_thinking():
    text, field, _ = llm_feasibility._select_ollama_generation_text(
        {"response": "final", "thinking": "draft"}
    )

    assert (text, field) == ("final", "response")


def test_parse_llm_response_strips_markdown_fence():
    response = "```json\n" + json.dumps(
        {"0": {"formula": "Si", "poscar": SILICON_POSCAR}}
    ) + "\n```"

    parsed = parse_llm_response(response, fmt="poscar")

    assert len(parsed) == 1


def test_parse_llm_response_rejects_multiple_candidates_from_one_attempt():
    response = json.dumps(
        {
            "0": {"formula": "Si", "poscar": SILICON_POSCAR},
            "1": {"formula": "Si", "poscar": SILICON_POSCAR},
        }
    )

    parsed = parse_llm_response(response, fmt="poscar")

    assert parsed == []
    assert parse_llm_response_detailed is not None
    detailed = parse_llm_response_detailed(response, fmt="poscar")
    assert detailed["failure_reasons"] == ["candidate_count_not_one:2"]


def test_parse_llm_response_returns_empty_for_malformed_or_nonperiodic_content():
    malformed = '{"0": {"formula": "Si", "poscar": "not a POSCAR"}}'

    assert parse_llm_response(malformed, fmt="poscar") == []
    assert parse_llm_response("explanation only", fmt="poscar") == []


def test_parseable_candidate_carries_basic_validation_reasons():
    collapsed = """Si2
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Si
2
Direct
0.0 0.0 0.0
0.05 0.0 0.0
"""
    response = json.dumps({"0": {"formula": "Si", "poscar": collapsed}})

    parsed = parse_llm_response(response)

    assert len(parsed) == 1
    assert parsed[0]["basic_valid"] is False
    assert "contact_below_0.6_radii_sum" in parsed[0]["basic_failure_reasons"]


@pytest.mark.parametrize(
    ("reported_formula", "structure", "expected_reason"),
    [
        (
            "Ge",
            Structure.from_str(SILICON_POSCAR, fmt="poscar"),
            "reported_formula_mismatch",
        ),
        (
            "Ar",
            Structure(Lattice.cubic(3.0), ["Ar"], [[0, 0, 0]]),
            "forbidden_element:Ar",
        ),
        (
            "Si21",
            Structure(
                Lattice.cubic(10.0),
                ["Si"] * 21,
                [[i / 21.0, 0.5, 0.5] for i in range(21)],
            ),
            "site_count_out_of_range:21",
        ),
        (
            "Si",
            Structure(
                Lattice([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 1e-10]]),
                ["Si"],
                [[0, 0, 0]],
            ),
            "singular_cell",
        ),
        (
            "Si",
            Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]]),
            "volume_per_atom_out_of_range",
        ),
        (
            "Si",
            Structure(
                Lattice.cubic(3.0, pbc=(True, True, False)),
                ["Si"],
                [[0, 0, 0]],
            ),
            "not_three_dimensional_periodic",
        ),
    ],
)
def test_basic_structure_validation_reports_each_contract_failure(
    reported_formula, structure, expected_reason
):
    assert validate_basic_structure is not None
    result = validate_basic_structure(structure, reported_formula=reported_formula)

    assert result["valid"] is False
    assert expected_reason in result["failure_reasons"]


def test_basic_structure_validation_accepts_dense_bonded_silicon():
    structure = Structure.from_str(SILICON_POSCAR, fmt="poscar")

    assert validate_basic_structure is not None
    result = validate_basic_structure(structure, reported_formula="Si")

    assert result == {"valid": True, "failure_reasons": []}


def test_forbidden_noble_gas_formula_check_does_not_require_electronegativity():
    structure = Structure(Lattice.cubic(3.0), ["Ar"], [[0, 0, 0]])

    assert validate_basic_structure is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validate_basic_structure(structure, reported_formula="Ar")

    assert "forbidden_element:Ar" in result["failure_reasons"]
    assert "reported_formula_invalid" not in result["failure_reasons"]


def test_forbidden_noble_gas_response_remains_parseable_with_itemized_reason():
    argon_poscar = """Ar
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Ar
1
Direct
0.0 0.0 0.0
"""
    response = json.dumps({"0": {"formula": "Ar", "poscar": argon_poscar}})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = parse_llm_response(response)

    assert len(parsed) == 1
    assert parsed[0]["basic_valid"] is False
    assert "forbidden_element:Ar" in parsed[0]["basic_failure_reasons"]


def test_structure_matcher_counts_reordered_and_translated_copies_once():
    original = Structure.from_str(SILICON_POSCAR, fmt="poscar")
    equivalent = original.copy()
    equivalent.translate_sites(range(len(equivalent)), [0.5, 0.5, 0.5])
    equivalent = Structure.from_sites(list(reversed(equivalent.sites)))
    distinct_composition = Structure(
        original.lattice,
        ["Ge", "Ge"],
        original.frac_coords,
    )

    assert unique_structure_count is not None
    assert unique_structure_count([original, equivalent, distinct_composition]) == 2


def _attempt(*, parsed: int, basic: int, wall_seconds: float) -> dict:
    return {
        "parsed_count": parsed,
        "basic_valid_count": basic,
        "wall_seconds": wall_seconds,
    }


def test_admission_requires_eight_parseable_basic_valid_unique_structures():
    attempts = [
        *[_attempt(parsed=1, basic=1, wall_seconds=50.0) for _ in range(8)],
        *[_attempt(parsed=0, basic=0, wall_seconds=50.0) for _ in range(2)],
    ]

    admitted = admission_summary(attempts, unique_basic_valid_count=8)
    too_few_parseable = admission_summary(
        [
            *[_attempt(parsed=1, basic=1, wall_seconds=50.0) for _ in range(7)],
            *[_attempt(parsed=0, basic=0, wall_seconds=50.0) for _ in range(3)],
        ],
        unique_basic_valid_count=7,
    )
    too_few_basic = admission_summary(
        [
            *[_attempt(parsed=1, basic=int(index < 7), wall_seconds=50.0) for index in range(8)],
            *[_attempt(parsed=0, basic=0, wall_seconds=50.0) for _ in range(2)],
        ],
        unique_basic_valid_count=7,
    )
    too_few_unique = admission_summary(attempts, unique_basic_valid_count=7)

    assert admitted["admitted"] is True
    assert admitted["parseable_attempts"] == 8
    assert admitted["basic_valid_structures"] == 8
    assert admitted["unique_basic_valid_structures"] == 8
    assert too_few_parseable["admitted"] is False
    assert too_few_basic["admitted"] is False
    assert too_few_unique["admitted"] is False


def test_admission_records_throughput_and_guarded_1024_projection():
    attempts = [_attempt(parsed=1, basic=1, wall_seconds=96.0) for _ in range(10)]

    summary = admission_summary(attempts, unique_basic_valid_count=8)

    assert summary["total_wall_seconds"] == pytest.approx(960.0)
    assert summary["seconds_per_parseable_structure"] == pytest.approx(96.0)
    assert summary["seconds_per_unique_basic_valid_structure"] == pytest.approx(120.0)
    assert summary["projected_1024_unique_wall_hours"] == pytest.approx(
        1.25 * 1024 * 120.0 / 3600.0
    )
    assert summary["admitted"] is True


def test_admission_projection_includes_matching_and_archive_overhead():
    attempts = [_attempt(parsed=1, basic=1, wall_seconds=80.0) for _ in range(10)]

    summary = admission_summary(
        attempts,
        unique_basic_valid_count=8,
        pipeline_wall_seconds=960.0,
    )

    assert summary["attempt_generation_wall_seconds"] == pytest.approx(800.0)
    assert summary["total_wall_seconds"] == pytest.approx(960.0)
    assert summary["post_generation_wall_seconds"] == pytest.approx(160.0)
    assert summary["seconds_per_unique_basic_valid_structure"] == pytest.approx(
        120.0
    )


def test_admission_projection_scales_pairwise_matching_quadratically():
    attempts = [_attempt(parsed=1, basic=1, wall_seconds=80.0) for _ in range(10)]

    summary = admission_summary(
        attempts,
        unique_basic_valid_count=8,
        pipeline_wall_seconds=864.0,
        matching_wall_seconds=56.0,
        archive_wall_seconds=8.0,
    )

    matching_projection = summary["projection_components_seconds"][
        "structure_matching"
    ]
    assert matching_projection == pytest.approx(
        56.0 * (1024 * 1023) / (8 * 7)
    )
    assert summary["projected_1024_unique_wall_hours"] > 48.0
    assert summary["admitted"] is False


def test_admission_fails_when_average_unique_candidate_cost_exceeds_120_seconds():
    attempts = [_attempt(parsed=1, basic=1, wall_seconds=96.1) for _ in range(10)]

    summary = admission_summary(attempts, unique_basic_valid_count=8)

    assert summary["seconds_per_unique_basic_valid_structure"] > 120.0
    assert summary["admitted"] is False


def test_admission_requires_explicit_structure_matcher_unique_count():
    attempts = [_attempt(parsed=1, basic=1, wall_seconds=1.0) for _ in range(10)]

    with pytest.raises(ValueError, match="explicit StructureMatcher unique count"):
        admission_summary(attempts)


def test_admission_uses_frozen_intermediate_100_profile_at_exact_boundaries():
    attempts = [
        *[_attempt(parsed=1, basic=1, wall_seconds=96.0) for _ in range(80)],
        *[_attempt(parsed=0, basic=0, wall_seconds=96.0) for _ in range(20)],
    ]

    summary = admission_summary(
        attempts,
        unique_basic_valid_count=80,
        profile_attempts=100,
    )

    assert summary["admission_profile"] == "intermediate-100-v1"
    assert summary["required_attempts"] == 100
    assert summary["required_parseable_attempts"] == 80
    assert summary["required_basic_valid_structures"] == 80
    assert summary["required_unique_basic_valid_structures"] == 80
    assert summary["maximum_seconds_per_structure"] == 120.0
    assert summary["maximum_projected_1024_unique_wall_hours"] == 48.0
    assert summary["projection_safety_factor"] == 1.25
    assert summary["seconds_per_parseable_structure"] == pytest.approx(120.0)
    assert summary["seconds_per_unique_basic_valid_structure"] == pytest.approx(
        120.0
    )
    assert summary["projected_1024_unique_wall_hours"] == pytest.approx(
        1.25 * 1024 * 120.0 / 3600.0
    )
    assert summary["admitted"] is True


@pytest.mark.parametrize(
    ("attempt_count", "parseable", "basic_valid", "unique", "failed_criterion"),
    [
        (99, 80, 80, 80, "attempts_completed"),
        (101, 80, 80, 80, "attempts_completed"),
        (100, 79, 79, 79, "parseable_structures"),
        (100, 80, 79, 79, "basic_valid_structures"),
        (100, 80, 80, 79, "unique_basic_valid_structures"),
    ],
)
def test_intermediate_100_profile_enforces_exact_attempt_and_count_boundaries(
    attempt_count,
    parseable,
    basic_valid,
    unique,
    failed_criterion,
):
    attempts = [
        _attempt(
            parsed=int(index < parseable),
            basic=int(index < basic_valid),
            wall_seconds=1.0,
        )
        for index in range(attempt_count)
    ]

    summary = admission_summary(
        attempts,
        unique_basic_valid_count=unique,
        profile_attempts=100,
    )

    assert summary["criteria_met"][failed_criterion] is False
    assert summary["admitted"] is False


def test_intermediate_100_profile_keeps_120_second_and_48_hour_limits():
    attempts = [
        *[_attempt(parsed=1, basic=1, wall_seconds=108.8) for _ in range(80)],
        *[_attempt(parsed=0, basic=0, wall_seconds=108.8) for _ in range(20)],
    ]

    summary = admission_summary(
        attempts,
        unique_basic_valid_count=80,
        profile_attempts=100,
    )

    assert summary["seconds_per_parseable_structure"] == pytest.approx(136.0)
    assert summary["seconds_per_unique_basic_valid_structure"] == pytest.approx(
        136.0
    )
    assert summary["projected_1024_unique_wall_hours"] > 48.0
    assert summary["criteria_met"]["seconds_per_parseable_structure"] is False
    assert (
        summary["criteria_met"]["seconds_per_unique_basic_valid_structure"]
        is False
    )
    assert summary["criteria_met"]["projected_1024_unique_wall_hours"] is False
    assert summary["admitted"] is False


@pytest.mark.parametrize("populate", [False, True])
def test_run_feasibility_refuses_any_existing_output_directory(
    tmp_path, monkeypatch, populate
):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    if populate:
        (output_dir / "old-candidate.cif").write_text("stale\n")

    def forbidden_generate(**_kwargs):
        raise AssertionError("generation must not start for an existing output")

    monkeypatch.setattr(llm_feasibility, "_ollama_generate", forbidden_generate)
    args = SimpleNamespace(
        output_dir=str(output_dir),
        attempts=1,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="offline",
        model="test-model",
        timeout_seconds=1,
    )

    with pytest.raises(FileExistsError):
        llm_feasibility.run_feasibility(args)


def test_candidate_archive_contains_only_basic_valid_structurematcher_unique_candidates(
    tmp_path, monkeypatch
):
    translated = Structure.from_str(SILICON_POSCAR, fmt="poscar")
    translated.translate_sites(range(len(translated)), [0.5, 0.5, 0.5])
    collapsed = """Si2
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Si
2
Direct
0.0 0.0 0.0
0.05 0.0 0.0
"""
    responses = iter(
        [
            SILICON_POSCAR,
            translated.to(fmt="poscar"),
            collapsed,
            *["not a POSCAR" for _ in range(7)],
        ]
    )

    def fake_generate(**_kwargs):
        poscar = next(responses)
        return {
            "response": json.dumps(
                {"0": {"formula": "Si", "poscar": poscar}}
            )
        }

    monkeypatch.setattr(llm_feasibility, "_ollama_generate", fake_generate)
    monkeypatch.setattr(
        llm_feasibility, "_ollama_show", lambda **_kwargs: MODEL_SHOW
    )
    output_dir = tmp_path / "fresh"
    args = SimpleNamespace(
        output_dir=str(output_dir),
        attempts=10,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="http://127.0.0.1:11434",
        model="test-model",
        timeout_seconds=1,
        prior_gate=None,
    )

    payload = llm_feasibility.run_feasibility(args)

    records = [record for attempt in payload["attempts"] for record in attempt["parsed"]]
    assert [record["basic_valid"] for record in records] == [True, True, False]
    assert records[0]["structure_sha256"] != records[1]["structure_sha256"]
    assert [record["included_in_candidate_archive"] for record in records] == [
        True,
        False,
        False,
    ]
    assert records[0]["candidate_archive_exclusion_reason"] is None
    assert records[1]["candidate_archive_exclusion_reason"] == (
        "structure_matcher_duplicate"
    )
    assert records[2]["candidate_archive_exclusion_reason"] == (
        "basic_structure_gate_failed"
    )
    assert payload["admission"]["unique_basic_valid_structures"] == 1
    assert payload["admission"]["admitted"] is False
    assert payload["artifacts"]["archive_status"] == "diagnostic_only"
    assert not any("formal" in key for key in payload["artifacts"])
    with ZipFile(output_dir / "unique_basic_valid_candidates_cif.zip") as archive:
        assert archive.namelist() == [records[0]["candidate_archive_member"]]


def test_manifest_records_frozen_intermediate_profile_and_thresholds(
    tmp_path, monkeypatch
):
    def fake_generate(**_kwargs):
        return {"response": "not JSON"}

    monkeypatch.setattr(
        llm_feasibility, "_ollama_show", lambda **_kwargs: MODEL_SHOW
    )
    output_dir = tmp_path / "intermediate-100"
    prior_gate = _run_admitted_pilot_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_feasibility, "_ollama_generate", fake_generate)
    args = SimpleNamespace(
        output_dir=str(output_dir),
        attempts=100,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="http://127.0.0.1:11434",
        model="test-model",
        timeout_seconds=1,
        prior_gate=str(prior_gate),
    )

    payload = llm_feasibility.run_feasibility(args)

    assert payload["protocol"]["admission_profile"] == "intermediate-100-v1"
    assert payload["protocol"]["admission_thresholds"] == {
        "required_attempts": 100,
        "required_parseable_attempts": 80,
        "required_basic_valid_structures": 80,
        "required_unique_basic_valid_structures": 80,
        "maximum_seconds_per_structure": 120.0,
        "maximum_projected_1024_unique_wall_hours": 48.0,
        "projection_safety_factor": 1.25,
    }
    assert payload["admission"]["admission_profile"] == "intermediate-100-v1"
    assert payload["admission"]["required_attempts"] == 100
    assert payload["admission"]["required_parseable_attempts"] == 80
    assert payload["admission"]["required_basic_valid_structures"] == 80
    assert payload["admission"]["required_unique_basic_valid_structures"] == 80


def test_cli_rejects_non_400_gpa_target_before_generation(tmp_path, monkeypatch):
    called = False

    def forbidden_run(_args):
        nonlocal called
        called = True
        raise AssertionError("non-frozen target must not reach generation")

    monkeypatch.setattr(llm_feasibility, "run_feasibility", forbidden_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llm_feasibility.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "test-model",
            "--target-bulk-modulus-gpa",
            "399.0",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        llm_feasibility.main()

    assert exc_info.value.code == 2
    assert called is False


def test_cli_defaults_bulk_modulus_target_to_400_gpa():
    assert build_parser is not None
    args = build_parser().parse_args(["--output-dir", "out", "--model", "model"])

    assert args.target_bulk_modulus_gpa == 400.0


def test_cli_accepts_frozen_100_attempt_profile_only_with_prior_gate(
    tmp_path, monkeypatch
):
    captured_attempts = None

    def fake_run(args):
        nonlocal captured_attempts
        captured_attempts = args.attempts
        return {"admission": {"admitted": True}}

    monkeypatch.setattr(llm_feasibility, "run_feasibility", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llm_feasibility.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "test-model",
            "--attempts",
            "100",
            "--prior-gate",
            str(tmp_path / "pilot-feasibility.json"),
        ],
    )

    assert llm_feasibility.main() == 0
    assert captured_attempts == 100


def test_cli_rejects_100_attempt_profile_without_prior_gate(
    tmp_path, monkeypatch
):
    called = False

    def forbidden_run(_args):
        nonlocal called
        called = True
        raise AssertionError("100-attempt generation must not start")

    monkeypatch.setattr(llm_feasibility, "run_feasibility", forbidden_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llm_feasibility.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "test-model",
            "--attempts",
            "100",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        llm_feasibility.main()

    assert exc_info.value.code == 2
    assert called is False


def test_run_feasibility_rejects_100_attempts_before_output_without_admitted_pilot(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "out"

    def forbidden_generate(**_kwargs):
        raise AssertionError("generation must not start")

    monkeypatch.setattr(llm_feasibility, "_ollama_generate", forbidden_generate)
    args = SimpleNamespace(
        output_dir=str(output_dir),
        attempts=100,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="http://127.0.0.1:11434",
        model="test-model",
        timeout_seconds=1,
        prior_gate=None,
    )

    with pytest.raises(ValueError, match="admitted 10-attempt prior gate"):
        llm_feasibility.run_feasibility(args)

    assert not output_dir.exists()


def test_model_identity_requires_completion_llm_architecture():
    identity = llm_feasibility.model_identity_from_show(MODEL_SHOW)

    assert identity["architecture"] == "qwen35moe"
    assert identity["model_blob_sha256"] == "1" * 64
    assert len(identity["descriptor_sha256"]) == 64
    with pytest.raises(ValueError, match="text-LLM architecture"):
        llm_feasibility.model_identity_from_show(
            {
                "capabilities": ["completion"],
                "details": {"format": "gguf", "family": "mattergen"},
                "model_info": {"general.architecture": "mattergen_diffusion"},
            }
        )


def test_run_feasibility_fails_if_model_blob_changes_during_generation(
    tmp_path, monkeypatch
):
    changed = json.loads(json.dumps(MODEL_SHOW))
    changed["details"]["parent_model"] = (
        "/var/lib/ollama/blobs/sha256-" + "2" * 64
    )
    descriptors = iter((MODEL_SHOW, changed))
    monkeypatch.setattr(
        llm_feasibility,
        "_ollama_show",
        lambda **_kwargs: next(descriptors),
    )
    monkeypatch.setattr(
        llm_feasibility,
        "_ollama_generate",
        lambda **_kwargs: {"response": "not JSON"},
    )
    output_dir = tmp_path / "changed-model"
    args = SimpleNamespace(
        output_dir=str(output_dir),
        attempts=10,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="http://127.0.0.1:11434",
        model="test-model",
        timeout_seconds=1,
        prior_gate=None,
    )

    with pytest.raises(RuntimeError, match="identity changed"):
        llm_feasibility.run_feasibility(args)

    assert not (output_dir / "feasibility.json").exists()


def test_run_feasibility_rejects_nonfrozen_target_and_endpoint_before_network(
    tmp_path, monkeypatch
):
    def forbidden_show(**_kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(llm_feasibility, "_ollama_show", forbidden_show)
    base = dict(
        attempts=10,
        seed=1,
        model="test-model",
        timeout_seconds=1,
        prior_gate=None,
    )
    with pytest.raises(ValueError, match="exactly 400 GPa"):
        llm_feasibility.run_feasibility(
            SimpleNamespace(
                **base,
                output_dir=str(tmp_path / "bad-target"),
                endpoint="http://127.0.0.1:11434",
                target_bulk_modulus_gpa=399.0,
            )
        )
    with pytest.raises(ValueError, match="frozen local Ollama endpoint"):
        llm_feasibility.run_feasibility(
            SimpleNamespace(
                **base,
                output_dir=str(tmp_path / "bad-endpoint"),
                endpoint="http://example.invalid:11434",
                target_bulk_modulus_gpa=400.0,
            )
        )


def test_forged_minimal_pilot_cannot_unlock_100_attempts(tmp_path, monkeypatch):
    prior = tmp_path / "forged.json"
    prior.write_text(
        json.dumps(
            {
                "protocol": {
                    "admission_profile": "pilot-10-v1",
                    "attempts": 10,
                    "model": "test-model",
                    "endpoint": "http://127.0.0.1:11434",
                    "target_bulk_modulus_gpa": 400.0,
                    "model_identity": llm_feasibility.model_identity_from_show(
                        MODEL_SHOW
                    ),
                },
                "admission": {"admitted": True},
            }
        )
    )
    monkeypatch.setattr(
        llm_feasibility, "_ollama_show", lambda **_kwargs: MODEL_SHOW
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        attempts=100,
        seed=1,
        target_bulk_modulus_gpa=400.0,
        endpoint="http://127.0.0.1:11434",
        model="test-model",
        timeout_seconds=1,
        prior_gate=str(prior),
    )

    with pytest.raises(ValueError, match="prior gate"):
        llm_feasibility.run_feasibility(args)

    assert not Path(args.output_dir).exists()


def test_cli_rejects_attempt_counts_outside_frozen_profiles():
    assert build_parser is not None

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(
            [
                "--output-dir",
                "out",
                "--model",
                "model",
                "--attempts",
                "50",
            ]
        )

    assert exc_info.value.code == 2
