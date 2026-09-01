from __future__ import annotations

from src.next551_hea_initial_cohort import (
    _project_csv_record,
    balanced_partition_map,
    cohort_hash,
)


def test_csv_projector_copies_only_requested_escaped_fields() -> None:
    record = b'a,"b,c","d""e",secret endpoint bytes\n'
    values, present, count = _project_csv_record(
        record, copy_indices={0, 1, 2}, presence_indices={3}
    )

    assert count == 4
    assert values == {0: b"a", 1: b"b,c", 2: b'd"e'}
    assert present == {3: True}
    assert b"secret" not in b"".join(values.values())


def test_csv_projector_handles_empty_fields_without_materializing_skipped_data() -> None:
    values, present, count = _project_csv_record(
        b"fid,,3,not copied\r\n", copy_indices={0, 2}, presence_indices={1, 3}
    )

    assert count == 4
    assert values == {0: b"fid", 2: b"3"}
    assert present == {1: False, 3: True}


def test_cohort_hash_and_balanced_system_split_are_frozen() -> None:
    assert cohort_hash("abc") == cohort_hash("abc")
    assert cohort_hash("abc") != cohort_hash("abd")
    rows = [
        {"chemical_system": "A-B", "size_family": "ordered"},
        {"chemical_system": "A-B", "size_family": "ordered"},
        {"chemical_system": "C-D", "size_family": "sqs"},
        {"chemical_system": "C-D", "size_family": "sqs"},
        {"chemical_system": "E-F", "size_family": "ordered"},
        {"chemical_system": "E-F", "size_family": "sqs"},
    ]
    first = balanced_partition_map(rows)
    second = balanced_partition_map(list(reversed(rows)))
    assert first == second
    assert set(first.values()) == {"development", "validation"}
