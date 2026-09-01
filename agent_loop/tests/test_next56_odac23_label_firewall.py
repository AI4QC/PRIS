from __future__ import annotations

import pandas as pd

from src.next56_odac23_label_firewall import partition_label_rows


def test_partition_label_rows_is_exact_and_disjoint() -> None:
    metadata = pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "partition_role": [
                "discovery",
                "internal_validation",
                "internal_replication",
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "material_id": ["c", "a", "b"],
            "framework_displacement_p95_median": [0.3, 0.1, 0.2],
        }
    )

    parts = partition_label_rows(metadata, labels)

    assert set(parts) == {
        "discovery",
        "internal_validation",
        "internal_replication",
    }
    assert parts["discovery"]["material_id"].tolist() == ["a"]
    assert parts["internal_validation"]["material_id"].tolist() == ["b"]
    assert parts["internal_replication"]["material_id"].tolist() == ["c"]
    assert sum(len(part) for part in parts.values()) == len(labels)
