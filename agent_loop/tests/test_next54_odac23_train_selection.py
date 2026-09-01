from __future__ import annotations

import pandas as pd

from src.next54_odac23_train_selection import select_representatives


def test_selection_is_label_independent_and_framework_isolated() -> None:
    metadata = pd.DataFrame(
        {
            "material_id": ["a1", "a2", "b1", "c1"],
            "framework_name": ["A", "A", "B", "C"],
            "geometry_sha256": ["f" * 64, "0" * 64, "1" * 64, "2" * 64],
            "natoms": [2, 2, 3, 4],
        }
    )

    selected = select_representatives(metadata)

    assert len(selected) == 3
    assert set(selected["framework_name"]) == {"A", "B", "C"}
    assert set(selected["partition_role"]) <= {
        "discovery",
        "internal_validation",
        "internal_replication",
    }
    assert not selected["framework_name"].duplicated().any()
    assert "label" not in " ".join(selected.columns).lower()
    assert selected.set_index("framework_name").at["A", "material_id"] in {"a1", "a2"}
