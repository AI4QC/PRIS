from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Lattice, Structure

from src.next543_lisi_random_relaxation_initial_cohort import (
    _extract_first_structure_dict,
    select_remote_objects,
)


def test_hash_selection_is_name_only_and_prefix_balanced() -> None:
    items = []
    for prefix in ("A", "B"):
        for index in range(8):
            items.append(
                {
                    "name": f"crystal-relaxations/{prefix}/data/{index}.json",
                    "size": str(10 + index),
                    "md5Hash": f"md5-{prefix}-{index}",
                    "generation": str(100 + index),
                }
            )
        items.append(
            {
                "name": f"crystal-relaxations/{prefix}/data/empty.json",
                "size": "0",
                "md5Hash": "empty",
                "generation": "1",
            }
        )

    selected = select_remote_objects(items, prefixes=("A", "B"), per_prefix=3)

    assert len(selected) == 6
    assert {row["prefix"] for row in selected} == {"A", "B"}
    assert all(int(row["size"]) > 0 for row in selected)
    assert all(row["object_name"].endswith(".json") for row in selected)
    assert all("selection_hash" in row for row in selected)


def test_initial_extractor_decodes_only_first_balanced_structure(tmp_path: Path) -> None:
    first = Structure(
        Lattice.cubic(4.0), ["Li", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).as_dict()
    second = Structure(
        Lattice.cubic(9.0), ["Li", "Si"], [[0, 0, 0], [0.2, 0.2, 0.2]]
    ).as_dict()
    payload = {
        "forces": [[[999.123]]],
        "stress": [[[888.456]]],
        "e_fr_energy": [777.789],
        "structure": [first, second],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload))

    extracted, audit = _extract_first_structure_dict(path)

    assert Structure.from_dict(extracted) == Structure.from_dict(first)
    assert audit["structure_objects_decoded"] == 1
    assert audit["first_structure_start"] < audit["first_structure_end"]
    assert audit["file_size"] > audit["first_structure_end"]
    assert "999.123" not in json.dumps(extracted)
    assert "777.789" not in json.dumps(extracted)
