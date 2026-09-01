from __future__ import annotations

import ast
from pathlib import Path

from ase import Atoms
import numpy as np
import pandas as pd

from src.next553_hea_development_search import (
    apply_frozen_endpoint_definition,
    extract_authorized_endpoint_payloads,
    symmetric_pair_score,
)


def test_endpoint_extractor_materializes_only_authorized_fids(tmp_path: Path) -> None:
    header = (
        "fid,reduced_formula,chemical_system,lattice,nelements,NIONS,"
        "space_group_number,volume_per_atom,pressure,stress,e_per_atom,Ef_per_atom,"
        "e_above_hull,magmom,charge,structure_ini_as_dict,structure_as_dict,kpt\n"
    )
    rows = [
        'dev,A,A,,,,,,,,,,0.2,,,"{}","{\'final\': \'development\'}",\n',
        'val,B,B,,,,,,,,,,999,,,"{}","{\'secret\': \'validation endpoint\'}",\n',
    ]
    path = tmp_path / "tiny.csv"
    path.write_text(header + "".join(rows))

    result, audit = extract_authorized_endpoint_payloads(path, {"dev"})

    assert set(result) == {"dev"}
    assert result["dev"]["e_above_hull"] == b"0.2"
    assert ast.literal_eval(result["dev"]["structure_as_dict"].decode()) == {
        "final": "development"
    }
    assert audit["endpoint_rows_materialized"] == 1
    assert audit["unauthorized_endpoint_rows_materialized"] == 0
    assert b"validation endpoint" not in b"".join(result["dev"].values())


def test_endpoint_definition_uses_predeclared_absolute_thresholds() -> None:
    table = pd.DataFrame(
        {
            "fid": ["stable", "energy", "move", "strain", "volume"],
            "e_above_hull": [0.01, 0.10, 0.01, 0.01, 0.01],
            "disp_p90": [0.05, 0.05, 0.25, 0.05, 0.05],
            "cell_logstrain_max": [0.01, 0.01, 0.01, 0.08, 0.01],
            "volume_logchange": [0.01, 0.01, 0.01, 0.01, 0.10],
        }
    )

    result = apply_frozen_endpoint_definition(table)

    assert result["dft_waste"].tolist() == [False, True, True, True, True]
    assert result["protected"].tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(result["waste_severity"], [0.2, 1.0, 1.0, 1.0, 1.0])


def test_symmetric_pair_catalogue_is_coefficient_free() -> None:
    u = np.array([0.2, 0.8])
    v = np.array([0.5, 0.4])

    np.testing.assert_allclose(symmetric_pair_score(u, v, "mean"), [0.35, 0.6])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "maximum"), [0.5, 0.8])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "union"), [0.6, 0.88])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "minimum"), [0.2, 0.4])
