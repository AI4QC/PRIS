from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.next53_odac23_train_cohort import sanitize_odac23_train_record


class _Record(SimpleNamespace):
    def keys(self):
        return list(self.__dict__)


def test_record_sanitizer_removes_adsorbates_and_computes_framework_displacement() -> None:
    record = _Record(
        fixed=torch.zeros(5),
        nco2=1,
        tags=torch.tensor([0, 0, 1, 2, 2]),
        atomic_numbers=torch.tensor([6.0, 6.0, 6.0, 8.0, 8.0]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [5.0, 5.0, 5.0], [6.1, 5.0, 5.0], [3.9, 5.0, 5.0]]
        ),
        nh2o=0,
        fid=torch.tensor([0]),
        cell=torch.tensor([np.diag([10.0, 10.0, 10.0])], dtype=torch.float32),
        defective=False,
        y_init=0.0,
        nads=1,
        y_relaxed=-1.0,
        sid=torch.tensor([1]),
        natoms=5,
        pos_relaxed=torch.tensor(
            [[0.1, 0.0, 0.0], [1.4, 0.0, 0.0], [5.2, 5.0, 5.0], [6.0, 5.0, 5.0], [3.8, 5.0, 5.0]]
        ),
        oms=False,
        name="TEST_0.00_0_w_CO2_1",
        supercell=np.asarray([1, 1, 1]),
        raw_y=-1.0,
    )

    sanitized = sanitize_odac23_train_record(record)

    assert sanitized.framework_name == "TEST_0.00_0"
    assert sanitized.atoms.get_chemical_symbols() == ["C", "C"]
    assert sanitized.atoms.info == {}
    assert set(sanitized.atoms.arrays) == {"numbers", "positions"}
    assert np.isclose(sanitized.framework_displacement_p95, 0.095, atol=1e-6)
    assert sanitized.adsorbate_atoms_removed == 3
