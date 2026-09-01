#!/usr/bin/env python3
"""PREREG-L4 §1: 判别矩阵(real + 再生扰动)的增强特征。

再生配方与 phys_law._bad 逐字一致:seed_of(sid)、S1–S5 固定顺序、共享 rng、
swapped_val。特征 = f3_features._feats(PREREG-F3 §4 定义)。
输出:law_real_aug.parquet(source_id 键)/ law_bad_aug.parquet((parent,kind) 键)。
"""
from __future__ import annotations
import os, sys, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f3_features import _feats  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
B = os.environ.get("PRIS_LAW_TABLES", "law_tables/")


def one_real(r):
    from pymatgen.core import Structure
    from discriminate import guess_oxi, read_blob_cif
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 80:
            return None
        val, ok = guess_oxi(st)
        if not ok:
            return None
        f = _feats(st, val)
        if f is None:
            return None
        f["source_id"] = r["sid"]
        return f
    except Exception:
        return None


def one_bad(r):
    from pymatgen.core import Structure
    from discriminate import guess_oxi, read_blob_cif
    from make_negatives import perturb, swapped_val
    from phys_law import seed_of
    out = []
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 80:
            return out
        val, ok = guess_oxi(st)
        if not ok:
            return out
        rng = np.random.default_rng(seed_of(r["sid"]))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            p = perturb(st, kind, rng, val)
            if p is None:
                continue
            pv = swapped_val(p, val)
            try:
                f = _feats(p, pv)
            except Exception:
                continue
            if f is None:
                continue
            f.update(parent=r["sid"], kind=kind)
            out.append(f)
    except Exception:
        pass
    return out


def main() -> int:
    rr = pd.read_parquet(B + "records_real.parquet")
    rb = pd.read_parquet(B + "records_bad.parquet")
    print(f"real records {len(rr)}, bad parents {len(rb)}", flush=True)

    jobs = [{"sid": t.source_id, "off": int(t.blob_offset), "ln": int(t.blob_length)}
            for t in rr.itertuples()]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, r in enumerate(ex.map(one_real, jobs, chunksize=8)):
            if r:
                rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  real {i+1}/{len(jobs)} -> {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(F + "law_real_aug.parquet", index=False)
    print(f"wrote {len(rows)} -> law_real_aug.parquet", flush=True)

    jobs = [{"sid": t.parent, "off": int(t.blob_offset), "ln": int(t.blob_length)}
            for t in rb.itertuples()]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, ch in enumerate(ex.map(one_bad, jobs, chunksize=4)):
            rows.extend(ch)
            if (i + 1) % 500 == 0:
                print(f"  bad {i+1}/{len(jobs)} -> {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(F + "law_bad_aug.parquet", index=False)
    print(f"wrote {len(rows)} -> law_bad_aug.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
