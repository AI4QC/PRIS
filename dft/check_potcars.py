#!/usr/bin/env python3
"""Confirm the cluster's PAW library is the one the packages were built against.

Standard library only, so it runs on a login node without numpy or pymatgen. `verify.py`
does the full pre-submission check but needs a scientific Python stack; this covers the one
thing that must be re-checked wherever the jobs will actually run: a differing potential
changes the physics silently, and a hash mismatch is the only way to see it.

    python3 check_potcars.py [--lib $VASP_PP_PATH] [--root .]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--lib", default=os.environ.get("VASP_PP_PATH", ""))
    a = ap.parse_args()
    if not a.lib:
        raise SystemExit("set --lib or VASP_PP_PATH to the PAW library")
    lib = Path(a.lib)

    wanted: dict[str, dict] = {}
    n_specs = 0
    for spec in sorted(Path(a.root).rglob("POTCAR.spec.json")):
        n_specs += 1
        for e in json.loads(spec.read_text())["order"]:
            wanted.setdefault(e["library_relative"], e)

    missing, mismatch, ok = [], [], 0
    for rel, e in sorted(wanted.items()):
        p = lib / rel
        if not p.is_file():
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != e["sha256"]:
            mismatch.append(rel)
        else:
            ok += 1

    print(f"library: {lib}")
    print(f"specs read: {n_specs}, distinct potentials required: {len(wanted)}")
    print(f"matching: {ok}, missing: {len(missing)}, differing: {len(mismatch)}")
    for rel in missing[:20]:
        print(f"  MISSING  {rel}")
    for rel in mismatch[:20]:
        print(f"  DIFFERS  {rel}  (the physics would not match the local build)")
    return 1 if (missing or mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())
