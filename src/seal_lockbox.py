#!/usr/bin/env python3
"""Split the data three ways and seal the lockbox. This is the anchor of the whole project's
credibility.

Why it is needed: searching for rules over 38,307 structures will inevitably turn up spurious
ones. The only defence is "the search never sees part of the data, and the final rules are
evaluated on that part once".

The three-way split (fixed by the sha256 of the sid, independent of order and of any file,
and reproducible):
  discovery   60%  -- search, tune, plot; look at it however you like
  calibration 25%  -- set thresholds, choose N, select models; may be inspected, but never
                      used to search for rules
  lockbox     15%  -- sealed, and may be opened at most 3 times in total

**Why it is not encrypted** (a design correction in v1.1): the first version encrypted the
lockbox sid list symmetrically with gpg, which was overkill and self-contradictory --
`splits.parquet` carries the full three-way table in the clear, so the encryption added
nothing. More fundamentally, `split_of()` is a pure function and the seed is written in the
seal in plain text, so anyone can recompute the whole assignment in one line of code.

What has to be guarded against here is **oneself** leaking information unnoticed while
repeatedly looking at results and retuning; it is not about outsiders. Guarding against
oneself relies on an auditable process, not on cryptography. So three things that do actually
work are kept:

  1. The seal LOCKBOX.sealed.json -- seed / timestamp / git commit / per-partition counts /
     sha256 of the sid list. It proves the split was fixed at this moment against this code,
     and cannot be changed afterwards.
  2. The audit log openings.log -- every touch of the lockbox must record a reason, and the
     paper reports the actual number of openings.
  3. The git tag on PREREG.md -- the criteria are frozen before the data are seen, and the
     tag's commit hash goes into the paper's Methods section.

Usage (no passphrase or interactive input at any point):
    python seal_lockbox.py --seal --seed 20260728        # seal for the first time
    python seal_lockbox.py --open --reason "MPU-6 final evaluation for the main paper"   # open (counts against the quota)
    python seal_lockbox.py --status                       # inspect the seal and the opening record
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FEATURES = Path(os.environ.get("PRIS_FEATURES", "features/"))
LOCKDIR = Path(os.environ.get("PRIS_LOCKBOX", "lockbox/"))
SEAL = LOCKDIR / "LOCKBOX.sealed.json"
AUDIT = LOCKDIR / "openings.log"
ENC = LOCKDIR / "lockbox_sids.txt"
MAX_OPENINGS = 3

# Partition fractions. Changing these three numbers means a different experiment, so they go
# into the seal rather than onto the command line.
FRAC = {"discovery": 0.60, "calibration": 0.25, "lockbox": 0.15}


def split_of(sid: str, seed: str) -> str:
    """sid -> partition. A pure function: the same (sid, seed) always gives the same
    partition, whatever the row order of the data files and however many times it is rerun."""
    h = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
    # take the first 8 bytes as an unsigned integer and map it into [0,1)
    x = int.from_bytes(h[:8], "big") / 2**64
    if x < FRAC["discovery"]:
        return "discovery"
    if x < FRAC["discovery"] + FRAC["calibration"]:
        return "calibration"
    return "lockbox"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "NO_GIT_COMMIT"


def load_sids() -> list[str]:
    """The sids of the analysis set. Uses provenance's source_id (icsd-N / cod-N), which is
    consistent across scripts and does not depend on sqlite's pk (which changes on a rebuild).
    """
    p = FEATURES / "provenance.parquet"
    if not p.exists():
        sys.exit(f"{p} is missing; run make week1 first")
    df = pd.read_parquet(p, columns=["source_id", "in_analysis_set"])
    sids = sorted(df.loc[df.in_analysis_set, "source_id"].astype(str).unique())
    if not sids:
        sys.exit("the analysis set is empty; nothing sealed")
    return sids


def do_seal(seed: str, force: bool) -> None:
    if SEAL.exists() and not force:
        sys.exit(f"a seal already exists: {SEAL}\n"
                 "Resealing invalidates every existing conclusion. Add --force to confirm.")
    LOCKDIR.mkdir(parents=True, exist_ok=True)
    sids = load_sids()
    assign = {s: split_of(s, seed) for s in sids}
    counts = {k: sum(1 for v in assign.values() if v == k) for k in FRAC}

    lock_sids = sorted(s for s, v in assign.items() if v == "lockbox")
    blob = "\n".join(lock_sids).encode()
    digest = hashlib.sha256(blob).hexdigest()

    # the sid list is stored in the clear. Encryption is pointless: the seed is in the seal and
    # split_of is a pure function, so it can be recomputed in one line. What actually works is
    # the seal and the audit log below.
    ENC.write_bytes(blob)

    # the full three-way table; downstream code filters on the split column.
    # Note that it includes the lockbox rows, deliberately -- hiding them would not prevent
    # recomputation, it would only force downstream code into detours and make errors likelier.
    pd.DataFrame({"source_id": list(assign), "split": list(assign.values())}) \
        .to_parquet(FEATURES / "splits.parquet", index=False)

    seal = {
        "seed": seed,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "n_total": len(sids),
        "counts": counts,
        "fractions": FRAC,
        "lockbox_sids_sha256": digest,
        "split_fn": "sha256(f'{seed}:{sid}')[:8] as uint64 / 2**64",
        "max_openings": MAX_OPENINGS,
        "sid_list": str(ENC),
    }
    SEAL.write_text(json.dumps(seal, indent=2, ensure_ascii=False))
    AUDIT.touch()
    print(json.dumps(seal, indent=2, ensure_ascii=False))
    print(f"\nseal written to {SEAL}")
    print("Next: commit PREREG.md and apply a git tag; the tag's commit hash goes into the "
          "paper's Methods section.")


def do_open(reason: str) -> None:
    if not SEAL.exists():
        sys.exit("nothing has been sealed, so there is nothing to open")
    seal = json.loads(SEAL.read_text())
    used = [l for l in AUDIT.read_text().splitlines() if l.strip()] if AUDIT.exists() else []
    if len(used) >= seal["max_openings"]:
        sys.exit(f"the opening quota is exhausted ({len(used)}/{seal['max_openings']}). "
                 f"The paper must report the actual number of openings; do not open again.")
    if not reason or len(reason) < 10:
        sys.exit("a substantive reason for opening is required (>=10 characters); "
                 "it goes into the audit log and the paper")

    if not ENC.exists():
        sys.exit(f"{ENC} is missing")
    blob = ENC.read_bytes()
    if hashlib.sha256(blob).hexdigest() != seal["lockbox_sids_sha256"]:
        sys.exit("!! the recovered sid list does not match the seal hash; the data has been "
                 "altered. Stopping.")

    rec = {"opened_at_utc": datetime.now(timezone.utc).isoformat(),
           "reason": reason, "git_commit": git_commit(),
           "opening_index": len(used) + 1}
    with AUDIT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = LOCKDIR / f"lockbox_sids_opening{rec['opening_index']}.txt"
    out.write_bytes(blob)
    print(f"opening {rec['opening_index']}/{seal['max_openings']}; wrote {out}")
    print(f"remaining quota: {seal['max_openings'] - rec['opening_index']}")


def do_status() -> None:
    if not SEAL.exists():
        print("not sealed yet")
        return
    print(json.dumps(json.loads(SEAL.read_text()), indent=2, ensure_ascii=False))
    if AUDIT.exists() and AUDIT.read_text().strip():
        print("\nopening record:")
        for l in AUDIT.read_text().splitlines():
            if l.strip():
                print("  " + l)
    else:
        print("\nopening record: none (never opened)")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--open", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--seed", default="20260728")
    ap.add_argument("--reason", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.seal:
        do_seal(a.seed, a.force)
    elif a.open:
        do_open(a.reason)
    else:
        do_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
