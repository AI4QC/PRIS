#!/usr/bin/env python3
"""Rebuild any figure of the manuscript by its displayed number.

    python figures/make.py "Fig. 1"      # one figure
    python figures/make.py S7            # the S prefix is enough
    python figures/make.py --list        # what makes what
    python figures/make.py --all         # everything the committed data supports

The script names in src/ predate the final figure numbering, so three of them
disagree with the figure they draw: fig6_deployment.py draws Fig. 5,
figS7_amplitude_response.py draws Fig. S5, and figS17_charge_coverage.py draws
Fig. S23.  manifest.json is the authority; this dispatcher reads it so nobody
has to remember the mismatch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())
BY_ID = {e["id"]: e for e in MANIFEST}


def resolve(token: str) -> str:
    """Accept 'Fig. S7', 'S7', 's7', '7' and return the manifest id."""
    t = token.strip().replace("Fig.", "").replace("fig", "").strip()
    for candidate in (f"Fig. {t}", f"Fig. {t.upper()}", token):
        if candidate in BY_ID:
            return candidate
    raise SystemExit(f"unknown figure {token!r}; try --list")


def show():
    width = max(len(e["generator"]) for e in MANIFEST)
    print(f"{'figure':<9} {'generator':<{width}}  panels  output")
    for e in MANIFEST:
        print(f"{e['id']:<9} {e['generator']:<{width}}  {e['panels']:^6}  {e['file']}")


def run(entry) -> int:
    script = ROOT / entry["generator"]
    if not script.exists():
        print(f"  {entry['id']}: {entry['generator']} is not in this repository", file=sys.stderr)
        return 1
    print(f"==> {entry['id']}  {entry['generator']}")
    return subprocess.call([sys.executable, str(script)], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help='e.g. "Fig. 1", S7, S25')
    ap.add_argument("--list", action="store_true", help="print the figure-to-script table")
    ap.add_argument("--all", action="store_true", help="run every distinct generator once")
    a = ap.parse_args()

    if a.list or not (a.figures or a.all):
        show()
        return 0
    if a.all:
        seen, bad = set(), 0
        for e in MANIFEST:
            if e["generator"] in seen:
                continue
            seen.add(e["generator"])
            bad += run(e) != 0
        return 1 if bad else 0
    return max((run(BY_ID[resolve(t)]) for t in a.figures), default=0)


if __name__ == "__main__":
    raise SystemExit(main())
