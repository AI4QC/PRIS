#!/usr/bin/env python3
"""Take only the last frame of each ELEMENTA relaxation trajectory (the converged structure).

ELEMENTA is MLIP training data: each `material` has one VASP ionic relaxation trajectory, and
a frame's comment line looks like
    material=Ac_01 formula=Ac structure=structure_01 ionic_step=0 nelm=11 \
    energy=-8.17267212 stress="..." magmom=... pbc="T T T"
The frames of one material are contiguous in the file with ionic_step increasing, so "take the
last frame" can be done in a stream: emit the previous frame whenever the material changes,
in O(1) memory.

Usage:
    zstd -dc ELEMENTA_open.extxyz.tar.zst | tar -xO | python3 elementa_endpoints.py \
        --out endpoints.extxyz --stats stats.json

    # binaries and above only (Pauling's rules are undefined on elemental structures):
    ... | python3 elementa_endpoints.py --out endpoints.extxyz --min-elements 2
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
# the streaming stage needs only the material and ionic_step fields. A full findall on each of
# 39 million frames measured as the main bottleneck (about 20-40 us each, 15-25 minutes in
# total), so a narrow regex picks those two out here and the full parse happens only when a
# last frame is actually written (about 2.9 million times).
RE_MATERIAL = re.compile(r'\bmaterial=(\S+)')
RE_STEP = re.compile(r'\bionic_step=(\d+)')


def parse_comment(line: str) -> dict:
    return {k: v.strip('"') for k, v in KV.findall(line)}


def n_elements(symbols) -> int:
    return len(set(symbols))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output extxyz path')
    ap.add_argument('--stats', default=None, help='path for the statistics JSON output')
    ap.add_argument('--min-elements', type=int, default=1,
                    help='minimum number of distinct elements. Pauling\'s rules are undefined '
                         'on elemental structures, so set 2 for a Pauling analysis')
    ap.add_argument('--max-sites', type=int, default=0,
                    help='maximum atom count; 0 means no limit. ChemEnv is very slow on large '
                         'cells')
    ap.add_argument('--progress-every', type=int, default=2_000_000)
    args = ap.parse_args()

    st = Counter()
    per_nelem = Counter()
    prev_key = None       # the previous frame's material
    prev_frame = None     # (natoms, comment, [atom_lines])
    step_of_prev = -1

    def flush(frame, key):
        """Write out frame as the last frame of a trajectory."""
        if frame is None:
            return
        natoms, comment, atoms = frame
        syms = [ln.split(None, 1)[0] for ln in atoms]
        ne = n_elements(syms)
        st['trajectories'] += 1
        per_nelem[ne] += 1
        if ne < args.min_elements:
            st['dropped_min_elements'] += 1
            return
        if args.max_sites and natoms > args.max_sites:
            st['dropped_max_sites'] += 1
            return
        # add a marker so downstream code can tell where it came from
        extra = f' source=elementa traj_id={key} n_ionic_steps={step_of_prev + 1}'
        out.write(f'{natoms}\n{comment.rstrip()}{extra}\n')
        out.writelines(atoms)
        st['written'] += 1

    out = open(args.out, 'w')
    it = sys.stdin
    try:
        while True:
            head = it.readline()
            if not head:
                break
            head = head.strip()
            if not head:
                continue
            try:
                natoms = int(head)
            except ValueError:
                st['bad_header'] += 1
                continue
            comment = it.readline()
            atoms = [it.readline() for _ in range(natoms)]
            if not comment or any(a == '' for a in atoms):
                st['truncated'] += 1
                break
            st['frames'] += 1

            m = RE_MATERIAL.search(comment)
            key = m.group(1) if m else ''
            m = RE_STEP.search(comment)
            step = int(m.group(1)) if m else -1

            if prev_key is not None and key != prev_key:
                flush(prev_frame, prev_key)
            prev_key, prev_frame, step_of_prev = key, (natoms, comment, atoms), step

            if args.progress_every and st['frames'] % args.progress_every == 0:
                print(f"  {st['frames']:,} frames -> {st['written']:,} last frames",
                      file=sys.stderr, flush=True)
        flush(prev_frame, prev_key)
    finally:
        out.close()

    st_d = dict(st)
    st_d['by_n_elements'] = dict(sorted(per_nelem.items()))
    st_d['compression_ratio'] = round(st['frames'] / max(st['trajectories'], 1), 2)
    print(json.dumps(st_d, ensure_ascii=False, indent=2), file=sys.stderr)
    if args.stats:
        with open(args.stats, 'w') as f:
            json.dump(st_d, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
