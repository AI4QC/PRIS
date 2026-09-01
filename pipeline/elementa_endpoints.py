#!/usr/bin/env python3
"""从 ELEMENTA 的弛豫轨迹里只取每条轨迹的最后一帧(收敛结构)。

ELEMENTA 是 MLIP 训练数据:每个 `material` 有一条 VASP 离子弛豫轨迹,
帧的注释行形如
    material=Ac_01 formula=Ac structure=structure_01 ionic_step=0 nelm=11 \
    energy=-8.17267212 stress="..." magmom=... pbc="T T T"
同一 material 的帧在文件里是连续的、ionic_step 递增。
所以"取末帧"可以流式做:material 变了就把上一帧吐出来,内存 O(1)。

用法:
    zstd -dc ELEMENTA_open.extxyz.tar.zst | tar -xO | python3 elementa_endpoints.py \
        --out endpoints.extxyz --stats stats.json

    # 只要二元及以上(泡林定律在单质上没有定义):
    ... | python3 elementa_endpoints.py --out endpoints.extxyz --min-elements 2
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
# 流式阶段只需要 material 和 ionic_step 两个字段。对 3900 万帧逐帧做完整 findall
# 实测是主要瓶颈(每次约 20-40 us,合计 15-25 分钟),所以这里用窄正则单独抓,
# 完整解析只在真正写出末帧时做(约 290 万次)。
RE_MATERIAL = re.compile(r'\bmaterial=(\S+)')
RE_STEP = re.compile(r'\bionic_step=(\d+)')


def parse_comment(line: str) -> dict:
    return {k: v.strip('"') for k, v in KV.findall(line)}


def n_elements(symbols) -> int:
    return len(set(symbols))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='输出 extxyz 路径')
    ap.add_argument('--stats', default=None, help='统计信息 JSON 输出路径')
    ap.add_argument('--min-elements', type=int, default=1,
                    help='最少元素种类数。泡林定律在单质上无定义,做泡林分析时设 2')
    ap.add_argument('--max-sites', type=int, default=0,
                    help='最大原子数,0 表示不限。ChemEnv 在大晶胞上会很慢')
    ap.add_argument('--progress-every', type=int, default=2_000_000)
    args = ap.parse_args()

    st = Counter()
    per_nelem = Counter()
    prev_key = None       # 上一帧的 material
    prev_frame = None     # (natoms, comment, [atom_lines])
    step_of_prev = -1

    def flush(frame, key):
        """把 frame 作为一条轨迹的末帧写出。"""
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
        # 补一个标记,便于下游区分来源
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
                print(f"  {st['frames']:,} 帧 → {st['written']:,} 末帧",
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
