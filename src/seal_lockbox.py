#!/usr/bin/env python3
"""三分数据并封存 lockbox。这是全项目可信度的锚点。

为什么需要:在 38,307 个结构上搜规则,必然搜得出伪规律。唯一的防线是
"搜索永远看不到一部分数据,最终规则在那部分上一次性评估"。

三分(按 sid 的 sha256 定,与顺序、与文件无关,可复现):
  discovery   60%  —— 搜索、调参、看图,随便看
  calibration 25%  —— 定阈值、选 N、做模型选择,可以看但不许用来搜规则
  lockbox     15%  —— 封存,全程只许开 3 次

**为什么不加密**(v1.1 的设计修正):初版用 gpg 对称加密 lockbox 的 sid 清单,
这是做过头了,而且自相矛盾——`splits.parquet` 里明文写着完整三分表,加密等于没加。
更根本的是 `split_of()` 是纯函数、seed 明文写在封条里,任何人一行代码就能重算出全部分配。

这里要防的是**自己**在反复"看结果再调参"的过程中不知不觉泄信息,不是防外人。
防自己靠的是流程可核查,不是密码学。所以保留三样真正起作用的:

  1. 封条 LOCKBOX.sealed.json —— seed / 时间戳 / git commit / 各分区计数 / sid 清单 sha256。
     证明"分法是在这个时间点、这份代码上定的",事后改不了
  2. 审计日志 openings.log —— 每次动 lockbox 必须写理由,论文里报实际开封次数
  3. PREREG.md 的 git tag —— 判据在看数据之前冻结,tag 的 commit hash 写进论文方法部分

用法(全程无需任何口令或交互输入):
    python seal_lockbox.py --seal --seed 20260728        # 首次封存
    python seal_lockbox.py --open --reason "MPU-6 主论文最终评估"   # 开封(计入配额)
    python seal_lockbox.py --status                       # 查看封条与开封记录
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

# 分区比例。改这三个数等于换了一套实验,所以它们进封条、不进命令行。
FRAC = {"discovery": 0.60, "calibration": 0.25, "lockbox": 0.15}


def split_of(sid: str, seed: str) -> str:
    """sid → 分区。纯函数:同样的 (sid, seed) 永远给同样的分区,
    与数据文件的行序、与是否重跑都无关。"""
    h = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
    # 取前 8 字节当无符号整数,映射到 [0,1)
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
    """分析集的 sid。用 provenance 的 source_id(icsd-N / cod-N),
    它跨脚本一致,而且不依赖 sqlite 的 pk(pk 会随重建变)。"""
    p = FEATURES / "provenance.parquet"
    if not p.exists():
        sys.exit(f"缺少 {p};先跑 make week1")
    df = pd.read_parquet(p, columns=["source_id", "in_analysis_set"])
    sids = sorted(df.loc[df.in_analysis_set, "source_id"].astype(str).unique())
    if not sids:
        sys.exit("分析集为空,不封")
    return sids


def do_seal(seed: str, force: bool) -> None:
    if SEAL.exists() and not force:
        sys.exit(f"封条已存在:{SEAL}\n重新封存会使既有结论失效。确认要重来请加 --force")
    LOCKDIR.mkdir(parents=True, exist_ok=True)
    sids = load_sids()
    assign = {s: split_of(s, seed) for s in sids}
    counts = {k: sum(1 for v in assign.values() if v == k) for k in FRAC}

    lock_sids = sorted(s for s, v in assign.items() if v == "lockbox")
    blob = "\n".join(lock_sids).encode()
    digest = hashlib.sha256(blob).hexdigest()

    # sid 清单明文存。加密没有意义:seed 在封条里、split_of 是纯函数,一行代码就能重算。
    # 真正起作用的是下面的封条与审计日志。
    ENC.write_bytes(blob)

    # 完整三分表,下游按 split 列过滤。
    # 注意:它包含 lockbox 行,这是有意的——遮住它并不能阻止重算,
    # 只会让下游代码不得不绕路,反而更容易出错。
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
    print(f"\n封条已写入 {SEAL}")
    print("下一步:把 PREREG.md 提交并打 git tag,tag 的 commit hash 要写进论文方法部分。")


def do_open(reason: str) -> None:
    if not SEAL.exists():
        sys.exit("还没封存,无从开启")
    seal = json.loads(SEAL.read_text())
    used = [l for l in AUDIT.read_text().splitlines() if l.strip()] if AUDIT.exists() else []
    if len(used) >= seal["max_openings"]:
        sys.exit(f"开封配额已用尽({len(used)}/{seal['max_openings']})。"
                 f"论文必须报告实际开封次数,不要再开。")
    if not reason or len(reason) < 10:
        sys.exit("必须给出实质性的开封理由(≥10 字),它会进审计日志和论文")

    if not ENC.exists():
        sys.exit(f"缺少 {ENC}")
    blob = ENC.read_bytes()
    if hashlib.sha256(blob).hexdigest() != seal["lockbox_sids_sha256"]:
        sys.exit("★ 解出的 sid 清单与封条哈希不符,数据被动过,停止")

    rec = {"opened_at_utc": datetime.now(timezone.utc).isoformat(),
           "reason": reason, "git_commit": git_commit(),
           "opening_index": len(used) + 1}
    with AUDIT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = LOCKDIR / f"lockbox_sids_opening{rec['opening_index']}.txt"
    out.write_bytes(blob)
    print(f"第 {rec['opening_index']}/{seal['max_openings']} 次开封,已写 {out}")
    print(f"剩余配额:{seal['max_openings'] - rec['opening_index']}")


def do_status() -> None:
    if not SEAL.exists():
        print("尚未封存")
        return
    print(json.dumps(json.loads(SEAL.read_text()), indent=2, ensure_ascii=False))
    if AUDIT.exists() and AUDIT.read_text().strip():
        print("\n开封记录:")
        for l in AUDIT.read_text().splitlines():
            if l.strip():
                print("  " + l)
    else:
        print("\n开封记录:无(从未开启)")


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
