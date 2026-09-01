# -*- coding: utf-8 -*-
"""LeakGuard:可证明无泄漏的特征白名单 + 非退化集合打分器。

背景
----
上一轮对抗复核证明:**人工判断"这个特征算不算泄漏"不可靠**。
`search_t1.univ_conn` 的作者在注释里写明了 `tgt = 1 <=> an_link > deg` 这条恒等式,
禁掉了 `deg` 却保留了 `an_link` —— 禁错了一半。`T_P5` 的 `cn_span_st<=0 => tgt=0`
是逻辑恒真,同样没被任何门拦住。

所以泄漏判定必须**机检**。本模块实现三级检测:

  A 经验确定性   : t 是否是单个 f 的确定性函数 / 是否存在零错分的单侧纯区
  B 单特征天花板 : 单个 f 喂无约束决策树,按 proto_id 分组 CV 的准确率
  C 构造溯源     : features.yaml 里每个特征声明 (atoms, op, scope, base),
                   与靶的声明做传递闭包 + 三条硬规则

验收标准(检测器自己的):必须复现上一轮的两个已知泄漏
  an_link  vs T_CONN
  cn_span_st vs T_P5

用法
----
  python leakguard.py prep      # 构建并缓存三个论域(只读 discovery)
  python leakguard.py audit     # 跑三级检测,写 report + whitelist
  python leakguard.py all
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEAT = os.environ.get("PRIS_FEATURES", "features/")
SCRATCH = ("/tmp/claude-1000/-home-zhilong-workspace-newpauling/"
           "6021b5d8-e7ef-4fa5-8416-1361cb973e90/scratchpad")
YAML_OUT = os.path.join(FEAT, "features_clean.yaml")
REPORT_OUT = os.path.join(FEAT, "leakguard_report.json")
YAML_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.yaml")

TARGETS = ("T_CONN", "T_P2", "T_P5")

# ---- 判定阈值(在看检测结果之前定死)-------------------------------------
EPS_DET = 1e-6          # 确定性函数的容错比例
PURE_MIN_COV = 0.005    # 单侧纯区的最小覆盖率(低于此不算泄漏,只是稀有子群)
TAIL_PURITY = 0.99      # 单侧"近纯"区的纯度阈值(cov >= TAIL_MIN_COV)
TAIL_MIN_COV = 0.01
CEIL_BLOCK = 0.98       # 单特征分组 CV 天花板 >= 此值 -> 泄漏
CEIL_REVIEW = 0.95


# ================================================================== 论域缓存

def prep(force=False):
    """构建三个 T1 论域并缓存到 scratchpad。只读 discovery(由 search_t1 保证)。"""
    import search_t1 as st
    todo = [t for t in TARGETS
            if force or not os.path.exists(os.path.join(SCRATCH, f"lg_{t}.parquet"))]
    if not todo:
        print("[prep] 缓存已存在,跳过")
        return
    t0 = time.time()
    X0, meta0, feat0, rnd0, orb, U = st.build_universes()

    def dump(tag, uni):
        X = uni["Xs"]["chemenv"].copy()
        for a in st.ALGOS:
            X["__y_" + a] = np.asarray(uni["ys"][a]).astype(np.int8)
        X["__ok"] = np.asarray(uni["ok"]).astype(bool)
        m = uni["meta"]
        for c in ("source_id", "element", "ox", "anion", "proto_id"):
            X["__m_" + c] = m[c].values
        X.to_parquet(os.path.join(SCRATCH, f"lg_{tag}.parquet"), index=False)
        json.dump({"feat": uni["feat"], "rnd": uni["rnd"]},
                  open(os.path.join(SCRATCH, f"lg_{tag}_cols.json"), "w"))
        print(f"[prep] {tag}: {len(X)} 行 / {len(uni['feat'])} 特征 "
              f"/ ok={int(X['__ok'].sum())} ({time.time()-t0:.0f}s)", flush=True)

    import gc
    if "T_CONN" in todo:
        dump("T_CONN", st.univ_conn(X0, meta0, orb, U)); gc.collect()
    if "T_P2" in todo:
        dump("T_P2", st.univ_p2(U, orb)); gc.collect()
    if "T_P5" in todo:
        dump("T_P5", st.univ_p5(X0, meta0, orb, U)); gc.collect()


def load(tag):
    X = pd.read_parquet(os.path.join(SCRATCH, f"lg_{tag}.parquet"))
    cols = json.load(open(os.path.join(SCRATCH, f"lg_{tag}_cols.json")))
    ok = X["__ok"].values.astype(bool)
    y = {a: X["__y_" + a].values.astype(np.int8)
         for a in ("chemenv", "crystalnn", "brunner")}
    meta = X[[c for c in X.columns if c.startswith("__m_")]].rename(
        columns=lambda c: c[4:])
    F = X[[c for c in cols["feat"] + cols["rnd"] if c in X.columns]]
    return F, y, ok, meta, cols["feat"], [c for c in cols["rnd"] if c in X.columns]


# ============================================== A 级:经验确定性 / 单侧纯区

def level_a(v, y, ok):
    """单特征经验确定性检测。

    返回
      err_thresh   : min over (θ, 极性) 的错分比例。== 0 -> t 是 f 的阈值函数
      err_value    : 按 f 的精确取值分组后的最小错分比例。== 0 -> t 是 f 的确定性函数
      pure_side    : 存在零错分单侧区(cov >= PURE_MIN_COV)时的 (侧, cov, 方向标签)
      tail_purity  : cov >= TAIL_MIN_COV 的单侧区能达到的最高纯度
    与 search_t1.tierA 同口径:NaN -> -1e18(落到低侧)。
    """
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=-1e18,
                      posinf=1e18, neginf=-1e18)[ok]
    y = np.asarray(y)[ok].astype(np.int64)
    n = v.size
    if n == 0:
        return None
    o = np.argsort(v, kind="stable")
    vs, ys = v[o], y[o]
    # 切点:相邻不同值之间
    cut = np.flatnonzero(vs[1:] != vs[:-1]) + 1          # 低侧样本数
    c1 = np.cumsum(ys)
    n_lo = cut.astype(np.float64)
    k1_lo = c1[cut - 1].astype(np.float64)               # 低侧的 y=1 个数
    tot1 = float(c1[-1])
    n_hi = n - n_lo
    k1_hi = tot1 - k1_lo
    k0_lo, k0_hi = n_lo - k1_lo, n_hi - k1_hi

    res = {"n": int(n), "n_unique": int(len(cut) + 1),
           "base_rate": float(tot1 / n)}
    if len(cut) == 0:
        res.update(err_thresh=1.0, err_value=1.0, pure_side=None,
                   tail_purity=float(max(res["base_rate"], 1 - res["base_rate"])),
                   tail_cov=1.0)
        return res

    # --- 阈值确定性:低侧预测 b、高侧预测 1-b
    err_a = k1_lo + k0_hi          # b=0 低侧 / b=1 高侧
    err_b = k0_lo + k1_hi
    res["err_thresh"] = float(np.minimum(err_a, err_b).min() / n)

    # --- 取值级确定性(t 是 f 的任意确定性函数)
    # 注意:近连续特征几乎每行一个取值,err_value 平凡地等于 0(每个取值一个样本
    # 当然纯)。负对照 rnd_u0/rnd_u3 就是这样被误判 BLOCK 的。因此本判据只在
    # 每个取值平均至少 MIN_PER_VALUE 个样本时才可用,否则置 NaN 弃用。
    MIN_PER_VALUE = 20
    _, inv = np.unique(vs, return_inverse=True)
    nv = inv.max() + 1
    g1 = np.bincount(inv, weights=ys, minlength=nv)
    gn = np.bincount(inv, minlength=nv).astype(np.float64)
    ev = float((gn - np.maximum(g1, gn - g1)).sum() / n)
    res["mean_per_value"] = float(n / nv)
    res["err_value"] = ev if (n / nv) >= MIN_PER_VALUE else float("nan")
    res["err_value_raw"] = ev

    # --- 零错分单侧区(恒真蕴含):某一侧全 0 或全 1
    best = None
    for side, nn, kk1, kk0, lab in (("lo", n_lo, k1_lo, k0_lo, "f<=theta"),
                                    ("hi", n_hi, k1_hi, k0_hi, "f>theta")):
        pure = ((kk1 == 0) | (kk0 == 0)) & (nn >= PURE_MIN_COV * n)
        if pure.any():
            j = int(np.flatnonzero(pure)[np.argmax(nn[pure])])
            cov = float(nn[j] / n)
            # 偶然纯的概率:纯区大小为 m 时 = max(p0,1-p0)^m,必须 < 1e-6
            p = max(res["base_rate"], 1 - res["base_rate"])
            log_p_chance = float(nn[j]) * math.log10(max(p, 1e-12))
            if log_p_chance < -6 and (best is None or cov > best["cov"]):
                best = {"side": side, "cov": cov, "body": int(kk1[j] > 0),
                        "theta": float(vs[cut[j] - 1]), "pred": lab,
                        "log10_p_chance": round(log_p_chance, 1)}
    res["pure_side"] = best

    # --- 近纯单侧区(cov >= TAIL_MIN_COV 时可达的最高纯度)
    tp, tc = 0.0, 0.0
    for nn, kk1 in ((n_lo, k1_lo), (n_hi, k1_hi)):
        m = nn >= TAIL_MIN_COV * n
        if m.any():
            pu = np.maximum(kk1[m], nn[m] - kk1[m]) / nn[m]
            j = int(np.argmax(pu))
            if pu[j] > tp:
                tp, tc = float(pu[j]), float(nn[m][j] / n)
    res["tail_purity"], res["tail_cov"] = tp, tc
    return res


# ====================================== B 级:单特征分组 CV 天花板(无约束树)

def level_b(v, y, ok, groups, n_folds=5, seed=20260728):
    """单个特征喂深度不限的决策树,按 proto_id 分组 K 折 CV。"""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GroupKFold
    x = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=-1e18,
                      posinf=1e18, neginf=-1e18)[ok].reshape(-1, 1)
    yy = np.asarray(y)[ok].astype(np.int8)
    gg = pd.factorize(np.asarray(groups)[ok])[0]
    if len(np.unique(yy)) < 2 or len(np.unique(gg)) < n_folds:
        return None
    gk = GroupKFold(n_splits=n_folds)
    acc, maj = [], []
    for tr, te in gk.split(x, yy, gg):
        t = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed).fit(
            x[tr], yy[tr])
        acc.append(float((t.predict(x[te]) == yy[te]).mean()))
        b = int(yy[tr].mean() >= 0.5)
        maj.append(float((yy[te] == b).mean()))
    return {"cv_acc": float(np.mean(acc)), "cv_sd": float(np.std(acc)),
            "cv_maj": float(np.mean(maj)),
            "cv_gain": float(np.mean(acc) - np.mean(maj))}


def joint_ceiling(F, cols, y, ok, groups, n_folds=3, seed=20260728):
    """一组特征联合的分组 CV 天花板(用来量化"公式项集合"能重建多少目标)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    cols = [c for c in cols if c in F.columns]
    if not cols:
        return None
    x = F[cols].values.astype(np.float64)[ok]
    yy = np.asarray(y)[ok].astype(np.int8)
    gg = pd.factorize(np.asarray(groups)[ok])[0]
    if len(np.unique(yy)) < 2 or len(np.unique(gg)) < n_folds:
        return None
    gk = GroupKFold(n_splits=n_folds)
    acc, maj = [], []
    for tr, te in gk.split(x, yy, gg):
        m = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                           early_stopping=False,
                                           random_state=seed).fit(x[tr], yy[tr])
        acc.append(float((m.predict(x[te]) == yy[te]).mean()))
        b = int(yy[tr].mean() >= 0.5)
        maj.append(float((yy[te] == b).mean()))
    return {"n_feat": len(cols), "cv_acc": float(np.mean(acc)),
            "cv_maj": float(np.mean(maj)),
            "cv_gain": float(np.mean(acc) - np.mean(maj))}


# ================================================== C 级:构造溯源(features.yaml)

ENVELOPE_OPS = {"span", "range", "ptp", "nunique", "std", "max", "min",
                "argmax", "argmin", "any", "all"}


def _closure(name, spec, atoms):
    """特征的原子传递闭包:atoms 字段 + 每个原子在 atoms 表里声明的上游。"""
    seen, stack = set(), list(spec.get("atoms", []))
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        stack.extend(atoms.get(a, {}).get("from", []))
    return seen


def level_c(cfg, tag):
    """按 features.yaml 的声明跑三条硬规则。返回 {feature: verdict_dict}。"""
    atoms = cfg["atoms"]
    rank = cfg["scope_rank"]             # 数字越小 = 聚合域越粗(包含更多个体)
    tspec = cfg["targets"][tag]
    tclos = set(tspec["forbidden_atoms"])
    tscope, tbase = tspec["scope"], tspec["base"]

    def contains(s_outer, s_inner):
        """s_outer 的聚合域是否包含 s_inner(靶的域)。"""
        if s_outer not in rank or s_inner not in rank:
            return False
        return rank[s_outer] <= rank[s_inner]

    feats = dict(cfg["features"][tag])
    for inh in tspec.get("inherit", []):
        for k, v in cfg["common"][inh].items():
            feats.setdefault(k, v)

    out = {}
    for f, spec in feats.items():
        clos = _closure(f, spec, atoms)
        hits = sorted(clos & tclos)
        r = {"atoms": sorted(clos), "op": spec.get("op"),
             "scope": spec.get("scope"), "base": spec.get("base"),
             "R1_atom_hit": hits,
             "R2_envelope": False, "R3_same_object": False, "note": spec.get("note")}
        # R1:特征的原子闭包碰到靶的禁用原子 -> 构造级泄漏
        r["R1"] = bool(hits)
        # 信息位:命中发生在比靶更粗的聚合域上(泄漏机制更弱,但仍不进白名单)
        r["R1_coarse"] = bool(hits and rank.get(spec.get("scope"), 9)
                              < rank.get(tscope, 9))
        # R2:包络算子 + 同底量 + 聚合域包含靶域 -> 恒真蕴含入口
        if (spec.get("base") == tbase and spec.get("op") in ENVELOPE_OPS
                and contains(spec.get("scope"), tscope)):
            r["R2"] = r["R2_envelope"] = True
        else:
            r["R2"] = False
        # R3:与靶同域同底量的任何统计量 -> 靶的充分统计量族,需经验清场
        r["R3"] = bool(spec.get("base") == tbase and spec.get("scope") == tscope
                       and not r["R2"])
        r["R3_same_object"] = r["R3"]
        r["verdict_c"] = ("BLOCK" if (r["R1"] or r["R2"]) else
                          "REVIEW" if r["R3"] else "PASS")
        out[f] = r
    return out


# ================================================================== 汇总判定

def combine(a, b, c):
    """三级结果 -> 最终判定。任一级 BLOCK 即 BLOCK。"""
    reasons = []
    if c is not None and c["R1"]:
        reasons.append("C.R1构造溯源:原子闭包命中靶的禁用原子 " + ",".join(c["R1_atom_hit"]))
    if c is not None and c["R2"]:
        reasons.append("C.R2包络恒真:同底量的包络算子聚合域包含靶域")
    if a is not None:
        ev = a.get("err_value")
        ev = 1.0 if (ev is None or ev != ev) else ev
        if ev <= EPS_DET:
            reasons.append(f"A.取值级恒等式 err={ev:.2e}"
                           f"(每取值 {a.get('mean_per_value', 0):.0f} 样本)")
        elif a.get("err_thresh", 1.0) <= EPS_DET:
            reasons.append(f"A.阈值级恒等式 err={a['err_thresh']:.2e}")
        if a.get("pure_side"):
            ps = a["pure_side"]
            reasons.append(f"A.零错分单侧区 {ps['pred']} theta={ps['theta']:.4g} "
                           f"=> tgt={ps['body']} cov={ps['cov']:.4f}")
    if b is not None and b["cv_acc"] >= CEIL_BLOCK:
        reasons.append(f"B.单特征分组CV天花板 {b['cv_acc']:.4f}")
    if reasons:
        return "BLOCK", reasons
    soft = []
    if c is not None and c["R3"]:
        soft.append("C.R3与靶同域同底量(靶的充分统计量族)")
    if a is not None and a.get("tail_purity", 0) >= TAIL_PURITY:
        soft.append(f"A.近纯单侧区 purity={a['tail_purity']:.4f} cov={a['tail_cov']:.4f}")
    if b is not None and b["cv_acc"] >= CEIL_REVIEW:
        soft.append(f"B.单特征CV {b['cv_acc']:.4f} >= {CEIL_REVIEW}")
    return ("REVIEW", soft) if soft else ("PASS", [])


def audit():
    import yaml
    cfg = yaml.safe_load(open(YAML_SRC))
    report = {"meta": {"eps_det": EPS_DET, "pure_min_cov": PURE_MIN_COV,
                       "tail_purity": TAIL_PURITY, "tail_min_cov": TAIL_MIN_COV,
                       "ceil_block": CEIL_BLOCK, "ceil_review": CEIL_REVIEW,
                       "split": "discovery only"}, "targets": {}}
    clean = {}
    for tag in TARGETS:
        t0 = time.time()
        F, y3, ok, meta, feats, rnds = load(tag)
        y = y3["chemenv"]
        grp = meta.proto_id.values
        cres = level_c(cfg, tag)
        rows = {}
        undeclared = [f for f in feats if f not in cres]
        for f in feats:
            a = level_a(F[f].values, y, ok)
            b = level_b(F[f].values, y, ok, grp)
            c = cres.get(f)
            v, why = combine(a, b, c)
            if c is None:
                v, why = "BLOCK", ["C.未在 features.yaml 声明(默认拒绝)"]
            rows[f] = {"A": a, "B": b, "C": c, "verdict": v, "reasons": why}
            if a and b:
                msg = (f"  [{tag}] {f:22s} {v:6s} "
                       f"errT={a['err_thresh']:.2e} errV={a['err_value']:.2e} "
                       f"pure={'Y' if a['pure_side'] else '-'} "
                       f"tail={a['tail_purity']:.4f} cv={b['cv_acc']:.4f}")
            else:
                msg = f"  [{tag}] {f:22s} {v:6s} (A/B 不可算)"
            print(msg + ("  <- " + "; ".join(why) if why else ""), flush=True)
        # 负对照:随机特征必须全部 PASS,否则检测器本身有问题
        neg = {}
        for f in rnds[:8]:
            a = level_a(F[f].values, y, ok)
            b = level_b(F[f].values, y, ok, grp)
            neg[f] = {"verdict": combine(a, b, None)[0],
                      "err_thresh": a["err_thresh"], "tail_purity": a["tail_purity"],
                      "cv_acc": None if b is None else b["cv_acc"]}
        passed = [f for f, r in rows.items() if r["verdict"] == "PASS"]
        review = [f for f, r in rows.items() if r["verdict"] == "REVIEW"]
        block = [f for f, r in rows.items() if r["verdict"] == "BLOCK"]
        # 公式项集合的联合天花板(量化"靶的定义式自身能重建多少")
        formula = [f for f in block if rows[f]["C"] and rows[f]["C"]["R1"]]
        jc = {"formula_terms": joint_ceiling(F, formula, y, ok, grp),
              "whitelist": joint_ceiling(F, passed, y, ok, grp)}
        report["targets"][tag] = {
            "n": int(ok.sum()), "base_rate": float(y[ok].mean()),
            "n_feat": len(feats), "n_pass": len(passed), "n_review": len(review),
            "n_block": len(block), "undeclared": undeclared,
            "pass": passed, "review": review, "block": block,
            "joint_ceiling": jc, "neg_control_rnd": neg, "features": rows}
        clean[tag] = {"scope": cfg["targets"][tag]["scope"],
                      "target": cfg["targets"][tag]["desc"],
                      "whitelist": passed, "review_excluded": review,
                      "blocked": {f: rows[f]["reasons"] for f in block}}
        print(f"[{tag}] PASS {len(passed)} / REVIEW {len(review)} / BLOCK {len(block)}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 验收:必须复现两个已知泄漏
    acc = {}
    for tag, f in (("T_CONN", "an_link"), ("T_P5", "cn_span_st")):
        r = report["targets"][tag]["features"].get(f)
        acc[f"{tag}::{f}"] = {"caught": bool(r and r["verdict"] == "BLOCK"),
                              "reasons": (r or {}).get("reasons")}
    report["acceptance"] = acc
    report["acceptance"]["all_caught"] = all(v["caught"] for k, v in acc.items()
                                             if isinstance(v, dict))
    json.dump(report, open(REPORT_OUT, "w"), indent=1, default=str)
    yaml.safe_dump({"generated": time.strftime("%Y-%m-%d %H:%M"),
                    "source": "leakguard.py", "split": "discovery",
                    "targets": clean}, open(YAML_OUT, "w"),
                   allow_unicode=True, sort_keys=False)
    print("\n验收:", json.dumps(acc, ensure_ascii=False, indent=1, default=str))
    print("写出", REPORT_OUT, "\n写出", YAML_OUT)
    return report


# ================================================== 修好的打分器(可被 search_t1 导入)

class AuditScorerV2:
    """合取语义下的非退化集合打分器。

    旧 `AuditScorer.matched()` 的病:所有触发成员里只按 `bits` 最短的那条给预测,
    成员 body 同极性时集合在覆盖域上退化成常数预测器,`acc_set` 恒等于 `maj_matched`。

    新语义(与 PREREG §3.3 的合取组合一致):
      * 每个成员在**自己的 guard 内**各自给出预测;
      * 覆盖域内若各成员预测**一致** -> 集合预测 = 该值;
      * **冲突则弃权**(abstain),弃权样本不计入 acc_set,但计入 `abstain` 率;
      * 报 `acc_set`(仅表决一致者)、`acc_set_all`(弃权按错算,保守口径)、
        `abstain`、以及**与 B1 和 B2 在同一批表决一致样本上的对比**。
    """

    def __init__(self, y, ok, masks, bodies, bits, deff):
        self.y, self.ok = np.asarray(y).astype(bool), np.asarray(ok).astype(bool)
        self.masks, self.bodies, self.bits = masks, [int(b) for b in bodies], bits
        self.n = int(self.ok.sum())
        self.deff = deff
        self.n_eff = self.n / deff
        self.pool = list(range(len(masks)))
        self.trig = [np.asarray(m).astype(bool) & self.ok for m in masks]
        self._cache = {}

    # ---- MDL 部分与旧类逐位一致(不动记账口径)
    def cells(self, S):
        pat = np.zeros(len(self.y), np.int64)
        cell = np.zeros(len(self.y), np.int64)
        for k, i in enumerate(S):
            t = self.trig[i]
            pat += (1 << k) * t
            cell = cell * 3 + np.where(t, 1 + self.bodies[i], 0)
        return cell * 8191 + pat

    def data_cost(self, S):
        k = tuple(sorted(S))
        if k not in self._cache:
            self._cache[k] = self._data_cost(k)
        return self._cache[k]

    def _data_cost(self, S):
        c = self.cells(S)[self.ok]
        t = self.y[self.ok].astype(float)
        _, cid = np.unique(c, return_inverse=True)
        nc = cid.max() + 1
        n1 = np.bincount(cid, weights=t, minlength=nc)
        nG = np.bincount(cid, minlength=nc).astype(float)
        n0 = nG - n1
        data = float(np.where(n0 > 0, n0 * np.log2(nG / np.maximum(n0, 1)), 0).sum() +
                     np.where(n1 > 0, n1 * np.log2(nG / np.maximum(n1, 1)), 0).sum())
        par = float(0.5 * np.log2(np.maximum(nG / self.deff, 2.0)).sum())
        par_reg = (2 ** max(len(S), 1) - 1) / 2.0 * math.log2(self.n_eff)
        return data / self.deff, dict(reg=par_reg, full=par, obs=par)

    def model_cost(self, S, M, lam):
        return lam * (sum(self.bits[i] for i in S) +
                      (math.log2(math.comb(M, len(S))) if len(S) else 0.0))

    def L_total(self, S, M, lam, par_mode="obs"):
        d, p = self.data_cost(S)
        return self.model_cost(S, M, lam) + p[par_mode] + d, d, p

    # ---- 修好的匹配覆盖率评估
    def vote(self, S):
        """返回 (cov, pred, abstain)。合取语义:成员各自预测,冲突弃权。"""
        n = len(self.y)
        n0 = np.zeros(n, np.int32)      # 预测 0 的成员数
        n1 = np.zeros(n, np.int32)
        for i in S:
            t = self.trig[i]
            if self.bodies[i]:
                n1 += t
            else:
                n0 += t
        cov = (n0 + n1) > 0
        abst = cov & (n0 > 0) & (n1 > 0)
        pred = np.where(n1 > 0, 1, 0)
        return cov, pred.astype(np.int8), abst

    def matched(self, S, base1=None, base2=None, maj_global=None):
        cov, pred, abst = self.vote(S)
        if not cov.any():
            return None
        dec = cov & ~abst                       # 表决一致(集合真正给出预测)的样本
        y = self.y
        r = {"cov": float(cov.sum() / self.n), "n_cov": int(cov.sum()),
             "abstain": float(abst.sum() / max(int(cov.sum()), 1)),
             "n_decided": int(dec.sum()),
             "cov_decided": float(dec.sum() / self.n)}
        r["acc_set"] = float((pred[dec] == y[dec]).mean()) if dec.any() else float("nan")
        # 保守口径:弃权按错算,分母仍是整个覆盖域
        r["acc_set_all"] = float((pred[cov] == y[cov]).mean() * 0
                                 + (pred[dec] == y[dec]).sum() / cov.sum())
        # 基线必须在**同一批已决样本**上比,否则分母不同,比较无效
        for nm, base in (("B1", base1), ("B2", base2)):
            if base is None:
                continue
            base = np.asarray(base)
            r[f"acc_{nm}_matched"] = float((base[dec] == y[dec]).mean()) if dec.any() else float("nan")
            r[f"gain_vs_{nm}"] = r["acc_set"] - r[f"acc_{nm}_matched"]
            r[f"acc_{nm}_cov"] = float((base[cov] == y[cov]).mean())
        # 退化诊断:集合在已决域上是不是常数预测器
        r["maj_matched"] = float(max(y[dec].mean(), 1 - y[dec].mean())) if dec.any() else float("nan")
        r["pred_entropy"] = float(_H(pred[dec].mean())) if dec.any() else 0.0
        r["is_constant_predictor"] = bool(dec.any() and len(np.unique(pred[dec])) == 1)
        r["gain_vs_maj"] = r["acc_set"] - r["maj_matched"]
        if maj_global is not None:
            r["gain_vs_maj_global"] = r["acc_set"] - float(maj_global)
        ntr = np.zeros(len(y), np.int16)
        for i in S:
            ntr += self.trig[i]
        r["overlap"] = float((ntr >= 2).sum() / max(self.n, 1))
        r["contra"] = float(abst[ntr >= 2].mean()) if (ntr >= 2).any() else 0.0
        r["n_members"] = len(S)
        r["bodies"] = [self.bodies[i] for i in S]
        r["single_polarity"] = bool(len(set(r["bodies"])) <= 1)
        return r


def _H(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def gate_G5(perm_z, gain_vs_B1, gain_vs_B2, z_min=5.0):
    """修好的 G5:必须同时打赢 B1 **和** B2。上一轮只比 B1,导致 vs B2 为负的规则混进来。"""
    return bool(perm_z >= z_min and gain_vs_B1 > 0 and gain_vs_B2 > 0)


# ================================================================== main

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("prep", "all"):
        prep(force="--force" in sys.argv)
    if cmd in ("audit", "all"):
        audit()


if __name__ == "__main__":
    main()
