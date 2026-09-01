#!/usr/bin/env python3
"""Recompute PU validation metrics from the frozen CSAgent-v2 artifacts.

This script is deliberately separate from the screening pipeline.  It does not
train or alter any model.  It reads the saved bag splits, OOB membership map,
best checkpoints/heads, graph packs and MatterSim embeddings, then writes small
audit tables and macro validation curves.  The validation labels are the PU
labels (experimental positive versus sampled-unlabelled pseudo-negative), not
observed synthesis-failure labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: str) -> list[tuple[str, int]]:
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row:
                out.append((row[0], int(row[1])))
    return out


def iter_pickle_dicts(path: str) -> Iterable[dict]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def make_head(dim: int, hidden: int = 256):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(0.3),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, 2), nn.LogSoftmax(dim=1),
    )


def metric_row(model: str, bag: int, y: np.ndarray, s: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    threshold = 0.5
    pred = (s >= threshold).astype(int)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return {
        "model": model,
        "bag": int(bag),
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_pseudo_negative": int((y == 0).sum()),
        "decision_threshold": threshold,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None,
        "accuracy_at_0.5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
        "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "positive_mean": float(np.mean(s[y == 1])) if np.any(y == 1) else None,
        "positive_median": float(np.median(s[y == 1])) if np.any(y == 1) else None,
        "negative_mean": float(np.mean(s[y == 0])) if np.any(y == 0) else None,
        "negative_median": float(np.median(s[y == 0])) if np.any(y == 0) else None,
    }


def curve_row(model: str, curve: str, bag: int, y: np.ndarray,
              s: np.ndarray, grid: np.ndarray) -> np.ndarray:
    from sklearn.metrics import precision_recall_curve, roc_curve
    if curve == "roc":
        x, v, _ = roc_curve(y, s)
        order = np.argsort(x, kind="mergesort")
        x, v = x[order], v[order]
        return np.interp(grid, x, v, left=v[0], right=v[-1])
    v, x, _ = precision_recall_curve(y, s)
    order = np.argsort(x, kind="mergesort")
    x, v = x[order], v[order]
    # precision-recall curves can be non-monotone; interpolation is only a
    # visualization summary, while per-bag AP remains the reported metric.
    return np.interp(grid, x, v, left=v[0], right=v[-1])


def summarize_curves(model: str, predictions: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    rows = []
    for curve, grid in (("roc", np.linspace(0.0, 1.0, 201)),
                        ("pr", np.linspace(0.0, 1.0, 201))):
        vals = np.stack([curve_row(model, curve, b, y, s, grid)
                         for b, (y, s) in sorted(predictions.items())])
        for j, x in enumerate(grid):
            rows.append({
                "model": model,
                "curve": curve,
                "x": float(x),
                "mean": float(np.mean(vals[:, j])),
                "sd": float(np.std(vals[:, j], ddof=1)),
                "q025": float(np.quantile(vals[:, j], 0.025)),
                "q975": float(np.quantile(vals[:, j], 0.975)),
            })
    return rows


def distribution_rows(model: str, predictions: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    rows = []
    for cls, name in ((1, "experimental_positive"), (0, "pseudo_negative")):
        x = np.concatenate([s[y == cls] for y, s in predictions.values()])
        rows.append({
            "model": model,
            "class": name,
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "q01": float(np.quantile(x, 0.01)),
            "q05": float(np.quantile(x, 0.05)),
            "q25": float(np.quantile(x, 0.25)),
            "median": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def load_protocol(pool: str, out: str) -> tuple[dict, dict[int, list[tuple[str, int]]], dict[int, set[str]]]:
    split = os.path.join(pool, "saved_splits")
    valid = {b: read_rows(os.path.join(split, f"id_prop_bag_{b}_valid.csv"))
             for b in range(1, 51)}
    train = {b: read_rows(os.path.join(split, f"id_prop_bag_{b}_train.csv"))
             for b in range(1, 51)}
    with open(os.path.join(split, "bag_members.json")) as f:
        members_raw = json.load(f)
    members = {int(b): set(v) for b, v in members_raw.items()}

    positives = {pid for pid, lab in valid[1] if lab == 1}
    same_pos = all({pid for pid, lab in valid[b] if lab == 1} == positives
                    for b in range(1, 51))
    overlap = {str(b): len(set(pid for pid, _ in train[b]) &
                            set(pid for pid, _ in valid[b])) for b in range(1, 51)}
    valid_neg_membership = {
        str(b): int(sum(pid in members[b] for pid, lab in valid[b] if lab == 0))
        for b in range(1, 51)
    }
    all_valid = {pid for rows in valid.values() for pid, _ in rows}
    oob_counts = {pid: 50 for pid in all_valid}
    for b in range(1, 51):
        for pid in members[b]:
            if pid in oob_counts:
                oob_counts[pid] -= 1
    idprop = read_rows(os.path.join(pool, "id_prop.csv"))
    n_pos = sum(lab == 1 for _, lab in idprop)
    n_unl = sum(lab == 0 for _, lab in idprop)
    files = [os.path.join(pool, "id_prop.csv"), os.path.join(split, "bag_members.json")]
    protocol = {
        "pool": pool,
        "n_pool_rows": len(idprop),
        "n_positive": n_pos,
        "n_unlabeled": n_unl,
        "n_bags": 50,
        "valid_rows_per_bag": {str(b): len(valid[b]) for b in range(1, 51)},
        "valid_positive_per_bag": {str(b): int(sum(l == 1 for _, l in valid[b])) for b in range(1, 51)},
        "valid_pseudo_negative_per_bag": {str(b): int(sum(l == 0 for _, l in valid[b])) for b in range(1, 51)},
        "train_rows_per_bag": {str(b): len(train[b]) for b in range(1, 51)},
        "positive_holdout_identical_across_bags": bool(same_pos),
        "train_valid_overlap_per_bag": overlap,
        "valid_negative_in_own_bag_membership": valid_neg_membership,
        "validation_oob_count_distribution": {
            "min": int(min(oob_counts.values())),
            "median": float(np.median(list(oob_counts.values()))),
            "max": int(max(oob_counts.values())),
            "n_validation_ids": len(oob_counts),
        },
        "independent_test_available": False,
        "independent_test_reason": "Only per-bag validation files exist; no true-negative or untouched test file was found.",
        "label_semantics": {
            "1": "experimental positive structure",
            "0": "unlabeled candidate sampled as a pseudo-negative for this bag",
        },
        "source_hashes": {os.path.relpath(p, pool): sha256(p) for p in files},
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out, "protocol_audit.json"), "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
    return protocol, valid, members


def evaluate_a(pool: str, scripts: str, valid: dict[int, list[tuple[str, int]]], out: str, device: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, scripts)
    from pu_common import InMemoryGraphDataset, load_model
    sys.path.insert(0, os.path.join(os.path.dirname(scripts), "script"))
    from cgcnn.data_PU_learning import collate_pool

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    models = [load_model(os.path.join(pool, "checkpoints", f"bag_{b}_best.pth")).to(dev).eval()
              for b in range(1, 51)]
    valid_pos = {pid for pid, lab in valid[1] if lab == 1}
    neg_sets = {b: {pid for pid, lab in valid[b] if lab == 0} for b in range(1, 51)}
    pos_scores = {b: [] for b in range(1, 51)}
    pos_labels = {b: [] for b in range(1, 51)}
    neg_scores = {b: [] for b in range(1, 51)}
    neg_labels = {b: [] for b in range(1, 51)}
    pos_path = os.path.join(pool, "bags", "positives.pkl")
    t0 = time.time()

    def score_rows(graphs: dict, rows: list[tuple[str, int]], use_models: list[int]):
        if not rows:
            return
        ds = InMemoryGraphDataset(rows, graphs)
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0,
                        collate_fn=collate_pool)
        with torch.no_grad():
            for (af, nf, ni, ci), _target, ids in dl:
                af, nf, ni = af.to(dev), nf.to(dev), ni.to(dev)
                ci = [x.to(dev) for x in ci]
                p = torch.stack([torch.exp(models[b](af, nf, ni, ci))[:, 1]
                                 for b in use_models], dim=1).cpu().numpy()
                for j, pid in enumerate(ids):
                    for k, model_index in enumerate(use_models):
                        b = model_index + 1
                        pos_scores[b].append(float(p[j, k]))
                        pos_labels[b].append(1)

    # Shared positive pack is read once and scored by all 50 models.
    for chunk_no, graphs in enumerate(iter_pickle_dicts(pos_path)):
        rows = [(pid, 1) for pid in valid_pos if pid in graphs]
        score_rows(graphs, rows, list(range(50)))
        if chunk_no % 10 == 0:
            print(f"A positives chunk {chunk_no}, rows={len(rows)}, elapsed={time.time()-t0:.1f}s", flush=True)

    # Each bag file contains its sampled unlabeled set; only its validation
    # pseudo-negatives are scored with that bag's selected checkpoint.
    for b in range(1, 51):
        path = os.path.join(pool, "bags", f"bag_{b}.pkl")
        n = 0
        for graphs in iter_pickle_dicts(path):
            rows = [(pid, 0) for pid in neg_sets[b] if pid in graphs]
            if rows:
                ds = InMemoryGraphDataset(rows, graphs)
                dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0,
                                collate_fn=collate_pool)
                with torch.no_grad():
                    for (af, nf, ni, ci), _target, ids in dl:
                        p = torch.exp(models[b-1](af.to(dev), nf.to(dev), ni.to(dev),
                                                   [x.to(dev) for x in ci]))[:, 1].cpu().numpy()
                        neg_scores[b].extend(float(x) for x in p)
                        neg_labels[b].extend(0 for _ in ids)
                        n += len(ids)
        print(f"A bag {b}: validation pseudo-negatives scored={n}", flush=True)

    predictions = {}
    metrics = []
    for b in range(1, 51):
        y = np.asarray(pos_labels[b] + neg_labels[b], dtype=int)
        s = np.asarray(pos_scores[b] + neg_scores[b], dtype=float)
        predictions[b] = (y, s)
        metrics.append(metric_row("CGCNN-PU", b, y, s))
    curves = summarize_curves("CGCNN-PU", predictions)
    dist = distribution_rows("CGCNN-PU", predictions)
    return metrics, curves, dist, {"device": str(dev), "elapsed_s": time.time()-t0,
                                   "checkpoint_dir": os.path.join(pool, "checkpoints")}


def evaluate_b(pool: str, valid: dict[int, list[tuple[str, int]]], out: str, device: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    import torch
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    valid_ids = {pid for rows in valid.values() for pid, _ in rows}
    embeddings = {}
    emb_dir = os.path.join(pool, "embeddings_mattersim")
    t0 = time.time()
    for j, fp in enumerate(sorted(Path(emb_dir).glob("emb_*.npz"))):
        z = np.load(fp, allow_pickle=False)
        ids = [str(x) for x in z["ids"]]
        for i, pid in enumerate(ids):
            if pid in valid_ids:
                embeddings[pid] = np.asarray(z["emb"][i], dtype=np.float32)
        if j % 20 == 0:
            print(f"B embeddings shard {j}, retained={len(embeddings)}", flush=True)
    if not embeddings:
        raise RuntimeError("No validation embeddings found")
    heads = []
    for b in range(1, 51):
        fp = os.path.join(pool, "mlp_heads_mattersim", f"bag_{b}_best.pt")
        h = make_head(128).to(dev)
        h.load_state_dict(torch.load(fp, map_location=dev))
        h.eval(); heads.append(h)
    predictions = {}
    metrics = []
    missing = {}
    with torch.no_grad():
        for b in range(1, 51):
            rows = [(pid, lab) for pid, lab in valid[b] if pid in embeddings]
            missing[str(b)] = len(valid[b]) - len(rows)
            y = np.asarray([lab for _, lab in rows], dtype=int)
            X = torch.tensor(np.stack([embeddings[pid] for pid, _ in rows]), device=dev)
            s = torch.exp(heads[b-1](X))[:, 1].cpu().numpy()
            predictions[b] = (y, s)
            metrics.append(metric_row("MatterSim-1M-MLP-PU", b, y, s))
            print(f"B bag {b}: evaluated={len(rows)} missing={missing[str(b)]}", flush=True)
    curves = summarize_curves("MatterSim-1M-MLP-PU", predictions)
    dist = distribution_rows("MatterSim-1M-MLP-PU", predictions)
    return metrics, curves, dist, {"device": str(dev), "elapsed_s": time.time()-t0,
                                   "embedding_dir": emb_dir, "n_validation_embeddings": len(embeddings),
                                   "missing_validation_rows_per_bag": missing,
                                   "checkpoint_dir": os.path.join(pool, "mlp_heads_mattersim")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", choices=("A", "B", "both"), default="both")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    protocol, valid, _members = load_protocol(args.pool, args.out)
    all_metrics, all_curves, all_dist = [], [], []
    run_meta = {"models_requested": args.models, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "protocol": protocol}
    if args.models in ("A", "both"):
        m, c, d, meta = evaluate_a(args.pool, args.scripts, valid, args.out, args.device)
        all_metrics.extend(m); all_curves.extend(c); all_dist.extend(d)
        run_meta["CGCNN-PU"] = meta
    if args.models in ("B", "both"):
        m, c, d, meta = evaluate_b(args.pool, valid, args.out, args.device)
        all_metrics.extend(m); all_curves.extend(c); all_dist.extend(d)
        run_meta["MatterSim-1M-MLP-PU"] = meta
    write_csv(os.path.join(args.out, "bag_validation_metrics.csv"), all_metrics)
    write_csv(os.path.join(args.out, "macro_validation_curves.csv"), all_curves)
    write_csv(os.path.join(args.out, "validation_score_distribution.csv"), all_dist)
    run_meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(args.out, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)
    print("AUDIT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
