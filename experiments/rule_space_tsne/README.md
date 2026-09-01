# rule_space_tsne — a map of the searched law space

One self-contained script (`run_rule_space.py`) that regenerates the programme's
documented candidate laws (SI Note S1.3 itemisation, 2,037,606 lower bound) from the
archived frozen search artefacts, renders each candidate as a one-line English rule
statement, embeds the statements, projects them with t-SNE, and draws a
publication-style map: the searched problems as territories, with the eight surviving
rules D1–D8 starred inside the crystal-plausibility region.

Nothing in `tex/`, `src/` or `paper/` is touched. All outputs go to `--out`
(default `./out`; `./out_smoke` for the smoke test).

## What is regenerated, and how honestly

| SI itemised entry | Archive source | Reconstruction | Status |
|---|---|---|---|
| 8,466 (L4 chain) | `src/l4_search.py` `candidates()` + `load_tables()` | exact re-enumeration; per-candidate single-rule satisfaction and detection rate evaluated on the discovery split | exact |
| 869,855 (sorbent threshold search) | `next72_..._v1` archived search record | term lists (anchor + singles + shortlist pairs) × 29 frozen rejection fractions, from the archived `guard_feature_ranking` / `pair_shortlist`; count verified == 869,855 | exact |
| 886,095 (sorbent threshold search) | `next78_..._v1` | same construction; count verified | exact |
| 435 | `next79_..._v1` | same construction; count verified | exact |
| 73,128 (2 × 36,564, SSSP / PBAAA framework guards) | `next534_..._v1`, `next540_..._v1` + archived label-free feature tables | label-free recomputation: 6 frozen weights × unique score values over supported development rows; no endpoint labels are read; counts verified | exact |
| 23,382 (CMVF: 2,077 + 12,127 + 9,178) | `next106/107/108_..._v1` per-candidate parquet | read back verbatim | exact |
| 22,592 (CMVO) | `next111_..._v1` parquet | read back verbatim | exact |
| 4,757 (DOBVR) | `next103_..._v1` parquet | read back verbatim | exact |
| 144,237 (mechanism-family grids) | **not uniquely relocated** | not regenerated | missing |
| 12,909; 402; 192; 21; 3 | **not uniquely relocated / not regenerated** | — | missing |

Exactly relocated itemised entries sum to **1,888,710 of 2,037,606 (92.7 %)**.

In addition, the archive contains per-candidate catalogues that are *not* individually
named in the SI itemisation (the itemisation is a lower bound); they are genuine
archived candidates and are plotted, tagged by their own round:
NEXT98b (12,111), NEXT114 (2,688), NEXT117 (11,349), NEXT121 (59,319),
NEXT122 (14,292), NEXT125 (57,178), NEXT127 (1,300), NEXT158 (176) — 158,413 extra.
Some of these very plausibly constitute the unmapped 144,237 "mechanism-family grids"
entry, but the correspondence could not be verified, so it is *not* claimed.
`out/coverage.json` records the full reconciliation. The archived NEXT76 tail search
(876,757 candidates, also absent from the itemisation) is included only with
`--include-next76`.

Default full-run population: **2,047,123 candidates** (1,888,710 itemised-exact
+ 158,413 archived extras).

## Pipeline stages (each cached; rerun any subset with `--stages`)

1. **enumerate** → `out/parts/*.parquet` (per-family cache), `out/candidates.parquet`,
   `out/stars.json` (D1–D8 → nearest L4 candidate), `out/coverage.json`.
   L4 metrics: `sat_disc` = fraction of real discovery structures satisfying the single
   predicate; `det_disc` = fraction of perturbed discovery structures it rejects.
2. **textualize** → `out/texts.parquet`. One-line English statements. The 94 L4
   features are glossed by the `FEATURE_GLOSS` dict (94/94 glossed, from the docstrings
   of `src/phys_law.py`, `src/geom_feat.py`, `src/elec_feat.py`, `src/discriminate.py`,
   `src/f3_features.py`). Other families use their archived term identifiers,
   prettified mechanically (`cmvf_core_log_scale_mismatch__high` → "high cmvf core log
   scale mismatch") — these are *not* hand-glossed.
3. **embed** → `out/embeddings.npy` (+ `embeddings_meta.json`).
   - `--backend st` (default): sentence-transformers, `--model` (default
     `sentence-transformers/all-MiniLM-L6-v2`, downloaded at runtime), `--device
     cuda|cpu`, `--batch-size`. Chunked into 50k-text shards under `out/emb_shards/`,
     fp16, **resumable** (finished shards are skipped on rerun).
   - `--backend tfidf`: purely local word(1–2) + char_wb(3–5) TF-IDF → TruncatedSVD
     (`--svd-dim`, default 256). Offline fallback; no downloads.
4. **project** → `out/tsne.csv`. openTSNE if installed (PCA-50 init, `--perplexity`
   default 40, seed 20260815, all cores); else sklearn t-SNE (≤100k points); else
   UMAP; else instructs you to `--subsample N` (stratified by family, survivors always
   kept).
5. **plot** → `out/rule_space_map.pdf/.png` (rasterized scatter, one colour per
   problem territory, D1–D8 as gold stars; plausibility points sized by detection rate
   where satisfaction ≥ 0.90) and `out/rule_space_plaus.pdf` (plausibility territory
   only, coloured by the 8 physical feature families in the `FAMILY` dict).

## Server setup and run

No local env currently has sentence-transformers (checked: newpauling,
matgenbench-py310, csagent; llm-extract's python segfaults), so create one:

```bash
conda create -y -n rulespace python=3.11 && conda activate rulespace
pip install numpy pandas pyarrow scikit-learn scipy matplotlib sentence-transformers openTSNE
```

The script needs the repo and the archive visible at the same paths as on the
workstation (override with `--repo` / `--archive`):
- repo: `<repo>/` (for `src/l4_search.py`, `src/next534/540_*.py`)
- archive: `$PRIS_ARCHIVE/` (frozen `next*_v1` dirs,
  `features/`, `next20260801/`)

Full run (GPU):

```bash
python run_rule_space.py --backend st --device cuda --batch-size 2048 --perplexity 40
```

Useful variants:

```bash
python run_rule_space.py --stages embed,project,plot          # after enumeration is cached
python run_rule_space.py --subsample 500000                   # lighter projection
python run_rule_space.py --backend tfidf                      # fully offline
python run_rule_space.py --smoke                              # 2,000-candidate end-to-end test (~10 s)
```

### Expected runtime / memory (2.05 M candidates)

| stage | GPU server (e.g. 4090/A100) | notes |
|---|---|---|
| enumerate | ~2–5 min | pure pandas/numpy + parquet reads; ~6 GB RAM peak |
| textualize | ~2–4 min | string building; candidates.parquet ~150 MB |
| embed (st, cuda, fp16) | ~20–40 min | MiniLM-L6, batch 2048; `embeddings.npy` ≈ 1.6 GB fp16; CPU instead: 8–20 h |
| embed (tfidf) | ~20–30 min, ~20 GB RAM | randomized SVD on 2M × 60k sparse |
| project (openTSNE, full 2 M) | ~1–3 h, ~10 GB RAM | FFT-accelerated; `--subsample 500000` cuts this to ~20 min |
| plot | ~1–2 min | rasterized PDF ~10–30 MB |

## Output files

- `candidates.parquet` — one row per candidate: `rule_id`, `problem`, `family`, and the
  family's parameterisation (L4: `feature`, `direction`, `threshold`, `guard`,
  `sat_disc`, `det_disc`; tails: correction terms + rejection fraction; guards:
  weight + threshold; catalogues: archived term/weight JSON + threshold).
- `texts.parquet` — `rule_id`, `problem`, `family`, `text`.
- `embeddings.npy`, `embeddings_meta.json`, `emb_shards/` — statement embeddings.
- `tsne.csv` — `rule_id`, `problem`, `family`, `x`, `y`.
- `stars.json` — D1–D8 → nearest L4 candidate (feature, target vs matched threshold,
  whether the guard matched exactly; D3's native guard is CN-conditioned and lies
  outside the L4 guard grid, so its star is the nearest unguarded predicate).
- `coverage.json` — per-family regenerated counts, exactness flags, and the list of
  SI itemisation entries that were not relocated.
- `rule_space_map.pdf/.png`, `rule_space_plaus.pdf` — the figures.

## Caveats

- Threshold values for the NEXT72/76/78/79 tail candidates are represented by their
  frozen rejection fractions (2–30 %), which is how the search grid was parameterised;
  the numeric score cut for each fraction is data-derived and does not change the
  statement's meaning.
- Feature-gloss coverage: 94/94 of the L4 feature names are hand-glossed; no raw-name
  fallbacks are currently exercised. Term identifiers of the sorbent and
  generative-screening families are machine-prettified, not hand-glossed.
- The smoke test subsamples families and therefore reports `exact=false` in its own
  coverage file by construction.
