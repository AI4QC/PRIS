# PRIS independent screening and the Fig. 4-5 integration plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Objective:** present the L4 laws and the frozen synthesis formula (S_{\mathrm{syn}}) as two
separately applied routes for pre-computation queue screening, and rearrange Fig. 5 so that the
physical meaning and the task-specific ranking role follow naturally from Fig. 4's screening
results.

**Overall approach:** L4 removes only structures that explicitly violate PRIS; (S_{\mathrm{syn}})
calibrates a continuous low-score threshold on experimental structures. The two routes decide
separately, and the main figure constructs no cascade, union or joint gate between L4 and the
formula. The low-scoring PU structures are a proxy queue chosen by a model, not negatives whose
synthesis has been shown to fail. The existing damage-validation figure stays in the SI; main-text
integration waits until the new figures and the narrative are confirmed.

**Stack:** Python, pandas, NumPy, PyArrow, Matplotlib, LaTeX, pytest, SHA-256 manifests.

## Frozen evidence and conventions of interpretation

- The experimental queue: 99,162 unique CIFs.
- The low-scoring PU queue: removing 179 duplicate CIFs from the 364,771 published records leaves
  364,592 unique CIFs; it is a model-selected proxy queue, not ground truth about failed synthesis
  experiments.
- L4 removes only `explicit_violation`; `no_verdict` always stays in the downstream queue.
- L4's fixed operating point: 80.6851% of experimental structures retained, and
  189,159/364,592 = 51.8824% of the PU queue screened out.
- At the same experimental retention, an independently calibrated (S_{\mathrm{syn}}) screens out
  305,075/364,592 = 83.6757% of the PU queue; at the more conservative 95% experimental-retention
  point it screens out 74,142/364,592 = 20.3356%.
- "The same retention" is understood against the actual strict thresholds: L4 retains
  80,009/99,162 and the formula retains 80,010/99,162, a difference of one experimental structure,
  so the caption says approximately matched.
- (S_{\mathrm{syn}}) uses frozen coefficients, frozen standardisation parameters and the frozen
  training-set median to impute missing terms. Structures with all six terms observed are only
  21,477/99,162 (21.66%) of the experimental group and 135/364,592 (0.037%) of the PU group; that
  level of support goes in the SI and Methods, and the whole-queue results may not be written as
  universal accuracy.
- In an independent mechanistic audit, D7 covers 186,741/189,159 = 98.72% of the PU structures L4
  screens out. That is an explanatory coverage, not a second screening route.

## New artefacts already generated

- `experiments/pu_synthesizability_20260821/independent_screening.py`: the computation and
  operating-point functions for the two independent decision routes; it returns no joint decision.
- `experiments/pu_synthesizability_20260821/plot_independent_screening.py`: generates the English
  Fig. 4 draft.
- `experiments/pu_synthesizability_20260821/plot_draft_fig5.py`: reuses the current frozen panel
  implementation to generate the English Fig. 5 draft in the proposed order.
- `outputs/20260822_pu_formula_scores/independent_choices_v1/`: CSVs, JSON, English PNG/PDF, a
  results note and the SHA-256 manifest.
- `tests/test_pu_synth_independent_screening.py`, `tests/test_pu_synth_draft_figures.py`: smoke
  tests for the decision logic and figure rendering.

## Task 1: keep the interface of the two independent decision routes

**Files:**

- Modify: `experiments/pu_synthesizability_20260821/independent_screening.py`
- Tests: `tests/test_pu_synth_independent_screening.py`

**Steps:**

1. Keep the behaviour of `build_independent_frontier()`: calibrate the formula threshold only, at
   each given retention, and add one natural L4 operating point; no `L4 OR S_syn` or
   `L4 AND S_syn` row may be added.
2. Keep the behaviour of `build_operating_point_summary()`: calibrate the formula separately at
   L4's experimental retention, and assert that every output row has `combined == False`.
3. Keep the totals of both the experimental and the PU queue on every CSV row, so that a structure
   count is not misread as a percentage.
4. Run `python -m pytest -q tests/test_pu_synth_independent_screening.py`, expecting 2 tests to
   pass.

## Task 2: freeze and verify the Fig. 4 draft

**Files:**

- Modify: `experiments/pu_synthesizability_20260821/plot_independent_screening.py`
- Generate: `outputs/20260822_pu_formula_scores/independent_choices_v1/`
- Tests: `tests/test_pu_synth_draft_figures.py`

**Steps:**

1. All reader-visible text in the figures is in English; any Chinese stays in the results report
   and the plan.
2. Keep the a -> b -> c -> d narrative chain:
   - **a, Independent pre-DFT operating choices:** the continuous operating curve of
     (S_{\mathrm{syn}}); L4 is a single fixed star.
   - **b, Matched experimental retention:** L4 and (S_{\mathrm{syn}}) computed separately at about
     80.69% experimental-structure retention.
   - **c, Queue length after pre-screening:** L4, (S_{\mathrm{syn}}) at the same retention, and
     (S_{\mathrm{syn}}) at 95% retention; the three bars are three independent choices and may not
     be read as the result of a cascade.
   - **d, Continuous score and mechanism:** the (S_{\mathrm{syn}}) distribution, the formula
     threshold corresponding to L4, a note on the frozen median imputation, and the independent
     explanatory coverage of D7.
3. Run:

   ```bash
   python -m experiments.pu_synthesizability_20260821.plot_independent_screening \
     --output-dir outputs/20260822_pu_formula_scores/independent_choices_v1
   ```

4. Run `python -m pytest -q tests/test_pu_synth_independent_screening.py tests/test_pu_synth_draft_figures.py`.
5. Check each PNG: no Chinese, no literal `\\n`, no overlapping text on the queue bar tops, and no
   curve or bar labelled as combined.

## Task 3: freeze the proposed Fig. 5 order

**Files:**

- Modify: `experiments/pu_synthesizability_20260821/plot_draft_fig5.py`
- Generate: `outputs/20260822_pu_formula_scores/independent_choices_v1/pris_draft_fig5_physical_meaning.{png,pdf}`

**Steps:**

1. Keep the keywords in the existing titles and use the following a -> b -> c -> d order:
   - **a, Two task-specific projections of PRIS mechanisms:** the standardised coefficients of
     (S_{\mathrm{stab}}) (hull-energy ranking) and (S_{\mathrm{syn}}) (experimental-record
     ranking). The main text must state that they are projections of the same continuous
     mechanistic descriptor space onto different supervision labels, and not an overall score
     obtained by weighting binary L4 states.
   - **b, Strong damage detection does not imply polymorph ranking:** the commitment plane; a
     "tie" means no unique choice, not an error.
   - **c, Confidence-dependent formula accuracy:** the confidence curves of the synthesis formula
     and of the hull energy, keeping the existing exploratory qualifier.
   - **d, Energy, phonons and experimental records select different structures:** the existing
     three-axis ladder, as the resolution of the tension.
2. If the layout is crowded, the tie-rate bars of the old Fig. 5b may become an inset of b or move
   to the SI; the wrong-hull-pair panel of the old Fig. 5c preferentially moves to the SI once
   Fig. 4 uses the downstream energy reference.
3. Run `python -m experiments.pu_synthesizability_20260821.plot_draft_fig5`, and check each figure
   for English-only text and no clipping in the lower part of Fig. 5d.

## Task 4: assess the downstream energy reference without writing it up as a third pre-screening route

**Files:**

- Inspect: `src/next15_basin_hull.py`,
  `outputs/20260802_next15_wbm_basin_hull_retrospective/`, the binary queue manifests and the
  local CIF data sources.
- If a new experiment turns out to be needed, add only a standalone pilot script and output
  directory; do not modify the main text.

**Known audit conclusions and rules of execution:**

1. The PU CIFs are locally complete: `CSAgent/data/from_hpc/release/negatives/train.csv` and
   `val.csv` hold 364,771 records in total, with `material_id` corresponding one to one with the
   binary queue; the experimental CIFs can be decoded from `structures.blob` through
   `experiments/pu_synthesizability_20260821/data.py::decode_blob_cif`.
2. The existing MatterSim protocol computes an **MLIP basin-hull proxy**: the MatterSim relaxed
   energy minus the MP reference hull. Without DFT energies for the same batch of structures, it
   must never be labelled DFT (E_{\mathrm{hull}}).
3. The current MP reference covers only 2,722/91,561 PU chemical systems (about 2.97%),
   corresponding to 12,239/364,771 records (about 3.35%). So the existing proxy curve does not go
   into the main Fig. 4 comparison; at most it appears in the SI, or in Fig. 4c in grey as a
   downstream reference, with the supported-subset coverage reported alongside.
4. If a bounded pilot is run, the threshold must be calibrated on experimental structures alone
   and then applied to PU; it is a downstream reference, not a third pre-DFT user option, and it
   may not be merged with L4 or (S_{\mathrm{syn}}).
5. If the data support or the GPU runs are unstable, record the blocking reason directly; do not
   fill the main figure with incomplete results.

## Task 5: integrate into the main text only after confirmation

**Files that may be modified only in future:**

- Create or modify: `src/fig4_pre_dft.py` (suggested to be kept separate from the old damage
  figure's plotting functions).
- Modify: the panel call order and caption data of `src/fig5_ranking.py`.
- Modify: `tex/body.tex`, `tex/front_body.tex`, `tex/si_body.tex`, `paper/FACTS.md`.

**Integration steps:**

1. First copy the current Fig. 4 damage-validation evidence into an SI figure or a compact inset;
   it may not be deleted silently.
2. The Fig. 4 caption keeps its existing full phrase and appends: `PRIS improves screening before
   expensive calculations: selected bounds detect damage types omitted during selection; the queue
   comparison adds two separately applied choices, an interpretable L4 gate and a continuous
   synthesis score.` If the old damage panels move entirely to the SI, the main-text title keeps
   the first half of the sentence and the second half becomes a new clause.
3. The Fig. 5 caption keeps its existing full phrase and appends: `What structural plausibility can
   and cannot decide. Task-specific formulas extend this diagnosis.`
4. After the Fig. 4 results, add a bridging sentence in English: the queue experiment shows the
   utility of two user choices, but not a synthesis success rate; L4 names the violated mechanism,
   and (S_{\mathrm{syn}}) provides a tunable ranking towards the experimental-record label.
5. Fig. 5a immediately explains the supervision task and the continuous-descriptor origin of the
   two formulas, and b/c/d then discuss polymorph ties, confidence-ordered ranking, and the tension
   between energy, phonons and the experimental record.
6. Delete or rewrite the vague "the four properties all differ" sentence in the main text,
   replacing it with a concrete bridge carrying the current queue counts, retentions and task
   labels.
7. Update every subfigure reference in the main text and the captions, checking a -> b -> c -> d
   one by one; every reference to the old Fig. 4 omission/damage panels points at the SI (if the
   decision is to move them out of the main text).
8. Finally rebuild the main-text and SI PDFs, and run the facts, render, `pdftotext`, PDF page
   count, SHA-256 and `git diff --check` audits.

## Verification commands that must be run

```bash
python -m pytest -q \
  tests/test_pu_synth_independent_screening.py \
  tests/test_pu_synth_draft_figures.py \
  tests/test_pu_synth_formula_score_results.py \
  tests/test_pu_synth_formula_score_figure.py
python -m experiments.pu_synthesizability_20260821.plot_independent_screening \
  --output-dir outputs/20260822_pu_formula_scores/independent_choices_v1
python -m experiments.pu_synthesizability_20260821.plot_draft_fig5
cd outputs/20260822_pu_formula_scores/independent_choices_v1
sha256sum -c SHA256SUMS
cd - >/dev/null
python - <<'PY'
from pathlib import Path
for p in Path('outputs/20260822_pu_formula_scores/independent_choices_v1').glob('*.pdf'):
    assert p.stat().st_size > 1000, p
print('draft PDFs have non-trivial size')
PY
```

This stage stops once the new data, reports and English drafts are generated; it does not modify
the canonical `tex/`, the main-text figure sources, or the existing PDFs. Before integrating into
the main text, the current dirty worktree must be preserved, and the old PDFs copied to the
SI/backup before the main figures are replaced.
