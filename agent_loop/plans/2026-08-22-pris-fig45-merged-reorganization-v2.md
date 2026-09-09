# PRIS Fig. 4-5 merged reorganization and PU diagnostics: implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** while preserving the physico-chemical meaning of PRIS, rearrange the existing Fig. 4/5
and the two PU experiments into one clear chain of evidence, and place the per-law contributions
and the held-out performance of both PU models fully in the SI.

**Architecture:** freeze the data conventions and the panel contract first, then generate one
merged main figure and the accompanying SI figures. The main text and the LaTeX are modified only
after the figure, fact and cross-reference audits pass. Binary PRIS, the continuous task formulas
and the PU model scores each serve a different task, and no new joint screening gate is
constructed.

**Tech stack:** Python 3.11, pandas, NumPy, PyArrow, SciPy, Matplotlib, pymatgen, MatterSim,
LaTeX, pytest, SHA-256 manifests.

---

## 0. Boundaries that must be respected this round

This file is an implementation plan; new experiments and drafts begin only once the plan is
written, and this round does not touch the main text directly.

1. Panel c of the merged main figure does not draw D7, does not use D7 to stand for PRIS as a
   whole, and does not show single-law contributions.
2. The contribution of each individual law goes into the per-law diagnostic figure in the SI.
3. Panel d of the merged main figure uses two stacked rows sharing one x axis: the relation
   between the PRIS laws and CLscore above, and the relation between the synthesis formula and the
   same CLscore bins below.
4. The Jang score is removed from the main figure. The legend writes both model names out in full:
   CGCNN-PU OOB and MatterSim-embedding PU OOB. If a mean CLscore is wanted, convert to a
   percentile or rank within each model first and then take the arithmetic mean; the raw
   quantities may not be averaged directly.
5. L4 and S_syn are two independent routes for the user to choose between; no L4 OR S_syn or
   L4 AND S_syn joint scheme is drawn.
6. Low-scoring PU structures may be called only a proxy queue of hard-to-synthesise candidates
   selected by the PU models, never syntheses shown experimentally to have failed.
7. MatterSim basin-hull serves only as a downstream reference. The old pilot is void because the
   reduced formula and the whole-cell energy were on inconsistent scales; the corrected GPU
   results may not be cited before they pass QA, and may not be labelled DFT E_hull.
8. Main-text figures use English only. The new figures and captions use uniform terminology --
   satisfaction, screening, damage detection, explicit violation -- avoiding internal words such
   as reject and retention appearing without warning.

## 1. The storyline and the order of the figures

The main figure is ordered along this chain of questions:

1. a-b show that binary mechanistic screening finds chemical damage before expensive calculations,
   and that the mechanisms are complementary.
2. c explains why a binary gate needs a continuous formula, and gives two practical pre-screening
   choices: L4 is conservative, S_syn is more aggressive but sacrifices experimental-structure
   retention.
3. d checks whether those mechanisms and formulas correlate with independent PU model scores. The
   upper and lower rows use exactly the same CLscore percentile x axis.
4. e shows that S_stab and S_syn address low-energy ranking and experimental-record ranking
   respectively, and cannot be merged into one aimless overall score.
5. f returns to the tension between energy, phonons and the experimental record, showing that site
   complexity still carries independent signal within the same energy and phonon state.

The commitment plane and tie rate of the old Fig. 5a are the first reason behind c, and the
standardised coefficients of the old Fig. 5e are a further reason for the physical origin of the
continuous formulas; both move to the SI, and the main text carries them over with one concrete
tie statistic and task definition rather than repeating them in the main figure.

## 2. The panel contract of the merged main figure

Call it Main Fig. M for now. Generate the complete large figure first; the final number should
preferentially stay Fig. 4, so as to keep the important main-text evidence of the current Fig. 4a/b,
with the old Fig. 5 panels moving to the SI.

| panel | contents | what the reader should take away | explicitly removed |
|---|---|---|---|
| a | the current Fig. 4a overall benchmark. Keep the 440 parents, 2,024 damaged structures, L4's 83.0% experimental satisfaction and 87.9% damage detection, and the distance-cutoff comparison. | PRIS finds chemical damage that a distance check misses, before expensive calculations. | do not mix the controlled-damage percentages with the PU queue percentages. |
| b | the current Fig. 4b heatmap by damage class, keeping the a -> b order. | the five mechanisms are complementary and a single distance threshold has a definite blind spot. | no per-law contributions; do not turn it into a D7 feature figure. |
| c | redraw c of docs/figures/pris_pu_task1_binary.png as a retention-screening frontier. x is experimental-structure retention and y is the screening rate of the PU hard-negative queue; compare L4, the two independent operating points of S_syn, the 0.5/0.7 A distance baseline, and, in grey, the corrected and QA-passed MatterSim basin-hull proxy as a reference. | L4 is a conservative mechanistic screen that retains most experimental structures; S_syn can compress hard-to-synthesise candidates more aggressively, but the two objectives cannot both be optimal and the user chooses by task. | a separate D7 bar, D7 coverage, D1-D8 contributions, any joint gate. D7 appears in c neither as a method label nor as a main conclusion. |
| d | the upper row plots the PRIS/L4 explicit-violation fraction against normalized CLscore; the lower row plots the S_syn mean against the same CLscore percentile bins. The independent curves of both models and the rank-normalized consensus must all be traceable to a full model name. | PRIS is related to the independent PU ranking but is not the same black-box score; the continuous formula shows a gradient the binary gate does not. | the Jang score, a direct average of the raw A/B values, different x binning, any statement of synthesis success probability. |
| e | two small axes: the group-equal accuracy of S_stab as the energy-difference threshold rises; and the held-out confidence-accuracy curve of S_syn, optionally with the DFT hull in grey as a reference. | the two formulas are task-specific projections of the same PRIS descriptor space onto two different scientific objectives. | listing coefficients with no performance; treating S_syn as a direct synthesis probability. |
| f | restore the historical four-cell D7/D1 panel: dynamically stable/on-hull, dynamically stable/metastable, imaginary modes/on-hull, imaginary modes/metastable; each cell split into experimentally recorded and computed-only. | with energy and phonon state fixed, the experimental record still favours a simpler site description, which fills out the explanation of the present crystallographic disagreement. | the current abstract full ladder, the old "91-99%" summary, and an excess of Wilson intervals. |

### The frozen operating points of c

Read the existing frozen results; do not retune any threshold:

- L4: about 80.69% experimental-structure retention, 51.88% of the PU queue screened out.
- S_syn: about 83.68% PU screening at the roughly 80.69% experimental-retention point; also show
  about 20.34% screening at the 95% experimental-retention point, as a conservative operating
  point.
- distance cutoff: 0% of the PU queue screened out.
- the MatterSim reference: rerun after the scaling is corrected; if the support coverage is
  insufficient or QA does not pass, keep it to the standalone report and out of the main figure.

Each point uses a distinct marker to indicate an independent user choice; no arrow suggests a
cascade. Write the structure counts and percentages beside the points directly, rather than only
abstract internal terms.

## 3. What goes in the SI

### SI-A/B: the two supporting reasons for a continuous formula

- SI-A keeps the commitment plane of the old Fig. 5a, with the tie rate as an inset; the main text
  cites only "binary PRIS frequently ties among same-composition pairs", with the detailed choice
  fraction, accuracy and tie counts in the SI.
- SI-B keeps the standardized coefficients of the old Fig. 5e, labelling the supervision objective,
  sign direction, sample size and standardisation of S_stab and S_syn separately, and explaining
  that the two expressions are not a decomposition of one overall score.
- The old Fig. 5b, the Fig. 5c wrong-hull-pair panel, the full ladder, and any Fig. 4c/d that does
  not reach the main figure all move to the SI with their internal references updated.

### SI-d: per-law contributions

Create a separate per-law figure, provisionally numbered SI-d (the final letter follows the SI
layout):

- the x axis lists D1-D8 in order of physical mechanism; where a combination has to be shown, list
  L1/L2/L4 separately, and never write a combination as a single law.
- the upper row shows the per-law satisfaction/explicit-violation fraction for the experimental and
  the PU hard-negative structures, with a decidable n.
- the lower row shows each law's incremental coverage of the shortening of the L4 queue, or its
  leave-one-law-out increment. Since one structure can violate several laws, use coverage rather
  than a mutually exclusive attribution.
- missing/no-verdict always stays in the queue and is never counted as a violation.

### SI-e/f: the held-out performance of the two PU models

Add two figures on the same template, one for each of:

1. the CGCNN-PU OOB scorer, from CSAgent's 03_train_bags.py and 04_predict_clscore.py.
2. the MatterSim-embedding PU OOB scorer, from CSAgent's 07_embed_pu_head.py.

Each figure contains at least ROC-AUC, PR-AUC (only where the labelling protocol permits), the
mean and uncertainty over bags/seeds, and the score distributions of held-out positives against
unlabelled/PU. Audit the original split, the OOB masks, the checkpoint hashes and the deduplication
manifest first; label something test only when it really is an independent test, and otherwise
label it validation explicitly. A performance figure may not be inferred backwards from full-pool
scores.

## 4. The statistical conventions of d

1. Use the 8,108,676 structures scored in common across the whole pool; the main figure does not
   read the Jang file.
2. Convert each model to a percentile separately and build
   consensus percentile = mean(percentile_CGCNN, percentile_MatterSim). The raw score ranges and
   the inter-model correlation go in the SI.
3. The upper and lower rows use the same decile or vigintile boundaries, the same n and the same
   bootstrap grouping; they must not appear to share an x axis while actually using different
   denominators.
4. The upper row's y axis is uniformly the PRIS L4 explicit-violation fraction, and the lower row's
   is uniformly the mean standardized S_syn, with the score direction and the formula support
   stated clearly.
5. Intervals bootstrap by source, chemical formula or structure group, so that 8.11 million
   structures do not create spurious precision. The relation figure expresses association only,
   not causation, probability calibration or a true failure label.

## 5. Tasks to execute once confirmed

### Task 1: freeze the data contract

**Files:** Create experiments/pu_synthesizability_20260821/fig45_reorganization_contract.py;
create outputs/20260822_fig45_reorganization_v1/FIGURE_CONTRACT.json; test
tests/test_fig45_reorganization_contract.py.

Write the tests first, verifying unique-CIF deduplication, the three-state denominators, the
independence of the L4 and S_syn routes, the absence of D7 from main-figure c, and the absence of
Jang from main-figure d. Then fix the input paths, versions, full model names, formula and
checkpoint hashes and random seeds, and generate the manifest and the panel labels.

### Task 2: correct and rerun MatterSim

**Files:** an additive runner under experiments/pu_synthesizability_20260821/; output
outputs/20260822_pu_mattersim_basin_hull_gpu_corrected/.

Correct the scaling between the full chemical formula and the whole-cell energy; rerun the
experimental/PU stratified pilot on the GPU queue; keep the Slurm record, the node, the checkpoint
and the input and output hashes. Complete the energy-zero, composition-scaling, duplicate-row,
ABSTAIN and extreme-tail QA. Only after it passes may the point enter c.

### Task 3: draw main-figure a-c

**Files:** create or refactor experiments/pu_synthesizability_20260821/plot_merged_fig45.py;
output outputs/20260822_fig45_reorganization_v1/main_fig_m.{png,pdf,svg}; test
tests/test_merged_fig45_panels.py.

Reuse the current Fig. 4a/b style; turn task1 c into the frontier, remove every D7 text, bar and
attribution, and add L4, S_syn, distance and the qualifying eHull proxy. Run the figure smoke
test, pdfinfo and a rasterized visual audit.

### Task 4: draw main-figure d-f

**Files:** reuse/port src/fig5_ranking.py; modify the additive plot module; output
main_fig_m_diagnostics.{png,pdf}; test tests/test_merged_fig45_diagnostics.py.

Generate d's two rows on a shared x, with Jang removed; generate e's two task-formula curves;
restore f's four-cell D7/D1, keeping the full ladder for the SI. Check that the bin edges, n and
CIs of the two rows agree exactly.

### Task 5: draw the SI and the PU model performance

**Files:** create experiments/pu_synthesizability_20260821/plot_si_fig45_diagnostics.py and
audit_pu_model_test_performance.py; output outputs/20260822_fig45_reorganization_v1/si/; tests
test_si_fig45_diagnostics.py and test_pu_model_test_performance.py.

Generate the SI-d per-law contribution figure; audit both models' held-out artefacts; generate one
same-template performance figure per model; and migrate the old Fig. 5a/e, the tie rate, the
wrong-hull panel, the full ladder and any validation panels that did not reach the main figure.

### Task 6: integrate into the main text only after the figures are confirmed

**Deferred files:** tex/body.tex, tex/si_body.tex, src/fig5_ranking.py, paper/FACTS.md, and the
figure cross-reference audit.

The Results are fixed as four conclusion-bearing subheadings: PRIS detects chemically damaged
structures before expensive calculations; Independent pre-DFT choices expose hard-to-synthesize
candidates; Continuous PRIS formulas connect mechanisms to independent PU scores; PRIS explains
residual disagreement among energy, phonons and experiments. Use the tie evidence of SI-A to
explain the first reason for introducing a formula in c, then the PU trade-off of c for the
second, with d/e/f completing the association, the performance and the physical explanation in
turn.

## 6. Acceptance criteria

- Reading a -> b -> c -> d -> e -> f gives "binary discovery -> the motivation for a continuous
  formula -> the PU relation -> task performance -> the energy/phonon/experiment tension".
- D7 does not appear in c; per-law contributions are in SI-d only.
- c states the two independent routes L4 and S_syn with concrete queue counts, and cannot be
  misread as joint screening.
- d's two rows share one x binning, the model sources are readable in full, and Jang has gone
  entirely.
- e retains both the difference in objective and the held-out performance of S_stab and S_syn.
- Each PU model has its own performance figure, with traceable labels, split, n, metrics and
  hashes.
- The MatterSim reference passes the scaling and energy-zero QA; if it does not, it stays out of
  the main figure.
- missing/no-verdict is never implicitly counted as a violation; the main figure contains no
  Chinese, no "reject" and no undefined "retention".
- The subfigure reference order is consistent across the main text, the captions and the SI; the
  PDF, fact list, render, SHA-256 and git diff --check audits all pass.

## 7. Where this round stops

Once the plan file is written, Tasks 1-5 begin their experiment and draft stage, but the main
text, the captions and the canonical PDFs are not modified until every data audit is complete.
Main-text integration is the final, separate stage.
