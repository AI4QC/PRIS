# PRIS Fig. 4-5: journal-style redraw and revision of the application storyline

## Objective

Rework the current additive c-f sketch into a complete a-f main figure that includes the original
Fig. 4a and b, and bring it into line with the Arial, white background, grid-free style and the
legacy PRIS palette already used in the main text. The storyline changes from "L4 is a
conservative application gate" to: binary PRIS supplies a physico-chemical explanation law by law
together with controlled-damage evidence, while the continuous synthesis score (S_{\rm syn})
derived from descriptors gives stronger pre-DFT queue compression at the same experimental-
structure retention.

## Main figure a-f

1. **a: overall comparison on the controlled-damage benchmark.** Keep the distance threshold,
   SMACT and L1-L4 results of the original Fig. 4a, highlighting L4's 87.9% damage detection at
   about 83% experimental-structure satisfaction; restore the detection figure at the top of each
   bar and the satisfaction figure above it, and remove the unnecessary long explanatory
   sentences.
2. **b: mechanistic complementarity across the five damage classes.** Keep the heatmap of the
   original Fig. 4b, the S1-S5 order, the value in every cell, the separators between groups and
   the black box marking the blind spot of distance screening. The small type here carries the
   complete measured values only, not long explanatory text.
3. **c: a fair queue comparison at the application level.** The x axis is experimental-structure
   satisfaction and the y axis is PU hard-negative screening. Show all four discrete law gates
   L1, L2, L3 and L4, then draw the (S_{\rm syn}) frontier, the distance baseline and the
   MatterSim basin-hull proxy. The four law points are independent operating choices and do not
   indicate a cascade. Highlight (S_{\rm syn})'s 83.7% against L4's 51.9% at the same 80.7%
   satisfaction -- +31.8 percentage points, shortening the queue from 175,433 to 59,517. The 95%
   satisfaction point serves as a conservative operating point, and "surpasses across the board"
   must not be written where the advantage does not hold over every retention range.
4. **d: correlation with the independent PU scores.** The two rows share a within-model CLscore
   percentile; each row draws three lines: CGCNN-PU, MatterSim-1M-MLP-PU, and the per-decile mean
   of the two. Write the model names out in full, with no undefined A/B or Jang; remove the grid
   and the long explanatory text, and use transparency to separate points from intervals.
5. **e: independent validation of the synthesis formula.** Keep only the held-out synthesis
   ranking of (S_{\rm syn}), compared in the same panel against the hull-energy baseline. The
   stability score is no longer in the main figure: its discriminating power under the full
   application convention is limited, and it easily leads a reader to take it for a
   synthesizability or phonon-stability score.
6. **f: the physico-chemical explanation.** Keep the four cells of on-hull/metastable x
   no-imaginary/imaginary, comparing recorded against computed-only D1/D7; remove the large
   sample counts under each bar and move them to the caption or SI.

### How the stability score is handled

The frozen stability score improves the energetic ranking only among same-composition pairs with
a clear energy separation (about 52.6% accuracy over all pairs, about 75.9% once the gap is at
least 0.10 eV atom$^{-1}$), while under the whole-pool, median-imputed application convention its
AUC is about 0.707 and it screens out only about 3.4% of the PU proxy at 95% experimental-
structure retention. It is not phonon stability, not experimental occurrence rate, and not a
synthesizability score; putting it in this paper would force two different questions together.
It is therefore removed from the main text, the main figures and the SI narrative in this round,
with only the raw result files kept as a research record.

## The SI model-performance figure

The two PU models are merged into one standard classification-model figure (a 2x2 layout), one
row per model:

- left: the ROC curves of the 50 bags, the mean ROC curve and the 95% bag interval, annotated
  with the mean ROC-AUC;
- right: the confusion matrix of the validation labels at the decision threshold 0.5 (TP, FP, TN,
  FN, with percentages);
- the caption states explicitly that the labels are experimental positives and sampled
  pseudo-negatives, and that OOB is a whole-pool scoring protocol, not an independent test;
- if the server recomputation did not emit the raw predictions, do not back out a confusion
  matrix from the AUC; keep the validation-only note instead.

## Visual gates

- Use the Arial, legacy blue/orange/red/green/grey palette and 0.6-1.0 pt line widths already in
  the main text.
- White background, no grid, and only the necessary left and bottom axis lines.
- The main figure carries no long explanations, model paths, n values or internal status words;
  exact counts and protocols go in the caption or SI.
- Panels a-f must all be present, and the figure reads a -> b -> c -> d -> e -> f.

## Data conventions

Fig. 4a and b are the 440-parent controlled-damage benchmark; Fig. 4c is the whole-pool PU
hard-negative proxy cohort. The two are never conflated. The matched-point results for
(S_{\rm syn}) come from the current frozen formula pipeline, and the full
observed/median-imputed fractions go in the SI.

This round produces only the additive figure, the SI figure and the plan and audit files; it does
not modify `tex/`, `paper/` or the canonical PDF. Integration into the main text comes after the
figures are confirmed.
