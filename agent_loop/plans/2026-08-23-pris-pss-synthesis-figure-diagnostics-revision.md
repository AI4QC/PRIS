# PRIS-PSS synthesizability narrative and the Fig. 3/4 diagnostics revision plan

## Aim of the revision

Rather than splitting the six issues into scattered patches, this round rebuilds one continuous
scholarly argument: PRIS first decides whether a structure meets testable physico-chemical
bounds and names the mechanism it violates; moving from a theoretical structure to an
experimental realisation additionally has to face synthesizability, so the agent then derives a
continuous PRIS-derived synthesis score (PSS) from PRIS-related descriptors; an independent
PU-learning model then tests the relation between PRIS and synthesizability; and finally a
genuinely executed MatterGen-UMA inverse-design task tests whether those judgements protect the
downstream validation queue and explain which structures are screened out.

## Evidence and terminology ledger

- `PRIS`: Plausibility Rules for Inorganic Structures -- discrete structural plausibility laws
  whose violations are attributable to a mechanism.
- `PSS`: the PRIS-derived synthesis score, written `S_syn` in the main text, a continuous score
  built from six PRIS-related descriptors.
- `PU learning`: the methodological basis is Jang et al., JACS 2020 (DOI `10.1021/jacs.0c07384`),
  in which a CGCNN emits a crystal-likeness score (CLscore).
- `CGCNN-PU`: a task-specific PU model following the structure-graph encoding idea above.
- `MatterSim-1M-MLP-PU`: a second PU model, an MLP head on a frozen MatterSim representation,
  used to test whether the correlation depends on the older CGCNN representation.
- `hard negatives`: proxy negatives that both PU models judge hard to synthesise; they may not be
  written as experimentally proven unsynthesisable.
- `MatterGen-UMA test`: the property-conditioned inverse-design test actually run in this work.
  MatterGen generates structures targeting 400 GPa, and UMA independently supplies a bulk-modulus
  surrogate value.

## Order of the argument in the main text

1. Fig. 4a-b first shows that PRIS combines screening efficiency with mechanistic diagnosis under
   controlled damage.
2. Introduce PSS naturally through "whether a theoretical candidate can reach experiment depends
   not only on structural plausibility but also on synthesizability", and delete the old
   sentences centred on candidate-pool size or computational cost.
3. Present PSS directly in the main text: the compact standardised linear expression first, then
   the six abbreviations and `z(x)` defined below the formula; emphasise that the six terms carry
   electrostatic stability, site parsimony, bond-valence conservation, dense packing and
   coordination-network topology respectively.
4. State the PU-learning/CLscore basis of the 2020 JACS paper first, then explain that this work
   enlarges the experimental and unlabelled structure stores and adds the MatterSim
   representation model, and only then give the 99,162 and 364,592 scale figures and the Fig. 4c
   results.
5. Fig. 4c writes the conservative mechanistic gating of L1-L4 and the continuously tunable
   screening of PSS as two complementary ways of using the work. At the same
   experimental-structure satisfaction, PSS raises the screening rate of the hard-to-synthesise
   proxy negatives.
6. Fig. 4d completes the key inference through a monotone relation consistent across both PU
   representations: no synthesis label entered the discovery of PRIS, yet its violation rate
   still falls as CLscore rises, and PSS rises with it, so the relation is not the product of one
   particular model.
7. Fig. 4e explains why the continuous PSS can also handle the same-composition ranking on which
   the discrete laws frequently tie.
8. Fig. 4f states plainly that "to test this use directly, we actually ran MatterGen". Describe
   the experimental design first, then the 1,081 unique structures and the UMA predictions, and
   finally the queue reduction and the retention of high-property candidates.
9. Fig. 4f adds a mechanistic diagnosis of the structures PSS screens out: within the current
   two-feature support domain, all 61 screened structures have every site inequivalent at the
   0.01 symmetry tolerance and lie in a larger volume-per-atom range; all of them pass the 0.7 A
   distance threshold, yet all of them trigger D7. Show this difference through one screened
   structure and one retained high-property structure.

## Figure changes

- Fig. 3a: first reproduce the S4 clipping with a projected-bounds regression test, then leave
  enough projection margin in the 3D view for the atomic radii and the cell lines, without
  changing the composition of the other four damage classes.
- Fig. 4f: keep the existing PSS curve and the L1-L4 operating points, and inset two small
  crystal schematics with very short data labels in the blank lower-left area; the full
  statistics go into the main text and the status file rather than occupying the figure with
  long explanations.
- Every new figure keeps the main text's uniform Arial, grid-free, semi-transparent fill and
  same-colour dark border style, and is copied into `tex/key-file/`.

## Tests first

1. In the LaTeX integration test, first assert that the main text contains the explicit PSS
   formula, the definitions of the six abbreviations, the 2020 JACS citation, the first
   introduction of the enlarged datasets, and the direct experimental statement "we ran
   MatterGen".
2. Add a Fig. 3 S4 projection-margin test, so that the current clipping implementation fails
   first.
3. Add a Fig. 4f diagnostic-data test, pinning the 61 screened structures, the 140/140 retention
   of high-property candidates, the 61/61 D7 violations, the 61/61 passes of the 0.7 A distance
   threshold, and the traceable provenance of the representative structures.
4. Add tests for the Fig. 4 status file and output assets, ensuring the diagnostic statistics and
   the two structure thumbnails genuinely reach the figure rather than appearing only in the text.

## Specific files

- Modify: `tex/body.tex`, `tex/methods.tex`, `tex/si_body.tex`, `tex/refs.bib`, and if necessary
  `tex/front_body.tex` and `tex/front_meta.tex`.
- Modify: `src/fig3_anatomy.py`.
- Modify: `experiments/pu_synthesizability_20260821/plot_merged_fig45_nature.py`, extracting a
  reusable inverse-design diagnostic function if needed.
- Add or extend: `tests/test_tex_pris_section24_integration.py`, and the test files for Fig. 3 and
  Fig. 4.
- Rebuild: `paper/figs/fig3_anatomy.*`, `tex/figs/fig4_validation_synthesis.pdf`, and the main and
  SI PDFs, keeping the key figures and numbered PDFs in `tex/key-file/` in sync.

## Acceptance

- The targeted tests and the full set of related tests all pass.
- Both the main text and the SI compile without error.
- A page-by-page render check confirms no clipping of S4 in Fig. 3a, no overlap between the
  Fig. 4f thumbnails and the curve, and no PSS formula overflowing the column width.
- A first-occurrence audit confirms that every model, dataset and number has an antecedent, and
  that the main-text subfigure references still run a -> b -> c -> d -> e -> f.
- `git diff --check` passes, and no unrelated user change in the working tree is overwritten.
