# PRIS Fig. 4--5 manuscript integration plan

## Objective

Rebuild Results section 2.4 as a continuous scientific argument that moves from
mechanism-resolved damage screening to synthesizability-aware triage, then to
property-conditioned inverse design and mechanistic deployment.  The main text
should state the scientific conclusions directly; protocol boundaries and
technical qualifications belong in Methods and Supplementary Information.

## Figure map

- Merge the former main-text Figs. 4 and 5 into a new six-panel Fig. 4.
- Move the former Fig. 6 forward to Fig. 5 without changing its scientific
  content.
- Move the displaced validation, polymorph-ranking, model-performance and
  energy--phonon--record panels into the SI at their point of first use.
- Store every canonical main-text and SI figure in `tex/figs/` with its rendered
  number in the filename.  Put the reproducible generation entry point and its
  manifest in `tex/figure_scripts/`.

## Main-text argument

1. Establish the pre-calculation screening problem with controlled damage
   (Fig. 4a) and show that the gain comes from complementary physicochemical
   mechanisms (Fig. 4b).
2. Introduce synthesizability as a natural requirement of ML materials design.
   Explain that the agents therefore derive the task-specific PRIS-derived
   synthesis score (PSS) from six PRIS-related descriptors, providing a
   continuous choice beside the discrete L1--L4 rules (Fig. 4c).
3. Only after both CGCNN-PU and MatterSim-1M-MLP-PU show the same continuous
   trend, conclude that label-blind PRIS plausibility has an unexpected
   population-level alignment with synthesizability and can pre-screen difficult
   structures (Fig. 4d).  Explain why the second model supplies a substantially
   richer pretrained structural representation and why agreement across the two
   encoders matters.
4. Show that PSS supplies high-confidence pre-DFT ordering while DFT
   \(E_{\mathrm{hull}}\) remains the all-pair energy comparator (Fig. 4e).
5. Translate this ordering into a shorter validation queue in a
   property-conditioned MatterGen--UMA inverse-design task (Fig. 4f).
6. Retain the energy--phonon--experimental-record distinction in prose and SI,
   then use Fig. 5 to explain the concrete site-complexity and wrong-occupant
   mechanisms missed by coordinate checks.

## Verification gates

- Regression tests require a--f first-citation order in the main text, complete
  main/SI figure citation, descriptive captions without result narration, and
  capitalized axis labels.
- Build SI first, regenerate `si-xr.tex`, then build the main paper.
- Audit all `\includegraphics` targets, unresolved references, figure order,
  PDF page counts, font embedding, and rendered-page appearance.

