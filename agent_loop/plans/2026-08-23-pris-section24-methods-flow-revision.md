# PRIS Section 2.4 and Methods Flow Revision Implementation Plan

> **For Codex:** Execute this plan with test-driven manuscript checks and rebuild both LaTeX deliverables before completion.

**Goal:** Turn Section 2.4 into one continuous evidence chain from pre-calculation screening to mechanistic explanation, connect Fig. 4 to Fig. 5 explicitly, and reduce every Methods subsection to one paragraph without losing reproducibility.

**Narrative architecture:** Preserve the four Results subsection titles and all figure-panel orders. In Section 2.4, make each paragraph close the question opened by the preceding paragraph and motivate the next panel. Keep scientific interpretation and decisive numbers in the main text. Keep cohort construction, model settings, thresholds, missing-data conventions, bootstrap details and implementation audits in the Supplementary Information.

**Files:**

- Modify: `tex/body.tex`
- Modify: `tex/methods.tex`
- Modify only if a displaced detail lacks a home: `tex/si_body.tex`
- Modify: `tests/test_tex_pris_section24_integration.py`
- Verify: `tex/main.pdf`, `tex/si.pdf`

## Task 1: Freeze the requested structure in tests

Add checks that the Fig. 4 evidence chain proceeds through screening, mechanism, synthesis-aware triage, cross-model association, polymorph prioritisation and inverse design before the Fig. 5 structural diagnosis. Require the unexpected synthesizability conclusion to occur only after panel d. Require the five Methods subsections to remain unchanged, contain one prose paragraph each and stay within a compact word budget.

Run the targeted test and confirm the new assertions fail on the current manuscript.

## Task 2: Rebuild Section 2.4 as a claim-escalation chain

Revise paragraph openings and closings so that panel a creates the mechanism question answered by b, b creates the synthesis-triage question answered by c, c motivates the two-representation test in d, d motivates prioritisation in e, and e motivates inverse design in f. State the conservative PRIS versus tunable PSS choice explicitly. Keep PU-model performance details in SI while giving enough main-text evidence to establish that both models are credible.

After Fig. 4, connect the energy--phonon--record contradiction to the need for structure-level mechanisms. Introduce Fig. 5 as the structural explanation of what decision-level screening exposed, then connect its panels in order through generator symmetry, GNoME ordering, relaxation controls, the merge intervention, wrong-site assignments and computational cost.

## Task 3: Compress Methods and preserve technical provenance

Rewrite each of the five Methods subsections as exactly one paragraph. Retain only the design choices needed to understand the evidence: source populations and split discipline, descriptor families and verdict semantics, controlled-damage and law-selection logic, external/PSS/PU/inverse-design evaluation, and autonomous-agent oversight. Point to the exact SI notes for thresholds, feature definitions, operators, coefficients, model settings and statistics. Add a concise SI sentence only if a displaced reproducibility detail is not already present there.

## Task 4: Verify scientific and editorial integrity

Run the focused and figure tests, build `main.pdf` and `si.pdf`, and inspect the logs for undefined references, overfull boxes and fatal errors. Confirm all main and SI figures are cited outside captions in panel order, captions contain no result replay, Results still have only the original four subsection titles, prose contains no semicolons, and every Methods subsection is a single paragraph. Render and inspect pages covering Section 2.4, the Fig. 4--5 transition and Methods.
