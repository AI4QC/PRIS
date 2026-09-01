# PRIS Fig. 5 and Crystal-Contradiction Narrative Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild Fig. 5 and restore the literature-grounded stories behind the five crystal-structure contradictions, while preserving the current Fig. 4, the four existing Results subsection titles and the manuscript's discovery-to-mechanism-to-application narrative.

**Architecture:** Keep Fig. 5 as one six-panel deployment-and-explanation figure. Move the charge-assignment coverage flow from the main figure to a new Supplementary panel, strengthen the relaxation comparison with L1--L4, and add a paired wrong-site panel. Reorder the Introduction and Section 2.4 so each contradiction begins with a documented field problem, then presents the PRIS test and mechanism. Cite original studies for events and datasets, later reanalyses for disputes, and this work only for new PRIS evidence.

**Tech Stack:** Python, pandas, NumPy, matplotlib, ASE/pymatgen where already used, LaTeX/BibTeX, pytest, latexmk.

## Task 1: Freeze figure evidence and citation boundaries

**Files:**
- Read: `src/fig6_deployment.py`
- Read: `tex/front_body.tex`
- Read: `tex/body.tex`
- Read: `tex/refs.bib`
- Create or update only if needed: `paper/data/fig6_*.csv`, `paper/data/fig6_*.json`

1. Verify all Fig. 5 values against their source tables, including the 500-structure MatterGen ladder and the parent-controlled wrong-site cohort.
2. Record exact denominators and define abstentions separately from satisfaction/failure.
3. Verify bibliographic metadata and attribution for Pauling-rule audits, distance validity, GNoME, A-Lab and its reanalyses, Harrison's falsified structures, PLATON/checkCIF, polymorph stability and phonon datasets.

## Task 2: Rebuild Fig. 5 and move the coverage flow to SI

**Files:**
- Modify: `src/fig6_deployment.py`
- Modify or add: `src/si_figs.py` or a dedicated SI figure script
- Modify: `tex/figure_scripts/generate_all.py`
- Modify: `tex/figure_scripts/figure_manifest.json`
- Add/modify: focused figure tests under `tests/`

1. Preserve panel a.
2. Move old panel b to SI and rename its outcomes as charge-dependent-rule coverage, not structural failure.
3. Move old panels c--e to b--d. Redesign c to show L1--L4 coverage/failure together with relaxation energy, so the panel directly establishes that small relaxation energy does not imply physicochemical plausibility.
4. Add panel e: a fixed-coordinate wrong-site pair plus the paired detection comparison for cation--cation and cation--anion exchanges.
5. Preserve panel f and the existing Nature-style visual language: no panel titles, no grids, restrained text, consistent colours, legible labels and vector output.
6. Regenerate and visually inspect the main and SI figures.

## Task 3: Restore the five literature-grounded stories

**Files:**
- Modify: `tex/front_body.tex`
- Modify: `tex/body.tex`
- Modify: `tex/refs.bib`
- Modify: `tex/si_body.tex`
- Modify: `tex/methods.tex` only where the moved SI panel requires a concise method pointer

1. Make the five field problems contiguous in the Introduction before presenting the detailed applications.
2. For each problem, use the sequence: documented observation or controversy, unresolved mechanistic question, PRIS test, quantitative result, physicochemical explanation.
3. Restore the agent-refutation story for polymorph ties without treating a tie as an error.
4. Distinguish GNoME's original report from later low-symmetry criticism, and distinguish A-Lab's original claims from later order/disorder reanalyses.
5. Present Harrison's genuine-diffraction/wrong-metal episode accurately; do not claim that checkCIF or PLATON never raised chemical alerts.
6. Use the Fig. 5 d-to-e bridge: merging similar labels restores equivalence, whereas wrong-site assignment changes chemical identity at fixed coordinates.
7. Update captions so they explain what each panel contains and the comparison protocol, without numerical results or discussion.

## Task 4: Integrate SI and remove duplicated or defensive prose

**Files:**
- Modify: `tex/si_body.tex`
- Modify: `tex/methods.tex`
- Modify: relevant tests

1. Add the moved charge-coverage panel and the detailed per-law wrong-site breakdown to SI.
2. Keep applicability details and technical caveats in SI where they interrupt the main story.
3. Preserve all current substantive Fig. 4/PSS/PU-learning/inverse-design results and the existing 2.1--2.4 subsection titles.
4. Ensure main-text terminology is consistent: satisfaction, damage detection, plausibility, synthesis score and experimental record.

## Task 5: Build and audit the complete deliverable

**Files:**
- Verify: `tex/main.pdf`
- Verify: `tex/si.pdf`
- Verify: all staged figure PDFs

1. Run focused figure and manuscript-integration tests.
2. Regenerate all staged figures.
3. Build main text and SI from a clean LaTeX auxiliary state without deleting user source data.
4. Check undefined citations/references, missing figure citations, a--f reference order, bibliography consistency and PDF page rendering.
5. Perform a final prose pass for logical transitions, unintroduced terms, unnecessary self-limitation and AI-like repetition.

**Working-tree note:** The current manuscript and figures contain extensive uncommitted user work. This implementation is therefore performed in place, as authorized, with narrowly scoped patches and without resetting or overwriting unrelated changes.
