# Post-mechanism Application Narrative Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the manuscript after Results section 2.3 so that the later analyses explain several unresolved or disputed crystal-structure phenomena through the five physicochemical mechanisms, with practical screening value presented as a consequence of that explanatory power rather than as a list of tool functions.

**Architecture:** Treat section 2.4 as the evidence that the learned constraints reflect chemistry rather than one damage recipe; section 2.5 as an explanation for why plausibility, thermodynamic stability, dynamical stability and experimental occurrence can disagree; and section 2.6 as a physicochemical account of three concrete puzzles---generated structures that pass distance checks yet have anomalously low symmetry, artificial ordering that inflates site complexity, and chemically wrong site assignments that survive coordinate-based validation. Practical pre-relaxation screening follows from these explanations. Draft the Results logic first, then rewrite the Introduction roadmap, Discussion synthesis and abstract against the same puzzle--explanation map. Preserve the existing figures and numerical evidence unless a factual audit shows that a display must change.

**Tech Stack:** LaTeX manuscript sources, `paper/FACTS.md` as numerical arbiter, Pandoc word counts, Tectonic build, Poppler/Ghostscript/PyMuPDF PDF checks.

## Result-allocation map

| Result or test | Crystal question or puzzle | Explanatory inference | Main-text decision |
|---|---|---|---|
| Omitted-damage tests | Are the laws merely fingerprints of the artificial damage procedures? | Flexible rules can learn the procedure, whereas simple physicochemical bounds retain signal under unfamiliar failures | keep in section 2.4 as the licence for later explanations |
| MatterSim relaxation of damaged parents | Were the benchmark changes physically consequential or only numerical perturbations? | The main coordinate perturbations are energetically severe, although this is not ground truth for each PRIS verdict | keep briefly in section 2.4; detailed values remain in SI/caption |
| Binary same-composition ranking | Why can a plausible structure still fail to be the most stable or experimentally observed polymorph? | PRIS is a gate for implausibility, not a general polymorph ranker | lead section 2.5 |
| Separate stability and synthesis fits | Why do stability and experimental occurrence favour different structures? | The same structural measurements carry different information about energy and experimental history | keep in section 2.5 |
| MP phonon population | Why do hull energy, phonons and experimental occurrence disagree? | Plausibility, thermodynamic stability, dynamical stability and experimental identification are empirically non-equivalent axes | make the central scientific inference of section 2.5 |
| Seven-generator comparison | Why do generated structures pass distance filters yet remain anomalously low-symmetry? | Excess distinct-site complexity, associated with whether symmetry is imposed, explains a failure hidden by minimum-distance checks | lead section 2.6 as a crystal-generation puzzle |
| GNoME similar-element merge | Can apparently stable low-symmetry entries reflect artificial ordering rather than an irreducibly complex lattice? | Merging similar elements generates a testable simpler-parent explanation for many sampled entries | keep in section 2.6 as a structural-ordering puzzle; state that it does not prove unsynthesizability |
| MatterGen relaxation with no D1-rejected group | Does this sample establish an energy difference between verdict classes? | It does not; no D1-rejected group exists | compress to one boundary sentence; detail in caption/SI |
| Coordinate-based checks versus site swaps | How can a crystallographically acceptable model still place the wrong element on a site? | Coordinates can remain acceptable while charge, electrostatics and bond valence expose a chemically wrong assignment | keep in section 2.6 as a validation puzzle |
| Five evaluable falsified depositions | Do historical falsified entries establish general sensitivity? | No; the applicable sample is too sparse | remove from main narrative; retain in SI |
| Runtime | Can these explanatory tests be applied before expensive relaxation? | The named mechanisms can be checked at catalogue scale before DFT and phonons | close section 2.6 as derived practical value, not the main claim |

### Task 1: Fix the claim map and evidence allocation

**Files:**
- Read: `tex/body.tex:350-677`
- Read: `paper/FACTS.md:545-782`
- Read: `tex/si_body.tex:1086-1843`

**Steps:**
1. Record the scientific question, new inference, decisive evidence, practical meaning and boundary for every post-mechanism test.
2. Classify each item as core discovery, necessary support, qualification, robustness or edge case.
3. Keep unfamiliar-damage tests, the disagreement among plausibility/stability/phonons/experimental occurrence, generated-structure low symmetry, GNoME ordering and chemical-site validation in the main text.
4. Compress or relocate checks that only defend an earlier claim, especially the null MatterGen relaxation contrast and the five evaluable falsified depositions.
5. Verify every retained number and boundary against `paper/FACTS.md`.

### Task 2: Rebuild Results sections 2.4--2.6

**Files:**
- Modify: `tex/body.tex:350-617`

**Steps:**
1. Rewrite section 2.4 around the question of whether the rules capture chemistry or corruption-generator fingerprints.
2. End section 2.4 with the inference that mechanism-based constraints retain information under unfamiliar errors, while the energy check only establishes perturbation severity.
3. Rewrite section 2.5 around the long-standing mismatch among thermodynamic stability, dynamical stability, structural plausibility and experimental occurrence.
4. Make binary ties evidence that PRIS is a screen rather than a polymorph ranker; use separate scores and the phonon population to establish the boundary.
5. Rewrite section 2.6 around three crystal-structure puzzles: distance-filtered generators with anomalous low symmetry, artificial ordering that inflates the number of distinct sites, and chemically wrong site assignments that survive coordinate-based checks.
6. Make explicit how each puzzle follows from one or more of the five equally important mechanisms in section 2.3.
7. Present catalogue-scale pre-relaxation screening only after the explanatory conclusions have been established.
8. Replace appended defensive detail before adding significance sentences; retain all conclusion-changing caveats.
9. Run Pandoc word counts and keep the revised combined sections no longer than the current combined baseline unless added text replaces equivalent detail.

### Task 3: Align Introduction and Discussion

**Files:**
- Modify: `tex/front_body.tex:4-66`
- Modify: `tex/body.tex:619-677`

**Steps:**
1. Make the Introduction pose the crystal puzzles that current distance, energy and coordinate checks leave unresolved: anomalous low symmetry, artificial ordering, disagreement among stability measures and experimental occurrence, and chemically wrong site assignments.
2. End the Introduction by proposing the discovered laws as a common physicochemical lens for these puzzles, without listing figure-level results.
3. Rewrite Discussion as synthesis: autonomous refutation, physical meaning, explanations of the crystal puzzles, derived screening value, then limitations.
4. Ensure every repeated claim has a different role: introduce, demonstrate, synthesize or bound.

### Task 4: Rewrite the abstract last

**Files:**
- Modify: `tex/front_meta.tex:11-29`

**Steps:**
1. Preserve the exact phrase `On held-out data` and the explicit Pauling comparison.
2. Replace the current application inventory with one connected explanatory consequence: PRIS accounts for discrepancies hidden by minimum-distance, energy, phonon or coordinate-only labels through contact, packing, electrostatic, bond-valence and chemical-order mechanisms.
3. State the derived value: PRIS is a rapid, interpretable screen before stability and synthesis assessment, not their surrogate.
4. Keep the abstract at no more than 200 words and retain all evidence boundaries.

### Task 5: Consistency, evidence and rendering verification

**Files:**
- Verify: `tex/front_meta.tex`
- Verify: `tex/front_body.tex`
- Verify: `tex/body.tex`
- Verify: `tex/si_body.tex`
- Verify: `paper/FACTS.md`

**Steps:**
1. Run `git diff --check` on all changed manuscript files.
2. Recompute the headline values used in the revised prose from the staged CSV, JSON and parquet sources.
3. Check terminology, British spelling, abstract length, figure-caption length, cross-references and the ban on Results mini-headings/bold prose.
4. Run `bash tex/build.sh` and require successful main/SI builds.
5. Validate PDF freshness, Ghostscript parsing, embedded fonts, unresolved references and banned visible terms.
6. Inspect all changed pages visually and require independent fact, language and render audits with no P0/P1 findings.
