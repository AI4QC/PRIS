# PRIS Property-Design Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify whether PRIS removes high-property materials before expensive validation and how much it shortens the validation queue in database screening and inverse crystal design.

**Architecture:** Use one mechanical-property task throughout: high bulk modulus. UMA-s-1p1 supplies a fixed-coordinate E(V) curvature proxy independent of both the database labels and the generation models. The forward experiment screens the 26,600-structure Materials Project phonon cohort and defines a good material as both high in this UMA proxy and carrying an experimental record. The inverse experiment treats MatterGen diffusion and direct LLM crystal generation as two independent generators. The LLM route never uses MatterGen as a decoder and is retained only if its local pilot is viable. Only an explicit PRIS violation removes a job. Satisfied structures and structures without a verdict both remain in the expensive-computation queue.

**Tech Stack:** Python 3.11, PyTorch, fairchem-core 2.22, UMA-s-1p1, ASE, pymatgen, pandas, SciPy, pytest, MatterGen 1.0.3, optional Ollama/MatLLMSearch.

## Frozen scientific definitions

1. The scientific target is high bulk modulus. The operational UMA score is the
   fixed-fractional-coordinate E(V) curvature proxy in GPa, not a relaxed
   equation-of-state calculation.
2. UMA uses the local checkpoint
   `<other-repo>/deps/fairchem_models/uma-s-1p1.pt`, the `omat`
   task head, and five isotropic volume factors: 0.96, 0.98, 1.00, 1.02 and
   1.04. The checkpoint SHA-256 is recorded in every run manifest.
3. The equation-of-state fit is accepted only when all five energies are finite,
   the fitted curvature is positive, the fitted minimum lies inside the sampled
   volume interval, and the fit has `R2 >= 0.98`. Failed fits do not enter a
   property-ranked candidate set or its queue-reduction denominator. They remain
   in a separate fallback-validation queue and never count as calculations saved
   by PRIS.
4. The forward source is
   `experiments/twoway_threeaxis/ladder_eval.parquet`, joined by `material_id` to
   the stored `structure_json` in
   `$PRIS_ARCHIVE/external_sources/mp_phonon_threeaxis/raw/summary_phonon.parquet`.
5. The primary forward high-property threshold is `UMA proxy >= 200 GPa`.
   Sensitivity thresholds are 150, 250, 300 and 400 GPa, together with the top
   1%, 5% and 10% of valid fits. This 200 GPa forward screening boundary is
   distinct from the 400 GPa MatterGen generation condition.
6. A forward good material requires `made == True` and membership in the chosen
   high-property set. Stability and phonon labels are reported separately and do
   not alter this definition.
7. For every PRIS rung, only an explicit PRIS violation is removed before
   expensive validation. Satisfied structures and structures without a verdict
   are both retained. Report the queue reduction, the fraction
   and count of good materials removed, Wilson intervals, and examples on both
   sides of the decision.
8. MatterGen uses the public `ml_bulk_modulus` checkpoint with a 400 GPa target,
   guidance factor 2.0 and frozen, non-overlapping random seeds. The formal
   cohort contains at least 1,024 parseable, StructureMatcher-non-equivalent
   structures, generated as recoverable 128-structure shards. Aggregation first
   removes exact structure hashes and then uses `ltol=0.2`, `stol=0.3` and
   `angle_tol=5 degrees`. The final denominator is checked only after both passes.
   Generate additional shards if duplicates leave the first eight below 1,024.
   Smaller runs are diagnostic only and never enter the formal denominator.
9. The independent LLM route uses a 400 GPa numerical target and is admitted
   only if 10 independent attempts yield at least eight parseable, eight
   physically basic-valid and eight StructureMatcher-unique candidates. Mean
   wall time per parseable and per unique basic-valid structure must each be no
   more than 120 seconds, and a 1.25-safety-factor projection to 1,024 unique
   candidates must be no more than 48 hours. Generation, parsing and archiving
   scale linearly in this projection, while the current pairwise structure
   matching scales quadratically. Freeze the service to the local Ollama
   endpoint, record the GGUF architecture and model blob digest before and after
   generation, and fail if the two identities differ. A passing ten-attempt pilot
   enters a 100-attempt intermediate gate only after its raw responses, basic
   checks, unique structures and archive have been replayed. That intermediate
   gate applies the same predeclared rates at tenfold scale: at least 80
   parseable, 80 physically basic-valid and 80 StructureMatcher-unique
   candidates, both mean per-structure times no more than 120 seconds, and the
   same 1.25-safety-factor projection no more than 48 hours. The closest
   published route is MatLLMSearch with a local Ollama-served Qwen model, but the
   prompt must state the numerical bulk-modulus objective because the current
   public MatLLMSearch prompt omits it and its oracle hard-codes a 100 GPa
   target. If the corrected direct-LLM route fails either gate, the final
   inverse experiment uses MatterGen alone and records why an LLM comparison
   was not scientifically usable.
10. Inverse design has no experimental-record label. Its primary endpoint is the
    percentage and absolute number of generated candidates removed from the
    expensive-validation queue. The kept fraction of candidates with UMA proxy
    values above 200, 300 and 400 GPa is auxiliary.
    No generated candidate is called a known good material.
11. No result from this experiment is written into `tex/` or `paper/` before the
    user approves the independent summary and manuscript-modification plan.

### Task 1: Equation-of-state core

**Files:**

- Create: `experiments/property_design_20260821/uma_bulk.py`
- Test: `tests/test_property_design_uma_bulk.py`

**Step 1: Write failing unit tests**

Test conversion from quadratic energy curvature to GPa, fail-closed handling of
negative curvature and an out-of-range minimum, and preservation of input atom
coordinates during isotropic scaling.

**Step 2: Verify RED**

Run: `pytest -q tests/test_property_design_uma_bulk.py`

Expected: collection fails because `uma_bulk` does not yet exist.

**Step 3: Implement the minimal pure functions**

Implement `scaled_atoms`, `fit_bulk_modulus`, `fit_quality`, and JSON-safe result
serialization. Model loading and batching remain behind a CLI boundary.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_property_design_uma_bulk.py`

Expected: all tests pass.

### Task 2: UMA batch inference and reference sanity check

**Files:**

- Modify: `experiments/property_design_20260821/uma_bulk.py`
- Test: `tests/test_property_design_uma_bulk.py`
- Create: `outputs/20260821_property_design/uma_reference_sanity.json`

**Step 1: Add a failing test for batch-to-structure energy alignment**

Use two small ASE structures and a deterministic fake prediction dictionary to
prove that result ordering is not mixed across structures or volume factors.

**Step 2: Implement fairchem batching**

Convert each ASE object with `AtomicData.from_ase(..., task_name="omat")`, batch
with `atomicdata_list_to_batch`, and map system energies back to the manifest.

**Step 3: Run the actual local checkpoint**

Evaluate diamond, silicon and rocksalt NaCl. Require the ordering
`B(diamond) > B(silicon) > B(NaCl)` and finite positive values. Record timing,
peak GPU memory, software versions and checkpoint hash.

### Task 3: Forward database screen

**Files:**

- Create: `experiments/property_design_20260821/forward_screen.py`
- Test: `tests/test_property_design_forward_screen.py`
- Create: `outputs/20260821_property_design/forward/manifest.json`
- Create: `outputs/20260821_property_design/forward/predictions.parquet`
- Create: `outputs/20260821_property_design/forward/summary.json`

**Step 1: Test joins and three-state accounting**

Prove that duplicate or missing material IDs fail closed, structures without a
verdict remain in the queue, and only an explicit violation counts as a saved
calculation.

**Step 2: Benchmark a stratified 128-structure pilot**

Measure throughput over the observed `nsites` distribution and set the largest
batch size that stays below 90% of available GPU memory.

**Step 3: Run the full 26,600-structure cohort**

Checkpoint every completed batch so the run is resumable. Never recompute rows
whose structure hash, model hash and protocol hash all match.

**Step 4: Produce frozen metrics**

For L1, L1-prime, L2, L3 and L4, report property-fit coverage, queue reduction,
good-material removal, experimental-record satisfaction, and enrichment for
`hull_50meV` and `dyn_stable`. List concrete high-B experimental structures that
are satisfied, show an explicit violation or receive no verdict.

### Task 4: MatterGen conditional generation

**Files:**

- Create: `experiments/property_design_20260821/run_mattergen.py`
- Test: `tests/test_property_design_generation_manifest.py`
- Create: `outputs/20260821_property_design/mattergen/manifest.json`
- Create: `outputs/20260821_property_design/mattergen/generated.extxyz`

**Step 1: Test manifest completeness and candidate identity hashing**

Require generator version, checkpoint hash, target value, guidance, seed and one
stable hash per generated structure.

**Step 2: Download only the public `ml_bulk_modulus` checkpoint**

Store it outside the repository and verify that it is a real checkpoint rather
than a Git-LFS pointer.

**Step 3: Generate and parse candidates**

Start with 4 diagnostic samples, then generate independently seeded
128-structure shards until at least 1,024 parseable structures remain after exact
hash deduplication and StructureMatcher deduplication with `ltol=0.2`, `stol=0.3`
and `angle_tol=5 degrees`. Preserve raw and parsed outputs, retain the first
deterministic representative of every equivalence class, and record all excluded
members plus shard-level and aggregate hashes.

### Task 5: LLM feasibility gate

**Files:**

- Create: `experiments/property_design_20260821/llm_feasibility.py`
- Test: `tests/test_property_design_llm_parse.py`
- Create: `outputs/20260821_property_design/llm/feasibility.json`

**Step 1: Test extraction of multiple POSCAR/CIF blocks and malformed responses**

**Step 2: Run ten independent local generations**

Use a property-explicit MatLLMSearch-style zero-shot prompt and the installed
Ollama model. Store prompts, raw responses, parse status and structure hashes.

**Step 3: Apply the admission rule**

Continue only if at least eight of ten responses are parseable, physically
basic-valid and StructureMatcher-unique, both measured per-structure costs are
at most 120 seconds, and the 1.25-guarded 1,024-structure projection is at most
48 hours. The measured wall time includes generation, parsing, StructureMatcher
deduplication and archive writing. Scale the current pairwise matching component
quadratically, while scaling generation, parsing and archive writing linearly.
A passing pilot enters a 100-attempt
intermediate gate requiring at
least 80 parseable, 80 physically basic-valid and 80 StructureMatcher-unique
candidates, the same two 120-second limits and the same guarded 48-hour
projection. The 100-attempt entry must replay the raw responses and archive from
an admitted 10-attempt artifact, then bind the recomputed gate to the same model
descriptor hash, local endpoint and 400 GPa target. Verify through Ollama model
metadata that the backend is a direct completion LLM, preserve the model blob
digest, and record that this driver does not call MatterGen or a diffusion decoder.
Otherwise stop the LLM branch and preserve any generated archive as diagnostic
only without changing the gate after seeing downstream scores.

### Task 6: Independent PRIS and UMA evaluation of generated candidates

**Files:**

- Create: `experiments/property_design_20260821/evaluate_generated.py`
- Create: `experiments/property_design_20260821/inverse_analysis.py`
- Create: `experiments/property_design_20260821/export_inverse_examples.py`
- Test: `tests/test_property_design_generated_evaluation.py`
- Test: `tests/test_property_design_inverse_analysis.py`
- Test: `tests/test_property_design_inverse_example_export.py`

**Step 1: Test inverse-design queue accounting**

Prove that the denominator is the generated cohort, not the database cohort, and
that no experimental-record terminology enters the output schema.

**Step 2: Evaluate every parsed candidate**

Run PRIS on the raw candidate first. Run UMA only for the factual analysis, while
also calculate the counterfactual number of expensive validation jobs that PRIS would have
avoided. Report the fraction of high-property candidates kept only as an
auxiliary metric.

### Task 7: Independent report and manuscript plan

**Files:**

- Create: `docs/2026-08-21-PRIS-property-design-results-zh.md`

**Step 1: Audit every number against saved rows**

Include exact denominators, missing/failed fits, uncertainty and model/protocol
hashes. Separate measured runtime from projected savings in expensive validation.

**Step 2: Select concrete examples**

Show at least one experimental high-B forward material retained by PRIS, every
experimental high-B material removed by PRIS if the count is small, and examples
of generated candidates removed before the expensive queue. Retain C11N4, BC7,
C3N2 and B2CN2 as numerical mechanism examples from forward screening. Retain
WC, BN and TiN to show why the strict site-complexity law is better used for
diagnosis than automatic deletion. Retain LiC12 as the highest-proxy experimental
L2 boundary case and report its small D3 excess numerically. Retain representative
Ir--Os--Ru and Re--Ir
MatterGen structures when the formal cohort confirms the pilot mechanism, and
retain a charge-resolved L2 violation with its actual triggering law values and
charge-assignment route. If a preferred composition has no eligible formal-cohort
candidate, keep the verified pilot example in the SI and label the deterministically
selected formal-cohort replacement as such. Make clear that these structures
illustrate individual rule decisions, whereas all performance percentages use the
complete prespecified cohorts. Export every named forward and inverse example as a
CIF together with its source identity, structure hash and exact CIF hash so that the
examples remain directly reusable in the manuscript and SI.

**Step 3: Propose, but do not apply, manuscript changes**

Specify the Results subsection, main/SI figure, Methods paragraph and Discussion
claims that the new evidence would support. Wait for user confirmation before any
canonical manuscript edit.

## Execution status on 2026-08-21

All required measurements are complete. Property-design results remain outside
`tex/` and `paper/` pending user approval.

- Forward screening evaluated 26,600 database structures. UMA produced 25,192
  valid fixed-coordinate E(V) curvature fits, and the independent report retains
  all eight frozen numerical examples: C11N4, BC7, C3N2, B2CN2, WC, BN, TiN and
  LiC12.
- The independent report gives Wilson 95% intervals for forward queue reduction,
  experimental-structure retention, inverse queue reduction and inverse
  high-proxy candidate retention. The interval parameters are also stored in the
  machine-readable inverse analysis artifact.
- MatterGen produced 1,664 parseable raw structures in 13 independently seeded
  shards. Exact hashes removed none. StructureMatcher removed 583 equivalent
  structures and left 1,081 formal unique candidates. The aggregate archive
  SHA-256 is
  `8765bfbb24e89767bc8dc6b9bc77dc7d1884ce549bee4f4734f7e7a3d3aa6f93`.
- UMA produced valid fits for all 1,081 MatterGen candidates. L4 removed 728
  explicit violations from the expensive-validation queue, reducing it from
  1,081 to 353. D7 alone reduced the queue by 67.25% at `symprec=0.01` and
  53.47% at `symprec=0.1`. The 0.5 and 0.7 angstrom distance thresholds removed
  none.
- Three formal inverse examples were exported with source provenance and exact
  CIF hashes: Re2Ir5Os7Ru2 `candidate_0939`, ReIr3 `candidate_0774` and B3W2
  `candidate_0660`.
- The first LLM diagnostic exposed an Ollama transport mismatch: this Qwen3.6
  template emitted its only text in `thinking` while `response` was empty. A
  tested deterministic fallback fixed the transport without changing any
  admission threshold. The corrected ten-attempt gate obtained 8 parseable
  structures in 123.61 seconds, but 0 passed the frozen basic-structure gate.
  The LLM branch therefore stopped before 100 attempts and did not enter the
  formal inverse denominator. Its gate artifact is
  `outputs/20260821_property_design/llm_bulk400_feasibility10_v2/feasibility.json`
  with SHA-256
  `079bae1975001a332ffb9d42ebaa37f7d74a3616037939d65725bcf17e274bf4`.
- The independent Chinese report and proposed manuscript insertion are in
  `docs/2026-08-21-PRIS-property-design-results-zh.md`. No property-design
  result has been inserted into the canonical manuscript.
