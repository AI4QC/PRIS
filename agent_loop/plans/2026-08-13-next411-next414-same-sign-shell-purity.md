# NEXT411--NEXT414 Same-sign shell purity implementation plan and freeze

> **For Codex:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`
> and execute this plan task by task without changing the frozen scientific
> choices after any feature value is observed.

**Goal:** Prospectively test one composition-plus-initial-geometry rule that
rewards an opposite-sign coordination shell free of closer same-sign periodic
neighbours.

**Architecture:** Reuse the frozen NEXT19 formal-charge-sign inference and
opposite-sign periodic Voronoi graph.  Convert its incident edge lengths into
one shell radius per site, compare that radius with the nearest same-sign
periodic neighbour, and expose exactly one fixed lower-tail order statistic.
Run the unchanged label-blind support, invariance, nondegeneracy, and novelty
gates before any outcome or formal full-table build is permitted.

**Tech stack:** Python, NumPy, ASE, pymatgen, pandas, pytest, and the existing
NeoPauling strict-loop infrastructure.

## Boundary and additive status

This is an additive exploratory branch.  It does not replace or modify any
existing script, result, report section, Pauling rule, or manuscript text.  Its
executable quantity may use composition and exactly one raw, initial,
unrelaxed periodic geometry.  It may use frozen empirical oxidation-state
tables, an electronegativity fallback, and deterministic periodic geometry.
It may not execute or read DFT, energy, force, stress, a learned proxy, an ML
potential, a relaxation, a trajectory, a later geometry, or a
same-composition alternative.  Discovery outcomes are permitted only as
offline labels after all label-blind and formal-build gates pass.  Validation
and replication remain sealed.

## Literature and repository audit

Langer and Kohlmann's 2026 test of Beck's extended coordination-number rule
uses distance ordering as an explicit coordination sanity condition: a
candidate cation coordination sphere is rejected when another cation lies
closer to the central cation than an included anion.  Their diagnostic audit
also separates structures in which a coordinating anion is closer to another
anion than to the central cation.  The reported ECN agreement changes from
50.6% in 705 structures with such an anion--anion intrusion to 79.6% in 333
structures without it.  These descriptive observations motivate a
prospective, energy-blind distance-order test; they do not establish its
predictive performance.

- https://journals.iucr.org/b/issues/2026/02/00/ra5166/
- DOI: `10.1107/S2052520626001794`
- https://github.com/Niklas-Langer/ExtendedCoordinationNumberRule-Publication

Three formulations were considered before observing any candidate value:

1. a cation-centred test matching the coordination-sphere construction;
2. an anion-endpoint test matching the paper's diagnostic split; and
3. a charge-sign-symmetric site test applying the same ordering to both ends.

The third is frozen because an initial-geometry plausibility law should not
choose one charge sign opportunistically, and because it subsumes the two
stated intrusion mechanisms with one definition.  The paper's exact
`C_gap = 1-d_gap2/d_gap1` alternative was rejected before implementation: the
repository already has the equivalent local shell-gap mechanism in P6/P6c.
The failed NEXT407 ECN statistic is not relaxed, rounded, or restricted to a
favourable chemistry here; this candidate needs charge signs but never integer
site valences or ECN classes.

CodeGraph and literal-text audits found absolute same-species/polyanion
separation, opposite-sign minimal cages, local shell gaps, packing, force
closure, bond-valence, and electrostatic features.  They found no feature that
compares every site's nearest same-sign periodic distance with its incident
opposite-sign Voronoi shell radius and aggregates the resulting dimensionless
shell-purity margin.

## Frozen formula

Apply the existing geometry-only firewall and reduced-cell validation.  Infer
one neutral formal-charge assignment with NEXT19; only `sign(z_i)` is used.
Build the unchanged opposite-sign periodic Voronoi multigraph.  For every site
`i`, define its incident opposite-sign shell radius

```text
R_i = max_(e incident on i) d_e.
```

Every site must have at least one incident opposite-sign edge.  Search exact
periodic neighbours only out to `R_i`; this is sufficient because the final
ratio is clipped at one.  Let

```text
D_i = min distance from i to a nonzero periodic image/site k
      with sign(z_k) = sign(z_i) and distance <= R_i,
```

and set `D_i=R_i` when that finite-radius search finds no same-sign neighbour.
The site purity is

```text
u_i = min(1, D_i/R_i),       0 < u_i <= 1.
```

The one and only candidate feature is

```text
sssp_same_sign_shell_purity_q10
    = round_1e-10(quantile_inverted_cdf({u_i}, 0.10)).
```

`quantile_inverted_cdf` is the empirical order statistic selected by NumPy's
`method="inverted_cdf"`.  It is unchanged by exact replication of the site
population.  The only allowed direction is `protected_high`: a value of one
means the lower tail has no same-sign neighbour inside its opposite-sign
Voronoi shell, while smaller values measure the worst prevalent intrusion.
No alternative centre sign, graph, cutoff, neighbour count, quantile,
aggregation, normalization, transform, direction, subgroup, conjunction, or
outcome-conditioned exception is available.

Malformed/nonperiodic geometry, failed or nonneutral sign inference, a
missing charge sign, missing incident opposite-sign contacts, zero distances,
or a nonfinite intermediate fails closed.  Symmetry metadata and every
outcome-like field are rejected by the existing geometry firewall.

## Label-blind admission gates

NEXT411 uses the unchanged deterministic 80+80 discovery-`x0` probe and all
numeric prior label-free controls through NEXT404.  It recomputes NEXT403 as a
same-row control and may recompute the already closed NEXT407 statistic solely
as an additional same-row label-free novelty control.  It opens no label,
endpoint, validation, or replication field.  Both sources must pass:

1. at least `72/80` finite supported rows;
2. exact domain `0 < feature <= 1`;
3. at least 20 values distinct at `1e-10`;
4. maximum error `<=1e-8` under the frozen equivalent-representation suite;
5. maximum adequate absolute Spearman correlation `<0.90` against all prior
   label-free controls, requiring at least 40 jointly finite rows per control.

Failure terminates this candidate without constructing NEXT412 or reading an
outcome.  A failed gate may not be repaired by changing the formula.

## Conditional formal build and outcome audit

If and only if every label-blind gate passes, NEXT412 builds exactly the one
frozen feature on complete SCIGEN and WyFormer discovery `x0` partitions.
Each source must have at least `0.95` finite coverage.  The builder records
immutable input, source, and output hashes and certifies all boundary flags.

If and only if both formal coverage gates pass, NEXT413 applies the unchanged
discovery-only rejected-extreme audit, reduced-formula five folds,
coverage/class requirements, and the single `protected_high` direction.
Validation and replication remain sealed.  Failure on either source terminates
the branch.  Only a cross-source-eligible frozen hypothesis may authorize a
NEXT414 formula search; otherwise NEXT414 must not exist.

## TDD execution tasks

### Task 1: Pure kernel contract

**Files:**

- Create: `tests/test_next411_same_sign_shell_purity.py`
- Create after the test fails: `src/next411_same_sign_shell_purity.py`

Write tests for the exact order statistic, monotonicity under an introduced
same-sign intrusion, permutation and exact-replication invariance, malformed
input failure, the single feature name/direction, and boundary flags.  Run

```text
python -m pytest \
  tests/test_next411_same_sign_shell_purity.py -q
```

first and require a missing-module failure.  Then write the minimum pure
kernel needed to make those tests pass.

### Task 2: Periodic structure adapter

Extend the same test file first with a real distorted ionic cell and frozen
equivalent representations.  Require geometry-only input, NEXT19 charge-sign
inference, NEXT19 Voronoi edges, exact finite-radius same-sign neighbour
search, fail-closed rows, and one output column.  Observe the new test fail,
then implement the adapter and rerun the focused tests.

### Task 3: Label-blind probe

**Files:**

- Create: `tests/test_next411_sssp_label_blind_probe.py`
- Create after the test fails:
  `experiments/next411_sssp_label_blind_probe.py`
- Conditional output only:
  `experiments/next411_sssp_label_blind_probe_result.full.json`

Test first that the interface has no outcome/later-geometry parameter, the
selection and gates are inherited unchanged, and source hashes cover the full
execution surface.  Observe the missing-module failure; implement the probe;
run focused tests; then execute exactly once on the frozen 80+80 inputs.

### Task 4: Conditional continuation and reporting

If NEXT411 fails any gate, stop before NEXT412 and append the negative result
to `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.  If it passes,
test and implement NEXT412, require complete discovery coverage, then and only
then test and run NEXT413.  In either path, preserve canonical documents and
run the focused branch tests followed by the full regression suite.

