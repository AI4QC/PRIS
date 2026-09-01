# NEXT371--NEXT374 periodic CHARDI return-consistency design

**Status:** frozen before feature values or discovery outcomes are opened.

## Purpose and boundary

This additive branch tests one literature-derived, geometry-only structure
validation hypothesis that is absent from the current repository: the backward
charge-return criterion of CHARDI.  It does not replace any existing script,
report, rule, or paper text.

The executable path may read only composition and one initial, raw, unrelaxed
periodic geometry.  It must not calculate or consume DFT energies, forces,
stresses, relaxed structures, trajectories, learned energy/force/stress
proxies, MLIPs, model potentials, or later geometries.  Outcome labels remain
sealed until this design, the implementation, the label-blind engineering
probe, and the full formal feature build have passed their frozen gates.
Validation and replication remain sealed throughout NEXT371--NEXT374.

## Prior art and missing mechanism

CHARDI distributes the formal charge of each polyhedron-centering atom among
its vertex atoms using Hoppe effective-coordination bond weights.  It then
returns the vertex over/under-bonding ratios to the centering atoms.  Agreement
between input and returned centering-atom charges is the method's internal
structure-quality criterion (Nespolo et al., Acta Cryst. B72, 51--66, 2016,
doi:10.1107/S2052520615019472; Nespolo & Guillot, J. Appl. Cryst. 49,
317--321, 2016, doi:10.1107/S1600576715024814).

The repository already contains Hoppe ECoN/MEFIR site summaries and NEXT19's
forward, worst-anion Pauling/CHARDI-style mismatch.  It does not contain the
CHARDI backward return followed by a cation input/output MAPD.  NEXT371 tests
only that missing quantity.  NEXT19, ECoN/MEFIR, bond-valence, and every prior
formal feature through NEXT367 are novelty controls, not inputs to the rule.

## Frozen applicability domain and graph

The structure must have a neutral NEXT19 formal-valence assignment, both
charge signs, and exactly one chemical species among the negative-valence
sites.  The latter restriction makes every cation-centred polyhedron
homoligand, so the published non-recursive CHARDI equations apply without an
invented heteroligand approximation.  Structures outside this domain abstain.

Use the existing NEXT19 opposite-sign periodic Voronoi multigraph.  Every
cation and anion must have at least one incident periodic edge.  The graph is
fixed before distances are weighted; no outcome-informed cutoff, radius,
threshold, or neighbour parameter is introduced.

## Frozen formula

For cation `c`, let its incident periodic distances be `d_ce`.  Starting from
`m_c^(0) = min_e d_ce`, iterate the published Hoppe weighted mean

```
w_ce^(t) = exp(1 - (d_ce / m_c^(t-1))^6)
m_c^(t)  = sum_e d_ce w_ce^(t) / sum_e w_ce^(t)
```

until the maximum mean-distance change is at most `1e-12`, with a hard limit
of 10,000 iterations.  Non-convergence abstains.  At convergence, normalize
the final bond weights within each cation star,

```
p_ce = w_ce / sum_f w_cf,
dq_ce = |q_c| p_ce.
```

The charge received by anion `a` and the charge returned to cation `c` are

```
Q_a = sum_(e incident on a) dq_ce,
Q_c(returned) = sum_(e incident on c) dq_ce |q_a| / Q_a.
```

The sole frozen hypothesis is the site-level fractional CHARDI MAPD

```
pchardi_cation_return_mapd
  = mean_c | |q_c| - Q_c(returned) | / |q_c|.
```

Its direction is `protected_low`.  The value is dimensionless, nonnegative,
and is quantized only at the final structure aggregate to a `1e-10` grid.
No quantile, alternate direction, second feature, conjunction, or transformed
version may be added after seeing values or outcomes.

## Label-blind engineering and novelty gates (NEXT371)

Select exactly 80 deterministic discovery rows from each of SCIGEN and
WyFormer using the existing frozen selector.  Open raw discovery geometry and
label-free controls only.  Require on both sources:

- support on at least 72/80 selected rows;
- finite nonnegative values;
- at least 20 distinct values after rounding to ten decimals;
- rigid rotation, translation, site permutation, unimodular cell rebase, and
  exact 2x1x1 replication error no larger than `1e-8`;
- absolute Spearman correlation below `0.90` against every eligible numeric
  base feature and formal feature through NEXT367, where a control is eligible
  only with at least 40 joint finite observations.

Failure of any gate terminates the branch before outcomes are opened.

## Formal build and discovery audit (NEXT372--NEXT374)

If the label-blind probe passes, build the one frozen feature for all 13,470
SCIGEN and 5,232 WyFormer discovery structures in a new physically isolated
directory.  Require at least 0.90 support per source and exact hashes and
boundary flags.  Then audit only the frozen `protected_low` hypothesis using
the unchanged NEXT268/NEXT368 discovery folds, cohort, quantile normalization,
and cross-source gates.  No threshold or direction may be changed after an
outcome is visible.

If the discovery audit passes, NEXT373 may freeze the existing finite formula
search without opening validation or replication; NEXT374 may run only if
that authorization exists.  Otherwise publish an explicit negative result and
create no NEXT373/NEXT374 search artifacts.

## Outputs

New code uses `src/next371_*` onward, tests use `tests/test_next371_*`, formal
artifacts live under `$PRIS_ARCHIVE/next371_*`,
and findings are appended only to the independent no-DFT search report.
Canonical paper/report/README/preregistration content remains untouched until
the user reviews the independent report.
