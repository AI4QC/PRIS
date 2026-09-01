# NEXT407--NEXT410 Beck ECN simplicity freeze

## Boundary and status

This is an additive exploratory branch.  It does not replace any existing
script, result, report, Pauling rule, or manuscript text.  The executable
quantity may use composition and exactly one raw, initial, unrelaxed periodic
geometry.  It may use frozen empirical oxidation-state tables, a deterministic
opposite-sign periodic neighbor graph, integer arithmetic, and deterministic
linear algebra.  It may not execute or read DFT, energy, force, stress, a
learned proxy, an ML potential, a relaxation, a trajectory, a later geometry,
or a same-composition alternative.  Discovery outcomes are permitted only as
offline labels after the label-blind and formal-build gates pass.  Validation
and replication remain sealed.

## Literature and repository audit

Langer and Kohlmann's 2026 large-scale test of Beck's extended coordination
number rule reports ECN-specific agreement for 59.9% of 1,038 ternary
fluorides, with a further 22.3% rationalized by no worse electrostatic
imbalance (Acta Cryst. B, DOI `10.1107/S2052520626002298`):

- https://journals.iucr.org/b/issues/2026/02/00/ra5166/
- https://github.com/Niklas-Langer/ExtendedCoordinationNumberRule-Publication

The rule starts from contact conservation and Beck's principle of simplicity:
the contacts supplied by each element-and-coordination class should be
distributed over the counter-ion sites using the two integers adjacent to the
mean.  The paper explicitly presents the rule as descriptive; predictive power
is left for future investigation.  The present branch therefore tests that
question prospectively rather than treating the literature benchmark as an
outcome claim for generated structures.

CodeGraph and literal-text audits found coordination-number, charge-spectrum,
Madelung/Ewald, shell-neutralization, Green-resistance, skeletal-growth,
bond-valence, and Pauling electrostatic-valence features, but no implementation
of Beck's simplest-numerical-solution integer-distribution excess.  A proposed
reverse-KL bond-valence distortion statistic was rejected before
implementation because it is a post-outcome variant of the already falsified
NEXT367 equal-valence-uniformity family.

## Frozen generalized formula

Use the same opposite-sign periodic Voronoi multigraph as the prior strict
loop.  Every cation site `i` has an ECN class

```text
g(i) = (element_i, formal_valence_i, periodic_coordination_number_i).
```

To retain the salt-like boundary while covering structures with more than one
counter-ion species, anion sites are partitioned by
`h(j)=(element_j, formal_valence_j)`.  No statistic is formed across distinct
anion types.  For one anion type `h` containing `M_h` sites and one cation ECN
class `g`, let `x_jg` be the integer number of periodic contacts from anion
site `j` to class `g`, `T_hg=sum_j x_jg`, and `mu_hg=T_hg/M_h`.  The observed
integer segregation sum is

```text
Q_obs(h,g) = sum_(j in h) (x_jg - mu_hg)^2.
```

Writing `T_hg = q M_h + r`, `0 <= r < M_h`, the smallest possible value over
all integer distributions with the same contact total is

```text
Q_min(h,g) = r(M_h-r)/M_h.
```

This is exactly the adjacent-integer content of Beck's simplest numerical
solution for each ECN column.  The generalized nonnegative excess and bounded
crystal statistic are frozen as

```text
Delta_Beck = sum_(h,g with T_hg>0) [Q_obs(h,g)-Q_min(h,g)],
B = 1 / (1 + Delta_Beck/E),
becns_beck_ecn_simplicity = round_1e-10(B),
```

where `E` is the total number of opposite-sign periodic contacts.  The only
direction is `protected_high`.  Perfect adjacent-integer distribution gives
`B=1`; increasing segregation at fixed ECN classes and contact totals strictly
lowers it.  Dividing by `E` makes the statistic invariant under exact disjoint
or crystallographic replication.  No alternate graph, neighbor cutoff, class
definition, normalization, transform, quantile, direction, subgroup,
conjunction, or outcome-conditioned exception is available.

## Label-blind admission gates

NEXT407 uses the unchanged deterministic 80+80 discovery-geometry probe and
all numeric prior label-free controls through NEXT404.  It may recompute the
closed NEXT403 feature only as a same-row novelty control.  It opens no label,
endpoint, validation, or replication field.  Both sources must pass:

1. at least `72/80` finite supported rows;
2. exact domain `0 < B <= 1`;
3. at least 20 values distinct at `1e-10`;
4. maximum error `<=1e-8` under the frozen equivalent-representation suite;
5. maximum adequate absolute Spearman correlation `<0.90` against all prior
   label-free controls, with at least 40 jointly finite rows per control.

Failure terminates the branch without building NEXT408 or opening outcomes.

## Formal build and offline outcome audit

If and only if the label-blind gates pass, NEXT408 builds the one frozen
feature on complete SCIGEN and WyFormer discovery `x0` partitions.  Each source
must have at least `0.95` finite coverage.  The builder must record immutable
input, source, and output hashes and certify all boundary flags.

If and only if both formal coverage gates pass, NEXT409 applies the unchanged
discovery-only rejected-extreme audit, the same reduced-formula five folds,
coverage/class requirements, and the single `protected_high` direction.
Validation and replication remain sealed.  Failure on either source terminates
the branch.  Only a cross-source-eligible frozen hypothesis may authorize a
NEXT410 formula search; otherwise NEXT410 must not exist.

All findings are reported first in the independent additive research report.
Canonical `paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` remain
unchanged until explicit user confirmation.
