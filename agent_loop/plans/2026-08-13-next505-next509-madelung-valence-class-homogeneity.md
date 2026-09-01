# NEXT505--NEXT509 Madelung valence-class homogeneity

> Additive frozen design. Preserve every prior script, result, report, and
> canonical document. Validation and replication remain sealed.

## 1. Scientific question and prior-work boundary

Hoppe's MAPLE programme uses the electrostatic part of lattice energy and the
partial Madelung contribution of individual ionic sites to test crystal
structures for plausibility and consistency. This motivates a still-unanswered
question in the current raw-structure loop: do sites assigned the same element
and formal valence receive mutually consistent electrostatic stabilization?

The repository already contains whole-structure normalized Madelung energy and
unconditioned site extrema/spread (NEXT21), approximate symmetry recovery
(NEXT33/NEXT190), same-element local-motif dispersion (NEXT46/NEXT199), charge
spectra (NEXT36/NEXT363), and graph-Poisson potentials (NEXT315). None partitions
Ewald site energies by the joint `(element, formal valence)` class and measures
the fraction of their dispersion explained by that chemical partition.

ChemEnv continuous symmetry measures and ideal coordination-polyhedron fitting
were rejected before this design because they already exist as P4 in
`src/next3_features.py` and were evaluated in the earlier six-family search.
This branch is therefore electrostatic/class-conditional rather than a renamed
angular or coordination descriptor.

The literature motivates partial-Madelung consistency, but the bounded ANOVA
score below is a new operational hypothesis of this project, not a claim that
Hoppe published this exact formula.

## 2. Hard executable boundary

The executable may read only element identities, deterministic NEXT19 formal
valences, the periodic cell, and positions from one initial raw unrelaxed
structure. It may evaluate the analytic point-charge Ewald sum with
`compute_forces=False`. It must not run or read DFT; use a learned or analytic
energy/force/stress proxy, MLIP, model potential, or relaxation; read a
trajectory/later structure/same-composition alternative; or access discovery
outcomes before the blind gates. Discovery outcomes may be used only as offline
labels after the design, implementation, blind engineering, blind novelty, and
full label-free feature build are frozen. Validation and replication remain
sealed throughout NEXT505--NEXT509.

The analytic Coulomb sum is the candidate formula itself, not a DFT result or
learned proxy. It contains no exchange-correlation, electron density, fitted
interatomic potential, empirical energy model, or geometry update.

## 3. Frozen MVCH formula

For Ewald site contributions `E_i`, cell volume `V`, formal charges `q_i`, and
`Q2=sum_i q_i^2`, define a reduced site contribution

```text
u_i = E_i V^(1/3) / (k_e Q2).
```

The common positive factor cancels from the score but is retained to make the
intermediate values dimensionless and auditable. Let `g(i)` be the exact joint
class `(atomic number, formal valence)`, let `mean_g` be the mean `u_i` in that
class, and `mean_all` the mean over all sites. Freeze

```text
S_within = sum_i (u_i - mean_{g(i)})^2
S_total  = sum_i (u_i - mean_all)^2

MVCH(x0) = 1                                  if S_total = 0,
           round_1e-10(1 - S_within/S_total) otherwise.
```

Roundoff values within `64 * eps * max(1, sum_i u_i^2)` of zero are treated as
zero. Values outside the same roundoff guard from `[0,1]` fail closed. The sole
feature is `mvch_madelung_valence_class_homogeneity`; its sole prospective
direction is `protected_high`. No alternative class definition, robust loss,
quantile, threshold, radius, Ewald component, direction, companion feature, or
formula search is available.

MVCH is the coefficient of determination of the element/formal-valence
partition for site Madelung contributions. One means every same-class site has
the same contribution; zero means the class partition explains none of the
site dispersion. Singleton classes contribute zero within-class residual by
definition. This is an explicit limitation, not a hidden eligibility filter.

## 4. Required analytic and firewall properties

Tests must establish the exact ANOVA identity, bounds, deterministic rounding,
class relabeling behavior, and invariance to uniform geometry scaling, uniform
charge-amplitude scaling, site order, disjoint replication, rigid motion,
translation, unimodular rebasing, and exact integer supercells. Malformed,
non-neutral, zero-charge, nonperiodic, nonfinite, calculator-bearing,
metadata-bearing, or extra-array inputs fail closed. Formal-valence inference
failure is explicit; no charge guess is learned from outcomes.

## 5. Frozen sequential gates

1. **NEXT505 engineering probe:** deterministically select 80 discovery records
   per source using the unchanged ordering. Without opening prior feature tables
   or outcomes, require at least `72/80` finite support, range `[0,1]`, at least
   20 distinct values at `1e-10`, and maximum equivalent-representation error
   `<=1e-8` in each source.
2. **NEXT505 novelty probe:** only after engineering passes, compare the same
   records with every frozen numeric label-free control and direct recomputations
   of NEXT21 normalized Madelung site spread/max/min/positive fraction and NEXT46
   same-element/global motif dispersion. Require at least 40 joint finite rows
   and maximum absolute Spearman `<0.90` in each source. No endpoint is opened.
3. **NEXT506 formal build:** only after both blind probes pass, build all
   discovery rows with coverage `>=0.95` in each source and immutable hashes.
4. **NEXT507 discovery audit:** use the unchanged NEXT224/NEXT413 rejected-extreme
   populations, empirical inverse-CDF aggregation, five-fold/source coverage and
   direction gates. Open discovery outcomes only as offline labels. If no frozen
   hypothesis is eligible, terminate the branch immediately.
5. **NEXT508/NEXT509:** run a bounded formula search and BROAD diagnostic only if
   NEXT507 explicitly authorizes them. Validation and replication remain sealed.

Any failed gate closes the whole MVCH branch. Direction, class definition,
zero rule, normalization, quantiles, or gates may not be repaired after seeing
the failure. The overall zero-DFT law search then continues with a physically
different mechanism.

