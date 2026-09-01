# NEXT251--NEXT254 SVTC second pre-outcome amendment

Date frozen: 2026-08-09 (America/Chicago)

Parent plan SHA-256:
`dea5e9391cdfcd38d8485e7115c13c99963ccc77cb8c8b4b15463f0903e1b8b3`

First amendment SHA-256:
`de29b931e7493d6f650ba106c19e87fed6c7990c99855978f9cfbc82b733b146`

## Timing and reason

This second amendment is frozen during synthetic engineering tests and before
any formal NEXT251 feature build or discovery-outcome access. The test showed
that the parent quantities `species_singleton_fraction` and
`species_excess_signature_density` change under exact supercell replication:
a signature observed once is copied, and `(U_s - 1) / N` changes when `N`
changes. This conflicts with the frozen supercell-invariance requirement.

## Exact amendment

For species `s`, let `p_st` be the empirical probability of signature `t`
among atoms of species `s`, and let

```text
H_raw_s = -sum_t p_st log(p_st).
```

Replace only the two noninvariant quantities and their feature names with

```text
species_signature_gini =
    sum_s (N_s / N) * (1 - sum_t p_st^2)

species_effective_signature_excess =
    sum_s (N_s / N) * (1 - exp(-H_raw_s)).
```

Both are in `[0, 1]`, equal zero when each element has one local signature,
and are invariant under exact replication. Replace the four feature names in
the parent universe as follows:

```text
svtc_raw_species_singleton_fraction
    -> svtc_raw_species_signature_gini
svtc_raw_species_excess_signature_density
    -> svtc_raw_species_effective_signature_excess
svtc_robust_species_singleton_fraction
    -> svtc_robust_species_signature_gini
svtc_robust_species_excess_signature_density
    -> svtc_robust_species_effective_signature_excess
```

Replace the corresponding four hypothesis names identically; all four retain
the frozen `protected_low` direction. The parent plan and first amendment
remain otherwise binding. No face definition, `1/64` threshold, other feature,
direction, gate, candidate grid, ranking, or stopping rule changes.

NEXT251 formal provenance must hash the immutable parent plan, first
amendment, and this second amendment. No further silent protocol correction is
allowed.
