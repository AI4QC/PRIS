# NEXT251--NEXT254 SVTC pre-outcome amendment

Date frozen: 2026-08-09 (America/Chicago)

Parent plan:
`docs/plans/2026-08-09-next251-next254-species-voronoi-topology-consistency.md`

Parent plan SHA-256:
`dea5e9391cdfcd38d8485e7115c13c99963ccc77cb8c8b4b15463f0903e1b8b3`

## Timing and reason

This amendment is frozen before any NEXT251 source, test, feature value,
formal output, or discovery outcome was computed or inspected. While deriving
the preregistered engineering tests, the parent definition

```text
H_s = -sum_t p_st log(p_st) / log(N_s)
```

was found to contradict the same plan's required supercell invariance: copying
the cell preserves all signature probabilities but changes `N_s`.

## Exact amendment

Replace only that definition with

```text
U_s = number of distinct signatures observed for species s
H_s = 0                                      if U_s <= 1
H_s = -sum_t p_st log(p_st) / log(U_s)       otherwise.
```

This is invariant to exact replication, lies in `[0, 1]`, and equals one for a
uniform distribution over the observed signature types. No feature name,
feature direction, Voronoi setting, face-area threshold, aggregation, audit
gate, candidate grid, ranking, or stopping rule changes. Every other sentence
of the parent plan remains binding.

NEXT251 formal provenance must hash both this amendment and the immutable
parent plan. Any further protocol change requires another separately frozen,
pre-outcome amendment; neither file may be edited.
