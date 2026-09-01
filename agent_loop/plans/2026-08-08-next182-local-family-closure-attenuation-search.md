# NEXT182 Local-Family Closure Attenuation Search

## Purpose

Test a mechanism-specific correction: strong-neighborhood closure may only
attenuate the `local_geometry` family contribution of the exact NEXT163 base.
It may not reduce charge-flow, valence-transport, contact-robustness, or other
family contributions. This avoids globally forgiving structures that are
locally rigid but fail another physical mechanism.

## Frozen formula universe

For the exact NEXT163 selected base, reconstruct each weighted physical term,
cap it at the frozen `0.5`, and average the capped terms whose IDs begin with
`cov_`, `scbv_`, or `sivr_`. Call this bounded quantity `L`.

Only when the original base satisfies `BROAD <= base < SAFE`, use

`score = max(0, base - alpha * L * strong_closure)`.

Otherwise, under missing strong closure, or without base support, keep the
exact original base score/support.

- Strong-closure features: the exact six NEXT180-eligible features.
- Attenuations: `0.25, 0.50, 0.75, 1.00`.
- Candidate count: one unchanged base plus six by four = 25.

The universe is frozen before evaluator execution. No candidate, gate,
threshold, term family, feature, or attenuation changes after results.

## Evaluation and boundaries

Use the unchanged cross-source source-AUC, SAFE, and BROAD discovery gates,
formula-group folds, and Pauling baselines. A freeze requires every gate.
Discovery outcomes are offline search labels only. No DFT calculation/value,
learned energy/force/stress proxy, or physical relaxation is used. Validation
and replication remain sealed unless a candidate passes all discovery gates.

## Outputs

Publish atomically under
`$PRIS_ARCHIVE/next182_local_family_closure_attenuation_search_v1`:

- `MANIFEST.json`;
- `NEXT182_LOCAL_FAMILY_CLOSURE_ATTENUATION_CATALOGUE.json`;
- `NEXT182_DISCOVERY_EVALUATION.json`;
- `NEXT182_FROZEN_CANDIDATE.json`;
- `next182_local_family_closure_attenuation_search.parquet`.

Append results only to the standalone report; preserve canonical documents.
