# NEXT67 monotone expanded additive search

## Correction to search catalogue

NEXT66 exposed a non-monotonic shortlist effect: adding strong coupling features
could remove previously evaluated pairs from the top-20 pool.  NEXT67 changes
only catalogue coverage; endpoint, features, directions, scaling, domain,
strata, thresholds, gates, ranking, and tie-break remain unchanged.

Search every one-term NEXT65 feature.  Search every pair among the top 40
directional features using second-term weights 0.25, 0.5, 1, 2, and 4.  Search
every triple among the top 18 using independent second/third weights 0.5, 1,
and 2.  Additionally union any term from the sealed NEXT61 and NEXT64 selected
formulas into both pools, guaranteeing that prior candidates remain reachable.
Thresholds remain rejection fractions 0.02--0.30 by 0.01.

Only NEXT60 robust discovery labels are used.  All lockboxes remain unopened.
A pass seals one formula for one-shot internal validation; a failure does not
advance.
