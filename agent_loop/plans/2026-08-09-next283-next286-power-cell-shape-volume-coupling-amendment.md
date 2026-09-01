# NEXT283--NEXT286 Power-Cell Shape--Volume Coupling Amendment 1

**Status:** frozen after the initial missing-module RED test and before any
NEXT283 feature value, formal table, or discovery outcome was computed.

## Correction

The parent plan incorrectly required invariance under uniform spatial scaling.
That is not a valid representation invariance for this law: tabulated atomic
radii are fixed physical length scales, so uniformly dilating or compressing a
raw crystal changes distances relative to those radii and must be allowed to
change the radical cells and their features.

The uniform-scale test is therefore removed. Required representation
invariances remain rigid rotation, periodic translation/wrapping, site
permutation, unimodular lattice rebasing, and integral periodic replication.

The amendment file becomes a required hashed NEXT283/NEXT284 formal input and
an explicit CLI argument. No feature name, formula, direction, quantile,
dataset, label, gate, or stopping condition changes.
