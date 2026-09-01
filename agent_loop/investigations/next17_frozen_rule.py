"""Immutable production decision for the development-promoted NEXT17 rule."""

from __future__ import annotations

import math


FROZEN_FORMULA = (
    "R64s(i) = E_MatterSim_strict_relaxed(i)/N_i "
    "- min_j_in_same_composition E_MatterSim_strict_relaxed(j)/N_j"
)
FROZEN_THRESHOLD_EV_PER_ATOM = 0.06
FROZEN_RELAXATION = {
    "optimizer": "FIRE",
    "filter": "FRECHETCELLFILTER",
    "fmax_ev_per_a": 0.005,
    "max_prediction_steps": 64,
    "atom_budget": 512,
}


def next17_frozen_decision(gap: object, *, group_supported: bool) -> str:
    """Return KEEP/REJECT/ABSTAIN without any configurable threshold."""

    if type(group_supported) is not bool:
        raise ValueError("group_supported must be an exact boolean")
    if not group_supported:
        return "ABSTAIN"
    try:
        value = float(gap)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("supported NEXT17 gap must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("supported NEXT17 gap must be finite")
    return "REJECT" if value >= FROZEN_THRESHOLD_EV_PER_ATOM else "KEEP"
