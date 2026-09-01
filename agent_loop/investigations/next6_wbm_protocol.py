#!/usr/bin/env python3
"""Frozen protocol helpers for unrelaxed WBM pre-screening.

This module contains no feature search and never reads relaxed structures.  It
only fixes composition-level splits, the official WBM stability label, triage
accounting, and a conservative calibration rule for a score where larger means
"more rejectable".
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Iterable

import numpy as np
from pymatgen.core import Composition
from scipy.stats import beta


VALID_DECISIONS = frozenset({"KEEP", "REJECT", "ABSTAIN"})


def reduced_formula_key(formula: str) -> str:
    """Return a canonical mathematical reduced-composition key.

    ``Composition.reduced_formula`` intentionally preserves some conventional
    peroxide formulas (for example ``Li2O2``).  A split key must instead reduce
    exact stoichiometric multiples so they cannot cross the boundary.
    """

    amounts = Composition(formula).element_composition.get_el_amt_dict()
    fractions = {
        symbol: Fraction(str(amount)).limit_denominator(10_000)
        for symbol, amount in amounts.items()
    }
    common_denominator = math.lcm(*(value.denominator for value in fractions.values()))
    integers = {
        symbol: value.numerator * (common_denominator // value.denominator)
        for symbol, value in fractions.items()
    }
    divisor = math.gcd(*integers.values())
    reduced = {symbol: count // divisor for symbol, count in integers.items()}
    return "".join(
        symbol + (str(reduced[symbol]) if reduced[symbol] != 1 else "")
        for symbol in sorted(reduced)
    )


def formula_split(formula: str) -> str:
    """Map an entire reduced composition to the frozen 20/80 WBM split."""

    key = reduced_formula_key(formula).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    return "calibration" if bucket < 20 else "test"


def wbm_stage(formula: str) -> str:
    """Nest formula selection and risk calibration inside the 20% outer split."""

    key = reduced_formula_key(formula)
    if formula_split(key) == "test":
        return "test"
    digest = hashlib.sha256(("stage:" + key).encode("utf-8")).digest()
    return "formula_selection" if int.from_bytes(digest[:8], "big") % 2 == 0 else "threshold_calibration"


def stable_from_wbm_hull(values: Iterable[float]) -> np.ndarray:
    """Official primary label: finite corrected WBM hull energy at or below 0."""

    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array <= 0.0)


def clopper_pearson_upper(
    errors: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    """One-sided exact binomial upper confidence bound."""

    if trials < 0 or errors < 0 or errors > trials:
        raise ValueError("require 0 <= errors <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if trials == 0 or errors == trials:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, trials - errors))


def evaluate_triage(stable: Iterable[bool], decisions: Iterable[str]) -> dict[str, float | int]:
    """Account for KEEP/REJECT/ABSTAIN without crediting abstentions as savings."""

    y = np.asarray(stable, dtype=bool)
    decision = np.asarray(decisions, dtype=object)
    if y.ndim != 1 or decision.ndim != 1 or len(y) != len(decision):
        raise ValueError("stable and decisions must be aligned one-dimensional arrays")
    unexpected = set(decision.tolist()) - VALID_DECISIONS
    if unexpected:
        raise ValueError(f"unexpected decisions: {sorted(unexpected)}")

    reject = decision == "REJECT"
    abstain = decision == "ABSTAIN"
    stable_reject = reject & y
    n = len(y)
    n_stable = int(y.sum())
    n_reject = int(reject.sum())
    false_rejects = int(stable_reject.sum())
    stable_recall = 1.0 - false_rejects / n_stable if n_stable else float("nan")
    reject_precision = (
        float((reject & ~y).sum() / n_reject) if n_reject else float("nan")
    )
    return {
        "n": n,
        "n_stable": n_stable,
        "n_reject": n_reject,
        "n_abstain": int(abstain.sum()),
        "stable_false_rejects": false_rejects,
        "stable_recall": float(stable_recall),
        "false_negative_rate": float(1.0 - stable_recall),
        "reject_precision": reject_precision,
        "dft_savings": float(n_reject / n) if n else 0.0,
        "abstention_rate": float(abstain.mean()) if n else 0.0,
    }


def select_rejection_threshold(
    scores: Iterable[float],
    stable: Iterable[bool],
    *,
    max_false_negative_ucb: float,
    confidence: float = 0.95,
) -> dict[str, float | int | bool]:
    """Choose maximum savings subject to a stable false-reject risk bound.

    Larger finite scores are more rejectable.  Non-finite scores always abstain.
    The threshold is selected only from observed finite values; ``inf`` is the
    no-rejection fallback when the calibration sample cannot certify the risk.
    """

    x = np.asarray(scores, dtype=float)
    y = np.asarray(stable, dtype=bool)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("scores and stable must be aligned one-dimensional arrays")
    if not 0.0 <= max_false_negative_ucb <= 1.0:
        raise ValueError("max_false_negative_ucb must lie in [0, 1]")

    finite = np.isfinite(x)
    n_stable = int(y.sum())
    n = len(y)
    n_abstain = int((~finite).sum())

    # For fixed n_stable the exact upper bound is monotone in the number of
    # false rejects.  Find the largest admissible count once, rather than call
    # beta.ppf at every one of tens of thousands of score thresholds.
    if clopper_pearson_upper(0, n_stable, confidence=confidence) > max_false_negative_ucb:
        max_admissible_errors = -1
    else:
        low, high = 0, n_stable
        while low < high:
            middle = (low + high + 1) // 2
            if (
                clopper_pearson_upper(middle, n_stable, confidence=confidence)
                <= max_false_negative_ucb
            ):
                low = middle
            else:
                high = middle - 1
        max_admissible_errors = low

    def make_row(
        threshold: float,
        n_reject: int,
        false_rejects: int,
    ) -> dict[str, float | int | bool]:
        stable_recall = (
            1.0 - false_rejects / n_stable if n_stable else float("nan")
        )
        upper = clopper_pearson_upper(
            false_rejects, n_stable, confidence=confidence
        )
        return {
            "n": n,
            "n_stable": n_stable,
            "n_reject": n_reject,
            "n_abstain": n_abstain,
            "stable_false_rejects": false_rejects,
            "stable_recall": float(stable_recall),
            "false_negative_rate": float(1.0 - stable_recall),
            "reject_precision": (
                float((n_reject - false_rejects) / n_reject)
                if n_reject
                else float("nan")
            ),
            "dft_savings": float(n_reject / n) if n else 0.0,
            "abstention_rate": float(n_abstain / n) if n else 0.0,
            "threshold": float(threshold),
            "false_negative_ucb": upper,
            "certified": bool(false_rejects <= max_admissible_errors),
        }

    fallback = make_row(np.inf, 0, 0)
    best_args: tuple[float, int, int] | None = (
        (np.inf, 0, 0) if fallback["certified"] else None
    )
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        order = finite_indices[np.argsort(-x[finite_indices], kind="stable")]
        sorted_scores = x[order]
        cumulative_false_rejects = np.cumsum(y[order], dtype=int)
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )
        for end in group_ends:
            false_rejects = int(cumulative_false_rejects[end])
            if false_rejects <= max_admissible_errors:
                best_args = (float(sorted_scores[end]), int(end + 1), false_rejects)

    return make_row(*best_args) if best_args is not None else fallback


__all__ = [
    "clopper_pearson_upper",
    "evaluate_triage",
    "formula_split",
    "reduced_formula_key",
    "select_rejection_threshold",
    "stable_from_wbm_hull",
    "wbm_stage",
]
