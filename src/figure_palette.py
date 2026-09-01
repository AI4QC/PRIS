#!/usr/bin/env python3
"""The manuscript colour scheme: Okabe-Ito, adapted to this paper's roles.

Importing this module registers the three sequential colour maps the figures use
and caps the opacity of bar fills.  The identity colours themselves live as
constants in the figure generators; this module is the single place that records
where they came from and enforces the two rules that are not expressible as a
constant.

Roles
-----
Set 1   #005B93   blue          Set 1'  #35A7D8   sky blue
Set 2   #9861B0   violet        Set 3   #D6564C   red
Set 4   #0A5A3C   bluish green  PSS     #E88A8E   light red

Set 1' is a sky blue rather than the purple of earlier drafts: it is a conditional
variant of Set 1, so a sibling of Set 1's blue reads correctly, and it frees the
violet for Set 2.  That matters because Fig. 4c and Fig. 4f used to draw Set 2 in
the same purple Fig. 1d gave to Set 1', so one colour carried two meanings inside
one figure.  PSS has its own light red instead of borrowing Set 2's colour.

No role sits in the yellow-amber or brown-terracotta bands, which have to be
darkened to stay legible on white and turn muddy when they are, and neither the red
nor the violet is a dark one.  The greens sit near OKLCH lightness 0.41 so that the
red and the green stay apart under protanopia, where hue alone cannot separate
them.

Provenance: the identity colours derive from the Okabe-Ito palette that Nature
Methods recommends for categorical data, snapped so that every pair a reader has to
tell apart inside one panel stays separable under normal, protanopic and
deuteranopic vision.  `tex/palette-options/audit_contrast.py` re-runs that check.
"""
from __future__ import annotations

import math

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------- identity roles
SET1 = "#005B93"
SET1P = "#35A7D8"
SET2 = "#9861B0"
SET3 = "#D6564C"
SET4 = "#0A5A3C"
PSS = "#E88A8E"
EHULL = "#6C7B92"          # DFT hull-energy baseline: a cool steel, not a second blue

# Bars carry the largest areas of flat colour in these figures and read as the most
# saturated thing on the page.  Markers, curves and labels keep full strength.
BAR_ALPHA = 0.82

# hue (OKLCH degrees) of the one-hue ramp that replaces each built-in colour map
_RAMP_HUE = {"density": None, "violation": None, "matrix": 250.0}


def _lin2s(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_oklch(h):
    r, g, b = (_s2lin(int(h[i:i + 2], 16) / 255) for i in (1, 3, 5))
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, math.hypot(a, bb), math.atan2(bb, a)


def _from_oklch(L, C, H):
    a, b = C * math.cos(H), C * math.sin(H)
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    return "#%02X%02X%02X" % tuple(
        max(0, min(255, round(_lin2s(max(0.0, min(1.0, v))) * 255))) for v in rgb)


def ramp(kind: str, n: int = 256) -> list[str]:
    """One hue, monotone light to dark.

    Magnitude then never reads as a rainbow, two cells can be ordered by eye, and
    the identity colours stay the only categorical signal.  The published figures
    used `Blues`, `Purples` and `RdYlBu_r`; the last is not monotone in lightness,
    so its cells cannot be ordered by eye or in greyscale at all.
    """
    if kind == "density":                     # Fig. 1b point cloud
        L1, C1, H = _to_oklch(SET1)
    elif kind == "violation":                 # Fig. 3b law-violation matrix
        L1, C1, H = _to_oklch(SET2)
    else:                                     # damage-detection matrices
        H = math.radians(_RAMP_HUE["matrix"])
        L1, C1 = 0.52, 0.11
    out = []
    for i in range(n):
        t = i / (n - 1)
        L = 0.985 - t * (0.985 - max(0.34, L1 - 0.16))
        C = 0.012 + t * (max(C1, 0.11) - 0.012)
        out.append(_from_oklch(L, C, H))
    return out


def _register() -> None:
    for kind, name in (("density", "palseq"), ("violation", "palseq2"),
                       ("matrix", "palmatrix")):
        cmap = LinearSegmentedColormap.from_list(name, ramp(kind))
        try:
            matplotlib.colormaps.register(cmap, force=True)
        except Exception:                     # matplotlib < 3.6
            matplotlib.cm.register_cmap(name=name, cmap=cmap)


def _cap_bar_opacity() -> None:
    for name in ("bar", "barh"):
        original = getattr(Axes, name)
        if getattr(original, "_pris_capped", False):
            continue

        def capped(self, *args, __original=original, **kwargs):
            kwargs["alpha"] = min(kwargs.get("alpha") or 1.0, BAR_ALPHA)
            return __original(self, *args, **kwargs)

        capped._pris_capped = True
        setattr(Axes, name, capped)


_register()
_cap_bar_opacity()
