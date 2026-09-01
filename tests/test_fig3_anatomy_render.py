from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np

from src import fig3_anatomy as anatomy


def test_expanded_s4_atoms_clear_the_three_dimensional_axes_clip() -> None:
    """The expanded cell must include marker radius, not only atom centres."""

    structure = anatomy.corrupt(
        anatomy.spinel(), "S4", np.random.default_rng(20260815)
    )
    supercell = structure.copy()
    supercell.make_supercell(anatomy.P2C)
    centre = supercell.lattice.matrix.sum(0) / 2
    cartesian = supercell.cart_coords - centre
    shared_extent = float(np.abs(cartesian).max() * 1.02)

    fig = plt.figure(figsize=(1.54, 1.54), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    anatomy.render(
        ax,
        structure,
        scale_ref=shared_extent,
        zoom=anatomy.CUBE_ZOOM,
    )
    fig.canvas.draw()
    axes_box = ax.get_window_extent()

    clearances: list[float] = []
    for coordinate, specie in zip(cartesian, supercell.species, strict=True):
        projected = proj3d.proj_transform(*coordinate, ax.get_proj())
        display_x, display_y = ax.transData.transform(projected[:2])
        marker_radius = (
            np.sqrt(anatomy.RAD[specie.symbol]) / 2 * fig.dpi / 72
        )
        clearances.append(
            min(
                display_x - axes_box.x0,
                axes_box.x1 - display_x,
                display_y - axes_box.y0,
                axes_box.y1 - display_y,
            )
            - marker_radius
        )
    plt.close(fig)

    assert min(clearances) >= 1.0
