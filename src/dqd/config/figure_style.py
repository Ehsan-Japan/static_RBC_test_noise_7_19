"""
figure_style.py — one place that decides how big every saved figure is.

For a paper the images have to be directly comparable, which means three
things must be identical from figure to figure:

  1. the canvas size in inches and the dpi  -> every file has the same
     physical size and the same pixel dimensions;
  2. the position of the plotting box inside that canvas -> the data area
     lands in the same place in every image, so figures can be stacked or
     placed side by side and the axes line up;
  3. the data-to-inches scale -> equal aspect plus the same voltage limits
     means 1 mV is the same number of millimetres on paper everywhere.

Previously each module picked its own figsize ((8,6), (10,8), (10,10),
(12,12), (14,6) …) and most saved with ``bbox_inches="tight"``, which crops
the canvas to whatever the labels happened to need — so no two images came
out the same size.  Nothing here uses "tight": the figure is saved exactly
as declared, and the fixed axes rectangle leaves room for the labels.

Usage
-----
    from ..config.figure_style import new_map_figure, save_figure

    fig, ax, cax = new_map_figure(with_colorbar=True)
    im = ax.imshow(...)
    fig.colorbar(im, cax=cax, label="Sensor Signal")
    save_figure(fig, out_path)

Multi-panel figures use ``new_figure(ncols=2)``: the canvas widens so each
panel keeps the same physical size as a single-panel figure.

Set the size ONCE at the start of the program — scripts/run_simulation.py
passes figure_width_in / figure_height_in / plot_dpi to DatasetPipeline,
which calls :func:`set_figure_style`.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt

DEFAULT_WIDTH_IN = 6.0
DEFAULT_HEIGHT_IN = 6.0
DEFAULT_DPI = 300

# Plotting box inside the canvas, as (left, bottom, width, height) fractions.
# Identical in every figure, so the data area is always in the same place.
# The right-hand strip is reserved for a colorbar whether or not one is drawn,
# so a figure with a colorbar and one without still share the same axes box.
AXES_RECT = (0.150, 0.130, 0.700, 0.800)
CBAR_RECT = (0.870, 0.130, 0.030, 0.800)


@dataclass
class FigureStyle:
    """Canvas geometry shared by every saved figure."""

    width_in: float = DEFAULT_WIDTH_IN
    height_in: float = DEFAULT_HEIGHT_IN
    dpi: int = DEFAULT_DPI

    def size(self, ncols: int = 1, nrows: int = 1) -> Tuple[float, float]:
        """Canvas size for an ncols x nrows panel grid, in inches."""
        return (self.width_in * ncols, self.height_in * nrows)


_ACTIVE = FigureStyle()


def set_figure_style(
    width_in: Optional[float] = None,
    height_in: Optional[float] = None,
    dpi: Optional[int] = None,
) -> FigureStyle:
    """Set the canvas geometry used by every figure. None = leave unchanged."""
    if width_in is not None:
        _ACTIVE.width_in = width_in
    if height_in is not None:
        _ACTIVE.height_in = height_in
    if dpi is not None:
        _ACTIVE.dpi = dpi
    return _ACTIVE


def get_figure_style() -> FigureStyle:
    """The active FigureStyle object."""
    return _ACTIVE


def figure_size(ncols: int = 1, nrows: int = 1) -> Tuple[float, float]:
    """Canvas size in inches — pass straight to ``figsize=``."""
    return _ACTIVE.size(ncols, nrows)


def figure_dpi() -> int:
    """The shared output dpi."""
    return _ACTIVE.dpi


def new_map_figure(with_colorbar: bool = False):
    """
    Figure for a voltage-space map: fixed canvas, fixed axes rectangle.

    Returns ``(fig, ax, cax)``.  ``cax`` is the colorbar axes when
    ``with_colorbar`` is True, else None — but the space it would occupy is
    reserved either way, so the main axes box never moves.
    """
    fig = plt.figure(figsize=figure_size())
    ax = fig.add_axes(AXES_RECT)
    cax = fig.add_axes(CBAR_RECT) if with_colorbar else None
    return fig, ax, cax


def new_figure(ncols: int = 1, nrows: int = 1, **kwargs):
    """
    Figure for anything that is not a voltage map (line plots, 3-D surfaces,
    multi-panel figures).  Same canvas size per panel; ``constrained_layout``
    keeps the labels inside the canvas instead of cropping it away.

    Returns whatever ``plt.subplots`` returns: ``(fig, ax)`` or ``(fig, axes)``.
    """
    kwargs.setdefault("figsize", figure_size(ncols, nrows))
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kwargs)


def apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax) -> None:
    """
    Give a voltage map the shared labels, limits and scale.

    Equal aspect + the same limits in every figure is what makes 1 mV the
    same physical distance across all of them.
    """
    from .axis_labels import x_label, y_label

    ax.set_xlabel(x_label())
    ax.set_ylabel(y_label())
    ax.set_xlim(vxmin, vxmax)
    ax.set_ylim(vymin, vymax)
    ax.set_aspect("equal", adjustable="box")


def save_figure(fig, path: str, dpi: Optional[int] = None) -> None:
    """
    Save at the shared dpi, WITHOUT ``bbox_inches="tight"``.

    Cropping to the ink is exactly what made the old images different sizes;
    the fixed axes rectangle already leaves room for the labels.
    """
    fig.savefig(path, dpi=dpi or _ACTIVE.dpi)
    plt.close(fig)
