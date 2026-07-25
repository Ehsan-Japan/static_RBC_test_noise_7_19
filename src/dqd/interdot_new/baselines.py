"""
baselines.py — what the ray method must beat.

The claim "80 % recall at 25 % coverage" is only meaningful next to "raster
scanning at 25 % coverage gives X".  Without that comparison the reader cannot
tell whether the ray geometry is doing any work, or whether *any* 25 % subset
of the grid plus a line-fitting step would do as well.

Four baselines, all measured through the same instrumented ``Device`` so their
budgets are counted identically:

  full_raster_hough  : every cell.  Reference ceiling; coverage = 100 %.
  uniform_hough      : every s-th cell in each direction.  Coverage = 1/s^2.
  random_hough       : n cells drawn uniformly at random.
  linescan_hough     : k complete rows, evenly spaced.  This is the realistic
                       hardware baseline — sweeping a fast axis and stepping a
                       slow one is what a coarse raster actually costs, and it
                       is *harder to beat* than random sampling because each
                       row is contiguous.

Every baseline shares one downstream pipeline so the comparison isolates the
sampling strategy:

  measured subset -> interpolate to full grid -> local-maxima binarisation
                  -> Hough line transform -> rasterise detected lines

The binarisation step reproduces the ridge-detection used in the original
work; the Hough stage follows the standard approach in the literature, which
requires a densely sampled grid as input and is therefore tied to raster cost.
A variant with the Hough stage disabled is also provided so the contribution
of line-fitting can be separated from that of sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
from skimage.transform import hough_line, hough_line_peaks

from .device import Device


# ---------------------------------------------------------------------------
# shared downstream pipeline
# ---------------------------------------------------------------------------


def local_maxima(img: np.ndarray) -> np.ndarray:
    """Ridge pixels: local maxima along rows or along columns."""
    v = np.zeros_like(img, dtype=bool)
    h = np.zeros_like(img, dtype=bool)
    v[1:-1, :] = (img[1:-1, :] > img[:-2, :]) & (img[1:-1, :] > img[2:, :])
    h[:, 1:-1] = (img[:, 1:-1] > img[:, :-2]) & (img[:, 1:-1] > img[:, 2:])
    return v | h


def hough_lines_to_pixels(
    binary: np.ndarray,
    max_lines: int = 12,
    threshold_frac: float = 0.35,
) -> np.ndarray:
    """Detect straight lines in a binary image and rasterise them."""
    ny, nx = binary.shape
    if binary.sum() == 0:
        return np.empty((0, 2), dtype=int)

    angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    hspace, theta, dist = hough_line(binary, theta=angles)
    if hspace.max() == 0:
        return np.empty((0, 2), dtype=int)

    _, angs, dists = hough_line_peaks(
        hspace, theta, dist,
        num_peaks=max_lines,
        threshold=threshold_frac * hspace.max(),
    )

    cols = np.arange(nx)
    rows_grid = np.arange(ny)
    out: List[np.ndarray] = []
    for a, d in zip(angs, dists):
        sin_a, cos_a = np.sin(a), np.cos(a)
        # Hough convention (skimage): d = x*cos(a) + y*sin(a), x=col, y=row
        if abs(sin_a) > 1e-6:
            r = (d - cols * cos_a) / sin_a
            m = (r >= 0) & (r < ny)
            out.append(np.column_stack([np.round(r[m]).astype(int), cols[m]]))
        if abs(cos_a) > 1e-6:
            c = (d - rows_grid * sin_a) / cos_a
            m = (c >= 0) & (c < nx)
            out.append(np.column_stack([rows_grid[m], np.round(c[m]).astype(int)]))

    if not out:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.vstack(out), axis=0)


def reconstruct(device: Device, cells: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate scattered measurements onto the full grid."""
    ny, nx = device.grid.shape
    if len(cells) == 0:
        return np.zeros((ny, nx))
    gi, gj = np.mgrid[0:ny, 0:nx]
    img = griddata(cells.astype(float), values, (gi, gj), method="linear")
    if np.any(np.isnan(img)):
        fill = griddata(cells.astype(float), values, (gi, gj), method="nearest")
        img = np.where(np.isnan(img), fill, img)
    return img


def _finish(
    device: Device,
    cells: np.ndarray,
    values: np.ndarray,
    use_hough: bool,
    max_lines: int,
) -> np.ndarray:
    img = reconstruct(device, cells, values)
    binary = local_maxima(img)
    if not use_hough:
        return np.argwhere(binary)
    return hough_lines_to_pixels(binary, max_lines=max_lines)


# ---------------------------------------------------------------------------
# sampling strategies
# ---------------------------------------------------------------------------


def _measure_cells(device: Device, cells: np.ndarray, stage: str) -> np.ndarray:
    device.stage(stage)
    return np.array([device.measure(int(i), int(j)) for i, j in cells])


def full_raster(device: Device, use_hough: bool = True,
                max_lines: int = 12) -> np.ndarray:
    device.reset_budget()
    ny, nx = device.grid.shape
    cells = np.array([(i, j) for i in range(ny) for j in range(nx)], dtype=int)
    vals = _measure_cells(device, cells, "raster")
    return _finish(device, cells, vals, use_hough, max_lines)


def uniform_raster(device: Device, stride: int, use_hough: bool = True,
                   max_lines: int = 12) -> np.ndarray:
    device.reset_budget()
    ny, nx = device.grid.shape
    cells = np.array(
        [(i, j) for i in range(0, ny, stride) for j in range(0, nx, stride)],
        dtype=int,
    )
    vals = _measure_cells(device, cells, "raster")
    return _finish(device, cells, vals, use_hough, max_lines)


def random_sample(device: Device, n_cells: int, seed: int = 0,
                  use_hough: bool = True, max_lines: int = 12) -> np.ndarray:
    device.reset_budget()
    ny, nx = device.grid.shape
    rng = np.random.default_rng(seed)
    flat = rng.choice(ny * nx, size=min(n_cells, ny * nx), replace=False)
    cells = np.column_stack([flat // nx, flat % nx]).astype(int)
    vals = _measure_cells(device, cells, "random")
    return _finish(device, cells, vals, use_hough, max_lines)


def line_scan(device: Device, n_lines: int, axis: int = 0,
              use_hough: bool = True, max_lines: int = 12) -> np.ndarray:
    """`n_lines` complete rows (axis=0) or columns (axis=1), evenly spaced."""
    device.reset_budget()
    ny, nx = device.grid.shape
    if axis == 0:
        idx = np.unique(np.linspace(0, ny - 1, n_lines).astype(int))
        cells = np.array([(i, j) for i in idx for j in range(nx)], dtype=int)
    else:
        idx = np.unique(np.linspace(0, nx - 1, n_lines).astype(int))
        cells = np.array([(i, j) for j in idx for i in range(ny)], dtype=int)
    vals = _measure_cells(device, cells, "linescan")
    return _finish(device, cells, vals, use_hough, max_lines)


# ---------------------------------------------------------------------------
# budget matching
# ---------------------------------------------------------------------------


def matched_to_budget(
    device: Device,
    target_cells: int,
    seed: int = 0,
    use_hough: bool = True,
    max_lines: int = 12,
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """Run every subsampling baseline at (approximately) `target_cells` cells.

    Returns {name: (predicted_pixels, budget_summary)}.

    Uniform raster can only hit budgets of the form (N/s)^2, so the closest
    achievable stride is used and its *actual* coverage is reported — never
    the nominal target.  Comparing at achieved rather than requested budget is
    the only way the comparison stays honest.
    """
    ny, nx = device.grid.shape
    total = ny * nx
    out: Dict[str, Tuple[np.ndarray, Dict]] = {}

    stride = max(1, int(round(np.sqrt(total / max(target_cells, 1)))))
    px = uniform_raster(device, stride, use_hough, max_lines)
    b = device.budget.summary(); b["param"] = stride
    out["uniform_raster"] = (px, b)

    px = random_sample(device, target_cells, seed, use_hough, max_lines)
    b = device.budget.summary(); b["param"] = target_cells
    out["random_sample"] = (px, b)

    n_lines = max(2, int(round(target_cells / nx)))
    px = line_scan(device, n_lines, 0, use_hough, max_lines)
    b = device.budget.summary(); b["param"] = n_lines
    out["line_scan"] = (px, b)

    return out
