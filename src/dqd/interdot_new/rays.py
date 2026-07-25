"""
rays.py — stage 1: ray scouting.

Fire a small number of rays from a fixed corner of gate-voltage space and
record the sensor signal along each one.  Where a ray crosses a charge
transition the sensor sees a Coulomb-blockade step, which appears as a local
maximum in the (otherwise monotonically decaying) trace.  Those maxima are the
seeds handed to the line-following stage.

Two changes relative to the original implementation:

1. ``prominence`` is now a real parameter.  ``scipy.signal.find_peaks`` with no
   threshold fires on every 1-sample wobble, so under noise the seed list is
   dominated by spurious maxima that each cost a full sweep.  Prominence is
   expressed as a fraction of the trace's peak-to-peak range, which makes it
   dimensionless and transferable across samples.

2. Every sample along a ray goes through ``Device.measure``, so ray cost is
   counted.  In the original, ray points were read from the precomputed array
   and never appeared in the coverage figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .device import Device


@dataclass
class RayTrace:
    angle_deg: float
    cells: np.ndarray        # (K,2) measured (row, col), in order along the ray
    values: np.ndarray       # (K,) sensor readings
    distance: np.ndarray     # (K,) normalised distance from the ray origin
    peak_idx: np.ndarray     # indices into `cells` flagged as local maxima

    @property
    def peak_cells(self) -> np.ndarray:
        return self.cells[self.peak_idx] if len(self.peak_idx) else np.empty((0, 2), int)


def ray_angles(n_rays: int) -> np.ndarray:
    """Angles strictly inside (0, 90) degrees, evenly spaced.

    Endpoints are excluded because a ray along a grid axis samples a single
    row or column and degenerates into a 1-D line scan.
    """
    return np.linspace(0.0, 90.0, n_rays + 2)[1:-1]


def scout(
    device: Device,
    n_rays: int = 6,
    n_points: int = 100,
    prominence: float = 0.01,
    origin: str = "max_corner",
    stage: str = "ray",
) -> List[RayTrace]:
    """Measure along `n_rays` rays and return the traces with peaks flagged.

    Parameters
    ----------
    prominence : float
        Minimum peak prominence, as a fraction of the peak-to-peak range of
        that ray's trace.  0.0 reproduces the original (unthresholded)
        behaviour.
    origin : {"max_corner", "min_corner"}
        Corner the rays are fired from.  "max_corner" = (V_P1_max, V_P2_max),
        i.e. the many-carrier corner, sweeping outward into depletion.
    """
    grid = device.grid
    device.stage(stage)

    if origin == "max_corner":
        start = np.array([grid.vx_max, grid.vy_max])
        sign = -1.0
    elif origin == "min_corner":
        start = np.array([grid.vx_min, grid.vy_min])
        sign = +1.0
    else:
        raise ValueError(f"unknown origin {origin!r}")

    traces: List[RayTrace] = []
    for angle in ray_angles(n_rays):
        theta = np.deg2rad(angle)
        direction = sign * np.array([np.cos(theta), np.sin(theta)])
        end = _box_exit(start, direction, grid)
        if end is None:
            continue

        t = np.linspace(0.0, 1.0, n_points)
        pts = start[None, :] + t[:, None] * (end - start)[None, :]

        cells, values = [], []
        for vx, vy in pts:
            val, cell = device.measure_voltage(float(vx), float(vy))
            cells.append(cell)
            values.append(val)
        cells = np.array(cells, dtype=int)
        values = np.array(values, dtype=float)

        span = float(values.max() - values.min())
        prom = prominence * span if span > 0 else None
        idx, _ = find_peaks(values, prominence=prom)

        traces.append(
            RayTrace(
                angle_deg=float(angle),
                cells=cells,
                values=values,
                distance=t,
                peak_idx=np.asarray(idx, dtype=int),
            )
        )
    return traces


def seed_cells(traces: List[RayTrace], dedup_radius: int = 2) -> np.ndarray:
    """Collect all ray peaks into a deduplicated seed list.

    Rays fired from a common origin converge near that origin, so two rays can
    flag the same transition twice.  Seeds closer than `dedup_radius` pixels
    are merged, because each duplicate would otherwise pay for a full sweep.
    """
    all_peaks = [t.peak_cells for t in traces if len(t.peak_idx)]
    if not all_peaks:
        return np.empty((0, 2), dtype=int)
    pts = np.vstack(all_peaks)

    kept: List[np.ndarray] = []
    for p in pts:
        if all(np.hypot(*(p - k)) > dedup_radius for k in kept):
            kept.append(p)
    return np.array(kept, dtype=int)


# ---------------------------------------------------------------------------


def _box_exit(start: np.ndarray, direction: np.ndarray, grid) -> Optional[np.ndarray]:
    """Where does the ray leave the voltage window?"""
    eps = 1e-12
    ts = []
    for axis, (lo, hi) in enumerate(
        [(grid.vx_min, grid.vx_max), (grid.vy_min, grid.vy_max)]
    ):
        if abs(direction[axis]) < eps:
            continue
        for bound in (lo, hi):
            t = (bound - start[axis]) / direction[axis]
            if t > eps:
                ts.append(t)
    if not ts:
        return None
    return start + min(ts) * direction
