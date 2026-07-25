"""
interdot.py — stage 3: interdot transition detection by break-point pairing.

Why interdot lines are hard
---------------------------
In a double dot, the charge-stability diagram has two kinds of boundary:

  * **dot-to-lead** lines, where a carrier enters or leaves a dot from the
    reservoir.  The total charge on the double-dot changes, so the nearby
    charge sensor sees a large potential step.  High contrast, easy to find.

  * **interdot** lines, where a carrier moves *between* the two dots:
    (m+1, n) <-> (m, n+1).  Total charge is unchanged; only the dipole moves.
    The sensor response is therefore weak — often an order of magnitude below
    the dot-to-lead contrast — and it scales with the sensor's *asymmetry*
    between the two dots (the difference of the two dot-to-sensor
    capacitances, ``Cds[0] - Cds[1]``).  A perfectly symmetric sensor sees
    nothing at all.

Any detector that thresholds on sensor contrast will therefore find the
dot-to-lead lines and miss the interdot lines.  That is the gap this stage
closes, and it does so without ever thresholding on contrast.

The idea: vertices are kinks, not endpoints
-------------------------------------------
The original formulation of this method reasoned that a line-follower fails
where the line it is tracking ends, and that dot-to-lead lines end at triple
points -- so a tracker's last successful cell ("break point") estimates a
vertex.

Measurement shows that premise is wrong, and it is worth being precise about
why.  The charge-stability boundary network is *connected*: at a triple point
three boundaries meet, so a boundary does not terminate there, it **branches**.
A tracker arriving at a vertex therefore does not lose the ridge -- it follows
the ridge around the corner onto the adjacent family.  In practice almost
every sweep runs until it leaves the voltage window, and pairing those exits
pairs window edges, not vertices.

The recoverable signal is the **kink**.  The two dot-to-lead families have
distinct slopes, fixed by the gate-to-dot capacitance ratios:

    dot-1 line :  dV_P2/dV_P1 = -C_g1d1 / C_g2d1     (steep)
    dot-2 line :  dV_P2/dV_P1 = -C_g1d2 / C_g2d2     (shallow)

so where a tracked polyline turns from one slope to the other, it has passed a
triple point.  Detecting that turn requires no extra measurement at all: it is
post-processing of the trajectory already recorded by the sweep.

Between two adjacent triple points lies exactly one interdot transition.  So
paired kinks give the interdot segment by geometry, with no contrast threshold
anywhere -- which is the point, because the interdot contrast is precisely
what is too weak to threshold on.

Algorithm
---------
1. **Collect** vertex candidates.  Primary source: kinks in each tracked
   polyline.  Secondary source (retained for ablation, and matching the
   original proposal): "lost_line" break points, i.e. sweeps that genuinely
   failed rather than exiting the window.
2. **Pair** candidates by mutual nearest neighbour within ``max_gap`` pixels.
   Mutual (rather than greedy one-directional) pairing is used because greedy
   pairing depends on iteration order and can chain three break points into a
   spurious pair.
3. **Verify** each candidate segment: measure the sensor along it, and along
   two reference segments displaced perpendicular by ``offset`` pixels.  Accept
   if the on-segment mean exceeds the off-segment mean by ``min_contrast``
   (in units of the local standard deviation).  This is a *relative* test, so
   it survives the weak absolute contrast of an interdot line.
4. **Emit** the rasterised segment as predicted transition pixels, and its two
   endpoints as estimated triple-point locations.

Step 4 is where this differs most from a naive implementation that returns
only the single brightest cell on the segment: the interdot transition is a
segment, not a point, so returning the segment is both more useful and
scores properly under a line-detection metric.

Every measurement in step 3 goes through ``Device.measure`` and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .device import Device


@dataclass
class InterdotSegment:
    a: Tuple[int, int]              # first break point (estimated triple point)
    b: Tuple[int, int]              # second break point
    pixels: np.ndarray              # (K,2) rasterised segment cells
    contrast: float                 # verification statistic (higher = stronger)
    accepted: bool


# ---------------------------------------------------------------------------
# vertex candidates from tracked polylines
# ---------------------------------------------------------------------------


def find_kinks(
    polyline: np.ndarray,
    window: int = 5,
    min_jump: float = 0.30,
    min_sep: int = 4,
) -> List[Tuple[int, int]]:
    """Locate slope discontinuities along one tracked transition line.

    ``polyline`` is the (K,2) sequence of (row, col) peaks produced by
    ``sweep.follow_line``, ordered along the walk.

    The local slope dcol/drow is smoothed over ``window`` steps and its
    first difference is taken; a kink is a local maximum of |d slope| above
    ``min_jump``.  ``min_jump`` is in columns-per-row, so 0.30 means the line
    must change direction by roughly a third of a cell per row -- comfortably
    above tracking jitter, comfortably below the slope difference between the
    two dot-to-lead families for any capacitance set that produces a visible
    honeycomb.

    Costs nothing: no device access.
    """
    p = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(p) < 2 * window + 2:
        return []

    dr = np.diff(p[:, 0])
    dc = np.diff(p[:, 1])
    slope = dc / np.where(np.abs(dr) < 1e-9, 1.0, dr)

    kernel = np.ones(window) / window
    smooth = np.convolve(slope, kernel, mode="valid")
    jump = np.abs(np.diff(smooth))
    if not len(jump):
        return []

    order = np.argsort(jump)[::-1]
    picked: List[int] = []
    for idx in order:
        if jump[idx] < min_jump:
            break
        if all(abs(idx - q) >= min_sep for q in picked):
            picked.append(int(idx))

    off = window // 2
    out: List[Tuple[int, int]] = []
    for idx in picked:
        k = min(len(p) - 1, idx + off + 1)
        out.append((int(p[k, 0]), int(p[k, 1])))
    return out


def find_kink_indices(
    polyline: np.ndarray,
    window: int = 5,
    min_jump: float = 0.30,
    min_sep: int = 4,
) -> List[int]:
    """Same detection as find_kinks, but returns polyline indices."""
    p = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(p) < 2 * window + 2:
        return []
    dr = np.diff(p[:, 0])
    dc = np.diff(p[:, 1])
    slope = dc / np.where(np.abs(dr) < 1e-9, 1.0, dr)
    smooth = np.convolve(slope, np.ones(window) / window, mode="valid")
    jump = np.abs(np.diff(smooth))
    if not len(jump):
        return []
    picked: List[int] = []
    for idx in np.argsort(jump)[::-1]:
        if jump[idx] < min_jump:
            break
        if all(abs(idx - q) >= min_sep for q in picked):
            picked.append(int(idx))
    off = window // 2
    return [min(len(p) - 1, i + off + 1) for i in picked]


def refine_vertex(
    polyline: np.ndarray,
    kink_index: int,
    fit_len: int = 8,
    min_slope_diff: float = 0.25,
) -> Optional[Tuple[float, float]]:
    """Sub-pixel triple point: intersect the two fitted line segments.

    Taking the kink *cell* as the vertex is accurate only to the tracker's
    step size, typically 1-4 px.  That is fatal here, because the interdot
    segment is itself only ~5 px long: a +/-3 px error on each endpoint can
    flip the sign of the segment's slope, and the positive-slope test -- which
    is exact in ground truth -- then rejects genuine pairs.

    Fitting instead uses every point of both incoming line arms.  Each arm is
    a straight transition line with a well-defined slope set by capacitance
    ratios, so ``col = a*row + b`` is fitted on each side and the vertex is
    their intersection.  Error falls as ~1/sqrt(fit_len) instead of being
    pinned to the step size.

    Returns (row, col) as floats, or None if the two arms are near-parallel
    (no resolvable corner).
    """
    p = np.asarray(polyline, dtype=float).reshape(-1, 2)
    k = int(kink_index)
    left = p[max(0, k - fit_len):k]
    right = p[k + 1:k + 1 + fit_len]
    if len(left) < 3 or len(right) < 3:
        return None

    def fit(arm):
        if np.ptp(arm[:, 0]) < 1e-9:
            return None
        a, b = np.polyfit(arm[:, 0], arm[:, 1], 1)
        return float(a), float(b)

    fl, fr = fit(left), fit(right)
    if fl is None or fr is None:
        return None
    a1, b1 = fl
    a2, b2 = fr
    if abs(a1 - a2) < min_slope_diff:
        return None

    row = (b2 - b1) / (a1 - a2)
    col = a1 * row + b1
    lo, hi = p[:, 0].min() - fit_len, p[:, 0].max() + fit_len
    if not (lo <= row <= hi):
        return None
    return (row, col)




def vertex_candidates(
    polylines: Sequence[np.ndarray],
    break_points: Optional[Sequence[Tuple[int, int]]] = None,
    window: int = 5,
    min_jump: float = 0.30,
    merge_radius: int = 3,
    refine: bool = True,
    fit_len: int = 8,
) -> np.ndarray:
    """All vertex candidates from kinks (+ optional lost_line break points)."""
    cands: List[Tuple[float, float]] = []
    for pl in polylines:
        for k in find_kink_indices(pl, window=window, min_jump=min_jump):
            if refine:
                v = refine_vertex(pl, k, fit_len=fit_len)
                if v is not None:
                    cands.append(v)
                    continue
            p = np.asarray(pl, dtype=float).reshape(-1, 2)
            cands.append((float(p[k, 0]), float(p[k, 1])))
    if break_points is not None:
        cands.extend(tuple(float(v) for v in bp) for bp in break_points)
    if not cands:
        return np.empty((0, 2), dtype=float)

    pts = np.array(cands, dtype=float)
    kept: List[np.ndarray] = []
    for q in pts:
        if all(np.hypot(*(q - k)) > merge_radius for k in kept):
            kept.append(q)
    return np.array(kept, dtype=float)


def detect(
    device: Device,
    candidates: Sequence[Tuple[int, int]],
    max_gap: int = 12,
    min_length: int = 2,
    offset: int = 3,
    min_contrast: float = 0.15,
    require_positive_slope: bool = True,
    stage: str = "interdot",
) -> List[InterdotSegment]:
    """Find interdot segments by pairing vertex candidates.

    Parameters
    ----------
    max_gap : int
        Maximum pixel separation for two break points to be considered the
        endpoints of one interdot transition.  Physically this is set by the
        interdot coupling: a larger ``C_m`` widens the honeycomb vertex
        splitting and lengthens the interdot segment.
    offset : int
        Perpendicular displacement, in pixels, of the two background reference
        segments used for verification.
    min_contrast : float
        Acceptance threshold on |contrast|, in units of the standard deviation
        of the reference segments.  Kept deliberately low: the pairing is
        carried by geometry, and this test exists only to reject pairs that
        span an empty charge region, not to detect the line.
    """
    device.stage(stage)
    pts = np.asarray(list(candidates), dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return []

    segments: List[InterdotSegment] = []
    for i, j in _mutual_nearest_pairs(pts, max_gap):
        a = (int(round(pts[i][0])), int(round(pts[i][1])))
        b = (int(round(pts[j][0])), int(round(pts[j][1])))
        fa, fb = pts[i], pts[j]
        if require_positive_slope:
            dr = float(fb[0] - fa[0])
            dc = float(fb[1] - fa[1])
            # dV_P2/dV_P1 = drow/dcol must be > 0
            if dc == 0 or (dr / dc) <= 0:
                continue

        line = _raster_line(a, b)
        if len(line) < min_length:
            continue
        line = _clip(line, device.grid.shape)
        if len(line) < min_length:
            continue

        contrast = _verify(device, a, b, line, offset)
        segments.append(
            InterdotSegment(
                a=a, b=b, pixels=line,
                contrast=contrast,
                # ABSOLUTE deviation: an interdot transition shifts the sensor
                # dot's Coulomb peak, so depending on the sensor's operating
                # point the segment appears as a ridge OR a trough.  Requiring
                # a positive-going ridge silently discards half of them.
                accepted=bool(abs(contrast) >= min_contrast),
            )
        )
    return segments


def accepted_pixels(segments: Sequence[InterdotSegment]) -> np.ndarray:
    """All pixels belonging to accepted interdot segments."""
    keep = [s.pixels for s in segments if s.accepted and len(s.pixels)]
    return np.vstack(keep) if keep else np.empty((0, 2), dtype=int)


def estimated_vertices(segments: Sequence[InterdotSegment]) -> np.ndarray:
    """Triple-point estimates: the endpoints of accepted segments."""
    pts: List[Tuple[int, int]] = []
    for s in segments:
        if s.accepted:
            pts.extend([s.a, s.b])
    return np.array(pts, dtype=int) if pts else np.empty((0, 2), dtype=int)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mutual_nearest_pairs(pts: np.ndarray, max_gap: float) -> List[Tuple[int, int]]:
    """Pair indices that are each other's nearest neighbour within max_gap."""
    n = len(pts)
    d = np.hypot(
        pts[:, 0][:, None] - pts[:, 0][None, :],
        pts[:, 1][:, None] - pts[:, 1][None, :],
    ).astype(float)
    np.fill_diagonal(d, np.inf)
    nearest = np.argmin(d, axis=1)

    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        j = int(nearest[i])
        if i < j and int(nearest[j]) == i and d[i, j] <= max_gap:
            pairs.append((i, j))
    return pairs


def _raster_line(a: Tuple[int, int], b: Tuple[int, int]) -> np.ndarray:
    """Integer cells along the segment a->b (dense, no gaps)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    rows = np.round(np.linspace(a[0], b[0], n)).astype(int)
    cols = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return np.unique(np.column_stack([rows, cols]), axis=0)


def _clip(cells: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    m = (
        (cells[:, 0] >= 0) & (cells[:, 0] < ny)
        & (cells[:, 1] >= 0) & (cells[:, 1] < nx)
    )
    return cells[m]


def _verify(
    device: Device,
    a: Tuple[int, int],
    b: Tuple[int, int],
    line: np.ndarray,
    offset: int,
) -> float:
    """Signal-to-background contrast of the candidate segment.

    Returns (mean_on - mean_off) / std_off.  Positive and large means the
    segment sits on a ridge of the sensor signal relative to the charge
    regions immediately either side of it.
    """
    d = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    norm = np.hypot(*d)
    if norm == 0:
        return -np.inf
    perp = np.array([-d[1], d[0]]) / norm   # unit normal

    on_vals = [device.measure(int(r), int(c)) for r, c in line]

    off_vals: List[float] = []
    for sign in (+1, -1):
        shifted = line + np.round(sign * offset * perp).astype(int)
        shifted = _clip(shifted, device.grid.shape)
        off_vals.extend(device.measure(int(r), int(c)) for r, c in shifted)

    if not off_vals or not on_vals:
        return -np.inf
    sd = float(np.std(off_vals))
    if sd < 1e-12:
        sd = 1e-12
    return (float(np.mean(on_vals)) - float(np.mean(off_vals))) / sd
