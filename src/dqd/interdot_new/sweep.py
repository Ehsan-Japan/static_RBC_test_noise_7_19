"""
sweep.py — stage 2: directional sweep line-following.

Given a seed cell known to sit on a transition line, walk along that line one
row at a time.  In each row, scan sideways from a start column until the first
local maximum is found; that maximum is the line's position in this row.  The
next row starts a small offset (`col_buffer`) back from the peak just found,
which is what keeps the scan short: the line moves by roughly one column per
row, so a window of a few columns is enough to recapture it.

The walk stops when a row yields no maximum.  That failure is not a nuisance —
it is *the signal that the line has ended*, which happens at a honeycomb
vertex.  The last successful cell is the **break point**, and break points are
the raw material for interdot detection in ``interdot.py``.

Sweep families
--------------
Four sweeps are launched per seed, in two pairs:

  dot-to-lead pairs   walk up-and-left / down-and-right.  These track the two
                      steep line families (the charge transitions of QD1 and
                      QD2 against their reservoirs).

  interdot pairs      start `col_buffer` rows away from the seed and
                      `2*col_buffer` columns to the outside, then scan inward.
                      The offset exists because at the vertex the interdot
                      segment and the dot-to-lead line are nearly on top of
                      each other; stepping away first lets them separate, and
                      approaching from outside means the scan reaches the
                      interdot peak before the dot-to-lead peak captures it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .device import Device


@dataclass
class SweepConfig:
    name: str
    start_row: int
    row_step: int
    start_col: int
    col_step: int
    col_buffer: int
    max_scan: int = 6    # cap on sideways scan length per row
    peak_margin: float = 0.0
    smooth: int = 1
    """Minimum height a candidate must exceed its neighbours by.

    With margin 0 the three-point test fires on any 1-LSB wobble, so under
    noise the tracker locks onto noise maxima, the seed list explodes, and the
    measurement budget grows without bound -- the failure mode is cost, not
    just accuracy.  Setting the margin to a few times the noise standard
    deviation restores a bounded budget.
    """


@dataclass
class SweepResult:
    name: str
    peaks: List[Tuple[int, int]] = field(default_factory=list)
    break_point: Optional[Tuple[int, int]] = None
    reason: str = "empty"
    """Why the walk stopped.  This distinction is essential:

        "lost_line" : a row yielded no local maximum.  The tracker failed
                      *because the line ended* — a triple-point candidate.
        "boundary"  : the walk simply reached the edge of the voltage window.
                      Nothing physical happened here.
        "empty"     : the very first row already failed.

    Treating a boundary exit as a triple point is the single easiest way to
    poison interdot detection, because most sweeps terminate at the window
    edge, not at a vertex.  Only "lost_line" break points are paired.
    """

    @property
    def peak_array(self) -> np.ndarray:
        return (
            np.array(self.peaks, dtype=int)
            if self.peaks
            else np.empty((0, 2), dtype=int)
        )


def follow_line(device: Device, cfg: SweepConfig, stage: str = "sweep") -> SweepResult:
    """Walk a transition line row by row.  Returns peaks and the break point."""
    device.stage(stage)
    ny, nx = device.grid.shape
    res = SweepResult(name=cfg.name)

    col_start = int(np.clip(cfg.start_col, 0, nx - 1))
    row = int(cfg.start_row)

    while 0 <= row < ny:
        peak_col = _scan_row(device, row, col_start, cfg.col_step,
                             cfg.max_scan, nx, cfg.peak_margin, cfg.smooth)
        if peak_col is None:
            res.reason = "lost_line" if res.peaks else "empty"
            return res
        res.peaks.append((row, peak_col))
        res.break_point = (row, peak_col)
        col_start = int(
            np.clip(
                peak_col - cfg.col_buffer if cfg.col_step > 0
                else peak_col + cfg.col_buffer,
                0, nx - 1,
            )
        )
        row += cfg.row_step

    res.reason = "boundary" if res.peaks else "empty"
    return res


def _scan_row(
    device: Device,
    row: int,
    start_col: int,
    col_step: int,
    max_scan: int,
    nx: int,
    margin: float = 0.0,
    smooth: int = 1,
) -> Optional[int]:
    """Scan sideways in one row; return the column of the first local maximum.

    A three-point test is used: the middle of the last three measurements is a
    peak if it exceeds both neighbours.  This costs one extra measurement past
    the peak, which is unavoidable for any causal (streaming) peak detector.
    """
    vals: List[float] = []
    cols: List[int] = []
    col = start_col
    for _ in range(max_scan):
        if not (0 <= col < nx):
            break
        vals.append(device.measure(row, col))
        cols.append(col)
        # Smoothing uses only values already measured, so it costs nothing.
        # Under *static* (frozen) noise this is the only averaging available:
        # re-measuring a cell returns the same corrupted value, so the average
        # must be taken across neighbouring cells, not across repeats.
        w = smooth if smooth >= 1 else 1
        need = 2 * w + 1
        if len(vals) >= need:
            a = float(np.mean(vals[-need:-need + w]))
            b = float(np.mean(vals[-need + w:-w])) if w > 1 else float(vals[-w - 1])
            c = float(np.mean(vals[-w:]))
            if b > a + margin and b > c + margin:
                return cols[-w - 1]
        col += col_step
    return None


def sweeps_for_seed(seed: Tuple[int, int], col_buffer: int,
                    margin: float = 0.0, smooth: int = 1) -> List[SweepConfig]:
    """The four sweep configurations launched from one seed cell."""
    i, j = int(seed[0]), int(seed[1])
    b = col_buffer
    return [
        SweepConfig("dotlead_up",   i,     +1, j + b,     -1, b, peak_margin=margin, smooth=smooth),
        SweepConfig("dotlead_down", i,     -1, j - b,     +1, b, peak_margin=margin, smooth=smooth),
        SweepConfig("interdot_up",   i + b, +1, j + 2 * b, -1, b, peak_margin=margin, smooth=smooth),
        SweepConfig("interdot_down", i - b, -1, j - 2 * b, +1, b, peak_margin=margin, smooth=smooth),
    ]
