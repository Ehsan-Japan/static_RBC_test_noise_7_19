"""
detector.py — the full ray-based detector.

    rays.scout      ->  seed cells on transition lines
    sweep.follow    ->  traced line pixels + break points
    interdot.detect ->  interdot segments from paired break points

Returns a ``Detection`` holding the predicted transition pixels, the estimated
triple points, and the measurement budget actually consumed.

Note on what counts as a "detection": exactly the cells the algorithm asserts
are on a transition line — one pixel each, no dilation.  Tolerance belongs in
the evaluation (see ``metrics.py``), not in the prediction.  Dilating
predictions by +/-k columns inflates recall and destroys precision while
changing nothing about what the algorithm actually knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import interdot, rays, sweep
from .device import Device


@dataclass
class DetectorConfig:
    n_rays: int = 6
    ray_points: int = 100
    ray_prominence: float = 0.01
    peak_margin_sigma: float = 0.0
    scan_smooth: int = 1
    """Peak-acceptance margin, in units of the noise sigma estimated from the
    ray traces.  0 disables it (original behaviour)."""
    seed_dedup_radius: int = 2
    col_buffer: int = 2
    max_scan: int = 10  # tight window: see sweep.py, this is what makes kinks detectable
    interdot_max_gap: int = 14
    kink_window: int = 5
    kink_min_jump: float = 0.30
    kink_refine: bool = True    # sub-pixel vertex by intersecting the two arms
    kink_fit_len: int = 8
    use_kinks: bool = True          # kink detection (this work)
    use_breakpoints: bool = False   # ablation: original break-point pairing
    interdot_offset: int = 3
    interdot_min_contrast: float = 0.15
    interdot_positive_slope: bool = True
    enable_interdot: bool = True


@dataclass
class Detection:
    pixels: np.ndarray                      # (K,2) predicted transition cells
    dotlead_pixels: np.ndarray
    interdot_pixels: np.ndarray
    vertices: np.ndarray                    # (V,2) estimated triple points
    break_points: np.ndarray
    seeds: np.ndarray
    traces: list = field(default_factory=list)
    polylines: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)


def run(device: Device, cfg: Optional[DetectorConfig] = None) -> Detection:
    cfg = cfg or DetectorConfig()
    device.reset_budget()

    # -- stage 1: rays ----------------------------------------------------
    traces = rays.scout(
        device,
        n_rays=cfg.n_rays,
        n_points=cfg.ray_points,
        prominence=cfg.ray_prominence,
    )
    seeds = rays.seed_cells(traces, dedup_radius=cfg.seed_dedup_radius)

    # Estimate the noise level from data already in hand.  The second
    # difference of a smooth trace is ~0, so its robust scale is dominated by
    # noise:  sigma ~ MAD(d2) / (sqrt(6) * 0.6745).
    sigma = 0.0
    if cfg.peak_margin_sigma > 0 and traces:
        d2 = np.concatenate([np.diff(t.values, n=2) for t in traces
                             if len(t.values) > 2] or [np.zeros(1)])
        sigma = float(np.median(np.abs(d2 - np.median(d2)))) / (np.sqrt(6) * 0.6745)
    margin = cfg.peak_margin_sigma * sigma

    # -- stage 2: sweeps --------------------------------------------------
    dotlead: List[np.ndarray] = []
    polylines: List[np.ndarray] = []
    breaks: List[tuple] = []
    for s in seeds:
        for scfg in sweep.sweeps_for_seed(tuple(s), cfg.col_buffer, margin, cfg.scan_smooth):
            scfg.max_scan = cfg.max_scan
            res = sweep.follow_line(device, scfg)
            if len(res.peaks):
                dotlead.append(res.peak_array)
                polylines.append(res.peak_array)
            # Only terminations caused by losing the line are triple-point
            # candidates; walks that ran off the edge of the window are not.
            if res.break_point is not None and res.reason == "lost_line":
                breaks.append(res.break_point)

    dotlead_px = (
        np.unique(np.vstack(dotlead), axis=0) if dotlead else np.empty((0, 2), int)
    )
    break_arr = (
        np.unique(np.array(breaks, dtype=int), axis=0)
        if breaks else np.empty((0, 2), int)
    )

    # -- stage 3: interdot ------------------------------------------------
    segments: list = []
    interdot_px = np.empty((0, 2), dtype=int)
    vertices = np.empty((0, 2), dtype=int)
    cands = interdot.vertex_candidates(
        polylines if cfg.use_kinks else [],
        break_points=break_arr if cfg.use_breakpoints else None,
        window=cfg.kink_window,
        min_jump=cfg.kink_min_jump,
        refine=cfg.kink_refine,
        fit_len=cfg.kink_fit_len,
    ) if cfg.enable_interdot else np.empty((0, 2), int)

    if cfg.enable_interdot and len(cands) >= 2:
        segments = interdot.detect(
            device,
            cands,
            max_gap=cfg.interdot_max_gap,
            offset=cfg.interdot_offset,
            min_contrast=cfg.interdot_min_contrast,
            require_positive_slope=cfg.interdot_positive_slope,
        )
        interdot_px = interdot.accepted_pixels(segments)
        vertices = interdot.estimated_vertices(segments)

    all_px = [p for p in (dotlead_px, interdot_px) if len(p)]
    pixels = np.unique(np.vstack(all_px), axis=0) if all_px else np.empty((0, 2), int)

    return Detection(
        pixels=pixels,
        dotlead_pixels=dotlead_px,
        interdot_pixels=interdot_px,
        vertices=vertices,
        break_points=cands,
        seeds=seeds,
        traces=traces,
        polylines=polylines,
        segments=segments,
        budget=device.budget.summary(),
    )
