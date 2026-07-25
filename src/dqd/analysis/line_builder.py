"""
line_builder.py — build STRAIGHT LINES out of the measured transition cells,
then read the break points off those lines.

Motivation
----------
The old detector (analysis/interdot_simple.py) looked for break points by
finite-differencing a thick, ragged band of cells directly.  That is fragile:
the band is 4-6 cells wide, doubles back on itself, and a single smoothing
window has to serve both short and long lines.  Break points get missed.

This module changes the representation first, then does the geometry:

  1.  CELLS      Take every transition cell the sweeps found (the red X's in
                 summary.png) and snap them to the voltage grid.
  2.  CHAINS     Link cells that touch into chains (a chain is one zig-zag
                 honeycomb boundary, not yet a straight line).
  3.  SEGMENTS   Inside each chain, pull out straight sub-segments with
                 RANSAC + total-least-squares.  These are the "big lines":
                 each one is an exact analytic line (point + direction), not
                 a cloud of cells.  ==> lines_fitted.png
  4.  BREAKS     A break point is where two fitted segments with clearly
                 different slopes have neighbouring ends.  Its position is
                 the INTERSECTION of the two analytic lines, so it is
                 sub-pixel accurate and does not depend on any smoothing.
  5.  INTERDOT   Connect break points that face each other across a gap; the
                 strongest sensor-gradient cell along that connector is the
                 interdot transition.  ==> lines_breakpoints.png

Only numpy + matplotlib are used, so this runs anywhere the pipeline runs.
It is standalone: it post-processes a finished sample folder and never
touches the sweep pipeline.
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]          # (row, col) on the voltage grid


# ======================================================================
# Segment container
# ======================================================================

class Segment:
    """One straight fitted line: analytic form plus the cells supporting it."""

    def __init__(self, seg_id: int, chain_id: int, cells: np.ndarray,
                 origin: np.ndarray, direction: np.ndarray):
        self.id = seg_id
        self.chain_id = chain_id
        self.cells = cells                  # (N, 2) float, (row, col)
        self.origin = origin                # (2,) point on the line
        self.direction = direction          # (2,) unit vector
        t = (cells - origin) @ direction
        self.start = origin + direction * t.min()
        self.end = origin + direction * t.max()
        self.length = float(t.max() - t.min())
        self.n_cells = len(cells)

    @property
    def angle_deg(self) -> float:
        """Direction angle in [0, 180), so a line equals its reverse."""
        a = np.degrees(np.arctan2(self.direction[0], self.direction[1]))
        return a % 180.0

    def endpoints(self) -> List[np.ndarray]:
        return [self.start, self.end]


def _angle_gap(a: float, b: float) -> float:
    """Smallest angle between two undirected directions, in degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# ======================================================================
# Public entry point
# ======================================================================

def build_lines(
    sample_dir: str,
    vx_min: float = -1.0,
    vx_max: float = 1.0,
    vy_min: float = -1.0,
    vy_max: float = 1.0,
    charge_sensing_path: Optional[str] = None,
    # --- step 2: cells -> chains -----------------------------------
    link_px: int = 2,
    min_chain_cells: int = 12,
    # --- step 3: chains -> straight segments ------------------------
    inlier_px: float = 1.5,
    min_segment_cells: int = 12,
    min_segment_length: float = 6.0,
    max_segments_per_chain: int = 12,
    seg_gap_px: float = 3.0,
    ransac_trials: int = 400,
    seed: int = 0,
    # --- step 4: segments -> break points ---------------------------
    min_angle_deg: float = 20.0,
    junction_px: float = 8.0,
    # --- step 5: break points -> interdot ---------------------------
    connect_px: float = 50.0,
    same_chain_connect: bool = False,
    plot_dpi: int = 200,
) -> Dict:
    """
    Fit straight lines to a sample's transition cells and derive break points.

    Parameters
    ----------
    sample_dir          : the sample folder (…/sample_i)
    vx_min/max, vy_min/max : voltage extent of the grid
    charge_sensing_path : charge_sensing_data.npy; if None it is looked up at
                          <sample_dir>/numpy/simulation/charge_sensing_data.npy.
                          Used only to refine interdot peaks on the gradient.

    link_px             : cells within this grid distance join the same chain.
                          Small = chains stay separate; large = everything
                          merges into one blob.
    min_chain_cells     : ignore chains with fewer cells than this (noise).

    inlier_px           : perpendicular distance for a cell to support a line.
                          Set it near the half-thickness of the measured band.
    min_segment_cells   : a fitted segment needs at least this many cells.
    min_segment_length  : …and must span at least this many grid pixels.
    max_segments_per_chain : safety cap on the RANSAC peel-off loop.
    seg_gap_px          : inliers separated by a bigger gap along the line
                          belong to different segments; only the longest
                          uninterrupted run is kept per RANSAC hit.
    ransac_trials       : random 2-point hypotheses per segment.
    seed                : RNG seed, so the fit is reproducible.

    min_angle_deg       : two segments count as "different slope" (and so can
                          form a break point) only above this angle.
    junction_px         : their endpoints must be at least this close.

    connect_px          : max length of a break-point-to-break-point connector.
    same_chain_connect  : allow connecting break points of the same chain.

    Returns
    -------
    dict with keys: cells, chains, segments, breaks, connections, interdot.
    Writes lines_fitted.png, lines_breakpoints.png and lines_report.json
    into sample_dir.
    """
    cells = load_transition_cells(sample_dir, vx_min, vx_max, vy_min, vy_max)
    if len(cells) < min_chain_cells:
        print(f"[line_builder] only {len(cells)} transition cells — skipping.")
        return {"cells": cells, "chains": [], "segments": [], "breaks": [],
                "connections": [], "interdot": []}

    # ---- 2. chains -------------------------------------------------
    chains = _chains(cells, link_px, min_chain_cells)

    # ---- 3. straight segments --------------------------------------
    segments: List[Segment] = []
    rng = np.random.default_rng(seed)
    for chain_id, chain in enumerate(chains):
        segments.extend(
            _fit_segments(
                np.array(chain, dtype=float), chain_id, len(segments), rng,
                inlier_px=inlier_px,
                min_cells=min_segment_cells,
                min_length=min_segment_length,
                max_segments=max_segments_per_chain,
                gap_px=seg_gap_px,
                trials=ransac_trials,
            )
        )

    # ---- 4. break points -------------------------------------------
    breaks = _break_points(segments, min_angle_deg, junction_px)

    # ---- 5. connectors + interdot peaks ----------------------------
    connections = _connect_breaks(breaks, connect_px, same_chain_connect)
    grad = _load_gradient(sample_dir, charge_sensing_path)
    interdot = _refine_on_gradient(connections, grad)

    print(f"[line_builder] {len(cells)} cells -> {len(chains)} chains -> "
          f"{len(segments)} straight lines -> {len(breaks)} break points -> "
          f"{len(connections)} connections -> {len(interdot)} interdot peaks")

    # ---- output -----------------------------------------------------
    res_y, res_x = _grid_shape(sample_dir, grad)
    extent = (vx_min, vx_max, vy_min, vy_max)
    _plot_lines(sample_dir, cells, segments, extent, res_x, res_y, plot_dpi)
    _plot_breaks(sample_dir, cells, segments, breaks, connections, interdot,
                 extent, res_x, res_y, plot_dpi)
    result = {
        "cells": cells,
        "chains": chains,
        "segments": segments,
        "breaks": breaks,
        "connections": connections,
        "interdot": interdot,
    }
    _write_report(sample_dir, result, extent, res_x, res_y)
    return result


# ======================================================================
# Step 1 — load the transition cells
# ======================================================================

def load_transition_cells(sample_dir: str, vx_min: float, vx_max: float,
                          vy_min: float, vy_max: float) -> List[Cell]:
    """
    Every transition cell found by the sweeps, as de-duplicated grid (row, col).

    Prefers all_scanned_peaks_overlay_binary.txt (exactly the red X's drawn in
    summary.png); falls back to walking cropped_results/ if it is absent.
    """
    res_x, res_y = _grid_resolution(sample_dir)
    voltages: List[Tuple[float, float]] = []

    txt = os.path.join(sample_dir, "all_scanned_peaks_overlay_binary.txt")
    if os.path.isfile(txt):
        voltages = _parse_peaks_section(open(txt).read())
    else:
        cropped = os.path.join(sample_dir, "cropped_results")
        for root, _, files in os.walk(cropped):
            if "interdot" in os.path.basename(root).lower():
                continue          # never re-ingest our own output
            for fname in files:
                if fname.lower() == "voltage_coordinates.txt":
                    with open(os.path.join(root, fname)) as f:
                        voltages.extend(_parse_peaks_section(f.read()))

    seen = set()
    for vx, vy in voltages:
        col = int(np.clip((vx - vx_min) / (vx_max - vx_min) * res_x, 0, res_x - 1))
        row = int(np.clip((vy - vy_min) / (vy_max - vy_min) * res_y, 0, res_y - 1))
        seen.add((row, col))
    return sorted(seen)


def _parse_peaks_section(txt: str) -> List[Tuple[float, float]]:
    """Pull the '(x, y)' pairs that follow the Peaks header."""
    idx = txt.find("Peaks (Voltage Coordinates):")
    if idx < 0:
        return []
    return [(float(a), float(b)) for a, b in
            re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", txt[idx:])]


def _grid_resolution(sample_dir: str) -> Tuple[int, int]:
    """(res_x, res_y) from hyperparameters.json, defaulting to 100x100."""
    path = os.path.join(sample_dir, "hyperparameters.json")
    if os.path.isfile(path):
        try:
            vs = json.load(open(path)).get("voltage_sweep", {})
            return int(vs.get("n_points_x", 100)), int(vs.get("n_points_y", 100))
        except Exception:
            pass
    return 100, 100


# ======================================================================
# Step 2 — cells -> chains  (union-find over a local neighbourhood)
# ======================================================================

def _chains(cells: Sequence[Cell], link_px: int,
            min_chain_cells: int) -> List[List[Cell]]:
    """Group touching cells.  Neighbour lookup is O(cells * link_px^2)."""
    index = {c: i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    offsets = [(dr, dc)
               for dr in range(-link_px, link_px + 1)
               for dc in range(-link_px, link_px + 1)
               if (dr, dc) != (0, 0) and dr * dr + dc * dc <= link_px * link_px]

    for (r, c), i in index.items():
        for dr, dc in offsets:
            j = index.get((r + dr, c + dc))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: Dict[int, List[Cell]] = {}
    for cell, i in index.items():
        groups.setdefault(find(i), []).append(cell)
    return [g for g in groups.values() if len(g) >= min_chain_cells]


# ======================================================================
# Step 3 — chain -> straight segments  (RANSAC, peeled off one at a time)
# ======================================================================

def _fit_segments(points: np.ndarray, chain_id: int, next_id: int,
                  rng: np.random.Generator, inlier_px: float, min_cells: int,
                  min_length: float, max_segments: int, gap_px: float,
                  trials: int) -> List[Segment]:
    """Repeatedly extract the best straight line, then remove its cells."""
    segments: List[Segment] = []
    remaining = points.copy()

    while len(remaining) >= min_cells and len(segments) < max_segments:
        hit = _ransac_line(remaining, inlier_px, trials, rng)
        if hit is None:
            break
        mask = hit
        idx = np.flatnonzero(mask)
        if len(idx) < min_cells:
            break

        # Keep only the longest uninterrupted run: a straight line can pick up
        # collinear cells from a far-away part of the chain, which would stretch
        # the segment across a gap that is not really on the line.
        run = _longest_run(remaining[idx], gap_px)
        if len(run) < min_cells:
            remaining = np.delete(remaining, idx, axis=0)
            continue
        keep = idx[run]

        origin, direction = _tls_fit(remaining[keep])
        seg = Segment(next_id + len(segments), chain_id, remaining[keep],
                      origin, direction)
        if seg.length >= min_length:
            segments.append(seg)
        remaining = np.delete(remaining, keep, axis=0)

    return segments


def _ransac_line(points: np.ndarray, inlier_px: float, trials: int,
                 rng: np.random.Generator) -> Optional[np.ndarray]:
    """Boolean inlier mask of the best 2-point line hypothesis."""
    n = len(points)
    if n < 2:
        return None
    best_mask, best_count = None, 0
    i1 = rng.integers(0, n, size=trials)
    i2 = rng.integers(0, n, size=trials)
    for a, b in zip(i1, i2):
        if a == b:
            continue
        p, q = points[a], points[b]
        d = q - p
        norm = np.hypot(*d)
        if norm < 1e-9:
            continue
        d = d / norm
        # perpendicular distance of every point to the line p + t*d
        rel = points - p
        perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        mask = perp <= inlier_px
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    return best_mask


def _longest_run(pts: np.ndarray, gap_px: float) -> np.ndarray:
    """Indices (into pts) of the longest gap-free run along the line."""
    origin, direction = _tls_fit(pts)
    t = (pts - origin) @ direction
    order = np.argsort(t)
    ts = t[order]
    breaks = np.flatnonzero(np.diff(ts) > gap_px)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [len(ts)]))
    best = max(zip(starts, stops), key=lambda se: se[1] - se[0])
    return order[best[0]:best[1]]


def _tls_fit(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line: centroid + unit principal direction."""
    origin = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - origin, full_matrices=False)
    return origin, vh[0]


# ======================================================================
# Step 4 — segments -> break points  (analytic line intersections)
# ======================================================================

def _break_points(segments: List[Segment], min_angle_deg: float,
                  junction_px: float) -> List[Dict]:
    """
    A break point is a meeting of two fitted lines whose slopes differ.

    Because both lines are analytic, the vertex is their exact intersection
    rather than a cell picked out of a noisy band.
    """
    breaks: List[Dict] = []
    for i, s1 in enumerate(segments):
        for s2 in segments[i + 1:]:
            if _angle_gap(s1.angle_deg, s2.angle_deg) < min_angle_deg:
                continue
            pair = min(
                ((np.hypot(*(e1 - e2)), e1, e2)
                 for e1 in s1.endpoints() for e2 in s2.endpoints()),
                key=lambda x: x[0],
            )
            dist, e1, e2 = pair
            if dist > junction_px:
                continue
            xp = _intersect(s1, s2)
            # An intersection far outside the junction means the lines are
            # nearly parallel or meet somewhere else entirely; fall back to
            # the midpoint of the two ends.
            if xp is None or np.hypot(*(xp - 0.5 * (e1 + e2))) > junction_px:
                xp = 0.5 * (e1 + e2)
            breaks.append({
                "point": xp,
                "seg_ids": (s1.id, s2.id),
                "chain_ids": (s1.chain_id, s2.chain_id),
                "angle": _angle_gap(s1.angle_deg, s2.angle_deg),
                "end_gap": float(dist),
            })
    return breaks


def _intersect(s1: Segment, s2: Segment) -> Optional[np.ndarray]:
    """Intersection of two infinite lines, or None if parallel."""
    a = np.array([s1.direction, -s2.direction]).T
    det = np.linalg.det(a)
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(a, s2.origin - s1.origin)
    return s1.origin + t[0] * s1.direction


# ======================================================================
# Step 5 — break points -> connectors -> interdot peaks
# ======================================================================

def _connect_breaks(breaks: List[Dict], connect_px: float,
                    same_chain: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Pair each break point with its nearest facing break point.

    Unlike the old greedy pass, a break point is not retired after one match:
    every break point gets its nearest partner, and duplicate pairs are
    removed at the end.  A honeycomb vertex that faces two neighbours
    therefore yields two connectors instead of one.
    """
    out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for i, bi in enumerate(breaks):
        best_j, best_d = None, float(connect_px)
        for j, bj in enumerate(breaks):
            if j == i:
                continue
            if not same_chain and set(bi["chain_ids"]) & set(bj["chain_ids"]):
                continue          # same honeycomb boundary — not an interdot
            d = float(np.hypot(*(bi["point"] - bj["point"])))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            continue
        key = (min(i, best_j), max(i, best_j))
        out.setdefault(key, (bi["point"], breaks[best_j]["point"]))
    return list(out.values())


def _refine_on_gradient(connections, grad: Optional[np.ndarray]) -> List[Cell]:
    """Strongest sensor-gradient cell along each connector."""
    if grad is None:
        return [tuple(np.round(0.5 * (p1 + p2)).astype(int))
                for p1, p2 in connections]
    m, n = grad.shape
    peaks: List[Cell] = []
    for p1, p2 in connections:
        steps = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), 1)) * 2 + 1
        rows = np.round(np.linspace(p1[0], p2[0], steps)).astype(int)
        cols = np.round(np.linspace(p1[1], p2[1], steps)).astype(int)
        seg = [(r, c) for r, c in zip(rows, cols) if 0 <= r < m and 0 <= c < n]
        if seg:
            peaks.append(max(seg, key=lambda rc: grad[rc[0], rc[1]]))
    return peaks


def _load_gradient(sample_dir: str,
                   charge_sensing_path: Optional[str]) -> Optional[np.ndarray]:
    """Gradient magnitude of the charge-sensor map, or None if unavailable."""
    if charge_sensing_path is None:
        charge_sensing_path = os.path.join(
            sample_dir, "numpy", "simulation", "charge_sensing_data.npy")
    if not os.path.isfile(charge_sensing_path):
        print("[line_builder] no charge-sensing data — "
              "interdot peaks fall back to connector midpoints.")
        return None
    data = np.load(charge_sensing_path)
    ux, uy = np.unique(data[:, 0]), np.unique(data[:, 1])
    current = data[:, 2].reshape(len(uy), len(ux))
    gy, gx = np.gradient(current)
    return np.hypot(gx, gy)


# ======================================================================
# Plotting
# ======================================================================

def _cells_to_v(cells, extent, res_x, res_y) -> np.ndarray:
    """Grid (row, col) -> voltage (vx, vy) at cell centres."""
    vx_min, vx_max, vy_min, vy_max = extent
    a = np.asarray(cells, dtype=float).reshape(-1, 2)
    vx = vx_min + (a[:, 1] + 0.5) * (vx_max - vx_min) / res_x
    vy = vy_min + (a[:, 0] + 0.5) * (vy_max - vy_min) / res_y
    return np.column_stack([vx, vy])


def _grid_shape(sample_dir: str, grad: Optional[np.ndarray]) -> Tuple[int, int]:
    if grad is not None:
        return grad.shape
    res_x, res_y = _grid_resolution(sample_dir)
    return res_y, res_x


def _base_axes(extent, title: str):
    vx_min, vx_max, vy_min, vy_max = extent
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")
    ax.set_xlim(vx_min, vx_max)
    ax.set_ylim(vy_min, vy_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Vx (V)")
    ax.set_ylabel("Vy (V)")
    ax.set_title(title)
    ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    return fig, ax


def _plot_lines(sample_dir, cells, segments, extent, res_x, res_y, dpi):
    """Clean image: measured cells in grey, every fitted straight line on top."""
    fig, ax = _base_axes(extent, "Transition cells fitted to straight lines")

    if cells:
        v = _cells_to_v(cells, extent, res_x, res_y)
        ax.scatter(v[:, 0], v[:, 1], s=6, c="0.75", zorder=1,
                   label=f"Transition cells ({len(cells)})")

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(segments), 1)))
    for k, seg in enumerate(segments):
        ends = _cells_to_v([seg.start, seg.end], extent, res_x, res_y)
        ax.plot(ends[:, 0], ends[:, 1], "-", color=colors[k % len(colors)],
                lw=3, solid_capstyle="round", zorder=3)
        mid = ends.mean(axis=0)
        ax.annotate(str(seg.id), mid, fontsize=7, color="black", zorder=4,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.7))
    ax.plot([], [], "-", color="tab:blue", lw=3,
            label=f"Fitted lines ({len(segments)})")

    ax.legend(loc="upper right", framealpha=0.95)
    out = os.path.join(sample_dir, "lines_fitted.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[line_builder] plot saved -> {out}")


def _plot_breaks(sample_dir, cells, segments, breaks, connections, interdot,
                 extent, res_x, res_y, dpi):
    """The same lines plus break points, connectors and interdot peaks."""
    fig, ax = _base_axes(extent, "Break points and interdot transitions")

    if cells:
        v = _cells_to_v(cells, extent, res_x, res_y)
        ax.scatter(v[:, 0], v[:, 1], s=5, c="0.85", zorder=1)

    for seg in segments:
        ends = _cells_to_v([seg.start, seg.end], extent, res_x, res_y)
        ax.plot(ends[:, 0], ends[:, 1], "-", color="tab:blue", lw=2, zorder=2)
    ax.plot([], [], "-", color="tab:blue", lw=2,
            label=f"Fitted lines ({len(segments)})")

    for p1, p2 in connections:
        v = _cells_to_v([p1, p2], extent, res_x, res_y)
        ax.plot(v[:, 0], v[:, 1], "--", color="green", lw=1.5, zorder=4)
    if connections:
        ax.plot([], [], "--", color="green", lw=1.5,
                label=f"Connections ({len(connections)})")

    if breaks:
        v = _cells_to_v([b["point"] for b in breaks], extent, res_x, res_y)
        ax.scatter(v[:, 0], v[:, 1], s=70, c="orange", marker="^",
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"Break points ({len(breaks)})")
    if interdot:
        v = _cells_to_v(interdot, extent, res_x, res_y)
        ax.scatter(v[:, 0], v[:, 1], s=110, c="red", marker="x", linewidths=2.5,
                   zorder=6, label=f"Interdot peaks ({len(interdot)})")

    ax.legend(loc="upper right", framealpha=0.95)
    out = os.path.join(sample_dir, "lines_breakpoints.png")
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[line_builder] plot saved -> {out}")


def _write_report(sample_dir, result, extent, res_x, res_y) -> None:
    """Machine-readable dump of every line, break point and interdot peak."""
    def v(p):
        return _cells_to_v([p], extent, res_x, res_y)[0].tolist()

    report = {
        "n_cells": len(result["cells"]),
        "n_chains": len(result["chains"]),
        "segments": [
            {
                "id": s.id,
                "chain_id": s.chain_id,
                "n_cells": s.n_cells,
                "length_px": round(s.length, 2),
                "angle_deg": round(s.angle_deg, 2),
                "start_v": v(s.start),
                "end_v": v(s.end),
            }
            for s in result["segments"]
        ],
        "break_points": [
            {
                "point_v": v(b["point"]),
                "seg_ids": list(b["seg_ids"]),
                "chain_ids": list(b["chain_ids"]),
                "angle_deg": round(b["angle"], 2),
                "end_gap_px": round(b["end_gap"], 2),
            }
            for b in result["breaks"]
        ],
        "interdot_v": [v(p) for p in result["interdot"]],
    }
    out = os.path.join(sample_dir, "lines_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[line_builder] report saved -> {out}")
