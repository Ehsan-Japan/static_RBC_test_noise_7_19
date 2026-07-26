"""
interdot_simple.py — SIMPLE sample-level interdot transition detector.

Runs AFTER the whole sweeping process is finished:

  1. Break points come from the SWEEPS themselves.  Each peak is swept twice —
     once walking up the transition line, once walking down — and each sweep
     traces out a slope.  If the two slopes differ by more than
     `min_slope_change`, the line bends at that peak: it is a honeycomb vertex,
     i.e. a break point.  (See :func:`slope_difference`, called per peak by
     DatasetPipeline.)
  2. Connect the two CLOSEST break points.  That short gap between two
     dot-to-lead lines is where an interdot transition sits.
  3. Scan the charge-sensor GRADIENT along that connecting segment and take
     the strongest point as the interdot peak.  (Interdot lines are almost
     invisible in the raw sensor amplitude but show up clearly in the
     gradient, which is why the amplitude-based row sweeps miss them.)

Earlier versions found break points by clustering all the sweep peaks into
lines and hunting for a kink along each one.  That needed four hyperparameters
(neighbour radius, kink window, window cap, min line points) and two extra
sweeps per peak.  Using the slopes the sweeps already produce needs none of
them.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

MIN_SLOPE_POINTS = 3   # peaks a sweep needs before its slope means anything

from ..config.figure_style import (
    apply_voltage_axes,
    new_map_figure,
    save_figure,
)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def slope_difference(sweep_results: Dict) -> Optional[float]:
    """
    |slope_up - slope_down| for one peak, or None if it cannot be measured.

    Each sweep tracks the transition line away from the peak and records one
    peak column per row.  A straight line gives both sweeps the same slope; a
    honeycomb vertex bends it, so the two disagree.  That disagreement is the
    break-point test.

    Slope is dcol/drow in grid-pixel units, from a least-squares fit over the
    cells the sweep actually locked onto.  A sweep that found fewer than
    MIN_SLOPE_POINTS peaks, or never moved off one row, has no measurable
    slope and the peak is skipped.
    """
    slopes = []
    for sr in sweep_results.values():
        pts = [(r, c) for r, c in sr.get("peaks", []) if c is not None]
        if len(pts) < MIN_SLOPE_POINTS:
            continue
        rows = np.array([p[0] for p in pts], dtype=float)
        cols = np.array([p[1] for p in pts], dtype=float)
        if np.ptp(rows) == 0:
            continue
        slopes.append(float(np.polyfit(rows, cols, 1)[0]))
    if len(slopes) < 2:
        return None
    return abs(slopes[0] - slopes[1])


def detect_interdot(
    sample_dir: str,
    charge_sensing_path: str,
    vx_min: float,
    vx_max: float,
    vy_min: float,
    vy_max: float,
    break_points: Optional[List[Tuple[int, int]]] = None,
    connect_px: int = 50,
) -> List[Tuple[float, float]]:
    """
    Connect break points and locate the interdot transition between them.

    Parameters
    ----------
    sample_dir          : the sample folder (…/sample_i)
    charge_sensing_path : path to charge_sensing_data.npy  [Vx, Vy, z]
    vx_min/max, vy_min/max : voltage extent (for plotting)
    break_points        : (row, col) grid cells flagged as break points by the
                          per-peak slope test.  Supplied by DatasetPipeline.
    connect_px          : max gap, in grid pixels, between two break points for
                          them to be connected

    Returns
    -------
    List of (vx, vy) interdot peaks.  Also writes interdot_transitions.txt,
    break_points.txt, cropped_results/interdot_peaks/voltage_coordinates.txt
    and summary_with_interdot.png.
    """
    break_points = list(break_points or [])
    if len(break_points) < 2:
        print(f"[interdot_simple] {len(break_points)} break point(s) — "
              "need at least 2 to connect; skipping.")
        return []

    # --- load the sensor grid and its gradient magnitude ---
    data = np.load(charge_sensing_path)
    ux = np.unique(data[:, 0])
    uy = np.unique(data[:, 1])
    current_2d = data[:, 2].reshape(len(uy), len(ux))
    gy, gx = np.gradient(current_2d)
    grad = np.hypot(gx, gy)                       # strong on every transition line
    m, n = current_2d.shape

    # Every break point comes from a different peak, so each gets its own id
    # and any two of them may be connected.
    break_pts = list(enumerate(break_points))

    # --- connect the closest break points ---
    connections = _connect(break_pts, connect_px)

    # --- scan the gradient along each connection, keep the strongest cell ---
    interdot_grid: List[Tuple[int, int]] = []
    for (r1, c1), (r2, c2) in connections:
        steps = max(abs(r2 - r1), abs(c2 - c1), 1) * 2 + 1
        rows = np.round(np.linspace(r1, r2, steps)).astype(int)
        cols = np.round(np.linspace(c1, c2, steps)).astype(int)
        seg = [(r, c) for r, c in zip(rows, cols) if 0 <= r < m and 0 <= c < n]
        if not seg:
            continue
        best = max(seg, key=lambda rc: grad[rc[0], rc[1]])
        interdot_grid.append(best)

    interdot_v = [_to_voltage(r, c, ux, uy) for r, c in interdot_grid]
    print(f"[interdot_simple] {len(break_pts)} break points -> "
          f"{len(connections)} connections -> "
          f"{len(interdot_v)} interdot peaks")

    peaks_v = _load_sample_peaks(sample_dir)        # for the plot only
    _write_txt(sample_dir, interdot_v)              # tracking log (sample level)
    _write_voltage_coords(sample_dir, interdot_v)   # feeds images + evaluation
    _write_break_points(                            # feeds the *_with_breaking_points images
        sample_dir,
        [_to_voltage(r, c, ux, uy) for _, (r, c) in break_pts],
        [(_to_voltage(r1, c1, ux, uy), _to_voltage(r2, c2, ux, uy))
         for (r1, c1), (r2, c2) in connections],
    )
    _plot(sample_dir, current_2d, peaks_v, break_pts, connections, interdot_v,
          ux, uy, vx_min, vx_max, vy_min, vy_max)
    return interdot_v


# ----------------------------------------------------------------------
# Connect the closest break points
# ----------------------------------------------------------------------

def _connect(
    break_pts: List[Tuple[int, Tuple[int, int]]], connect_px: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    connections = []
    used: set = set()
    for i, (li, pi) in enumerate(break_pts):
        if i in used:
            continue
        best_j, best_d = None, float(connect_px)
        for j, (lj, pj) in enumerate(break_pts):
            if j == i or j in used or lj == li:      # must be a DIFFERENT line
                continue
            d = np.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            continue
        used.add(i)
        used.add(best_j)
        connections.append((pi, break_pts[best_j][1]))
    return connections


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _to_grid(vx, vy, ux, uy):
    return (int(np.argmin(np.abs(uy - vy))), int(np.argmin(np.abs(ux - vx))))


def _to_voltage(row, col, ux, uy):
    return (float(ux[col]), float(uy[row]))


def _load_sample_peaks(sample_dir: str) -> List[Tuple[float, float]]:
    """
    Collect every sweep-detected peak in the sample by walking
    cropped_results/ray_*/peak_*/voltage_coordinates.txt.

    The interdot_peaks folder (our own output) is skipped so re-running the
    pipeline never feeds previously-detected interdot peaks back in as if they
    were sweep peaks.
    """
    peaks: List[Tuple[float, float]] = []
    cropped = os.path.join(sample_dir, "cropped_results")
    if not os.path.isdir(cropped):
        print(f"[interdot_simple] cropped_results not found: {cropped}")
        return peaks

    for root, _, files in os.walk(cropped):
        if "interdot" in os.path.basename(root).lower():
            continue
        for fname in files:
            if fname.lower() == "voltage_coordinates.txt":
                peaks.extend(_parse_peaks(os.path.join(root, fname)))
    return peaks


def _parse_peaks(path: str) -> List[Tuple[float, float]]:
    """Extract the Peaks section of a voltage_coordinates.txt file."""
    with open(path) as f:
        txt = f.read()
    idx = txt.find("Peaks (Voltage Coordinates):")
    if idx < 0:
        return []
    seg = txt[idx:]
    return [
        (float(a), float(b))
        for a, b in re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", seg)
    ]


def _write_voltage_coords(
    sample_dir: str, interdot_v: List[Tuple[float, float]]
) -> None:
    """
    Write interdot peaks in the standard voltage_coordinates.txt format under
    cropped_results/interdot_peaks/.  Both the sample overlays and the
    evaluator walk cropped_results, so this is all that is needed for the
    interdot peaks to appear in the final images and count in evaluation.txt.
    """
    out_dir = os.path.join(sample_dir, "cropped_results", "interdot_peaks")
    os.makedirs(out_dir, exist_ok=True)
    peaks_str = ", ".join(f"({vx:.6f}, {vy:.6f})" for vx, vy in interdot_v)
    with open(os.path.join(out_dir, "voltage_coordinates.txt"), "w") as f:
        f.write("Scanned Cells (Voltage Coordinates):\n\n")
        f.write("Peaks (Voltage Coordinates):\n")
        f.write(peaks_str + "\n")


BREAK_POINTS_FILE = "break_points.txt"


def _write_break_points(
    sample_dir: str,
    break_v: List[Tuple[float, float]],
    connections_v: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> None:
    """
    Save the break points and the pairs that were connected, in voltage
    coordinates, so the sample overlays can draw them without re-running the
    detection.  Read back with :func:`load_break_points`.
    """
    out = os.path.join(sample_dir, BREAK_POINTS_FILE)
    with open(out, "w") as f:
        f.write("Break Points (Voltage Coordinates):\n")
        f.write(", ".join(f"({vx:.6f}, {vy:.6f})" for vx, vy in break_v) + "\n\n")
        f.write("Connections (Voltage Coordinate Pairs):\n")
        for (vx1, vy1), (vx2, vy2) in connections_v:
            f.write(f"({vx1:.6f}, {vy1:.6f}) -> ({vx2:.6f}, {vy2:.6f})\n")
    print(f"[interdot_simple] saved -> {out}")


def load_break_points(
    sample_dir: str,
) -> Tuple[List[Tuple[float, float]],
           List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Inverse of :func:`_write_break_points`; ([], []) if the file is missing."""
    path = os.path.join(sample_dir, BREAK_POINTS_FILE)
    if not os.path.isfile(path):
        return [], []
    with open(path) as f:
        txt = f.read()

    def _pairs(seg: str) -> List[Tuple[float, float]]:
        return [(float(a), float(b)) for a, b in
                re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", seg)]

    head, _, tail = txt.partition("Connections (Voltage Coordinate Pairs):")
    break_v = _pairs(head)
    connections_v = []
    for line in tail.splitlines():
        pts = _pairs(line)
        if len(pts) == 2:
            connections_v.append((pts[0], pts[1]))
    return break_v, connections_v


def _write_txt(sample_dir: str, interdot_v: List[Tuple[float, float]]) -> None:
    out = os.path.join(sample_dir, "interdot_transitions.txt")
    with open(out, "w") as f:
        f.write("Interdot Transition Peaks (Voltage Coordinates)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total detected: {len(interdot_v)}\n\n")
        for vx, vy in interdot_v:
            f.write(f"({vx:.6f}, {vy:.6f})\n")
    print(f"[interdot_simple] saved -> {out}")


def _plot(sample_dir, current_2d, peaks_v, break_pts, connections, interdot_v,
          ux, uy, vx_min, vx_max, vy_min, vy_max) -> None:
    x_edges = np.linspace(vx_min, vx_max, current_2d.shape[1] + 1)
    y_edges = np.linspace(vy_min, vy_max, current_2d.shape[0] + 1)

    fig, ax, _ = new_map_figure()
    ax.pcolormesh(x_edges, y_edges, current_2d, cmap="hot")

    if peaks_v:
        a = np.array(peaks_v)
        ax.scatter(a[:, 0], a[:, 1], c="cyan", s=8, label="Sweep peaks")
    if break_pts:
        a = np.array([_to_voltage(r, c, ux, uy) for _, (r, c) in break_pts])
        ax.scatter(a[:, 0], a[:, 1], c="orange", s=40, marker="^",
                   label="Break points")
    for (r1, c1), (r2, c2) in connections:
        v1 = _to_voltage(r1, c1, ux, uy)
        v2 = _to_voltage(r2, c2, ux, uy)
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], "g--", lw=1)
    if interdot_v:
        a = np.array(interdot_v)
        ax.scatter(a[:, 0], a[:, 1], c="red", s=90, marker="x", linewidths=2,
                   label="Interdot peaks")

    apply_voltage_axes(ax, vx_min, vx_max, vy_min, vy_max)
    ax.legend(loc="upper right")
    out = os.path.join(sample_dir, "summary_with_interdot.png")
    save_figure(fig, out)
    print(f"[interdot_simple] plot saved -> {out}")
