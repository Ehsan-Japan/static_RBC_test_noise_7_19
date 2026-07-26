"""
interdot_simple.py — SIMPLE sample-level interdot transition detector.

Runs AFTER the whole sweeping process is finished.  The idea (exactly the
break-point method):

  1. Take every transition peak the sweeps found in the sample.
  2. Group the peaks into lines (points close together = same dot-to-lead line).
  3. Each line has two ends; those ends are the "break points" — the places
     where a tracked line stops / its slope breaks at a honeycomb vertex.
  4. Connect the two CLOSEST break points that belong to DIFFERENT lines.
     That short gap between two lines is where an interdot transition sits.
  5. Scan the charge-sensor GRADIENT along that connecting segment and take
     the strongest point as the interdot peak.  (Interdot lines are almost
     invisible in the raw sensor amplitude but show up clearly in the
     gradient, which is why the amplitude-based row sweeps miss them.)

This module is intentionally small and self-contained.  It does not touch the
existing sweep pipeline; it only post-processes its results.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple

from ..config.axis_labels import x_label, y_label


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def detect_interdot(
    sample_dir: str,
    charge_sensing_path: str,
    vx_min: float,
    vx_max: float,
    vy_min: float,
    vy_max: float,
    neighbor_px: int = 4,
    connect_px: int = 50,
    min_slope_change: float = 0.8,
    kink_window_frac: float = 1.0 / 6.0,
    kink_window_max: int = 0,
    min_line_pts: int = 5,
) -> List[Tuple[float, float]]:
    """
    Detect interdot transition peaks for one sample and save results.

    Parameters
    ----------
    sample_dir          : the sample folder (…/sample_i)
    charge_sensing_path : path to charge_sensing_data.npy  [Vx, Vy, z]
    vx_min/max, vy_min/max : voltage extent (for plotting)
    neighbor_px         : two peaks this close (grid pixels) are the same line.
                          Smaller keeps nearly-touching lines separate (more
                          break-point pairs survive the "different line" rule);
                          larger merges thick/broken bands into one line.
    connect_px          : max gap between break points of different lines
    min_slope_change    : a line only yields a break point if its local slope
                          changes by more than this (grid-pixel slope units).
                          THE dominant reason break points go undetected —
                          shallow honeycomb vertices change the slope by only
                          ~0.3-0.6 and are silently rejected at 0.8.
    kink_window_frac    : slope-comparison half-window as a fraction of the
                          line length.  Bigger = more smoothing = kinks on
                          long lines get straightened away.
    kink_window_max     : hard cap on that half-window in pixels (0 = no cap).
                          Capping it keeps long lines as kink-sensitive as
                          short ones.
    min_line_pts        : a group needs at least this many points (and this
                          many distinct steps along its long axis) to be
                          considered a line at all.

    Returns
    -------
    List of (vx, vy) interdot peaks.  Also writes:
       interdot_transitions.txt   and   summary_with_interdot.png
    """
    peaks_v = _load_sample_peaks(sample_dir)
    if len(peaks_v) < 2:
        print("[interdot_simple] not enough peaks — skipping.")
        return []

    # --- load the sensor grid and its gradient magnitude ---
    data = np.load(charge_sensing_path)
    ux = np.unique(data[:, 0])
    uy = np.unique(data[:, 1])
    current_2d = data[:, 2].reshape(len(uy), len(ux))
    gy, gx = np.gradient(current_2d)
    grad = np.hypot(gx, gy)                       # strong on every transition line
    m, n = current_2d.shape

    # --- voltage -> grid (row, col), de-duplicated ---
    grid_peaks = list({_to_grid(vx, vy, ux, uy) for vx, vy in peaks_v})

    # --- 2. group peaks into lines ---
    lines = _group_lines(grid_peaks, neighbor_px)

    # --- 3. break points = the two far ends of every line ---
    break_pts = _break_points(                    # list of (line_id, (row, col))
        lines,
        min_slope_change=min_slope_change,
        kink_window_frac=kink_window_frac,
        kink_window_max=kink_window_max,
        min_line_pts=min_line_pts,
    )

    # --- 4. connect closest break points from different lines ---
    connections = _connect(break_pts, connect_px)

    # --- 5. scan the gradient along each connection, keep the strongest cell ---
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
    print(f"[interdot_simple] {len(lines)} lines -> "
          f"{len(break_pts)} break points -> "
          f"{len(connections)} connections -> "
          f"{len(interdot_v)} interdot peaks")

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
# Step 2 — group peaks into lines (union-find on pixel distance)
# ----------------------------------------------------------------------

def _group_lines(
    grid_peaks: List[Tuple[int, int]], neighbor_px: int
) -> List[List[Tuple[int, int]]]:
    parent = list(range(len(grid_peaks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pts = np.array(grid_peaks, dtype=float)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(*(pts[i] - pts[j])) <= neighbor_px:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i, pk in enumerate(grid_peaks):
        groups.setdefault(find(i), []).append(pk)
    return list(groups.values())


# ----------------------------------------------------------------------
# Step 3 — break points: the kink of each line (where its slope changes)
# ----------------------------------------------------------------------

def _break_points(
    lines: List[List[Tuple[int, int]]],
    min_slope_change: float = 0.8,
    kink_window_frac: float = 1.0 / 6.0,
    kink_window_max: int = 0,
    min_line_pts: int = 5,
) -> List[Tuple[int, Tuple[int, int]]]:
    """
    One break point per line: the peak where the line's local slope changes
    the most (a honeycomb vertex).  Lines that are essentially straight
    (no slope change above min_slope_change) contribute no break point.
    """
    breaks: List[Tuple[int, Tuple[int, int]]] = []
    for line_id, line in enumerate(lines):
        kink = _line_kink(line, min_slope_change,
                          kink_window_frac, kink_window_max, min_line_pts)
        if kink is not None:
            breaks.append((line_id, kink))
    return breaks


def _line_kink(line, min_slope_change,
               kink_window_frac=1.0 / 6.0, kink_window_max=0, min_line_pts=5):
    """Return the (row, col) where the line bends the most, or None."""
    pts = np.array(line, dtype=float)
    if len(pts) < min_line_pts:
        return None

    # Track the line along whichever axis it spans more (row or col),
    # collapsing the thick band to one value per step (its mean).
    key, val = (0, 1) if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else (1, 0)
    uk = np.unique(pts[:, key])
    if len(uk) < min_line_pts:
        return None
    tv = np.array([pts[pts[:, key] == u, val].mean() for u in uk])

    # Compare the slope a window before vs a window after each interior point;
    # the largest change in slope is the kink.
    w = max(1, int(len(uk) * kink_window_frac))
    if kink_window_max > 0:
        w = min(w, kink_window_max)
    best = None
    for i in range(w, len(uk) - w):
        s_before = (tv[i] - tv[i - w]) / (uk[i] - uk[i - w] + 1e-9)
        s_after = (tv[i + w] - tv[i]) / (uk[i + w] - uk[i] + 1e-9)
        change = abs(s_after - s_before)
        if change > min_slope_change and (best is None or change > best[0]):
            best = (change, i)
    if best is None:
        return None

    i = best[1]
    return ((int(round(uk[i])), int(round(tv[i]))) if key == 0
            else (int(round(tv[i])), int(round(uk[i]))))


# ----------------------------------------------------------------------
# Step 4 — connect the closest break points of different lines
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

    fig, ax = plt.subplots(figsize=(10, 10))
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

    ax.set_xlim(vx_min, vx_max)
    ax.set_ylim(vy_min, vy_max)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label())
    ax.set_ylabel(y_label())
    ax.legend(loc="upper right")
    out = os.path.join(sample_dir, "summary_with_interdot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[interdot_simple] plot saved -> {out}")
