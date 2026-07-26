"""
InterdotDetector — sample-level post-processing that finds interdot transition
lines by analysing the GLOBAL set of already-detected sweep peaks.

Algorithm
---------
1. Load all detected peaks (voltage coords) from the sample's sweep output.
2. Convert to grid (row, col) coordinates.
3. Build connected components: peaks within *neighbor_px* of each other
   belong to the same line segment.
4. For every component find two kinds of "breaking points":
     a) Endpoints  — peaks with ≤ 1 in-component neighbour (line terminus).
     b) Kink points — peaks where the local direction changes sharply
        (slope sign flip or |Δslope| > kink_threshold).
   Both mark honeycomb vertices where an interdot transition begins.
5. Pair breaking points from DIFFERENT components that are within
   *connect_px* of each other (same vertex, different line families).
6. For each connected pair, scan the charge-sensor signal along the
   straight segment; the argmax is the interdot peak.
7. Write results and plot an updated overlay.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ..config.axis_labels import x_label, y_label


class InterdotDetector:
    """
    Detects interdot transition lines at the sample level.

    Parameters
    ----------
    sample_dir          : root directory of the sample (e.g. .../sample_1)
    charge_sensing_path : path to charge_sensing_data.npy  [Vx, Vy, z]
    vx_min/max          : voltage extent x
    vy_min/max          : voltage extent y
    neighbor_px         : max grid-pixel distance between consecutive peaks
                          that belong to the same transition line  (default 4)
    connect_px          : max grid-pixel distance between breaking points from
                          different components to be treated as the same
                          honeycomb vertex  (default 12)
    kink_threshold      : minimum absolute slope change (Δ col/row) that
                          counts as a kink (direction reversal)  (default 1.5)
    """

    def __init__(
        self,
        sample_dir: str,
        charge_sensing_path: str,
        vx_min: float,
        vx_max: float,
        vy_min: float,
        vy_max: float,
        neighbor_px: int = 4,
        connect_px: int = 12,
        kink_threshold: float = 1.5,
    ):
        self.sample_dir = sample_dir
        self.charge_sensing_path = charge_sensing_path
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max
        self.neighbor_px = neighbor_px
        self.connect_px = connect_px
        self.kink_threshold = kink_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> List[Tuple[float, float]]:
        """
        Execute the full interdot detection pipeline.

        Returns
        -------
        List of (vx, vy) voltage-coordinate tuples for detected interdot peaks.
        """
        # ---- 1. Load sweep peaks ----
        txt_path = os.path.join(
            self.sample_dir, "all_scanned_peaks_overlay_binary.txt"
        )
        if not os.path.isfile(txt_path):
            print(f"[InterdotDetector] peaks file not found: {txt_path}")
            return []

        _, peaks_v = self._parse_peaks_txt(txt_path)
        if len(peaks_v) < 4:
            print("[InterdotDetector] too few peaks — skipping.")
            return []

        # ---- 2. Load charge-sensor grid ----
        data = np.load(self.charge_sensing_path)
        Vx_flat, Vy_flat, z_flat = data[:, 0], data[:, 1], data[:, 2]
        unique_Vx = np.unique(Vx_flat)
        unique_Vy = np.unique(Vy_flat)
        current_2d = z_flat.reshape(len(unique_Vy), len(unique_Vx))
        m, n = current_2d.shape

        # ---- 3. Convert voltage → grid, deduplicate ----
        grid_peaks = list({
            self._v2g(vx, vy, unique_Vx, unique_Vy)
            for vx, vy in peaks_v
        })

        # ---- 4. Connected components ----
        components = self._connected_components(grid_peaks, self.neighbor_px)
        print(
            f"[InterdotDetector] {len(grid_peaks)} peaks → "
            f"{len(components)} line segments"
        )

        # ---- 5. Breaking points (endpoints + kinks) ----
        breaking_pts = self._breaking_points(
            components, self.neighbor_px, self.kink_threshold
        )
        print(f"[InterdotDetector] {len(breaking_pts)} breaking points found")

        # ---- 6. Connect nearest pairs from different components ----
        connections = self._connect(breaking_pts, self.connect_px)
        print(f"[InterdotDetector] {len(connections)} interdot connections")

        if not connections:
            print("[InterdotDetector] no connections found — done.")
            return []

        # ---- 7. Sweep along each connection, find peak ----
        interdot_grid: List[Tuple[int, int]] = []
        for (r1, c1), (r2, c2) in connections:
            n_pts = max(abs(r2 - r1), abs(c2 - c1), 1) * 2 + 1
            rows = np.round(np.linspace(r1, r2, n_pts)).astype(int)
            cols = np.round(np.linspace(c1, c2, n_pts)).astype(int)
            valid = [
                (r, c) for r, c in zip(rows, cols)
                if 0 <= r < m and 0 <= c < n
            ]
            if not valid:
                continue
            vals = [float(current_2d[r, c]) for r, c in valid]
            interdot_grid.append(valid[int(np.argmax(vals))])

        # ---- 8. Grid → voltage ----
        interdot_v = [
            self._g2v(r, c, unique_Vx, unique_Vy)
            for r, c in interdot_grid
        ]

        print(f"[InterdotDetector] {len(interdot_v)} interdot peaks detected")

        # ---- 9. Write outputs ----
        self._write_txt(interdot_v)
        self._write_plot(current_2d, peaks_v, interdot_v, breaking_pts,
                         connections, unique_Vx, unique_Vy)

        return interdot_v

    # ------------------------------------------------------------------
    # Step 4 — connected components (union-find)
    # ------------------------------------------------------------------

    @staticmethod
    def _connected_components(
        grid_peaks: List[Tuple[int, int]],
        threshold: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Group peaks into components: two peaks belong to the same component
        if their Euclidean distance is ≤ threshold pixels.
        """
        parent = list(range(len(grid_peaks)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        arr = np.array(grid_peaks, dtype=float)
        for i in range(len(grid_peaks)):
            for j in range(i + 1, len(grid_peaks)):
                dr = arr[i, 0] - arr[j, 0]
                dc = arr[i, 1] - arr[j, 1]
                if (dr * dr + dc * dc) ** 0.5 <= threshold:
                    union(i, j)

        comps: Dict[int, List[Tuple[int, int]]] = {}
        for i, pk in enumerate(grid_peaks):
            cid = find(i)
            comps.setdefault(cid, []).append(pk)
        return comps

    # ------------------------------------------------------------------
    # Step 5 — breaking points
    # ------------------------------------------------------------------

    def _breaking_points(
        self,
        components: Dict[int, List[Tuple[int, int]]],
        neighbor_px: int,
        kink_threshold: float,
    ) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Return list of (component_id, (row, col)) for every breaking point.

        A breaking point is either:
          • an endpoint  — a peak with ≤ 1 in-component neighbour, OR
          • a kink point — a peak where the sorted trajectory's local slope
                           changes by more than kink_threshold.
        """
        ext = neighbor_px * 1.5
        breaking: List[Tuple[int, Tuple[int, int]]] = []

        for cid, peaks in components.items():
            arr = np.array(peaks, dtype=float)

            # -- endpoints --
            if len(peaks) == 1:
                breaking.append((cid, peaks[0]))
                continue
            for i, pk in enumerate(peaks):
                dists = np.sqrt(np.sum((arr - np.array(pk)) ** 2, axis=1))
                dists[i] = np.inf
                if np.sum(dists <= ext) <= 1:
                    breaking.append((cid, pk))

            # -- kink points --
            if len(peaks) < 3:
                continue
            # Sort along the principal axis of the component
            centroid = arr.mean(axis=0)
            _, _, Vt = np.linalg.svd(arr - centroid)
            axis = Vt[0]                         # principal direction
            projections = (arr - centroid) @ axis
            order = np.argsort(projections)
            sorted_peaks = arr[order]

            # Compute consecutive slopes (Δrow / Δcol, protect div-by-zero)
            def local_slope(p1, p2):
                dc = p2[1] - p1[1]
                dr = p2[0] - p1[0]
                if abs(dc) < 1e-6:
                    return np.sign(dr) * 1e6
                return dr / dc

            slopes = [
                local_slope(sorted_peaks[k], sorted_peaks[k + 1])
                for k in range(len(sorted_peaks) - 1)
            ]
            for k in range(1, len(slopes)):
                if abs(slopes[k] - slopes[k - 1]) > kink_threshold:
                    # The peak between the two segments is the kink
                    kink_idx = order[k]
                    kink_pk = (int(arr[kink_idx, 0]), int(arr[kink_idx, 1]))
                    breaking.append((cid, kink_pk))

        return breaking

    # ------------------------------------------------------------------
    # Step 6 — connect breaking points from different components
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(
        breaking_pts: List[Tuple[int, Tuple[int, int]]],
        connect_px: int,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Greedy nearest-neighbour pairing: each breaking point is paired
        with the closest breaking point from a DIFFERENT component that is
        within connect_px pixels.
        """
        connections: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        used: set = set()

        for i, (ci, pi) in enumerate(breaking_pts):
            if i in used:
                continue
            best_j: Optional[int] = None
            best_d = float(connect_px)

            for j, (cj, pj) in enumerate(breaking_pts):
                if j == i or j in used or cj == ci:
                    continue
                dr = pi[0] - pj[0]
                dc = pi[1] - pj[1]
                d = (dr * dr + dc * dc) ** 0.5
                if d < best_d:
                    best_d = d
                    best_j = j

            if best_j is None:
                continue
            used.add(i)
            used.add(best_j)
            connections.append((pi, breaking_pts[best_j][1]))

        return connections

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _v2g(
        vx: float, vy: float,
        unique_Vx: np.ndarray, unique_Vy: np.ndarray,
    ) -> Tuple[int, int]:
        col = int(np.argmin(np.abs(unique_Vx - vx)))
        row = int(np.argmin(np.abs(unique_Vy - vy)))
        return (row, col)

    @staticmethod
    def _g2v(
        row: int, col: int,
        unique_Vx: np.ndarray, unique_Vy: np.ndarray,
    ) -> Tuple[float, float]:
        return (float(unique_Vx[col]), float(unique_Vy[row]))

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_txt(self, interdot_v: List[Tuple[float, float]]) -> None:
        out = os.path.join(self.sample_dir, "interdot_transitions.txt")
        with open(out, "w") as f:
            f.write("Interdot Transition Peaks — Voltage Coordinates\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total detected: {len(interdot_v)}\n\n")
            f.write("Peaks (Voltage Coordinates):\n")
            for vx, vy in interdot_v:
                f.write(f"({vx:.6f}, {vy:.6f})\n")
        print(f"[InterdotDetector] saved → {out}")

    def _write_plot(
        self,
        current_2d: np.ndarray,
        sweep_peaks_v: List[Tuple[float, float]],
        interdot_v: List[Tuple[float, float]],
        breaking_pts: List[Tuple[int, Tuple[int, int]]],
        connections: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        unique_Vx: np.ndarray,
        unique_Vy: np.ndarray,
    ) -> None:
        vxmin, vxmax = self.vx_min, self.vx_max
        vymin, vymax = self.vy_min, self.vy_max
        x_edges = np.linspace(vxmin, vxmax, current_2d.shape[1] + 1)
        y_edges = np.linspace(vymin, vymax, current_2d.shape[0] + 1)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.pcolormesh(x_edges, y_edges, current_2d,
                      cmap="gray_r", edgecolors="k", linewidth=0.3)

        # Sweep-detected peaks
        if sweep_peaks_v:
            arr = np.array(sweep_peaks_v)
            ax.scatter(arr[:, 0], arr[:, 1], c="blue", s=8,
                       alpha=0.5, label="Sweep Peaks", zorder=3)

        # Breaking points
        if breaking_pts:
            bp_v = [
                self._g2v(r, c, unique_Vx, unique_Vy)
                for _, (r, c) in breaking_pts
            ]
            arr = np.array(bp_v)
            ax.scatter(arr[:, 0], arr[:, 1], c="orange", s=40,
                       marker="^", label="Break Points", zorder=4)

        # Connection lines
        for (r1, c1), (r2, c2) in connections:
            v1 = self._g2v(r1, c1, unique_Vx, unique_Vy)
            v2 = self._g2v(r2, c2, unique_Vx, unique_Vy)
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]],
                    "g--", lw=1, alpha=0.7, zorder=4)

        # Interdot peaks
        if interdot_v:
            arr = np.array(interdot_v)
            ax.scatter(arr[:, 0], arr[:, 1], c="red", s=80,
                       marker="x", linewidths=1.5,
                       label="Interdot Peaks", zorder=5)

        ax.set_xlabel(x_label())
        ax.set_ylabel(y_label())
        ax.set_xlim(vxmin, vxmax)
        ax.set_ylim(vymin, vymax)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")

        out = os.path.join(self.sample_dir, "summary_with_interdot.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[InterdotDetector] plot saved → {out}")

    # ------------------------------------------------------------------
    # Parser  (same format as OverlayRenderer._parse_voltage_coordinates_global)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_peaks_txt(
        txt_path: str,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        scanned: List[Tuple[float, float]] = []
        peaks: List[Tuple[float, float]] = []
        scanned_sec = False
        peaks_sec = False

        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Scanned Cells (Voltage Coordinates):"):
                    scanned_sec, peaks_sec = True, False
                    continue
                if line.startswith("Peaks (Voltage Coordinates):"):
                    scanned_sec, peaks_sec = False, True
                    continue
                if (scanned_sec or peaks_sec) and line.startswith("("):
                    try:
                        x_str, y_str = line[1:-1].split(",")
                        coord = (float(x_str.strip()), float(y_str.strip()))
                        (scanned if scanned_sec else peaks).append(coord)
                    except Exception:
                        continue
        return scanned, peaks
