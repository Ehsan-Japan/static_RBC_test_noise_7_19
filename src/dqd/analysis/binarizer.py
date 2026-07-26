"""
ChargeSensorBinarizer — converts charge-sensor data to a binary local-maxima
image and optionally overlays sweep results.
Absorbs the old binarization_charge_sensor.py.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Set, Tuple

from ..config.axis_labels import x_label as _x_label, y_label as _y_label


class ChargeSensorBinarizer:
    """
    Loads a charge-sensor .npy file, detects local maxima (vertical + horizontal),
    and saves binary images with or without overlay markers.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        data_path: str,
        output_no_overlay: str,
        output_with_overlay: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        center_row: Optional[int] = None,
        center_col: Optional[int] = None,
        summary_path: Optional[str] = None,
        axis_vxmin: Optional[float] = None,
        axis_vxmax: Optional[float] = None,
        axis_vymin: Optional[float] = None,
        axis_vymax: Optional[float] = None,
        voltage_coords_path: Optional[str] = None,
    ) -> None:
        """
        Convert charge-sensor data to binary and save two images.

        Parameters
        ----------
        data_path            : path to .npy file with columns [Vx, Vy, Current]
        output_no_overlay    : save path for binary image without overlay
        output_with_overlay  : save path for binary image with sweep overlay
        xlabel / ylabel      : axis labels
        center_row/col       : 1-based centre cell index in the data grid
        summary_path         : path to local summary .txt (row/col overlay, legacy)
        axis_vxmin/vxmax     : override x-axis limits (e.g. full sweep range -1..1)
        axis_vymin/vymax     : override y-axis limits (e.g. full sweep range -1..1)
        voltage_coords_path  : path to voltage_coordinates.txt; when given, overlay
                               markers are placed using actual voltage values instead
                               of row/col indices from summary_path
        """
        xlabel = xlabel or _x_label()
        ylabel = ylabel or _y_label()
        try:
            data = np.load(data_path)
            Vx, Vy, Current = data.T
            unique_Vx = np.unique(Vx)
            unique_Vy = np.unique(Vy)
            num_cols = len(unique_Vx)
            num_rows = len(unique_Vy)
            current_2d = Current.reshape(num_rows, num_cols)

            binary = self._detect_local_maxima(current_2d, num_rows, num_cols)

            vxmin, vxmax = unique_Vx.min(), unique_Vx.max()
            vymin, vymax = unique_Vy.min(), unique_Vy.max()
            x_edges = np.linspace(vxmin, vxmax, num_cols + 1)
            y_edges = np.linspace(vymin, vymax, num_rows + 1)

            # Use caller-supplied axis limits when given, else fall back to data range
            ax_xmin = axis_vxmin if axis_vxmin is not None else vxmin
            ax_xmax = axis_vxmax if axis_vxmax is not None else vxmax
            ax_ymin = axis_vymin if axis_vymin is not None else vymin
            ax_ymax = axis_vymax if axis_vymax is not None else vymax

            self._save_no_overlay(binary, x_edges, y_edges, xlabel, ylabel,
                                  ax_xmin, ax_xmax, ax_ymin, ax_ymax, output_no_overlay)

            # Prefer voltage_coords_path for overlay; fall back to row/col summary_path
            if voltage_coords_path is not None and os.path.isfile(voltage_coords_path):
                scanned_v, peaks_v = self._parse_voltage_coords(voltage_coords_path)
                self._save_with_overlay_voltages(
                    binary, x_edges, y_edges, xlabel, ylabel,
                    num_cols, num_rows, vxmin, vxmax, vymin, vymax,
                    ax_xmin, ax_xmax, ax_ymin, ax_ymax,
                    scanned_v, peaks_v, center_row, center_col, output_with_overlay,
                )
            elif summary_path is not None:
                self._save_with_overlay(
                    binary, x_edges, y_edges, xlabel, ylabel,
                    num_cols, num_rows, vxmin, vxmax, vymin, vymax,
                    ax_xmin, ax_xmax, ax_ymin, ax_ymax,
                    summary_path, center_row, center_col, output_with_overlay,
                )
        except Exception as e:
            print(f"Error in ChargeSensorBinarizer.convert: {e}")

    # ------------------------------------------------------------------
    # Local-maxima detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_local_maxima(current_2d, num_rows, num_cols) -> np.ndarray:
        vertical = np.zeros_like(current_2d, dtype=bool)
        horizontal = np.zeros_like(current_2d, dtype=bool)

        for col in range(num_cols):
            for row in range(1, num_rows - 1):
                val = current_2d[row, col]
                if val > max(current_2d[row - 1, col], current_2d[row + 1, col]):
                    vertical[row, col] = True

        for row in range(num_rows):
            for col in range(1, num_cols - 1):
                val = current_2d[row, col]
                if val > max(current_2d[row, col - 1], current_2d[row, col + 1]):
                    horizontal[row, col] = True

        return np.logical_or(vertical, horizontal).astype(float)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _save_no_overlay(self, binary, x_edges, y_edges, xlabel, ylabel,
                         ax_xmin, ax_xmax, ax_ymin, ax_ymax, out_path):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolormesh(x_edges, y_edges, binary, cmap="binary",
                      edgecolors="gray", linewidth=0.2, vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Local Maxima (No Overlay)")
        ax.set_xlim(ax_xmin, ax_xmax)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_with_overlay(
        self, binary, x_edges, y_edges, xlabel, ylabel,
        num_cols, num_rows, vxmin, vxmax, vymin, vymax,
        ax_xmin, ax_xmax, ax_ymin, ax_ymax,
        summary_path, center_row, center_col, out_path,
    ):
        scanned_cells, detected_peaks = self._parse_summary(summary_path)
        scanned_pts = [(r - 1, c - 1) for (r, c) in scanned_cells]
        detected_pts = [(r - 1, c - 1) for (r, c) in detected_peaks]

        cell_w = (vxmax - vxmin) / num_cols
        cell_h = (vymax - vymin) / num_rows

        def to_voltage(pts):
            x = [vxmin + (c + 0.5) * cell_w for (r, c) in pts]
            y = [vymin + (r + 0.5) * cell_h for (r, c) in pts]
            return x, y

        sx, sy = to_voltage(scanned_pts)
        dx, dy = to_voltage(detected_pts)

        max_dim = max(num_cols, num_rows)
        s_scan = 200 / max_dim
        s_peak = 400 / max_dim

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolormesh(x_edges, y_edges, binary, cmap="binary",
                      edgecolors="gray", linewidth=0.2, vmin=0, vmax=1)
        ax.set_aspect("equal")

        if sx:
            ax.scatter(sx, sy, color="blue", s=s_scan, marker="s", alpha=0.5, label="Measured Cells")
        if dx:
            ax.scatter(dx, dy, color="red", s=s_peak, edgecolors="black",
                       linewidths=0.8, marker="o", label="Detected Peaks")
        if center_row and center_col:
            cx = vxmin + (center_col - 0.5) * cell_w
            cy = vymin + (center_row - 0.5) * cell_h
            ax.scatter(cx, cy, color="lime", s=s_peak * 1.5, marker="*", edgecolors="black")

        ax.set_xlim(ax_xmin, ax_xmax)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Local Maxima with Overlay")
        ax.legend(loc="upper right")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Overlay using direct voltage coordinates  (new path)
    # ------------------------------------------------------------------

    def _save_with_overlay_voltages(
        self, binary, x_edges, y_edges, xlabel, ylabel,
        num_cols, num_rows, vxmin, vxmax, vymin, vymax,
        ax_xmin, ax_xmax, ax_ymin, ax_ymax,
        scanned_v, peaks_v, center_row, center_col, out_path,
    ):
        max_dim = max(num_cols, num_rows)
        s_scan = 200 / max_dim
        s_peak = 400 / max_dim

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolormesh(x_edges, y_edges, binary, cmap="binary",
                      edgecolors="gray", linewidth=0.2, vmin=0, vmax=1)
        ax.set_aspect("equal")

        if scanned_v:
            sx = [v[0] for v in scanned_v]
            sy = [v[1] for v in scanned_v]
            ax.scatter(sx, sy, color="blue", s=s_scan, marker="s", alpha=0.5,
                       label="Measured Cells")
        if peaks_v:
            dx = [v[0] for v in peaks_v]
            dy = [v[1] for v in peaks_v]
            ax.scatter(dx, dy, color="red", s=s_peak, edgecolors="black",
                       linewidths=0.8, marker="o", label="Detected Peaks")
        if center_row and center_col:
            cell_w = (vxmax - vxmin) / num_cols
            cell_h = (vymax - vymin) / num_rows
            cx = vxmin + (center_col - 0.5) * cell_w
            cy = vymin + (center_row - 0.5) * cell_h
            ax.scatter(cx, cy, color="lime", s=s_peak * 1.5, marker="*",
                       edgecolors="black", label="Peak Centre")

        ax.set_xlim(ax_xmin, ax_xmax)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Local Maxima with Overlay")
        ax.legend(loc="upper right")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _parse_voltage_coords(path: str):
        """Parse voltage_coordinates.txt → (scanned_list, peaks_list) of (vx,vy) tuples."""
        import re
        scanned, peaks = [], []
        section = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Scanned Cells"):
                    section = "scanned"
                    continue
                if line.startswith("Peaks"):
                    section = "peaks"
                    continue
                for m in re.findall(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", line):
                    coord = (float(m[0]), float(m[1]))
                    if section == "scanned":
                        scanned.append(coord)
                    elif section == "peaks":
                        peaks.append(coord)
        return scanned, peaks

    # ------------------------------------------------------------------
    # Summary-file parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_summary(summary_path: str) -> Tuple[Set, Set]:
        """Parse summary_local.txt and return (scanned, peaks) as 1-based (row,col) sets."""
        scanned: Set[Tuple[int, int]] = set()
        peaks: Set[Tuple[int, int]] = set()
        current_row = None

        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Row="):
                    current_row = int(line.split("=")[1].strip())
                elif "Scanned Columns:" in line:
                    cols_str = line.split(": [")[1].rstrip("]")
                    if cols_str:
                        for c in cols_str.split(", "):
                            scanned.add((current_row, int(c)))
                elif "Detected Peaks:" in line:
                    peak_str = line.split(": [")[1].rstrip("]")
                    if peak_str:
                        for pair in peak_str.split("), "):
                            r, c = pair.lstrip("(").rstrip(")").split(", ")
                            peaks.add((int(r), int(c)))
                elif "LOCAL Peak Coordinates:" in line:
                    peak_str = line.split(": [")[1].rstrip("]")
                    if peak_str:
                        for pair in peak_str.split("), ("):
                            r, c = pair.lstrip("(").rstrip(")").split(", ")
                            peaks.add((int(r), int(c)))
        return scanned, peaks
