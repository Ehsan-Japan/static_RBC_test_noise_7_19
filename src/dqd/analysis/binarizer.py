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
from ..config.figure_style import (
    LABEL_PEAK,
    LABEL_SCANNED,
    MARKER_CENTRE,
    MARKER_PEAK,
    MARKER_SCANNED,
    apply_voltage_axes,
    draw_ground_truth_map,
    new_map_figure,
    save_figure,
)


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
        data_is_ground_truth: bool = False,
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
        data_is_ground_truth : data_path is double_dot_data.npy, i.e. the exact
                               binary charge-state-change map.  It is thresholded
                               directly instead of being run through local-maxima
                               detection, so the image matches
                               double_dot_stability_diagram.jpg exactly.
                               Local-maxima detection on the SENSOR signal cannot
                               reproduce it: interdot transitions barely change the
                               sensor amplitude, so they are missing from it.
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

            if data_is_ground_truth:
                binary = (current_2d > 0.5).astype(int)
                title = "Ground Truth Transitions"
            else:
                binary = self._detect_local_maxima(current_2d, num_rows, num_cols)
                title = "Local Maxima"

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
                                  ax_xmin, ax_xmax, ax_ymin, ax_ymax,
                                  output_no_overlay, title)

            # Prefer voltage_coords_path for overlay; fall back to row/col summary_path
            if voltage_coords_path is not None and os.path.isfile(voltage_coords_path):
                scanned_v, peaks_v = self._parse_voltage_coords(voltage_coords_path)
                self._save_with_overlay_voltages(
                    binary, x_edges, y_edges, xlabel, ylabel,
                    num_cols, num_rows, vxmin, vxmax, vymin, vymax,
                    ax_xmin, ax_xmax, ax_ymin, ax_ymax,
                    scanned_v, peaks_v, center_row, center_col, output_with_overlay,
                    title,
                )
            elif summary_path is not None:
                self._save_with_overlay(
                    binary, x_edges, y_edges, xlabel, ylabel,
                    num_cols, num_rows, vxmin, vxmax, vymin, vymax,
                    ax_xmin, ax_xmax, ax_ymin, ax_ymax,
                    summary_path, center_row, center_col, output_with_overlay,
                    title,
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
                         ax_xmin, ax_xmax, ax_ymin, ax_ymax, out_path,
                         title="Local Maxima"):
        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, binary)
        apply_voltage_axes(ax, ax_xmin, ax_xmax, ax_ymin, ax_ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (No Overlay)")
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        save_figure(fig, out_path)

    def _save_with_overlay(
        self, binary, x_edges, y_edges, xlabel, ylabel,
        num_cols, num_rows, vxmin, vxmax, vymin, vymax,
        ax_xmin, ax_xmax, ax_ymin, ax_ymax,
        summary_path, center_row, center_col, out_path,
        title="Local Maxima",
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

        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, binary)

        if sx:
            ax.scatter(sx, sy, label=LABEL_SCANNED, **MARKER_SCANNED)
        if dx:
            ax.scatter(dx, dy, label=LABEL_PEAK, **MARKER_PEAK)
        if center_row and center_col:
            cx = vxmin + (center_col - 0.5) * cell_w
            cy = vymin + (center_row - 0.5) * cell_h
            ax.scatter(cx, cy, label="Peak Centre", **MARKER_CENTRE)

        apply_voltage_axes(ax, ax_xmin, ax_xmax, ax_ymin, ax_ymax)
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} with Overlay")
        ax.legend(loc="upper right")
        save_figure(fig, out_path)

    # ------------------------------------------------------------------
    # Overlay using direct voltage coordinates  (new path)
    # ------------------------------------------------------------------

    def _save_with_overlay_voltages(
        self, binary, x_edges, y_edges, xlabel, ylabel,
        num_cols, num_rows, vxmin, vxmax, vymin, vymax,
        ax_xmin, ax_xmax, ax_ymin, ax_ymax,
        scanned_v, peaks_v, center_row, center_col, out_path,
        title="Local Maxima",
    ):
        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, binary)

        if scanned_v:
            sx = [v[0] for v in scanned_v]
            sy = [v[1] for v in scanned_v]
            ax.scatter(sx, sy, label=LABEL_SCANNED, **MARKER_SCANNED)
        if peaks_v:
            dx = [v[0] for v in peaks_v]
            dy = [v[1] for v in peaks_v]
            ax.scatter(dx, dy, label=LABEL_PEAK, **MARKER_PEAK)
        if center_row and center_col:
            cell_w = (vxmax - vxmin) / num_cols
            cell_h = (vymax - vymin) / num_rows
            cx = vxmin + (center_col - 0.5) * cell_w
            cy = vymin + (center_row - 0.5) * cell_h
            ax.scatter(cx, cy, label="Peak Centre", **MARKER_CENTRE)

        apply_voltage_axes(ax, ax_xmin, ax_xmax, ax_ymin, ax_ymax)
        ax.set_xticks(np.linspace(ax_xmin, ax_xmax, 5))
        ax.set_yticks(np.linspace(ax_ymin, ax_ymax, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} with Overlay")
        ax.legend(loc="upper right")
        save_figure(fig, out_path)

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
