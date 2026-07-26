"""
OverlayRenderer — overlays scanned cells, peaks, and rays on charge-sensor images.
Absorbs the old plot_overlay.py, utility4.py, and utility6.py.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ..config.figure_style import (
    LABEL_PEAK,
    LABEL_SCANNED,
    MARKER_PEAK,
    MARKER_SCANNED,
    apply_voltage_axes,
    draw_ground_truth_map,
    new_map_figure,
    save_figure,
)


class OverlayRenderer:
    """
    Generates overlay images that highlight scanned cells, detected peaks,
    and ray paths on top of charge-sensor or double-dot background images.

    All methods are stateless; pass required parameters explicitly.
    """

    # ------------------------------------------------------------------
    # Voltage-coordinate file  (shared parser)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_voltage_coordinates(
        file_path: str,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Parse a voltage_coordinates.txt file.

        Returns
        -------
        (scanned_cells, peaks) — each is a list of (Vx, Vy) float tuples.
        """
        with open(file_path, "r") as f:
            content = f.read()

        scanned_match = re.search(
            r"Scanned Cells\s*\(Voltage Coordinates\):\s*(.+?)"
            r"(?:Peaks\s*\(Voltage Coordinates\):|$)",
            content,
            re.DOTALL,
        )
        peaks_match = re.search(
            r"Peaks\s*\(Voltage Coordinates\):\s*(.+)", content, re.DOTALL
        )

        def _extract(coord_string: str) -> List[Tuple[float, float]]:
            coords = []
            for m in re.findall(
                r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", coord_string
            ):
                try:
                    coords.append((float(m[0]), float(m[1])))
                except Exception as e:
                    print(f"Error parsing coordinate {m}: {e}")
            return coords

        scanned = _extract(scanned_match.group(1)) if scanned_match else []
        peaks = _extract(peaks_match.group(1)) if peaks_match else []
        return scanned, peaks

    # ------------------------------------------------------------------
    # Single-file voltage-coordinate overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_voltage_coordinates(
        npy_file: str,
        voltage_coords_file: str,
        output_file: str,
        is_double_dot: bool = False,
    ) -> None:
        """
        Replot the full charge-sensor image and overlay scanned cells / peaks
        read from *voltage_coords_file*.

        Parameters
        ----------
        npy_file             : path to charge_sensing_data.npy (columns Vx,Vy,z)
        voltage_coords_file  : path to voltage_coordinates.txt
        output_file          : where to save the PNG
        is_double_dot        : if True, binarise z and use gray_r cmap
        """
        array = np.load(npy_file)
        Vx_flat, Vy_flat, z_flat = array[:, 0], array[:, 1], array[:, 2]

        cmap = "gray_r" if is_double_dot else "hot"
        if is_double_dot:
            z_flat = np.where(z_flat > 0.5, 1.0, 0.0).astype(np.float32)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = None, None

        Vx_values = np.unique(Vx_flat)
        Vy_values = np.unique(Vy_flat)
        z_2d = z_flat.reshape((len(Vy_values), len(Vx_values)))
        extent = [Vx_values[0], Vx_values[-1], Vy_values[0], Vy_values[-1]]

        scanned_cells, peaks = OverlayRenderer.parse_voltage_coordinates(
            voltage_coords_file
        )
        print(
            f"Parsed {len(scanned_cells)} scanned cells and {len(peaks)} peaks "
            f"from {voltage_coords_file}"
        )

        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(
            z_2d,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if scanned_cells:
            sc = np.array(scanned_cells)
            ax.scatter(sc[:, 0], sc[:, 1], color="blue", alpha=0.5, s=10,
                       label="Scanned Cells")
        if peaks:
            pk = np.array(peaks)
            ax.scatter(pk[:, 0], pk[:, 1], marker="x", color="red", s=50,
                       linewidths=1, label="Peaks")

        apply_voltage_axes(ax, extent[0], extent[1], extent[2], extent[3])
        ax.legend(loc="upper right")
        fig.colorbar(
            im, cax=cax,
            label="Current (nA)" if is_double_dot else "Sensor Signal",
        )
        save_figure(fig, output_file)
        print(f"Highlighted charge-sensor image saved to: {output_file}")

    # ------------------------------------------------------------------
    # All-rays overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_all_rays_in_sample(
        npy_file: str,
        paired_dict: Dict,
        angles: List[float],
        output_file: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
        is_double_dot: bool = False,
    ) -> None:
        """
        Plot the background image from *npy_file* and overlay each ray's peaks
        in a different colour.

        Parameters
        ----------
        npy_file    : .npy file (columns Vx, Vy, z)
        paired_dict : {angle_str: [[vx, vy], ...], ...}
        angles      : list of angles (used for legend labels)
        output_file : where to save the PNG
        vxmin/max, vymin/max : voltage range for imshow extent
        is_double_dot : binarise and use hot cmap if True
        """
        if not os.path.isfile(npy_file):
            print(f"File not found: {npy_file}. Aborting.")
            return

        array = np.load(npy_file)
        Vx_flat, Vy_flat, z_flat = array[:, 0], array[:, 1], array[:, 2]
        Vx_values = np.unique(Vx_flat)
        Vy_values = np.unique(Vy_flat)
        z_2d = z_flat.reshape((len(Vy_values), len(Vx_values)))

        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(
            z_2d,
            extent=[vxmin, vxmax, vymin, vymax],
            origin="lower",
            aspect="auto",
            cmap="hot",
        )
        apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax)

        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(angles)))
        for i, angle in enumerate(angles):
            angle_str = str(float(angle))
            if angle_str not in paired_dict:
                continue
            peak_coords = np.array(paired_dict[angle_str])
            if len(peak_coords) > 0:
                ax.scatter(
                    peak_coords[:, 0], peak_coords[:, 1],
                    marker="x", color=color_cycle[i], s=50, linewidths=1,
                    label=f"Ray {round(angle)}°",
                )

        cbar_label = "Sensor Signal" if not is_double_dot else "Occupation (binary)"
        fig.colorbar(im, cax=cax, label=cbar_label)
        ax.legend(loc="upper right")
        save_figure(fig, output_file)
        print(f"[All Rays Overlay] Saved combined plot to: {output_file}")

    # ------------------------------------------------------------------
    # All-scanned + all-peaks overlay with row/col txt  (utility6.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_all_scanned_and_peaks_in_sample(
        sample_dir: str,
        output_file: str,
        is_double_dot: bool = False,
        charge_sensing_filename: str = "charge_sensing_data.npy",
    ) -> None:
        """
        Walk *sample_dir/cropped_results/* collecting every
        voltage_coordinates.txt, then overlay all scanned cells and peaks on
        the full charge-sensor background.  Also writes a .txt summary with
        row/col indices alongside voltage coordinates.

        Parameters
        ----------
        sample_dir               : e.g. ".../sample_1"
        output_file              : path of the output PNG
        is_double_dot            : binarise z if True
        charge_sensing_filename  : .npy file name inside numpy/simulation/
        """
        all_scanned: List[Tuple[float, float]] = []
        all_peaks: List[Tuple[float, float]] = []

        cropped_results_dir = os.path.join(sample_dir, "cropped_results")
        if not os.path.isdir(cropped_results_dir):
            print(f"No cropped_results folder found at {cropped_results_dir}.")
            return

        for root, _, files in os.walk(cropped_results_dir):
            for fname in files:
                if fname.lower() == "voltage_coordinates.txt":
                    s, p = OverlayRenderer.parse_voltage_coordinates(
                        os.path.join(root, fname)
                    )
                    all_scanned.extend(s)
                    all_peaks.extend(p)

        num_scanned = len(all_scanned)
        num_peaks = len(all_peaks)
        print(
            f"[highlight_all_scanned_and_peaks_in_sample] Found "
            f"{num_scanned} scanned cells and {num_peaks} peaks in {sample_dir}."
        )

        npy_path = os.path.join(
            sample_dir, "numpy", "simulation", charge_sensing_filename
        )
        if not os.path.isfile(npy_path):
            print(f"File not found: {npy_path}. Cannot generate overlay.")
            return

        array = np.load(npy_path)
        Vx_flat, Vy_flat, z_flat = array[:, 0], array[:, 1], array[:, 2]

        if is_double_dot:
            z_flat = np.where(z_flat > 0.5, 1.0, 0.0).astype(np.float32)
            cmap, vmin, vmax = "binary", 0, 1
            cbar_label = "Occupation (binary)"
        else:
            cmap, vmin, vmax = "hot", None, None
            cbar_label = "Sensor Signal"

        Vx_values = np.unique(Vx_flat)
        Vy_values = np.unique(Vy_flat)
        z_2d = z_flat.reshape((len(Vy_values), len(Vx_values)))
        extent = [Vx_values[0], Vx_values[-1], Vy_values[0], Vy_values[-1]]

        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(
            z_2d, extent=extent, origin="lower", aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
        )

        if num_scanned > 0:
            sa = np.array(all_scanned)
            ax.scatter(sa[:, 0], sa[:, 1], marker="o", facecolors="none",
                       edgecolors="cyan", s=30, label="All Scanned Cells")
        if num_peaks > 0:
            pa = np.array(all_peaks)
            ax.scatter(pa[:, 0], pa[:, 1], marker="x", color="red",
                       s=40, label="All Detected Peaks")

        apply_voltage_axes(ax, extent[0], extent[1], extent[2], extent[3])
        ax.legend(loc="upper right")
        fig.colorbar(im, cax=cax, label=cbar_label)
        save_figure(fig, output_file)
        print(
            f"[highlight_all_scanned_and_peaks_in_sample] Saved overlay to: {output_file}"
        )

        # Write summary txt with row/col
        def _v_to_rc(vx, vy):
            col = int(np.abs(Vx_values - vx).argmin())
            row = int(np.abs(Vy_values - vy).argmin())
            return row, col

        output_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("[highlight_all_scanned_and_peaks_in_sample] Summary\n")
            f.write(f"Total scanned cells: {num_scanned}\n")
            f.write(f"Total peaks:         {num_peaks}\n\n")
            f.write("Scanned Cells (Voltage + row/col):\n")
            for vx, vy in all_scanned:
                r, c = _v_to_rc(vx, vy)
                f.write(f"V=({vx}, {vy}), row={r}, col={c}\n")
            f.write("\nPeaks (Voltage + row/col):\n")
            for vx, vy in all_peaks:
                r, c = _v_to_rc(vx, vy)
                f.write(f"V=({vx}, {vy}), row={r}, col={c}\n")
        print(
            f"[highlight_all_scanned_and_peaks_in_sample] Summary saved to: {txt_path}"
        )

    # ------------------------------------------------------------------
    # Binary-mode all-scanned overlay (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_all_scanned_and_peaks_in_sample_in_binary(
        sample_dir: str,
        output_file: str,
        is_double_dot: bool = False,
        charge_sensing_filename: str = "charge_sensing_data.npy",
    ) -> None:
        """
        Same walk as :meth:`highlight_all_scanned_and_peaks_in_sample`, but
        only saves the .txt summary (no image).  Used for binary ground-truth
        workflows.
        """
        all_scanned: List[Tuple[float, float]] = []
        all_peaks: List[Tuple[float, float]] = []

        cropped_results_dir = os.path.join(sample_dir, "cropped_results")
        if not os.path.isdir(cropped_results_dir):
            print(f"No cropped_results folder found at {cropped_results_dir}.")
            return

        for root, _, files in os.walk(cropped_results_dir):
            for fname in files:
                if fname.lower() == "voltage_coordinates.txt":
                    s, p = OverlayRenderer.parse_voltage_coordinates(
                        os.path.join(root, fname)
                    )
                    all_scanned.extend(s)
                    all_peaks.extend(p)

        num_scanned = len(all_scanned)
        num_peaks = len(all_peaks)
        print(
            f"[highlight_all_scanned_and_peaks_in_sample_in_binary] Found "
            f"{num_scanned} scanned cells and {num_peaks} peaks in {sample_dir}."
        )

        output_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            f.write("[highlight_all_scanned_and_peaks_in_sample] Summary\n")
            f.write(f"Total scanned cells: {num_scanned}\n")
            f.write(f"Total peaks:         {num_peaks}\n\n")
            f.write("Scanned Cells (Voltage Coordinates):\n")
            if num_scanned > 0:
                for vx, vy in all_scanned:
                    f.write(f"({vx}, {vy})\n")
            else:
                f.write("None\n")
            f.write("\nPeaks (Voltage Coordinates):\n")
            if num_peaks > 0:
                for vx, vy in all_peaks:
                    f.write(f"({vx}, {vy})\n")
            else:
                f.write("None\n")
        print(
            f"[highlight_all_scanned_and_peaks_in_sample_in_binary] Summary saved to: {txt_path}"
        )

    # ------------------------------------------------------------------
    # Ground-truth binary array  (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_ground_truth_array(
        data_path: str,
        output_npy_path: str,
        use_double_dot: bool = False,
    ) -> np.ndarray:
        """
        Build a binary ground-truth array and save it as a .npy file.

        Parameters
        ----------
        data_path       : .npy file with columns [Vx, Vy, z]
        output_npy_path : destination .npy file
        use_double_dot  : if True, treat *data_path* as double_dot_data.npy
                          whose z column is already an exact binary transition
                          label (1 = charge-state change, 0 = background).
                          This preserves interdot transition lines which have
                          no measurable charge-sensor response and are therefore
                          missed by local-maxima detection on the sensor signal.
                          If False (legacy), detect local maxima in the sensor
                          current to approximate the transition lines.

        Returns
        -------
        binary_array : np.ndarray of shape (num_rows, num_cols), dtype uint8
        """
        data = np.load(data_path)
        Vx, Vy, Current = data.T
        unique_Vx = np.unique(Vx)
        unique_Vy = np.unique(Vy)
        current_2d = Current.reshape(len(unique_Vy), len(unique_Vx))

        if use_double_dot:
            # double_dot_data.npy already encodes exact charge-state-change
            # boundaries computed by DotArray — all transition types included.
            binary_array = (current_2d > 0.5).astype(np.uint8)
        else:
            # Legacy path: approximate transitions via local-maxima in the
            # charge-sensor signal.  Misses interdot lines (weak sensor response).
            vertical = np.zeros_like(current_2d, dtype=bool)
            horizontal = np.zeros_like(current_2d, dtype=bool)

            for col in range(current_2d.shape[1]):
                for row in range(1, current_2d.shape[0] - 1):
                    if current_2d[row, col] > max(
                        current_2d[row - 1, col], current_2d[row + 1, col]
                    ):
                        vertical[row, col] = True

            for row in range(current_2d.shape[0]):
                for col in range(1, current_2d.shape[1] - 1):
                    if current_2d[row, col] > max(
                        current_2d[row, col - 1], current_2d[row, col + 1]
                    ):
                        horizontal[row, col] = True

            binary_array = np.logical_or(vertical, horizontal).astype(np.uint8)

        np.save(output_npy_path, binary_array)
        print(f"Ground truth array saved to: {output_npy_path}")
        return binary_array

    # ------------------------------------------------------------------
    # 2-D grid visualisation  (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_grid_2d(
        file_path: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
        ray_resolution: int,
        output_file: str,
    ) -> None:
        """
        Visualise a 2-D binary ground-truth grid with scanned cells and peaks
        overlaid (reads coordinates from the sibling txt file).

        Parameters
        ----------
        file_path      : .npy file containing the binary ground-truth array
        vxmin/max      : voltage extent in x
        vymin/max      : voltage extent in y
        ray_resolution : unused but kept for API compatibility
        output_file    : where to save the PNG
        """
        data = np.load(file_path)
        res_y, res_x = data.shape

        sample_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        txt_path = os.path.join(sample_dir, "all_scanned_peaks_overlay_binary.txt")

        scanned, peaks = OverlayRenderer._parse_voltage_coordinates_global(txt_path)

        x_edges = np.linspace(vxmin, vxmax, res_x + 1)
        y_edges = np.linspace(vymin, vymax, res_y + 1)
        cell_width_x = (vxmax - vxmin) / res_x
        cell_width_y = (vymax - vymin) / res_y

        def _to_centers(coords):
            xs, ys = [], []
            for x, y in coords:
                col = int(np.clip((x - vxmin) / (vxmax - vxmin) * res_x, 0, res_x - 1))
                row = int(np.clip((y - vymin) / (vymax - vymin) * res_y, 0, res_y - 1))
                xs.append(vxmin + (col + 0.5) * cell_width_x)
                ys.append(vymin + (row + 0.5) * cell_width_y)
            return xs, ys

        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, data)

        if scanned:
            xs, ys = _to_centers(scanned)
            ax.scatter(xs, ys, label=LABEL_SCANNED, **MARKER_SCANNED)
        if peaks:
            xs, ys = _to_centers(peaks)
            ax.scatter(xs, ys, label=LABEL_PEAK, **MARKER_PEAK)

        apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax)
        ax.legend()
        save_figure(fig, output_file)

    # ------------------------------------------------------------------
    # Combined sweeping + rays summary  (summary_total.png)
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_grid_2d_with_rays(
        file_path: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
        ray_resolution: int,
        output_file: str,
        rays_dict: Dict,
        peaks_dict: Dict,
        show_measured_points: bool = True,
        show_ray_peaks: Optional[bool] = None,
        ray_peak_color: Optional[str] = None,
        ray_peak_size: float = 80,
        ray_peak_marker: str = "x",
        ray_peak_linewidths: float = 2,
        ray_peak_edgecolor: Optional[str] = None,
        ray_peak_label: Optional[str] = None,
        legend_fontsize: int = 8,
    ) -> None:
        """
        Like :meth:`visualize_grid_2d` but also overlays ray paths and ray peaks.

        Parameters
        ----------
        file_path      : .npy binary ground-truth array
        vxmin/max      : voltage extent in x
        vymin/max      : voltage extent in y
        ray_resolution : unused but kept for API compatibility
        output_file    : where to save summary_total.png
        rays_dict      : {angle_float: {"X": ..., "Y": ..., "Current": ...}}
        peaks_dict     : {angle_float: {"vx": [...], "vy": [...]}}
        show_measured_points : draw the swept/measured cells (blue dots)
        show_ray_peaks : draw the per-ray peak crosses (defaults to
                         ``show_measured_points``)
        ray_peak_color : if given, EVERY ray-peak cross is drawn in this single
                         colour and gets one shared legend entry.  When ``None``
                         each ray angle keeps its own tab10 colour and the
                         crosses stay out of the legend (original behaviour).
        ray_peak_size / ray_peak_marker / ray_peak_linewidths /
        ray_peak_edgecolor  : marker styling for the ray-peak crosses.
        ray_peak_label : legend label used with ``ray_peak_color``.
        legend_fontsize : font size of the legend (bigger for publication).
        """
        if show_ray_peaks is None:
            show_ray_peaks = show_measured_points
        data = np.load(file_path)
        res_y, res_x = data.shape

        sample_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        txt_path = os.path.join(sample_dir, "all_scanned_peaks_overlay_binary.txt")

        scanned, peaks = OverlayRenderer._parse_voltage_coordinates_global(txt_path)

        x_edges = np.linspace(vxmin, vxmax, res_x + 1)
        y_edges = np.linspace(vymin, vymax, res_y + 1)
        cell_width_x = (vxmax - vxmin) / res_x
        cell_width_y = (vymax - vymin) / res_y

        def _to_centers(coords):
            xs, ys = [], []
            for x, y in coords:
                col = int(np.clip((x - vxmin) / (vxmax - vxmin) * res_x, 0, res_x - 1))
                row = int(np.clip((y - vymin) / (vymax - vymin) * res_y, 0, res_y - 1))
                xs.append(vxmin + (col + 0.5) * cell_width_x)
                ys.append(vymin + (row + 0.5) * cell_width_y)
            return xs, ys

        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, data)

        # ---- Sweeping results ----
        if show_measured_points and scanned:
            xs, ys = _to_centers(scanned)
            ax.scatter(xs, ys, label=LABEL_SCANNED, **MARKER_SCANNED)
        if peaks:
            xs, ys = _to_centers(peaks)
            ax.scatter(xs, ys, color="red", s=50, marker="x",
                       linewidths=1, label="Detected Transitions")

        # ---- Ray traces and ray peaks (snapped to grid cell centres) ----
        sorted_angles = sorted(rays_dict.keys())
        color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_angles), 1)))
        uniform_ray_peaks: List[Tuple[float, float]] = []
        for idx, angle in enumerate(sorted_angles):
            ray_data = rays_dict[angle]
            if show_measured_points:
                ray_coords = list(zip(ray_data["X"].tolist(), ray_data["Y"].tolist()))
                rxs, rys = _to_centers(ray_coords)
                ax.scatter(rxs, rys, color="blue", s=10, alpha=0.5,
                           label="_nolegend_")
            if show_ray_peaks and angle in peaks_dict:
                pvx = peaks_dict[angle]["vx"]
                pvy = peaks_dict[angle]["vy"]
                if pvx:
                    peak_coords = list(zip(pvx, pvy))
                    if ray_peak_color is not None:
                        # One colour for all of them -> collect and draw once so
                        # the legend gets a single entry.
                        uniform_ray_peaks.extend(peak_coords)
                    else:
                        pxs, pys = _to_centers(peak_coords)
                        ax.scatter(pxs, pys, marker=ray_peak_marker,
                                   color=color_cycle[idx], s=ray_peak_size,
                                   linewidths=ray_peak_linewidths,
                                   zorder=6, label="_nolegend_")

        if uniform_ray_peaks:
            pxs, pys = _to_centers(uniform_ray_peaks)
            scatter_kw = dict(
                marker=ray_peak_marker,
                s=ray_peak_size,
                linewidths=ray_peak_linewidths,
                zorder=7,
                label=ray_peak_label or "Sweep Start Points",
            )
            if ray_peak_edgecolor is not None:
                scatter_kw.update(facecolors=ray_peak_color,
                                  edgecolors=ray_peak_edgecolor)
            else:
                scatter_kw.update(color=ray_peak_color)
            ax.scatter(pxs, pys, **scatter_kw)

        apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax)

        from matplotlib.patches import Patch
        transition_patch = Patch(facecolor="black", edgecolor="black",
                                 label="Transition Lines (Ground Truth)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[transition_patch] + handles,
                  labels=["Transition Lines (Ground Truth)"] + labels,
                  loc="upper right", fontsize=legend_fontsize)

        save_figure(fig, output_file)
        print(f"[summary_total] Saved to: {output_file}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_voltage_coordinates_global(
        file_path: str,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Line-by-line parser for files where each coordinate is on its own line."""
        scanned: List[Tuple[float, float]] = []
        peaks: List[Tuple[float, float]] = []
        scanned_section = False
        peaks_section = False

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Scanned Cells (Voltage Coordinates):"):
                    scanned_section, peaks_section = True, False
                    continue
                if line.startswith("Peaks (Voltage Coordinates):"):
                    scanned_section, peaks_section = False, True
                    continue
                if (scanned_section or peaks_section) and line.startswith("("):
                    try:
                        x_str, y_str = line[1:-1].split(",")
                        coord = (float(x_str.strip()), float(y_str.strip()))
                        (scanned if scanned_section else peaks).append(coord)
                    except Exception:
                        continue

        return scanned, peaks
