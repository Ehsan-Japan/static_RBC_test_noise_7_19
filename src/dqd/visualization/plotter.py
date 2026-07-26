"""
Plotter — Matplotlib-based visualisations for charge-sensor data and rays.
Absorbs the old plotting.py and plotting_functions.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
from typing import Dict, Optional

from ..config.axis_labels import x_label as _x_label, y_label as _y_label
from ..config.figure_style import (
    apply_voltage_axes,
    figure_size,
    new_map_figure,
    save_figure,
)


class Plotter:
    """
    Generates and saves visualisation images for charge-stability diagrams,
    rays, and current-vs-distance profiles.
    """

    # ------------------------------------------------------------------
    # Charge-stability diagram  (from old plotting.py)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_charge_stability(
        z: np.ndarray,
        dz: np.ndarray,
        voltage_sweep: Dict,
        save_paths: Dict,
        dpi: int = 300,
    ) -> None:
        """
        Plot z and dz/dV side by side and save.

        Parameters
        ----------
        z, dz         : 2-D arrays
        voltage_sweep : dict with vx_min/max, vy_min/max
        save_paths    : dict with "charge_sensing_save_path"
        dpi           : output resolution
        """
        vx_min = voltage_sweep["vx_min"]
        vx_max = voltage_sweep["vx_max"]
        vy_min = voltage_sweep["vy_min"]
        vy_max = voltage_sweep["vy_max"]
        extent = [vx_min, vx_max, vy_min, vy_max]

        fig, axes = plt.subplots(1, 2, figsize=figure_size(ncols=2),
                                 constrained_layout=True)

        im0 = axes[0].imshow(z, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_xlabel(_x_label())
        axes[0].set_ylabel(_y_label())
        axes[0].set_title("$z$")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(
            "Charge Sensor Output"
        )

        im1 = axes[1].imshow(dz, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_xlabel(_x_label())
        axes[1].set_ylabel(_y_label())
        axes[1].set_title(r"$\frac{dz}{dVx} + \frac{dz}{dVy}$")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label(
            "Gradient of Charge Sensor Output"
        )

        save_figure(fig, save_paths["charge_sensing_save_path"], dpi=dpi)

    # ------------------------------------------------------------------
    # Normalised 3-D surface plot  (from old utility1.py / sweeping_plot.py)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_normalized_3d(
        data_path: str,
        save_path: str,
        peak_i: int,
        peak_j: int,
        plot_config: Optional[Dict] = None,
    ) -> None:
        """
        Generate a normalised 3-D surface plot with the peak highlighted.

        Parameters
        ----------
        data_path   : .npy file with columns [Vx, Vy, Current]
        save_path   : directory to save the PNG
        peak_i/j    : 0-based (row, col) of the peak
        plot_config : optional dict with keys 'surface', 'point', 'view'
        """
        data = np.load(data_path)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Data must have shape (N,3).")

        Vx, Vy, Current = data[:, 0], data[:, 1], data[:, 2]
        unique_cols = np.unique(Vx)
        unique_rows = np.unique(Vy)
        num_cols = len(unique_cols)
        num_rows = len(unique_rows)

        if num_cols * num_rows != len(Current):
            raise ValueError(f"Grid mismatch: {num_cols}×{num_rows} ≠ {len(Current)}")

        current = Current.reshape((num_rows, num_cols))
        pixel_cols, pixel_rows = np.meshgrid(
            np.arange(1, num_cols + 1), np.arange(1, num_rows + 1)
        )

        if not (0 <= peak_i < num_rows) or not (0 <= peak_j < num_cols):
            raise IndexError(f"Peak indices ({peak_i},{peak_j}) out of bounds "
                             f"for grid ({num_rows},{num_cols}).")

        def min_max_scale(z):
            lo, hi = np.nanmin(z), np.nanmax(z)
            return np.zeros_like(z) if hi == lo else (z - lo) / (hi - lo)

        current_norm = min_max_scale(current)
        cfg = plot_config or {
            "surface": {"cmap": "hot", "alpha": 0.7, "linewidth": 0, "vmin": 0, "vmax": 1, "zorder": 1},
            "point":   {"s": 100, "color": "red", "edgecolor": "black", "linewidths": 1.5, "zorder": 10},
            "view":    {"elev": 30, "azim": -45},
        }
        os.makedirs(save_path, exist_ok=True)

        fig = plt.figure(figsize=figure_size(), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(pixel_cols, pixel_rows, current_norm, **cfg["surface"])
        ax.scatter(peak_j + 1, peak_i + 1, current_norm[peak_i, peak_j], **cfg["point"])
        ax.view_init(**cfg["view"])
        ax.set_title(
            f"Normalised Charge Sensor\n(Col={peak_j + 1}, Row={peak_i + 1})",
            fontsize=14, pad=20,
        )
        ax.set_xlabel("Column Index", labelpad=15, fontsize=12)
        ax.set_ylabel("Row Index", labelpad=15, fontsize=12)
        ax.set_zlabel("Normalised Value", labelpad=15, fontsize=12)
        ax.set_xticks(np.arange(1, num_cols + 1))
        ax.set_yticks(np.arange(1, num_rows + 1))
        ax.set_xlim(0.5, num_cols + 0.5)
        ax.set_ylim(0.5, num_rows + 0.5)
        cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label("Normalised Value", rotation=270, labelpad=20)
        save_figure(fig, os.path.join(save_path, "normalized_3d_surface.png"))

    # ------------------------------------------------------------------
    # Ray plots  (from old plotting_functions.py)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_original_data(
        data_array: np.ndarray,
        save_dir: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        cmap: str = "hot",
        prefix: str = "original",
    ) -> None:
        X, Y, I = data_array[:, 0], data_array[:, 1], data_array[:, 2]
        fig, ax, cax = new_map_figure(with_colorbar=True)
        sc = ax.scatter(X, Y, c=I, cmap=cmap)
        fig.colorbar(sc, cax=cax, label="Current (nA)")
        apply_voltage_axes(ax, X.min(), X.max(), Y.min(), Y.max())
        ax.set_xlabel(x_label or _x_label())
        ax.set_ylabel(y_label or _y_label())
        save_figure(fig, os.path.join(save_dir, f"{prefix}.png"))

    @staticmethod
    def plot_rays_only(
        rays_data: np.ndarray,
        save_dir: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        cmap: str = "hot",
        prefix: str = "rays_only",
    ) -> None:
        Xr, Yr, Ir = rays_data[:, 0], rays_data[:, 1], rays_data[:, 2]
        fig, ax, cax = new_map_figure(with_colorbar=True)
        sc = ax.scatter(Xr, Yr, c=Ir, cmap=cmap, s=30, edgecolor="black",
                        vmin=Ir.min(), vmax=Ir.max())
        fig.colorbar(sc, cax=cax, label="Current (nA)")
        ax.grid(True)
        apply_voltage_axes(ax, Xr.min(), Xr.max(), Yr.min(), Yr.max())
        ax.set_xlabel(x_label or _x_label())
        ax.set_ylabel(y_label or _y_label())
        save_figure(fig, os.path.join(save_dir, f"{prefix}.png"))

    @staticmethod
    def plot_combined(
        data_array: np.ndarray,
        rays_data: np.ndarray,
        save_dir: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        cmap: str = "hot",
        prefix: str = "combined_plot",
    ) -> None:
        X, Y, I = data_array[:, 0], data_array[:, 1], data_array[:, 2]
        Xr, Yr, Ir = rays_data[:, 0], rays_data[:, 1], rays_data[:, 2]
        combined_I = np.concatenate([I, Ir])
        vmin, vmax = combined_I.min(), combined_I.max()
        fig, ax, cax = new_map_figure(with_colorbar=True)
        ax.scatter(X, Y, c=I, cmap=cmap, s=20, edgecolor="none", vmin=vmin, vmax=vmax)
        sc = ax.scatter(Xr, Yr, c=Ir, cmap=cmap, s=40, edgecolor="black",
                        marker="o", vmin=vmin, vmax=vmax)
        fig.colorbar(sc, cax=cax, label="Current (nA)")
        apply_voltage_axes(ax, min(X.min(), Xr.min()), max(X.max(), Xr.max()),
                           min(Y.min(), Yr.min()), max(Y.max(), Yr.max()))
        ax.set_xlabel(x_label or _x_label())
        ax.set_ylabel(y_label or _y_label())
        save_figure(fig, os.path.join(save_dir, f"{prefix}.png"))

    @staticmethod
    def plot_current_vs_distance(
        rays_dict: Dict,
        save_dir: str,
        prefix: str = "rays_current_vs_distance",
        xlabel: str = "Normalised Distance",
        ylabel: str = "Current (nA)",
        cmap: str = "viridis",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        fig = plt.figure(figsize=figure_size(), constrained_layout=True)
        colors = get_cmap(cmap)(np.linspace(0, 1, len(rays_dict)))
        for idx, angle in enumerate(sorted(rays_dict.keys())):
            current = rays_dict[angle]["Current"]
            dist = np.linspace(0, 1, len(current))
            plt.plot(dist, current, label=f"{round(angle)}°", color=colors[idx],
                     linewidth=2, marker="o", markersize=4)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(title="Angle", fontsize=12, loc="best")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlim(0, 1)
        save_figure(fig, os.path.join(save_dir, f"{prefix}.png"))

    @staticmethod
    def plot_each_ray_separately(
        rays_dict: Dict,
        save_dir: str,
        prefix: str = "ray",
        ylabel: str = "Current (nA)",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for angle, data in rays_dict.items():
            current = data["Current"]
            dist = np.linspace(0, 1, len(current))
            a = round(angle)
            fig = plt.figure(figsize=figure_size(), constrained_layout=True)
            plt.plot(dist, current, label=f"Ray {a}°", color="blue",
                     linewidth=2, marker="o", markersize=6)
            peaks_idx, _ = find_peaks(current)
            plt.plot(dist[peaks_idx], current[peaks_idx], "ro", markersize=6, label="Peaks")
            plt.xlim(0, 1)
            plt.xlabel("Normalised Distance", fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            save_figure(fig, os.path.join(save_dir, f"{prefix}_{a}deg.png"))

    @staticmethod
    def plot_rays_with_peaks(
        data_array: np.ndarray,
        rays_dict: Dict,
        save_dir: str,
        prefix: str = "rays_with_peaks",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: str = "hot",
        peak_marker_style: str = "x",
        peak_marker_size: int = 100,
        peak_marker_color: str = "black",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        X, Y, I = data_array[:, 0], data_array[:, 1], data_array[:, 2]
        fig, ax, cax = new_map_figure(with_colorbar=True)
        plt.sca(ax)
        sc = ax.scatter(X, Y, c=I, cmap=cmap, s=10, edgecolor="none", alpha=1)
        fig.colorbar(sc, cax=cax, label="Current (nA)")
        apply_voltage_axes(ax, X.min(), X.max(), Y.min(), Y.max())
        ax.set_xlabel(xlabel or _x_label(), fontsize=14)
        ax.set_ylabel(ylabel or _y_label(), fontsize=14)
        peaks_added = False
        for angle, ray in rays_dict.items():
            rx, ry, rc = ray["X"], ray["Y"], ray["Current"]
            plt.plot(rx, ry, linewidth=2, label=f"{round(angle)}°")
            idx, _ = find_peaks(rc)
            if len(idx) > 0:
                plt.scatter(rx[idx], ry[idx], marker=peak_marker_style,
                            s=peak_marker_size, c=peak_marker_color, zorder=6,
                            label="Peaks" if not peaks_added else "_nolegend_")
                peaks_added = True
        save_figure(fig, os.path.join(save_dir, f"{prefix}.png"))
