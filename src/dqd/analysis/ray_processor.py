"""
RayProcessor — loads charge-sensor data, fires rays at specified angles,
detects peaks along each ray, and generates plots.
Absorbs the old data_processing.pre_process / collect_real_data_rays /
rays_to_dictionary and the plotting calls in process_plot.py.
"""
import os
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple

from ..visualization.plotter import Plotter


class RayProcessor:
    """
    Processes a single .npy charge-sensor data file through the ray-detection
    pipeline.

    Parameters
    ----------
    file_path_numpy : path to the .npy file (columns [Vx, Vy, Current])
    angles_deg      : list of ray angles (degrees); rays are fired from (max_x, max_y)
    resolution      : number of sample points per ray
    save_dir        : directory where plots are saved
    """

    def __init__(
        self,
        file_path_numpy: str,
        angles_deg: List[float],
        resolution: int = 50,
        save_dir: Optional[str] = None,
    ):
        self.file_path_numpy = file_path_numpy
        self.angles_deg = angles_deg
        self.resolution = resolution
        self.save_dir = save_dir or os.path.dirname(file_path_numpy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Full pipeline: load → normalise → collect rays → convert to dict →
        generate plots.

        Returns
        -------
        results_dict : dict with keys real_data, rays_data, rays_dict,
                       peaks_dict, save_dir, barrier_x, barrier_y, …
        """
        # Load
        real_data = np.load(self.file_path_numpy)
        barrier_x, barrier_y = self._extract_barriers(self.file_path_numpy)

        # Normalise current (in-place)
        lo, hi = real_data[:, 2].min(), real_data[:, 2].max()
        if hi > lo:
            real_data[:, 2] = (real_data[:, 2] - lo) / (hi - lo)

        min_x, max_x = real_data[:, 0].min(), real_data[:, 0].max()
        min_y, max_y = real_data[:, 1].min(), real_data[:, 1].max()

        # Collect rays and peaks
        rays_data, peaks_dict = self._collect_rays(
            real_data, min_x, max_x, min_y, max_y
        )
        rays_dict = self._rays_to_dict(rays_data)

        os.makedirs(self.save_dir, exist_ok=True)

        # Save numpy arrays
        self._save_all_data(real_data, rays_dict, self.save_dir)

        # Generate plots
        Plotter.plot_original_data(
            data_array=real_data,
            save_dir=self.save_dir,
            prefix="original",
        )
        Plotter.plot_rays_only(
            rays_data=rays_data,
            save_dir=self.save_dir,
            prefix="rays_only",
        )
        Plotter.plot_combined(
            data_array=real_data,
            rays_data=rays_data,
            save_dir=self.save_dir,
            prefix="combined_plot",
        )
        Plotter.plot_current_vs_distance(
            rays_dict=rays_dict,
            save_dir=self.save_dir,
            prefix="rays_current_vs_distance",
        )
        Plotter.plot_each_ray_separately(
            rays_dict=rays_dict,
            save_dir=self.save_dir,
            prefix="original_ray",
        )
        return {
            "base_filename": os.path.splitext(
                os.path.basename(self.file_path_numpy)
            )[0],
            "barrier_x": barrier_x,
            "barrier_y": barrier_y,
            "real_data": real_data,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "rays_data": rays_data,
            "rays_dict": rays_dict,
            "peaks_dict": peaks_dict,
            "save_dir": self.save_dir,
        }

    # ------------------------------------------------------------------
    # Ray collection
    # ------------------------------------------------------------------

    def _collect_rays(
        self,
        data_array: np.ndarray,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> Tuple[np.ndarray, Dict]:
        """
        For each angle in self.angles_deg fire a ray and detect peaks.

        Returns
        -------
        rays_data  : (M, 4) array  [X, Y, Current, Angle_deg]
        peaks_dict : {angle_float: {"vx": [...], "vy": [...]}, …}
        """
        all_rays: List[np.ndarray] = []
        peaks_dict: Dict = {}

        for angle in self.angles_deg:
            angle_rad = np.deg2rad(angle)
            try:
                ray_data, peak_array = self._measure_ray(
                    data_array, angle_rad, min_x, max_x, min_y, max_y
                )
                all_rays.append(ray_data)
                if peak_array.size > 0:
                    peaks_dict[float(angle)] = {
                        "vx": peak_array[:, 0].tolist(),
                        "vy": peak_array[:, 1].tolist(),
                    }
            except ValueError as exc:
                print(f"  Skipping angle {angle:.1f}° → {exc}")

        if not all_rays:
            raise ValueError("No valid rays found for given angles.")
        return np.vstack(all_rays), peaks_dict

    def _measure_ray(
        self,
        data_array: np.ndarray,
        angle_rad: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fire a single ray from (max_x, max_y) in direction (-cos θ, -sin θ),
        sample *resolution* points, detect peaks, and return
        (ray_array (K,4), peak_array (P,4)).
        """
        start = np.array([max_x, max_y])
        direction = np.array([-np.cos(angle_rad), -np.sin(angle_rad)])
        angle_deg = np.degrees(angle_rad)
        eps = 1e-10

        # Find where the ray exits the bounding box
        intersections = []
        if abs(direction[0]) > eps:
            t = (min_x - start[0]) / direction[0]
            if t > 0:
                y_c = start[1] + t * direction[1]
                if min_y <= y_c <= max_y:
                    intersections.append(np.array([min_x, y_c]))
        if abs(direction[1]) > eps:
            t = (min_y - start[1]) / direction[1]
            if t > 0:
                x_c = start[0] + t * direction[0]
                if min_x <= x_c <= max_x:
                    intersections.append(np.array([x_c, min_y]))

        if not intersections:
            raise ValueError(f"No valid boundary intersection for {angle_deg:.1f}°")

        end = min(intersections, key=lambda p: np.linalg.norm(p - start))
        vec = end - start
        length = np.linalg.norm(vec)
        unit = vec / length

        d_vals = np.linspace(0, length, self.resolution)
        pts = np.clip(
            start + d_vals.reshape(-1, 1) * unit,
            [min_x, min_y],
            [max_x, max_y],
        )

        currents = np.array(
            [self._nearest_current(x, y, data_array) for x, y in pts]
        )

        ray_array = np.column_stack([pts, currents, np.full(len(pts), angle_deg)])

        peak_idx, _ = find_peaks(currents)
        if peak_idx.size > 0:
            peak_pts = np.clip(
                start + d_vals[peak_idx].reshape(-1, 1) * unit,
                [min_x, min_y],
                [max_x, max_y],
            )
            peak_array = np.column_stack(
                [peak_pts, currents[peak_idx], np.full(len(peak_idx), angle_deg)]
            )
        else:
            peak_array = np.empty((0, 4))

        return ray_array, peak_array

    # ------------------------------------------------------------------
    # Data persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rays_to_dict(rays_data: np.ndarray) -> Dict:
        """
        Convert (M,4) array [X, Y, Current, Angle_deg] →
        {angle: {"X": …, "Y": …, "Current": …}}
        """
        result: Dict = {}
        for ang in np.unique(rays_data[:, 3]):
            mask = rays_data[:, 3] == ang
            s = rays_data[mask]
            result[float(ang)] = {"X": s[:, 0], "Y": s[:, 1], "Current": s[:, 2]}
        return result

    @staticmethod
    def _save_all_data(
        real_data: np.ndarray,
        rays_dict: Dict,
        output_dir: str,
    ) -> None:
        """
        Save the normalised real_data and each ray's (Vx, Vy, Current) arrays
        to numpy/original_numpy/ (mirrors old save_all_data behaviour).
        """
        save_path = os.path.join(output_dir, "original_numpy")
        os.makedirs(save_path, exist_ok=True)

        np.save(
            os.path.join(save_path, "original_voltage_current.npy"),
            np.column_stack([real_data[:, 0], real_data[:, 1], real_data[:, 2]]),
        )

        for angle, data in rays_dict.items():
            safe_key = f"{angle:.2f}".replace(".", "_").replace("-", "neg_")
            ray_arr = np.stack([data["X"], data["Y"], data["Current"]], axis=0)
            np.save(
                os.path.join(save_path, f"original_ray_{safe_key}_voltage_current.npy"),
                ray_arr,
            )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_current(x: float, y: float, data_array: np.ndarray) -> float:
        diffs = data_array[:, :2] - np.array([x, y])
        return float(data_array[np.argmin(np.sum(diffs ** 2, axis=1)), 2])

    @staticmethod
    def _extract_barriers(file_path: str) -> Tuple[str, str]:
        base = os.path.splitext(os.path.basename(file_path))[0]
        parts = base.split("_")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return "Vx", "Vy"
