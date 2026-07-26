"""
SampleEvaluator — writes evaluation.txt and skipped_peaks.txt for a sample.

evaluation.txt  : precision/recall/F1/accuracy against ground-truth binary
                  array, plus measurement statistics (rays, coverage, …)
skipped_peaks.txt : ray-detected peaks whose sweep found zero confirmed peaks
"""
import os
import json
import numpy as np
from typing import List, Set, Tuple


class SampleEvaluator:
    """
    Generates evaluation.txt and skipped_peaks.txt for a fully processed sample.

    Parameters
    ----------
    sample_dir     : path to the sample directory (e.g. …/sample_1)
    num_angles     : number of ray angles used
    ray_resolution : number of points per ray
    x_resolution   : voltage-grid resolution in x (columns)
    y_resolution   : voltage-grid resolution in y (rows)
    vx_min/max     : voltage range in x
    vy_min/max     : voltage range in y
    peak_neighbor_cols : number of extra columns to mark on each side of every
                         detected peak (0 = exact pixel only).  Increasing this
                         trades precision for recall (fewer false negatives).
    """

    def __init__(
        self,
        sample_dir: str,
        num_angles: int,
        ray_resolution: int,
        x_resolution: int,
        y_resolution: int,
        vx_min: float,
        vx_max: float,
        vy_min: float,
        vy_max: float,
        peak_neighbor_cols: int = 0,
    ):
        self.sample_dir = sample_dir
        self.num_angles = num_angles
        self.ray_resolution = ray_resolution
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max
        self.peak_neighbor_cols = peak_neighbor_cols

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Write evaluation.txt and skipped_peaks.txt into sample_dir."""
        self._write_evaluation()
        self._write_skipped_peaks()

    # ------------------------------------------------------------------
    # Coordinate helper
    # ------------------------------------------------------------------

    def _voltage_to_pixel(self, vx: float, vy: float) -> Tuple[int, int]:
        """
        Convert voltage (vx, vy) → 0-based (row, col) in the full grid.
        row corresponds to vy, col corresponds to vx.
        """
        col = int(round(
            (vx - self.vx_min) / (self.vx_max - self.vx_min) * (self.x_resolution - 1)
        ))
        row = int(round(
            (vy - self.vy_min) / (self.vy_max - self.vy_min) * (self.y_resolution - 1)
        ))
        col = max(0, min(self.x_resolution - 1, col))
        row = max(0, min(self.y_resolution - 1, row))
        return row, col

    # ------------------------------------------------------------------
    # Data collectors
    # ------------------------------------------------------------------

    def _collect_voltage_coords(self) -> Tuple[
        List[Tuple[float, float]], List[Tuple[float, float]]
    ]:
        """
        Walk cropped_results/ and return all (scanned_cells, peaks) as
        flat lists of (Vx, Vy) tuples collected from every
        voltage_coordinates.txt file.
        """
        from ..visualization.overlay import OverlayRenderer

        all_scanned: List[Tuple[float, float]] = []
        all_peaks: List[Tuple[float, float]] = []

        cropped_dir = os.path.join(self.sample_dir, "cropped_results")
        if not os.path.isdir(cropped_dir):
            return all_scanned, all_peaks

        for root, _, files in os.walk(cropped_dir):
            for fname in files:
                if fname.lower() == "voltage_coordinates.txt":
                    s, p = OverlayRenderer.parse_voltage_coordinates(
                        os.path.join(root, fname)
                    )
                    all_scanned.extend(s)
                    all_peaks.extend(p)

        return all_scanned, all_peaks

    # ------------------------------------------------------------------
    # evaluation.txt
    # ------------------------------------------------------------------

    def _write_evaluation(self) -> None:
        gt_path = os.path.join(
            self.sample_dir, "numpy", "simulation", "ground_truth_labels.npy"
        )
        if not os.path.isfile(gt_path):
            print(
                f"[Evaluator] Ground truth not found at {gt_path}; "
                "skipping evaluation.txt"
            )
            return

        # Ground truth: set of (row, col) pixels with value 1
        gt = np.load(gt_path)  # shape (y_resolution, x_resolution), dtype uint8
        gt_pixels: Set[Tuple[int, int]] = set(zip(*np.where(gt == 1)))
        n_gt = len(gt_pixels)

        # Sweep-detected peaks and scanned cells (deduplicated by pixel)
        all_scanned_v, all_peaks_v = self._collect_voltage_coords()

        detected_pixels: Set[Tuple[int, int]] = set()
        for vx, vy in all_peaks_v:
            detected_pixels.add(self._voltage_to_pixel(vx, vy))

        scanned_pixels: Set[Tuple[int, int]] = set()
        for vx, vy in all_scanned_v:
            scanned_pixels.add(self._voltage_to_pixel(vx, vy))

        # Classification metrics
        tp_pixels = detected_pixels & gt_pixels
        fp_pixels = detected_pixels - gt_pixels
        fn_pixels = gt_pixels - detected_pixels

        tp = len(tp_pixels)
        fp = len(fp_pixels)
        fn = len(fn_pixels)
        n_detected = len(detected_pixels)

        total_cells = self.x_resolution * self.y_resolution
        tn = total_cells - tp - fp - fn

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy  = (tp + tn) / total_cells * 100

        n_scanned = len(scanned_pixels)
        coverage  = n_scanned / total_cells * 100

        out_path = os.path.join(self.sample_dir, "evaluation.txt")
        with open(out_path, "w") as f:
            f.write("Peak Detection Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Ground Truth Peaks : {n_gt}\n")
            f.write(f"Detected Peaks     : {n_detected}\n")
            f.write(f"True Positives     : {tp}\n")
            f.write(f"False Positives    : {fp}\n")
            f.write(f"False Negatives    : {fn}\n\n")
            f.write(f"Precision : {precision:.2f}%\n")
            f.write(f"Recall    : {recall:.2f}%\n")
            f.write(f"F1 Score  : {f1:.2f}%\n")
            f.write(f"Accuracy  : {accuracy:.2f}%\n\n")
            f.write("Measurement Statistics:\n")
            f.write(f"  Number of Rays : {self.num_angles}\n")
            f.write(f"  Ray Resolution : {self.ray_resolution}\n")
            f.write(f"  Scanned Cells  : {n_scanned} (out of {total_cells})\n")
            f.write(f"  Grid Coverage  : {coverage:.2f}%\n")

        print(f"[Evaluator] evaluation.txt saved to {out_path}")

    # ------------------------------------------------------------------
    # skipped_peaks.txt
    # ------------------------------------------------------------------

    def _write_skipped_peaks(self) -> None:
        """
        For each ray angle, list ray-detected peaks whose peak folder has
        zero sweep-confirmed peaks in voltage_coordinates.txt.
        """
        from ..visualization.overlay import OverlayRenderer

        peaks_paired_path = os.path.join(self.sample_dir, "peaks_paired.json")
        if not os.path.isfile(peaks_paired_path):
            print(
                "[Evaluator] peaks_paired.json not found; skipping skipped_peaks.txt"
            )
            return

        with open(peaks_paired_path, "r") as f:
            paired_dict = json.load(f)

        lines = [
            "Skipped Peaks Report\n",
            "=" * 40 + "\n",
            "Peaks detected by ray algorithm but not confirmed by sweep analysis.\n\n",
        ]

        any_skipped = False

        for angle_str, peaks in paired_dict.items():
            angle_f = float(angle_str)
            ray_folder = os.path.join(
                self.sample_dir, "cropped_results", f"ray_{angle_f:.0f}"
            )
            skipped: List[Tuple[int, float, float]] = []

            for peak_idx, (vx, vy) in enumerate(peaks, 1):
                vc_path = os.path.join(
                    ray_folder, f"peak_{peak_idx}", "voltage_coordinates.txt"
                )
                if not os.path.isfile(vc_path):
                    # Folder missing / not processed → treat as skipped
                    skipped.append((peak_idx, vx, vy))
                    continue
                _, sweep_peaks = OverlayRenderer.parse_voltage_coordinates(vc_path)
                if len(sweep_peaks) == 0:
                    skipped.append((peak_idx, vx, vy))

            angle_label = f"Angle {angle_f:.1f}\u00b0"
            if skipped:
                any_skipped = True
                lines.append(f"{angle_label}:\n")
                for idx, vx, vy in skipped:
                    lines.append(f"  Peak {idx}: ({vx:.6f}, {vy:.6f})\n")
            else:
                lines.append(f"{angle_label}: (none)\n")

        if not any_skipped:
            lines.append(
                "\nAll ray-detected peaks were confirmed by sweep analysis.\n"
            )

        out_path = os.path.join(self.sample_dir, "skipped_peaks.txt")
        with open(out_path, "w") as f:
            f.writelines(lines)

        print(f"[Evaluator] skipped_peaks.txt saved to {out_path}")
