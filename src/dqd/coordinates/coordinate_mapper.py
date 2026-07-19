"""
CoordinateMapper — all coordinate-system conversions for the DQD pipeline.
Absorbs the old utility1.py, utility2.py, and the mapping parts of crop_image.py.
"""
import os
import json
import numpy as np
from typing import Optional, Tuple, Dict


class CoordinateMapper:
    """
    Handles three coordinate spaces used throughout the pipeline:
      - Voltage space   (Vx, Vy)  in millivolts
      - Pixel / global  (col, row) in 0-based full-image indices
      - Local / cropped (col, row) in 1-based indices inside a cropped region

    All methods are stateless; pass the required parameters explicitly.
    """

    # ------------------------------------------------------------------
    # Voltage ↔ Pixel conversions  (from old utility1.py)
    # ------------------------------------------------------------------

    @staticmethod
    def voltage_to_pixel(
        vx: float, vy: float,
        vx_min: float, vx_max: float,
        vy_min: float, vy_max: float,
        x_res: int, y_res: int,
    ) -> Tuple[int, int]:
        """
        Map a voltage coordinate to 0-based pixel indices (pixel_x, pixel_y).
        Both indices are clipped to [0, res-1].
        """
        vx = float(np.clip(vx, vx_min, vx_max))
        vy = float(np.clip(vy, vy_min, vy_max))

        vx_step = (vx_max - vx_min) / (x_res - 1)
        vy_step = (vy_max - vy_min) / (y_res - 1)

        pixel_x = int(round((vx - vx_min) / vx_step))
        pixel_y = int(round((vy - vy_min) / vy_step))

        pixel_x = int(np.clip(pixel_x, 0, x_res - 1))
        pixel_y = int(np.clip(pixel_y, 0, y_res - 1))
        return pixel_x, pixel_y

    # ------------------------------------------------------------------
    # Center indices from JSON  (from old utility1.py)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_center_indices(json_path: str) -> Optional[Tuple[int, int]]:
        """
        Read center_indices.json and return (row_index, col_index) for
        charge_sensing_data.npy, or None if not found.
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            entry = data.get("charge_sensing_data.npy", {})
            cropped = entry.get("cropped", {})
            row = cropped.get("row_index")
            col = cropped.get("column_index")
            if row is not None and col is not None:
                return (row, col)
            print("Center indices not found in JSON.")
            return None
        except FileNotFoundError:
            print(f"JSON not found: {json_path}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON: {json_path}")
            return None

    # ------------------------------------------------------------------
    # Local (1-based cropped) → Voltage  (from old utility2.py)
    # ------------------------------------------------------------------

    @staticmethod
    def local_to_voltage(
        summary_file: str,
        mapping_file: str,
        output_file: str,
    ) -> None:
        """
        Convert LOCAL row/col indices from a summary file to voltage
        coordinates using the grid-mapping file and write to output_file.

        Parameters
        ----------
        summary_file  : path to summary_local.txt
        mapping_file  : path to charge_sensing_cropped_grid_mapping.txt
        output_file   : where to write voltage_coordinates.txt
        """
        mapping = CoordinateMapper._load_mapping(mapping_file)
        scanned_local, peaks_local = CoordinateMapper._parse_summary(summary_file)

        scanned_v = [mapping[c] for c in scanned_local if c in mapping]
        peaks_v = [mapping[c] for c in peaks_local if c in mapping]

        with open(output_file, "w") as f:
            f.write("Scanned Cells (Voltage Coordinates):\n")
            f.write(", ".join(f"({vx:.6f}, {vy:.6f})" for vx, vy in scanned_v) + "\n")
            f.write("\nPeaks (Voltage Coordinates):\n")
            f.write(", ".join(f"({vx:.6f}, {vy:.6f})" for vx, vy in peaks_v) + "\n")

    # ------------------------------------------------------------------
    # Grid mapping file I/O  (from old crop_image.py)
    # ------------------------------------------------------------------

    @staticmethod
    def write_cropped_grid_mapping(
        cropped_Vx: np.ndarray,
        cropped_Vy: np.ndarray,
        output_dir: str,
        base_name: str,
    ) -> str:
        """
        Write a tab-separated file that maps 1-based (row, col) → (Vx, Vy).
        Returns the path of the written file.
        """
        mapping_path = os.path.join(output_dir, f"{base_name}_cropped_grid_mapping.txt")
        with open(mapping_path, "w") as f:
            f.write("row\tcolumn\tVx\tVy\n")
            for i, vy in enumerate(cropped_Vy):
                for j, vx in enumerate(cropped_Vx):
                    f.write(f"{i + 1}\t{j + 1}\t{vx}\t{vy}\n")
        print(f"Cropped grid mapping saved to: {mapping_path}")
        return mapping_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mapping(mapping_file: str) -> Dict[Tuple[int, int], Tuple[float, float]]:
        mapping: Dict[Tuple[int, int], Tuple[float, float]] = {}
        with open(mapping_file, "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                row, col = int(parts[0]), int(parts[1])
                vx, vy = float(parts[2]), float(parts[3])
                mapping[(row, col)] = (vx, vy)
        return mapping

    @staticmethod
    def _parse_summary(
        summary_file: str,
    ) -> Tuple[list, list]:
        """Parse summary_local.txt → (scanned_coords, peak_coords) as 1-based (row,col) tuples."""
        scanned, peaks = [], []
        current_row = None
        with open(summary_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Row="):
                    current_row = int(line.split("=")[1])
                elif line.startswith("Scanned Columns:"):
                    cols_str = line.split(": [")[1].split("]")[0]
                    for c in cols_str.split(", "):
                        if c:
                            scanned.append((current_row, int(c)))
                elif line.startswith("Detected Peaks:"):
                    peaks_str = line.split(": ")[1].strip()
                    if peaks_str == "[]":
                        continue
                    peaks_str = peaks_str[1:-1]
                    if not peaks_str:
                        continue
                    for p in peaks_str.split("), "):
                        p = p.strip("()")
                        r, c = p.split(", ")
                        peaks.append((int(r), int(c)))
        return scanned, peaks
