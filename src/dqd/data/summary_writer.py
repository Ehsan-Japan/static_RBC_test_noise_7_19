"""
SummaryWriter — aggregates sweep results from text files into a unified
LOCAL or GLOBAL coordinate summary.
Absorbs the old parse_all.py.
"""
import os
import re
from typing import Optional


class SummaryWriter:
    """
    Reads all sweep_params_*.txt files from a given directory and
    writes a unified summary file with LOCAL or GLOBAL coordinates.
    """

    def write(
        self,
        directory: str,
        global_start_row: Optional[int] = None,
        global_start_col: Optional[int] = None,
        output_filename: str = "summary.txt",
        peak_neighbor_cols: int = 0,
    ) -> str:
        """
        Create a summary text file.

        Parameters
        ----------
        directory         : folder containing the sweep parameter files
        global_start_row  : if provided, switch to GLOBAL coordinate mode
        global_start_col  : if provided, switch to GLOBAL coordinate mode
        output_filename   : name for the output summary file

        Returns
        -------
        Path of the written summary file.
        """
        is_global = (global_start_row is not None) or (global_start_col is not None)
        gsr = global_start_row or 0
        gsc = global_start_col or 0

        coord_info = {
            "system": "GLOBAL" if is_global else "LOCAL",
            "origin": (f"Crop Origin: Row {gsr}, Column {gsc}"
                       if is_global else "Local Coordinate System"),
            "row_label": "Global Row" if is_global else "Row",
            "col_label": "Global Column" if is_global else "Column",
        }

        row_aggregate: dict = {}
        all_peaks: set = set()

        sweep_files = sorted(
            f for f in os.listdir(directory)
            if f.startswith("sweep_params_") and f.endswith(".txt")
        )
        for filename in sweep_files:
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                for line in f:
                    if not line.startswith("Row="):
                        continue
                    row_info = self._parse_row_line(line.strip())
                    if not row_info:
                        continue

                    local_row = row_info["row"]
                    cur_row = gsr + local_row if is_global else local_row

                    if cur_row not in row_aggregate:
                        row_aggregate[cur_row] = {"scanned": set(), "peaks": set()}

                    scanned_cols = [
                        gsc + c if is_global else c
                        for c in row_info.get("scanned", [])
                    ]
                    row_aggregate[cur_row]["scanned"].update(scanned_cols)

                    if row_info.get("peak") is not None:
                        peak_col = gsc + row_info["peak"] if is_global else row_info["peak"]
                        for dc in range(-peak_neighbor_cols, peak_neighbor_cols + 1):
                            nc = peak_col + dc
                            if nc >= 1:
                                all_peaks.add((cur_row, nc))
                                row_aggregate[cur_row]["peaks"].add((cur_row, nc))

        total_measured = sum(len(v["scanned"]) for v in row_aggregate.values())
        sys = coord_info["system"]

        lines = [
            f"{sys} Coordinate Summary",
            "=" * (len(sys) + 18),
            coord_info["origin"],
            "",
            f"Row-wise Details ({sys} Coordinates):",
            "-" * (len(sys) + 20),
        ]
        for row in sorted(row_aggregate):
            data = row_aggregate[row]
            lines.append(f"{coord_info['row_label']}={row}")
            lines.append(f"  Scanned {coord_info['col_label']}s: {sorted(data['scanned'])}")
            lines.append(f"  Detected Peaks: {sorted(data['peaks'])}")
            lines.append("")

        lines += [
            "",
            "Summary Statistics:",
            "-" * 20,
            f"Total {sys} Cells Measured: {total_measured}",
            f"{sys} Peaks Detected: {len(all_peaks)}",
            f"{sys} Peak Coordinates: {sorted(all_peaks)}",
        ]

        output_path = os.path.join(directory, output_filename)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_row_line(line: str) -> dict:
        parts = re.split(r",\s*(?![^\[\]]*\])", line.strip())
        row_info: dict = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key, value = key.strip(), value.strip()
            if key == "Row":
                row_info["row"] = int(value)
            elif key == "scanned":
                s = value.strip("[]")
                row_info["scanned"] = list(map(int, s.split(", "))) if s else []
            elif key == "peak":
                row_info["peak"] = None if value == "None" else int(value)
            elif key == "val":
                row_info["val"] = float(value) if value != "None" else None
        return row_info
