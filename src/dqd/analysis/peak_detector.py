"""
PeakDetector — directional (bottom-up / top-down) peak sweeping.
Absorbs the old sweeping_plot.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple


class PeakDetector:
    """
    Detects peaks in a 2-D charge-sensor array by scanning rows in a
    configurable direction and stopping at the first local maximum per row.

    Parameters
    ----------
    output_dir    : directory where plots / GIFs / text files are saved
    save_plots    : generate PNG coverage images
    save_gifs     : generate animated GIFs
    save_txt      : write sweep parameter text files
    col_buffer    : column offset applied between consecutive rows
    scanned_voltages : list of (vx, vy) pairs already scanned (termination check)
    threshold     : distance threshold for early termination in voltage space
    """

    def __init__(
        self,
        output_dir: str,
        save_plots: bool = True,
        save_gifs: bool = True,
        save_txt: bool = True,
        col_buffer: int = 3,
        scanned_voltages: Optional[List[Tuple[float, float]]] = None,
        threshold: float = 0.05,
    ):
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.save_gifs = save_gifs
        self.save_txt = save_txt
        self.col_buffer = col_buffer
        self.scanned_voltages = scanned_voltages or []
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_path: str, hyperparams: Dict) -> Dict:
        """
        Run all sweep configurations defined in hyperparams["sweeps"].

        Parameters
        ----------
        data_path  : path to .npy file with columns [Vx, Vy, Current]
        hyperparams: dict with keys:
                       sweeps         – list of sweep config dicts
                       save_plots     – bool (overrides constructor if present)
                       save_gifs      – bool
                       save_txt       – bool
                       col_buffer     – int
                       scanned_voltages – list
                       threshold      – float
                       start_pixel_x, start_pixel_y  (used externally)
                       global_pixel_x, global_pixel_y (used externally)

        Returns
        -------
        dict with keys "data_shape", "sweeps", "output_files"
        """
        os.makedirs(self.output_dir, exist_ok=True)

        data = np.load(data_path)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Data at {data_path} must have shape (N,3).")

        Vx, Vy, Current = data[:, 0], data[:, 1], data[:, 2]
        unique_Vx = np.unique(Vx)
        unique_Vy = np.unique(Vy)
        num_Vx, num_Vy = len(unique_Vx), len(unique_Vy)
        if num_Vx * num_Vy != len(Current):
            raise ValueError(f"Grid mismatch: {num_Vx}×{num_Vy} ≠ {len(Current)}")
        current_2d = Current.reshape((num_Vy, num_Vx))

        col_buffer = hyperparams.get("col_buffer", self.col_buffer)
        scanned_voltages = hyperparams.get("scanned_voltages", self.scanned_voltages)
        threshold = hyperparams.get("threshold", self.threshold)

        results = {"data_shape": current_2d.shape, "sweeps": {}, "output_files": []}

        for sweep_cfg in hyperparams["sweeps"]:
            name = sweep_cfg["name"]
            peaks, states = self._sweep(
                current_2d=current_2d,
                start_row=sweep_cfg["start_row"],
                row_step=sweep_cfg["row_step"],
                start_col=sweep_cfg["start_col"],
                col_step=sweep_cfg["col_step"],
                col_buffer=col_buffer,
                unique_Vx=unique_Vx,
                unique_Vy=unique_Vy,
                scanned_voltages=scanned_voltages,
                threshold=threshold,
            )

            sr = {
                "peaks": peaks,
                "states": states,
                "measured_cells": sum(len(s["scanned_cols"]) for s in states),
                "outputs": {},
            }

            if hyperparams.get("save_gifs", self.save_gifs):
                gif_path = os.path.join(self.output_dir, f"peak_sweep_{name}.gif")
                try:
                    self._animate(current_2d, states, gif_path)
                    sr["outputs"]["animation"] = gif_path
                except Exception as e:
                    print(f"Warning: GIF animation failed for {name} ({e}). Skipping.")

            if hyperparams.get("save_txt", self.save_txt):
                txt_path = os.path.join(self.output_dir, f"sweep_params_{name}.txt")
                self._save_txt(states, peaks, current_2d, txt_path, sweep_cfg)
                sr["outputs"]["parameters_file"] = txt_path

            results["sweeps"][name] = sr
            results["output_files"].extend(list(sr["outputs"].values()))

        # ------------------------------------------------------------------
        # Post-processing: interdot transitions via break-point connection
        #
        # Each sweep trajectory has a "break point" — the last grid cell where
        # it successfully tracked a transition line before losing it.  At a
        # honeycomb vertex two different sweep families both lose their line at
        # nearby locations.  Connecting the nearest pair of break points gives
        # the interdot transition segment; scanning along it finds its peak.
        # ------------------------------------------------------------------
        interdot_threshold = hyperparams.get(
            "interdot_threshold_px", col_buffer * 4
        )
        interdot_peaks = self._find_interdot_by_breakpoints(
            current_2d, results["sweeps"], threshold_px=interdot_threshold
        )
        results["interdot_peaks"] = interdot_peaks

        if interdot_peaks and hyperparams.get("save_txt", self.save_txt):
            txt_path = os.path.join(
                self.output_dir, "sweep_params_interdot_connections.txt"
            )
            self._save_interdot_txt(interdot_peaks, current_2d, txt_path)
            results["output_files"].append(txt_path)

        return results

    # ------------------------------------------------------------------
    # Row-major sweep
    # ------------------------------------------------------------------

    def _sweep(
        self, current_2d, start_row, row_step, start_col, col_step,
        col_buffer, unique_Vx, unique_Vy, scanned_voltages, threshold,
    ) -> Tuple[List, List]:
        m, n = current_2d.shape
        states, peaks = [], []

        # Clamp the start cell into the grid.  Sweep configs deliberately offset
        # the start by col_buffer, which can land just outside a small crop; an
        # unclamped start row indexes current_2d out of bounds and takes the
        # whole peak down with it.
        start_row = max(0, min(m - 1, start_row))
        current_start_col = max(0, min(n - 1, start_col))

        for r in range(start_row, m if row_step > 0 else -1, row_step):
            current_start_col = max(0, min(n - 1, current_start_col))
            peak_col, scanned_cols, measured_vals = self._scan_row(
                current_2d[r, :], current_start_col, col_step,
                unique_Vx, unique_Vy, r, scanned_voltages, threshold,
            )
            states.append({"row": r, "peak_col": peak_col,
                           "scanned_cols": scanned_cols, "measured_vals": measured_vals})
            peaks.append((r, peak_col))
            if peak_col is None:
                break
            next_start = (peak_col - col_buffer if col_step > 0
                          else peak_col + col_buffer)
            current_start_col = max(0, min(n - 1, next_start))

        return peaks, states

    def _scan_row(self, row_data, start_col, col_step,
                  unique_Vx, unique_Vy, row_index, scanned_voltages, threshold):
        scanned_cols, measured_vals = [], []
        peak_col = None
        col_range = (range(start_col, len(row_data), col_step)
                     if col_step > 0 else range(start_col, -1, col_step))

        for c in col_range:
            vx = unique_Vx[c]
            vy = unique_Vy[row_index]

            # Early termination if already scanned nearby
            terminate_early = False
            for (ex_vx, ex_vy) in scanned_voltages:
                if ((vx - ex_vx) ** 2 + (vy - ex_vy) ** 2) ** 0.5 < threshold:
                    scanned_cols.append(c)
                    measured_vals.append(row_data[c])
                    terminate_early = True
                    break
            if terminate_early:
                return None, scanned_cols, measured_vals

            scanned_cols.append(c)
            measured_vals.append(row_data[c])

            # Local-maximum check (three-point)
            if len(measured_vals) >= 3:
                if measured_vals[-2] > measured_vals[-3] and measured_vals[-2] > measured_vals[-1]:
                    peak_col = scanned_cols[-2]
                    break

        return peak_col, scanned_cols, measured_vals

    # ------------------------------------------------------------------
    # Interdot break-point detection (post-processing)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_interdot_by_breakpoints(
        current_2d: np.ndarray,
        sweep_results: Dict,
        threshold_px: int = 12,
    ) -> List[Tuple[int, int]]:
        """
        Connect 'break points' of different sweep trajectories to find
        interdot transition lines.

        A break point is the last grid cell (row, col) where a sweep
        successfully tracked a transition line before losing it.  At a
        honeycomb vertex, two different sweep families both terminate nearby;
        the short segment between them is the interdot transition.

        Algorithm
        ---------
        1. Extract the last valid (row, col) peak from each sweep.
        2. Pair each break point with its nearest unmatched partner that is
           within *threshold_px* pixels.
        3. For each connected pair, sample points along the straight line
           between them and return the grid cell with the maximum current as
           the interdot peak.

        Returns
        -------
        List of (row, col) integer tuples in grid coordinates.
        """
        # Step 1 — collect break points
        break_pts: List[Tuple[str, int, int]] = []   # (sweep_name, row, col)
        for name, sr in sweep_results.items():
            last_valid = None
            for r, c in sr["peaks"]:
                if c is not None:
                    last_valid = (int(r), int(c))
            if last_valid is not None:
                break_pts.append((name, last_valid[0], last_valid[1]))

        if len(break_pts) < 2:
            return []

        m, n = current_2d.shape
        interdot_peaks: List[Tuple[int, int]] = []
        used: set = set()

        # Step 2 — greedy nearest-neighbour pairing within threshold
        for i in range(len(break_pts)):
            if i in used:
                continue
            best_j, best_dist = None, float(threshold_px)
            for j in range(len(break_pts)):
                if j == i or j in used:
                    continue
                dr = break_pts[i][1] - break_pts[j][1]
                dc = break_pts[i][2] - break_pts[j][2]
                d = (dr ** 2 + dc ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is None:
                continue

            used.add(i)
            used.add(best_j)

            r1, c1 = break_pts[i][1], break_pts[i][2]
            r2, c2 = break_pts[best_j][1], break_pts[best_j][2]

            # Step 3 — scan along the connecting segment, find the peak
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
            peak_idx = int(np.argmax(vals))
            interdot_peaks.append(valid[peak_idx])

        return interdot_peaks

    @staticmethod
    def _save_interdot_txt(
        interdot_peaks: List[Tuple[int, int]],
        current_2d: np.ndarray,
        filename: str,
    ) -> None:
        """
        Write interdot peaks in the same sweep_params_*.txt format that
        SummaryWriter consumes, so they appear in summary_local.txt and
        subsequently in voltage_coordinates.txt and all overlays.
        """
        with open(filename, "w") as f:
            f.write("Sweep Parameters and Results\n")
            f.write("=" * 40 + "\n\n")
            f.write("Sweep Configuration:\n")
            f.write("-" * 40 + "\n")
            f.write("Name: interdot_connections (break-point post-processing)\n\n")
            f.write("Row-wise Measurement Details:\n")
            f.write("-" * 40 + "\n")
            for r, c in interdot_peaks:
                try:
                    val_str = f"{current_2d[r, c]:.3e}"
                except IndexError:
                    val_str = "Out of Bounds"
                # Write as 1-based indices to match the format _parse_row_line expects
                f.write(
                    f"Row={r + 1}, scanned=[{c + 1}], peak={c + 1}, val={val_str}\n"
                )
            f.write("\nSummary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total measured cells: {len(interdot_peaks)}\n")
            f.write(
                f"Valid peaks detected: {len(interdot_peaks)}/{len(interdot_peaks)}\n"
            )
            corrected = [(r + 1, c + 1) for r, c in interdot_peaks]
            f.write(f"Peak coordinates: {corrected}\n")

    # ------------------------------------------------------------------
    # GIF animation
    # ------------------------------------------------------------------

    def _animate(self, current_2d, states, out_path):
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(current_2d, cmap="hot", origin="lower", aspect="auto")
        fig.colorbar(cax, ax=ax, shrink=0.7)
        ax.set_title("Row-major Peak Sweep")

        num_rows, num_cols = current_2d.shape
        ax.set_xticks(np.arange(num_cols))
        ax.set_yticks(np.arange(num_rows))
        ax.set_xticklabels(np.arange(1, num_cols + 1))
        ax.set_yticklabels(np.arange(1, num_rows + 1))
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)

        row_line, = ax.plot([], [], "r-", lw=1)
        scanned_sc, = ax.plot([], [], "ko", ms=3)
        peak_sc, = ax.plot([], [], "ro", ms=6)

        def init():
            row_line.set_data([], [])
            scanned_sc.set_data([], [])
            peak_sc.set_data([], [])
            return row_line, scanned_sc, peak_sc

        def update(frame):
            s = states[frame]
            r, cols, pcol = s["row"], s["scanned_cols"], s["peak_col"]
            row_line.set_data([0, current_2d.shape[1] - 1], [r, r])
            scanned_sc.set_data(cols, [r] * len(cols))
            peak_sc.set_data([pcol] if pcol is not None else [], [r] if pcol is not None else [])
            ax.set_title(f"Row={r}, peak col={pcol}")
            return row_line, scanned_sc, peak_sc

        try:
            if len(states) == 0:
                raise ValueError("No states to animate.")
            ani = animation.FuncAnimation(
                fig, update, frames=len(states), init_func=init,
                interval=700, blit=False, repeat=False,
            )
            ani.save(out_path, writer="imagemagick", fps=1, dpi=150)
        except Exception as e:
            print(f"Warning: could not save GIF ({e}). Skipping.")
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Text output
    # ------------------------------------------------------------------

    def _save_txt(self, states, peaks, current_2d, filename, sweep_config):
        total_measured = sum(len(s["scanned_cols"]) for s in states)
        valid_peaks = sum(1 for p in peaks if p[1] is not None)

        with open(filename, "w") as f:
            f.write("Sweep Parameters and Results\n")
            f.write("=" * 40 + "\n\n")
            f.write("Sweep Configuration:\n")
            f.write("-" * 40 + "\n")

            sr = sweep_config.get("start_row", 0) + 1
            sc = sweep_config.get("start_col", 0) + 1
            rs = sweep_config.get("row_step", 1)
            cs = sweep_config.get("col_step", 1)
            f.write(f"Start Row (1-based): {sr}\n")
            f.write(f"Start Column (1-based): {sc}\n")
            f.write(f"Row Step Direction: {'Up' if rs > 0 else 'Down'}\n")
            f.write(f"Column Step Direction: {'Right' if cs > 0 else 'Left'}\n\n")

            if states:
                last = states[-1]
                er = last["row"] + 1
                ec = (last["scanned_cols"][-1] + 1) if last["scanned_cols"] else "N/A"
            else:
                er = ec = "N/A"
            f.write(f"End Row (1-based): {er}\n")
            f.write(f"End Column (1-based): {ec}\n\n")

            f.write("Row-wise Measurement Details:\n")
            f.write("-" * 40 + "\n")
            for s in states:
                r = s["row"] + 1
                scanned = [c + 1 for c in s["scanned_cols"]]
                pcol = s["peak_col"]
                if pcol is not None:
                    pc1 = pcol + 1
                    try:
                        val_str = f"{current_2d[s['row'], pcol]:.3e}"
                    except IndexError:
                        val_str = "Out of Bounds"
                    f.write(f"Row={r}, scanned={scanned}, peak={pc1}, val={val_str}\n")
                else:
                    f.write(f"Row={r}, scanned={scanned}, peak=None\n")

            f.write("\nSummary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total measured cells: {total_measured}\n")
            f.write(f"Grid coverage: {total_measured}/{current_2d.size} "
                    f"({total_measured/current_2d.size:.1%})\n")
            f.write(f"Valid peaks detected: {valid_peaks}/{len(peaks)}\n")
            corrected = [(r + 1, c + 1) if c is not None else (r + 1, c) for r, c in peaks]
            f.write(f"Peak coordinates: {corrected}\n")
