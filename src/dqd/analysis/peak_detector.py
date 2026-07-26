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

from ..config.figure_style import draw_ground_truth_map


_GIF_WRITER: Optional[str] = None


def _gif_writer() -> str:
    """
    Writer used for the sweep GIFs, decided once per process.

    ImageMagick is preferred when installed; otherwise Pillow, which ships
    with matplotlib and produces the same GIF.  Resolving it once keeps
    matplotlib from printing "MovieWriter imagemagick unavailable; using
    Pillow instead." for every single GIF of the run.
    """
    global _GIF_WRITER
    if _GIF_WRITER is None:
        _GIF_WRITER = ("imagemagick"
                       if animation.writers.is_available("imagemagick")
                       else "pillow")
        print(f"[PeakDetector] GIF writer: {_GIF_WRITER}")
    return _GIF_WRITER


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
    gif_dpi       : resolution of the sweep GIFs.  Every frame is rendered
                    from scratch, so this is the main cost driver: 150 dpi
                    is ~230 ms/frame and ~2.4 MB per GIF, 100 dpi is
                    ~190 ms/frame and ~0.75 MB.
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
        gif_dpi: int = 150,
    ):
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.save_gifs = save_gifs
        self.save_txt = save_txt
        self.col_buffer = col_buffer
        self.scanned_voltages = scanned_voltages or []
        self.threshold = threshold
        self.gif_dpi = gif_dpi

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
                    self._animate(
                        current_2d, states, gif_path,
                        dpi=hyperparams.get("gif_dpi", self.gif_dpi),
                        background=hyperparams.get("gif_background"),
                        offset=hyperparams.get("gif_offset", (0, 0)),
                    )
                    sr["outputs"]["animation"] = gif_path
                except Exception as e:
                    print(f"Warning: GIF animation failed for {name} ({e}). Skipping.")

            if hyperparams.get("save_txt", self.save_txt):
                txt_path = os.path.join(self.output_dir, f"sweep_params_{name}.txt")
                self._save_txt(states, peaks, current_2d, txt_path, sweep_cfg)
                sr["outputs"]["parameters_file"] = txt_path

            results["sweeps"][name] = sr
            results["output_files"].extend(list(sr["outputs"].values()))

        # One extra GIF with every sweep of this peak running together, so the
        # way the four of them combine to cover the peak is visible at a glance.
        if hyperparams.get("save_gifs", self.save_gifs):
            combined_path = os.path.join(self.output_dir, "peak_sweep_ALL.gif")
            try:
                self._animate_combined(
                    current_2d,
                    [(n, results["sweeps"][n]["states"]) for n in results["sweeps"]],
                    combined_path,
                    dpi=hyperparams.get("gif_dpi", self.gif_dpi),
                    background=hyperparams.get("gif_background"),
                    offset=hyperparams.get("gif_offset", (0, 0)),
                )
                results["output_files"].append(combined_path)
            except Exception as e:
                print(f"Warning: combined GIF failed ({e}). Skipping.")

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
    # GIF animation
    # ------------------------------------------------------------------

    # Roughly how many tick labels to show per axis in the sweep animation.
    # One tick per cell (the old behaviour) overlaps into an unreadable smear
    # as soon as the crop is more than ~15 cells wide.
    _GIF_MAX_TICKS = 10

    # Colour per sweep in the combined animation, in sweep-config order.
    _SWEEP_COLORS = ("tab:blue", "tab:green", "tab:orange", "tab:purple")

    def _animate_combined(self, current_2d, sweeps, out_path,
                          dpi: Optional[int] = None,
                          background=None, offset=(0, 0)):
        """
        One animation with EVERY sweep of a peak advancing together, so you can
        see how the four of them combine to cover the peak.

        sweeps : ordered [(name, states), ...].  Sweeps are different lengths;
                 a short one holds its final state while the others finish, so
                 its tracked line stays on screen instead of vanishing.
        """
        sweeps = [(n, st) for n, st in sweeps if st]
        if not sweeps:
            return
        n_frames = max(len(st) for _, st in sweeps)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        r_off, c_off = offset

        if background is not None:
            num_rows, num_cols = background.shape
            draw_ground_truth_map(ax, np.arange(num_cols + 1) - 0.5,
                                  np.arange(num_rows + 1) - 0.5, background)
            crop_rows, crop_cols = current_2d.shape
            # Magenta dashed: not one of the sweep colours, so the crop
            # outline is never confused with a sweep.
            ax.add_patch(plt.Rectangle(
                (c_off - 0.5, r_off - 0.5), crop_cols, crop_rows,
                fill=False, edgecolor="magenta", ls="--", lw=2, zorder=5,
            ))
        else:
            num_rows, num_cols = current_2d.shape
            ax.imshow(current_2d, cmap="hot", origin="lower", aspect="auto")

        def _ticks(n):
            step = max(1, int(np.ceil(n / self._GIF_MAX_TICKS)))
            return np.arange(0, n, step)

        xt, yt = _ticks(num_cols), _ticks(num_rows)
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xticklabels(xt + 1); ax.set_yticklabels(yt + 1)
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)
        ax.set_aspect("equal", adjustable="box")

        artists = []
        for i, (name, _) in enumerate(sweeps):
            color = self._SWEEP_COLORS[i % len(self._SWEEP_COLORS)]
            row_line, = ax.plot([], [], "-", color=color, lw=1, alpha=0.8,
                                zorder=6, label=name)
            scanned, = ax.plot([], [], "o", color=color, ms=2.5, alpha=0.5,
                               zorder=7)
            found, = ax.plot([], [], "x", color=color, ms=7, mew=1.8, zorder=8)
            artists.append((row_line, scanned, found))

        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

        def init():
            for row_line, scanned, found in artists:
                row_line.set_data([], []); scanned.set_data([], [])
                found.set_data([], [])
            return [a for trio in artists for a in trio]

        def update(frame):
            for (name, states), (row_line, scanned, found) in zip(sweeps, artists):
                k = min(frame, len(states) - 1)      # short sweeps hold their end
                s = states[k]
                gr = s["row"] + r_off
                row_line.set_data([0, num_cols - 1], [gr, gr])
                gcols = [c + c_off for c in s["scanned_cols"]]
                scanned.set_data(gcols, [gr] * len(gcols))
                # every peak this sweep has found so far — its tracked line
                px = [st["peak_col"] + c_off for st in states[:k + 1]
                      if st["peak_col"] is not None]
                py = [st["row"] + r_off for st in states[:k + 1]
                      if st["peak_col"] is not None]
                found.set_data(px, py)
            ax.set_title(f"All sweeps — frame {frame + 1}/{n_frames}")
            return [a for trio in artists for a in trio]

        try:
            ani = animation.FuncAnimation(
                fig, update, frames=n_frames, init_func=init,
                interval=700, blit=False, repeat=False,
            )
            ani.save(out_path, writer=_gif_writer(), fps=1,
                     dpi=dpi or self.gif_dpi)
        except Exception as e:
            print(f"Warning: combined GIF failed ({e}). Skipping.")
        finally:
            plt.close(fig)

    def _animate(self, current_2d, states, out_path, dpi: Optional[int] = None,
                 background=None, offset=(0, 0)):
        """
        Animate the row-major sweep.

        background : full-grid binary ground-truth map (the same array as
                     binary_no_overlay.png / the stability diagram).  When
                     given, the sweep is drawn on the WHOLE stability diagram
                     instead of the little crop, so you can see where in the
                     honeycomb the scan is happening.
        offset     : (row_offset, col_offset) that maps a cropped index onto
                     that full grid — the sweep states are in crop coordinates.
        """
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        r_off, c_off = offset

        if background is not None:
            # Same house style as binary_no_overlay.png: white background,
            # black transition cells, visible cell grid.
            num_rows, num_cols = background.shape
            edges_x = np.arange(num_cols + 1) - 0.5
            edges_y = np.arange(num_rows + 1) - 0.5
            draw_ground_truth_map(ax, edges_x, edges_y, background)
            # Outline the crop the sweep is confined to.
            crop_rows, crop_cols = current_2d.shape
            # Magenta dashed: not one of the sweep colours, so the crop
            # outline is never confused with a sweep.
            ax.add_patch(plt.Rectangle(
                (c_off - 0.5, r_off - 0.5), crop_cols, crop_rows,
                fill=False, edgecolor="magenta", ls="--", lw=2, zorder=5,
            ))
        else:
            # No colorbar: the sweep animation is about WHERE the scan is, not
            # the absolute sensor value.
            num_rows, num_cols = current_2d.shape
            ax.imshow(current_2d, cmap="hot", origin="lower", aspect="auto")
        ax.set_title("Row-major Peak Sweep")

        def _ticks(n):
            """At most _GIF_MAX_TICKS evenly spaced cell indices (0-based)."""
            step = max(1, int(np.ceil(n / self._GIF_MAX_TICKS)))
            return np.arange(0, n, step)

        xt, yt = _ticks(num_cols), _ticks(num_rows)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels(xt + 1)          # cell indices stay 1-based
        ax.set_yticklabels(yt + 1)
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)
        ax.set_aspect("equal", adjustable="box")

        # Marker colours match every other figure in the set.
        row_line, = ax.plot([], [], "-", color="red", lw=1, zorder=6)
        scanned_sc, = ax.plot([], [], "o", color="blue", ms=3, alpha=0.6, zorder=7)
        peak_sc, = ax.plot([], [], "x", color="red", ms=9, mew=2, zorder=8)

        def init():
            row_line.set_data([], [])
            scanned_sc.set_data([], [])
            peak_sc.set_data([], [])
            return row_line, scanned_sc, peak_sc

        def update(frame):
            s = states[frame]
            r, cols, pcol = s["row"], s["scanned_cols"], s["peak_col"]
            gr = r + r_off                                   # onto the full grid
            gcols = [c + c_off for c in cols]
            gpcol = None if pcol is None else pcol + c_off
            row_line.set_data([0, num_cols - 1], [gr, gr])
            scanned_sc.set_data(gcols, [gr] * len(gcols))
            peak_sc.set_data([gpcol] if gpcol is not None else [],
                             [gr] if gpcol is not None else [])
            ax.set_title(f"Row={gr + 1}, peak col="
                         f"{'-' if gpcol is None else gpcol + 1}")
            return row_line, scanned_sc, peak_sc

        try:
            if len(states) == 0:
                raise ValueError("No states to animate.")
            ani = animation.FuncAnimation(
                fig, update, frames=len(states), init_func=init,
                interval=700, blit=False, repeat=False,
            )
            ani.save(out_path, writer=_gif_writer(), fps=1,
                     dpi=dpi or self.gif_dpi)
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
