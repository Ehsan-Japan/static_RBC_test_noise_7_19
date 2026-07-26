import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.pipeline.dataset_pipeline import DatasetPipeline

def main():
    pipeline = DatasetPipeline(
        # ── Output ────────────────────────────────────────────────────
        base_save_dir="training_data",

        # ── Axis labels ───────────────────────────────────────────────
        # Names and units of the two gate axes.  Set here ONCE: every figure
        # the pipeline produces (stability diagram, overlays, binary images,
        # interdot plots) is labelled from these, so all images agree.
        # Rendered as "<name> (<unit>)", e.g. "P1 (mV)"; set the unit to ""
        # to print the bare name.
        x_axis_name="P1",
        y_axis_name="P2",
        x_axis_unit="mV",
        y_axis_unit="mV",

        # ── Figure geometry ───────────────────────────────────────────
        # Canvas size of EVERY saved figure, in inches.  Images are saved at
        # exactly this size (no tight-bbox cropping) and the plotting box sits
        # at the same place inside it, so all figures have the same physical
        # size and the same data scale — they can be compared or placed side
        # by side in the paper without rescaling.  Pixel size = size x dpi
        # (12 in x 300 dpi = 3600 px).  Panel figures widen per panel.
        #
        # 12 x 12 is what summary_total.png / summary_peaks_only.png always
        # used.  SHRINKING THIS DOES NOT SHRINK THE CONTENTS: legend text,
        # tick labels and markers are sized in points, so a smaller canvas
        # only makes them look bigger relative to the frame.
        figure_width_in=12.0,
        figure_height_in=12.0,

        # ── Dataset size ──────────────────────────────────────────────
        n_samples=70,

        # ── Ray parameters ────────────────────────────────────────────
        num_angles=6,
        ray_resolution=100,
        # ── Voltage grid ──────────────────────────────────────────────
        x_resolution=100,
        y_resolution=100,
        vx_min=-1.0,
        vx_max=1.0,
        vy_min=-1.0,
        vy_max=1.0,

        # ── Peak processing ───────────────────────────────────────────
        crop_size=1,
        col_buffer=2,

        # ── Simulator physics ─────────────────────────────────────────
        coulomb_peak_width=0.01,
        temperature=0.00001,
        noise_std=0.01,

        # ── Output options ────────────────────────────────────────────
        plot_dpi=300,
        # GIF ON/OFF.  True writes one animated sweep GIF per sweep per peak
        # (peak_sweep_<name>.gif inside every cropped_results/ray_*/peak_*
        # folder) showing the row-by-row scan.  Useful for debugging a sweep,
        # slow and bulky for a full dataset — needs the ImageMagick writer, and
        # just warns and skips if it is missing.
        save_gifs=False,

        # ── Evaluation ────────────────────────────────────────────────
        peak_neighbor_cols=2,

        # ── Interdot break-point detection ────────────────────────────
        # Grouping: peaks within this many grid pixels count as ONE line.
        # Lower (2-3) keeps nearly-touching lines separate, so more
        # break-point pairs survive the "must be different lines" rule.
        interdot_neighbor_px=4,
        # Max gap (grid pixels) allowed between the break points that get
        # connected.  Raise it if valid vertex pairs sit far apart.
        interdot_connect_px=50,
        # MAIN SENSITIVITY KNOB.  A line only produces a break point if its
        # local slope changes by more than this (grid-pixel slope units).
        # At 0.8 shallow honeycomb vertices are silently rejected — this is
        # the usual reason break points go undetected.  Try 0.25-0.4.
        interdot_min_slope_change=0.8,
        # Slope-comparison half-window, as a fraction of the line length.
        # Larger = more smoothing = kinks get straightened away.
        interdot_kink_window_frac=1.0 / 6.0,
        # Hard cap on that half-window in pixels (0 = uncapped).  Capping at
        # ~4-6 keeps long lines as kink-sensitive as short ones.
        interdot_kink_window_max=0,
        # Minimum points / distinct steps for a group to count as a line.
        interdot_min_line_pts=5,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
