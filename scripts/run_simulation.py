import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.pipeline.dataset_pipeline import DatasetPipeline

def main():
    pipeline = DatasetPipeline(
        # ── Output ────────────────────────────────────────────────────
        base_save_dir="training_data",

        # ── Dataset size ──────────────────────────────────────────────
        n_samples=45,

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
        crop_size=2.0,
        col_buffer=2,

        # ── Simulator physics ─────────────────────────────────────────
        coulomb_peak_width=0.01,
        temperature=0.00001,
        noise_std=0.01,

        # ── Output options ────────────────────────────────────────────
        plot_dpi=300,
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
