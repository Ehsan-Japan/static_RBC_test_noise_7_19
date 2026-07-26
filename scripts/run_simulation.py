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
        # the pipeline produces (stability diagram, overlays, binary images)
        # is labelled from these, so all images agree.
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
        n_samples=201,

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
        # slow and bulky for a full dataset.  Uses ImageMagick when installed,
        # otherwise Pillow, which ships with matplotlib.
        save_gifs=True,
        # GIF resolution — the single biggest cost of a run when save_gifs=True,
        # because every frame is re-rendered from scratch.  Measured on a
        # 25-row sweep: 150 dpi = 5.8 s and 2.4 MB per GIF, 100 dpi = 4.7 s and
        # 0.75 MB, 75 dpi = 3.9 s.  Four sweeps per peak, so ~23 s per peak at
        # 150 dpi.  Drop to 100 if the GIFs are only for eyeballing the sweep.
        gif_dpi=151,

        # ── Evaluation ────────────────────────────────────────────────
        peak_neighbor_cols=2,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
