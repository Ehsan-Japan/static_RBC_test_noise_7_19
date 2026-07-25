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
        n_samples=30,

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
    )
    pipeline.run()


if __name__ == "__main__":
    main()
