"""
rebuild_publication_figures.py — regenerate the publication figures for samples
that have ALREADY been simulated, without re-running the sweeps.

Produces, per sample:
    summary_total_all_crosses.png                  (all sweep-start crosses,
                                                    one colour, large, legended)

The existing summary_total.png / summary_peaks_only.png are left untouched.

Usage
-----
    python scripts/rebuild_publication_figures.py <path>

<path> may be a single sample_* folder, a dataset folder containing sample_*
folders, or training_data/ itself.  With no argument every sample under
training_data/ is rebuilt.
"""
import os
import re
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from dqd.visualization.overlay import OverlayRenderer          # noqa: E402

# Same styling the pipeline uses for the publication figures.
CROSS_KWARGS = dict(
    ray_peak_color="magenta",
    ray_peak_marker="X",
    ray_peak_size=420,
    ray_peak_linewidths=1.6,
    ray_peak_edgecolor="black",
    ray_peak_label="Directional Sweep Start Points",
    legend_fontsize=14,
)


def _load_rays(sample_dir: str, angles) -> dict:
    """rays_dict {angle: {"X":…, "Y":…, "Current":…}} from original_numpy/."""
    rays = {}
    for angle in angles:
        tag = f"{angle:.2f}".replace(".", "_")
        path = os.path.join(sample_dir, "original_numpy",
                            f"original_ray_{tag}_voltage_current.npy")
        if not os.path.isfile(path):
            print(f"  [skip ray] missing {os.path.basename(path)}")
            continue
        arr = np.load(path)
        rays[angle] = {"X": arr[0], "Y": arr[1], "Current": arr[2]}
    return rays


def rebuild_sample(sample_dir: str) -> bool:
    gt = os.path.join(sample_dir, "numpy", "simulation", "ground_truth_labels.npy")
    peaks_json = os.path.join(sample_dir, "peaks.json")
    hp_json = os.path.join(sample_dir, "hyperparameters.json")
    for p in (gt, peaks_json, hp_json):
        if not os.path.isfile(p):
            print(f"[skip] {sample_dir}: missing {os.path.basename(p)}")
            return False

    vs = json.load(open(hp_json))["voltage_sweep"]
    peaks_dict = {float(k): v for k, v in json.load(open(peaks_json)).items()}
    rays_dict = _load_rays(sample_dir, sorted(peaks_dict))
    if not rays_dict:
        print(f"[skip] {sample_dir}: no ray arrays found")
        return False

    ray_kwargs = dict(
        file_path=gt,
        vxmin=vs["vx_min"], vxmax=vs["vx_max"],
        vymin=vs["vy_min"], vymax=vs["vy_max"],
        ray_resolution=len(rays_dict[next(iter(rays_dict))]["X"]),
        rays_dict=rays_dict,
        peaks_dict=peaks_dict,
    )

    OverlayRenderer.visualize_grid_2d_with_rays(
        output_file=os.path.join(sample_dir, "summary_total_all_crosses.png"),
        **CROSS_KWARGS, **ray_kwargs,
    )
    return True


def _collect_samples(path: str):
    if os.path.basename(path.rstrip("/\\")).startswith("sample_"):
        return [path]
    found = sorted(glob.glob(os.path.join(path, "sample_*")))
    if found:
        return found
    return sorted(glob.glob(os.path.join(path, "*", "sample_*")))


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "training_data"
    samples = _collect_samples(target)
    if not samples:
        print(f"No sample_* folders found under {target}")
        return
    ok = sum(rebuild_sample(s) for s in samples)
    print(f"\nRebuilt {ok}/{len(samples)} samples.")


if __name__ == "__main__":
    main()
