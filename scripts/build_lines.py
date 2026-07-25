"""
build_lines.py — fit straight lines to already-measured transition cells and
read the break points off them.

Runs on FINISHED sample folders, so you can retune the line/break-point
hyperparameters in seconds without re-running the simulation.

    python scripts/build_lines.py <run_dir_or_sample_dir> [sample numbers...]

Examples
--------
    # every sample in a run
    python scripts/build_lines.py training_data/num_40_rays_6_res_100_image_res_100

    # only samples 12, 13 and 18
    python scripts/build_lines.py training_data/num_40_rays_6_res_100_image_res_100 12 13 18

    # a single sample folder
    python scripts/build_lines.py training_data/.../sample_13

Writes into each sample folder:
    lines_fitted.png       cells + every fitted straight line  (the clean view)
    lines_breakpoints.png  the same lines + break points + interdot peaks
    lines_report.json      all of it as numbers
"""
import os
import re
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dqd.analysis.line_builder import build_lines


# ── Hyperparameters — tweak these ─────────────────────────────────────
PARAMS = dict(
    # --- cells -> chains ---
    # Cells within this grid distance join the same chain.  Raise it if a
    # honeycomb boundary is broken into pieces; lower it if separate
    # boundaries get glued together.
    link_px=2,
    # Chains smaller than this are discarded as noise.
    min_chain_cells=12,

    # --- chains -> straight lines ---
    # Perpendicular distance for a cell to count as "on" a line.  Set it near
    # the half-thickness of the measured band (bands look ~4-6 cells wide).
    inlier_px=1.5,
    # A fitted line needs at least this many supporting cells...
    min_segment_cells=12,
    # ...and must span at least this many grid pixels.  Raise both to get
    # fewer, bigger lines; lower them to catch short segments near vertices.
    min_segment_length=6.0,
    # Safety cap on how many lines are peeled off one chain.
    max_segments_per_chain=12,
    # Inliers separated by a bigger gap along the line are treated as
    # different segments, so a line cannot bridge an empty stretch.
    seg_gap_px=3.0,
    # Random 2-point hypotheses per line.  More = steadier fits, slower.
    ransac_trials=400,
    # RNG seed, so a given setting always reproduces the same figure.
    seed=0,

    # --- lines -> break points ---
    # Two lines form a break point only if their directions differ by more
    # than this.  THE sensitivity knob now — and it is an honest angle in
    # degrees, not a finite-difference slope threshold.
    min_angle_deg=20.0,
    # ...and only if their nearest endpoints are this close.
    junction_px=8.0,

    # --- break points -> interdot ---
    # Longest allowed connector between two break points.
    connect_px=50.0,
    # Allow connecting break points that share a chain (normally off: two
    # vertices of the SAME boundary are not an interdot transition).
    same_chain_connect=False,

    plot_dpi=200,
)

# Voltage extent; overridden per sample by hyperparameters.json when present.
EXTENT = dict(vx_min=-1.0, vx_max=1.0, vy_min=-1.0, vy_max=1.0)


def _extent_for(sample_dir):
    path = os.path.join(sample_dir, "hyperparameters.json")
    if os.path.isfile(path):
        try:
            vs = json.load(open(path))["voltage_sweep"]
            return {k: float(vs[k]) for k in
                    ("vx_min", "vx_max", "vy_min", "vy_max")}
        except Exception:
            pass
    return dict(EXTENT)


def _sample_dirs(target, wanted):
    if os.path.isdir(os.path.join(target, "cropped_results")):
        return [target]                       # a single sample folder
    dirs = []
    for name in sorted(os.listdir(target)):
        m = re.fullmatch(r"sample_(\d+)", name)
        if not m:
            continue
        if wanted and int(m.group(1)) not in wanted:
            continue
        dirs.append(os.path.join(target, name))
    return dirs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    wanted = {int(a) for a in sys.argv[2:]}
    dirs = _sample_dirs(target, wanted)
    if not dirs:
        print(f"No sample folders found under {target}")
        sys.exit(1)

    for sample_dir in dirs:
        print(f"\n=== {os.path.basename(sample_dir)} ===")
        try:
            build_lines(sample_dir, **_extent_for(sample_dir), **PARAMS)
        except Exception as exc:
            import traceback
            print(f"  [WARN] failed: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
