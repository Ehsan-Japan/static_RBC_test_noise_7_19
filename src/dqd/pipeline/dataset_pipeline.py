"""
DatasetPipeline — end-to-end orchestrator for the DQD training-data pipeline.
Absorbs the old run_simulation.py, dqd_processor.py, and generate_training_dataset.py.
"""
import os
import json
import shutil
import numpy as np
from typing import Dict, Optional

from ..config.capacitance_config import CapacitanceConfig
from ..config.axis_labels import set_axis_labels, x_label, y_label
from ..config.figure_style import (
    NO_LEGEND_DIRNAME,
    set_figure_style,
    set_no_legend_dir,
)
from ..simulation.matrix_generator import CapacitanceMatrixGenerator
from ..simulation.dqd_simulator import DQDSimulator
from ..analysis.peak_detector import PeakDetector
from ..analysis.binarizer import ChargeSensorBinarizer
from ..coordinates.coordinate_mapper import CoordinateMapper
from ..data.sample import Sample
from ..data.cropped_region import CroppedRegion
from ..data.summary_writer import SummaryWriter
from ..visualization.plotter import Plotter
from ..visualization.overlay import OverlayRenderer


class DatasetPipeline:
    """
    Generates N simulation samples and processes each through the full
    DQD charge-sensing analysis pipeline.

    Parameters
    ----------
    base_save_dir        : root directory where per-sample folders are created
    n_samples            : number of samples to generate
    num_angles           : number of ray angles (linearly spaced inside (0°, 90°))
    ray_resolution       : number of points per ray
    x_resolution         : voltage-grid resolution in x
    y_resolution         : voltage-grid resolution in y
    crop_size            : half-width (in mV) of the cropping window around each peak
    col_buffer           : column offset from peak centre used as sweep start / stop
    vx_min / vx_max      : voltage range for the x (P1) axis
    vy_min / vy_max      : voltage range for the y (P2) axis
    coulomb_peak_width   : Lorentzian width of Coulomb peaks in the simulator
    temperature          : electron temperature used by the simulator
    noise_std            : white-noise standard deviation added to the sensor output
    plot_dpi             : DPI for all saved figures
    save_gifs            : whether to save sweep-animation GIFs per peak
    gif_dpi              : resolution of those GIFs; the dominant cost of a
                           run when save_gifs is on (see below)
    fixed_matrices       : if provided, every sample uses this capacitance dict
                           instead of random generation (keys: Cdd, Cgd, Cds, Cgs)
    config               : optional CapacitanceConfig; uses defaults if None
    x_axis_name          : name of the x (horizontal) gate axis on every plot
    y_axis_name          : name of the y (vertical) gate axis on every plot
    x_axis_unit          : unit shown after the x-axis name, e.g. "mV"
    y_axis_unit          : unit shown after the y-axis name, e.g. "mV"
    figure_width_in      : canvas width of every saved figure, in inches
    figure_height_in     : canvas height of every saved figure, in inches
    """

    def __init__(
        self,
        base_save_dir: str,
        n_samples: int = 100,
        num_angles: int = 4,
        ray_resolution: int = 50,
        x_resolution: int = 100,
        y_resolution: int = 100,
        crop_size: float = 2.0,
        col_buffer: int = 3,
        vx_min: float = -1.0,
        vx_max: float = 1.0,
        vy_min: float = -1.0,
        vy_max: float = 1.0,
        coulomb_peak_width: float = 0.01,
        temperature: float = 0.00001,
        noise_std: float = 0.01,
        plot_dpi: int = 300,
        save_gifs: bool = True,
        gif_dpi: int = 150,
        fixed_matrices: Optional[Dict] = None,
        config: Optional[CapacitanceConfig] = None,
        # ── Axis labels (shared by every figure) ──────────────────────
        x_axis_name: str = "P1",
        y_axis_name: str = "P2",
        x_axis_unit: str = "mV",
        y_axis_unit: str = "mV",
        # ── Figure geometry (shared by every saved figure) ────────────
        figure_width_in: float = 12.0,
        figure_height_in: float = 12.0,
        # ── Evaluation hyperparameters ────────────────────────────────
        peak_neighbor_cols: int = 0,
    ):
        self.base_save_dir = base_save_dir
        self.n_samples = n_samples
        self.num_angles = num_angles
        self.ray_resolution = ray_resolution
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.crop_size = crop_size
        self.col_buffer = col_buffer
        self.voltage_sweep = {
            "vx_min": vx_min,
            "vx_max": vx_max,
            "vy_min": vy_min,
            "vy_max": vy_max,
            "n_points_x": x_resolution,
            "n_points_y": y_resolution,
        }
        self.coulomb_peak_width = coulomb_peak_width
        self.temperature = temperature
        self.noise_std = noise_std
        self.plot_dpi = plot_dpi
        self.save_gifs = save_gifs
        self.gif_dpi = gif_dpi
        self.fixed_matrices = fixed_matrices
        self.config = config or CapacitanceConfig()
        # Publish the axis names / units so every figure in the pipeline —
        # simulator, overlays, binariser — labels its axes the same way.
        self.axis_labels = set_axis_labels(
            x_name=x_axis_name,
            y_name=y_axis_name,
            x_unit=x_axis_unit,
            y_unit=y_axis_unit,
        )
        # Same idea for the canvas: one size and one dpi for every figure,
        # so the saved images are directly comparable in a paper.
        self.figure_style = set_figure_style(
            width_in=figure_width_in,
            height_in=figure_height_in,
            dpi=plot_dpi,
        )
        self.angles = np.linspace(0, 90, num_angles + 2)[1:-1]
        self.peak_neighbor_cols = peak_neighbor_cols

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full pipeline for all N samples."""
        run_folder = (
            f"num_{self.n_samples}"
            f"_rays_{self.num_angles}"
            f"_res_{self.ray_resolution}"
            f"_image_res_{self.x_resolution}"
        )
        save_dir = os.path.join(self.base_save_dir, run_folder)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(1, self.n_samples + 1):
            print(f"\n{'='*60}")
            print(f"  Sample {i} / {self.n_samples}")
            print(f"{'='*60}")
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            try:
                self._run_sample(i, sample_dir)
            except Exception as exc:
                print(f"[ERROR] Sample {i} failed: {exc}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Per-sample orchestration
    # ------------------------------------------------------------------

    def _run_sample(self, sample_idx: int, sample_dir: str) -> None:
        vs = self.voltage_sweep

        # Every sample-level figure is also saved without its legend into
        # <sample>/figures_no_legend/, ready for legends drawn by hand for
        # the paper.  The normal figures keep theirs.
        set_no_legend_dir(os.path.join(sample_dir, NO_LEGEND_DIRNAME))

        # ---- 1. Generate (or reuse fixed) capacitance matrices ----
        if self.fixed_matrices is not None:
            matrices = self.fixed_matrices
        else:
            matrices = CapacitanceMatrixGenerator().generate_all(self.config)

        # ---- 2. Simulate charge-sensing data ----
        sim_params = {
            "save_dir": sample_dir,
            "capacitance": matrices,
            "model_params": {"coulomb_peak_width": self.coulomb_peak_width, "T": self.temperature},
            "xlabel": x_label(),
            "ylabel": y_label(),
            "voltage_sweep": vs,
            "optimal_Vg": [0.0, 0.0, 0.0],
            "noise_params": {"std": self.noise_std},
            "plot_options": {
                "charge_sensing_save_path": os.path.join(
                    sample_dir, "charge_sensing.jpg"
                ),
                "charge_sensing_grad_save_path": os.path.join(
                    sample_dir, "charge_sensing2.jpg"
                ),
                "dpi": self.plot_dpi,
            },
        }

        # ---- 2a. Record every hyperparameter used for this sample ----
        # Written before the simulator runs so the record survives a crash.
        self._write_hyperparameters(sample_dir, sim_params)

        DQDSimulator(sim_params).run()

        # npy files are now in sample_dir (before organize)
        numpy_file_path = os.path.join(sample_dir, "charge_sensing_data.npy")

        # ---- 3. Ray detection (process_dqd_data equivalent) ----
        ray_params = {
            "file_path_numpy": numpy_file_path,
            "angles_deg": self.angles,
            "resolution": self.ray_resolution,
            "common_params_rays": {
                "distance_intervals": 10,
                "marker_style": "o",
                "marker_size": 50,
                "marker_color": "white",
                "marker_edge_color": "black",
                "annotation_fontsize": 12,
            },
            "common_params_rays_and_peaks": {
                "distance_intervals": 10,
                "marker_style": "o",
                "marker_size": 50,
                "marker_color": "white",
                "marker_edge_color": "black",
                "annotation_fontsize": 12,
                "peak_marker_style": "x",
                "peak_marker_size": 100,
                "peak_marker_color": "black",
                "peak_label_fontsize": 8,
            },
            "target_directory": sample_dir,
        }
        results_dict = self._run_ray_processing(ray_params)
        peaks_dict = results_dict["peaks_dict"]

        # ---- 4. Organise sample directory (moves npy → numpy/simulation/) ----
        Sample(sample_dir).organize()
        numpy_sim_dir = os.path.join(sample_dir, "numpy", "simulation")
        charge_sensing_path = os.path.join(numpy_sim_dir, "charge_sensing_data.npy")
        double_dot_path = os.path.join(numpy_sim_dir, "double_dot_data.npy")
        ground_truth_path = os.path.join(numpy_sim_dir, "ground_truth_labels.npy")

        # ---- 5. Build paired_dict and save JSONs ----
        paired_dict = {
            str(angle): [[vx, vy] for vx, vy in zip(d["vx"], d["vy"])]
            for angle, d in peaks_dict.items()
        }
        with open(os.path.join(sample_dir, "peaks.json"), "w") as f:
            json.dump(peaks_dict, f, indent=4)
        with open(os.path.join(sample_dir, "peaks_paired.json"), "w") as f:
            json.dump(paired_dict, f, indent=4)

        # ---- 6. Per-peak processing ----
        for angle_str, peaks in paired_dict.items():
            ray_folder = os.path.join(
                sample_dir, "cropped_results", f"ray_{float(angle_str):.0f}"
            )
            os.makedirs(ray_folder, exist_ok=True)

            for peak_idx, (vx, vy) in enumerate(peaks, 1):
                peak_folder = os.path.join(ray_folder, f"peak_{peak_idx}")
                os.makedirs(peak_folder, exist_ok=True)
                try:
                    self._process_peak(
                        vx=vx,
                        vy=vy,
                        peak_idx=peak_idx,
                        peak_folder=peak_folder,
                        sample_dir=sample_dir,
                        charge_sensing_path=charge_sensing_path,
                        double_dot_path=double_dot_path,
                        ground_truth_path=ground_truth_path,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [WARN] Peak {peak_idx} (angle {angle_str}) failed: {exc}")
                    traceback.print_exc()

        # ---- 7. Sample-level overlays ----
        try:
            self._generate_sample_overlays(
                sample_dir=sample_dir,
                charge_sensing_npy=charge_sensing_path,
                ground_truth_path=ground_truth_path,
                paired_dict=paired_dict,
                rays_dict=results_dict.get("rays_dict"),
                peaks_dict=peaks_dict,
            )
        except Exception as exc:
            import traceback
            print(f"  [WARN] Sample overlays failed: {exc}")
            traceback.print_exc()

        # ---- 8. Evaluation (evaluation.txt + skipped_peaks.txt) ----
        try:
            from ..evaluation.evaluator import SampleEvaluator
            SampleEvaluator(
                sample_dir=sample_dir,
                num_angles=self.num_angles,
                ray_resolution=self.ray_resolution,
                x_resolution=self.x_resolution,
                y_resolution=self.y_resolution,
                vx_min=vs["vx_min"],
                vx_max=vs["vx_max"],
                vy_min=vs["vy_min"],
                vy_max=vs["vy_max"],
                peak_neighbor_cols=self.peak_neighbor_cols,
            ).run()
        except Exception as exc:
            import traceback
            print(f"  [WARN] Evaluation failed: {exc}")
            traceback.print_exc()


    # ------------------------------------------------------------------
    # GIF gathering
    # ------------------------------------------------------------------

    GIF_DIRNAME = "gifs"
    GIF_COMBINED_DIRNAME = "combined"      # the all-sweeps-together animations
    GIF_PER_SWEEP_DIRNAME = "per_sweep"    # one animation per individual sweep

    def _collect_gifs(self, sample_dir: str, peak_folder: str) -> None:
        """
        Copy every GIF in *peak_folder* into <sample_dir>/gifs/, split in two:

            gifs/combined/   peak_sweep_ALL.gif  — all four sweeps at once
            gifs/per_sweep/  one file per sweep

        Keeping them apart means the combined animations, which are the ones
        to look at first, are not buried among four times as many per-sweep
        files.  The flattened name keeps the ray and peak it came from, e.g.
        ``ray_26__peak_2__ALL.gif``.
        """
        try:
            rel = os.path.relpath(peak_folder, sample_dir)       # cropped_results/ray_x/peak_y
            prefix = "__".join(rel.split(os.sep)[1:])            # ray_x__peak_y
            gif_root = os.path.join(sample_dir, self.GIF_DIRNAME)
            combined_dir = os.path.join(gif_root, self.GIF_COMBINED_DIRNAME)
            per_sweep_dir = os.path.join(gif_root, self.GIF_PER_SWEEP_DIRNAME)

            for name in os.listdir(peak_folder):
                if not name.lower().endswith(".gif"):
                    continue
                short = name[len("peak_sweep_"):] if name.startswith("peak_sweep_") else name
                dest_dir = (combined_dir if short.upper().startswith("ALL")
                            else per_sweep_dir)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(os.path.join(peak_folder, name),
                            os.path.join(dest_dir, f"{prefix}__{short}"))
        except Exception as exc:
            print(f"  [WARN] Could not gather GIFs for {peak_folder}: {exc}")

    # ------------------------------------------------------------------
    # Hyperparameter record
    # ------------------------------------------------------------------

    def _write_hyperparameters(self, sample_dir: str, sim_params: Dict) -> None:
        """
        Write hyperparameters.json into the sample folder.

        This file spells the analysis knobs out under their full parameter
        names with a "meaning" string next to each, so a sample is
        self-describing even when it is copied out of its run folder.  JSON
        has no comment syntax, hence the {"value": …, "meaning": …} pairs.

        The simulator parameters (voltage_sweep, capacitance, …) are kept at
        the TOP level of the file exactly as DQDSimulator used to write them,
        because rebuild_publication_figures.py reads
        ``hyperparameters.json["voltage_sweep"]``.
        """
        hyperparams = {
            "crop_size": {
                "value": self.crop_size,
                "folder_tag": None,
                "meaning": (
                    "Half-width (in mV) of the cropping window taken around "
                    "each detected peak before the directional sweeps run."
                ),
            },
            "figure_width_in": {
                "value": self.figure_style.width_in,
                "folder_tag": None,
                "meaning": (
                    "Canvas width of every saved figure, in inches. Every "
                    "image is saved at exactly this size (no tight-bbox "
                    "cropping) so figures can be compared side by side."
                ),
            },
            "figure_height_in": {
                "value": self.figure_style.height_in,
                "folder_tag": None,
                "meaning": (
                    "Canvas height of every saved figure, in inches. Panel "
                    "figures widen by one canvas per panel."
                ),
            },
            "plot_dpi": {
                "value": self.plot_dpi,
                "folder_tag": None,
                "meaning": (
                    "Output resolution. Canvas size x dpi = pixel size of "
                    "every saved image."
                ),
            },
            "x_axis_name": {
                "value": self.axis_labels.x_name,
                "folder_tag": None,
                "meaning": (
                    "Name of the x (horizontal) gate axis. Every figure in the "
                    "pipeline is labelled with it, so all images agree."
                ),
            },
            "y_axis_name": {
                "value": self.axis_labels.y_name,
                "folder_tag": None,
                "meaning": "Name of the y (vertical) gate axis, on every figure.",
            },
            "x_axis_unit": {
                "value": self.axis_labels.x_unit,
                "folder_tag": None,
                "meaning": (
                    "Unit printed after the x-axis name, giving "
                    f"\"{self.axis_labels.x_label}\". Empty string = no unit."
                ),
            },
            "y_axis_unit": {
                "value": self.axis_labels.y_unit,
                "folder_tag": None,
                "meaning": (
                    "Unit printed after the y-axis name, giving "
                    f"\"{self.axis_labels.y_label}\". Empty string = no unit."
                ),
            },
        }
        payload = dict(sim_params)          # voltage_sweep, capacitance, … as before
        payload["_about"] = (
            "Hyperparameters used to generate this sample. The top-level keys "
            "are the simulator parameters; the 'hyperparameters' block "
            "documents the analysis knobs, where 'folder_tag' is the "
            "abbreviation the same value appears under in the run folder name "
            "(null = not part of the folder name)."
        )
        payload["hyperparameters"] = hyperparams
        out_path = os.path.join(sample_dir, "hyperparameters.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4, default=str)
        print(f"Hyperparameters saved: {out_path}")

    # ------------------------------------------------------------------
    # Per-peak processing
    # ------------------------------------------------------------------

    def _process_peak(
        self,
        vx: float,
        vy: float,
        peak_idx: int,
        peak_folder: str,
        sample_dir: str,
        charge_sensing_path: str,
        double_dot_path: str,
        ground_truth_path: str,
    ) -> None:
        vs = self.voltage_sweep

        # Pixel coordinate of this peak in the full grid
        global_pixel_x, global_pixel_y = CoordinateMapper.voltage_to_pixel(
            vx, vy,
            vs["vx_min"], vs["vx_max"],
            vs["vy_min"], vs["vy_max"],
            vs["n_points_x"], vs["n_points_y"],
        )
        print(
            f"  Peak {peak_idx}: voltage=({vx:.4f}, {vy:.4f}), "
            f"pixel=({global_pixel_x}, {global_pixel_y})"
        )

        # Crop data around this peak
        cropper = CroppedRegion(peak_folder)
        results = cropper.process_batch(
            base_dir=os.path.join(sample_dir, "numpy", "simulation"),
            x_center=vx,
            y_center=vy,
            crop_size=self.crop_size,
        )

        cropped_path = os.path.join(peak_folder, "charge_sensing_cropped.npy")
        center_info = results.get("charge_sensing_data.npy", {}).get("cropped") or {}
        center_i = center_info.get("row_index")
        center_j = center_info.get("column_index")
        # The sweeps run on charge_sensing_cropped.npy, so they must be given
        # the peak's position INSIDE the crop.  Using the full-grid index only
        # works when the crop starts at index 0 (peaks at the lower-left edge);
        # anywhere else it starts the sweep on the wrong cell, and once the
        # index exceeds the crop it raises IndexError and the whole peak is
        # lost — which is why smaller crop_size used to empty out peak folders.
        local_i = center_info.get("local_row_index")
        local_j = center_info.get("local_column_index")

        if center_i is None or center_j is None:
            print(f"  [WARN] No center indices for peak {peak_idx}; skipping sweep.")
            return
        if local_i is None or local_j is None:      # older runs without them
            local_i, local_j = center_i, center_j

        # Binariser and the full-grid overlays work on the UNCROPPED data,
        # so they keep the global index (1-based).
        center_row = center_i + 1
        center_col = center_j + 1

        # Generate ground-truth array from the double-dot stability diagram.
        # The DotArray output (double_dot_data.npy) is an exact binary map of
        # ALL charge-state-change boundaries, including interdot transition lines.
        # Using the charge sensing signal instead (old behaviour) misses interdot
        # lines because their charge-sensor response is negligible.
        OverlayRenderer.generate_ground_truth_array(
            data_path=double_dot_path,
            output_npy_path=ground_truth_path,
            use_double_dot=True,
        )

        # Full-grid ground-truth map used as the GIF background (skipped when
        # GIFs are off, since loading and thresholding it is pure waste then).
        gif_background = None
        if self.save_gifs and os.path.isfile(double_dot_path):
            dd = np.load(double_dot_path)
            dd_nx = len(np.unique(dd[:, 0]))
            dd_ny = len(np.unique(dd[:, 1]))
            gif_background = (dd[:, 2].reshape(dd_ny, dd_nx) > 0.5).astype(int)

        # Sweep analysis
        analysis_params = {
            "sweeps": [
                # Two sweeps, both starting AT the peak and walking in
                # opposite directions along the dot-to-lead line.
                {
                    "name": "bottom_up_right_left",
                    "start_row": local_i,
                    "row_step": +1,
                    "start_col": local_j + self.col_buffer,
                    "col_step": -1,
                    "col_buffer": self.col_buffer,
                },
                {
                    "name": "top_down_left_right",
                    "start_row": local_i,
                    "row_step": -1,
                    "start_col": local_j - self.col_buffer,
                    "col_step": +1,
                    "col_buffer": self.col_buffer,
                },
            ],
            "save_plots": True,
            "save_gifs": self.save_gifs,
            "gif_dpi": self.gif_dpi,
            # Draw the sweep on the WHOLE stability diagram (the same binary
            # ground-truth map as binary_no_overlay.png) instead of the little
            # crop, so the animation shows where in the honeycomb the scan is.
            # The sweep states are in crop coordinates, hence the offset.
            "gif_background": gif_background,
            "gif_offset": (center_i - local_i, center_j - local_j),
            "save_txt": True,
            "start_pixel_x": local_i,
            "start_pixel_y": local_j,
            "global_pixel_x": global_pixel_x,
            "global_pixel_y": global_pixel_y,
        }

        detector = PeakDetector(output_dir=peak_folder)
        detector.run(data_path=cropped_path, hyperparams=analysis_params)

        # Gather every GIF of this peak into <sample>/gifs/ as well, so all the
        # sweeps of the whole sample can be browsed in one place instead of
        # opening dozens of peak folders.  Copies, so the peak folders keep
        # theirs; names are flattened to stay unique.
        if self.save_gifs:
            self._collect_gifs(sample_dir, peak_folder)

        # Local summary → binarisation → voltage coordinates
        summary_writer = SummaryWriter()
        summary_local_path = os.path.join(peak_folder, "summary_local.txt")
        summary_writer.write(peak_folder, output_filename="summary_local.txt",
                             peak_neighbor_cols=self.peak_neighbor_cols)

        # Copy full data into peak folder first (needed for binarizer + overlay)
        for src in [charge_sensing_path, double_dot_path]:
            if os.path.exists(src):
                shutil.copy(src, peak_folder)

        double_dot_npy_in_peak = os.path.join(peak_folder, "double_dot_data.npy")
        charge_sensing_npy_in_peak = os.path.join(
            peak_folder, "charge_sensing_data.npy"
        )

        # Convert local row/col → voltage coordinates (needed by binarizer overlay)
        mapping_path = os.path.join(
            peak_folder, "charge_sensing_cropped_grid_mapping.txt"
        )
        voltage_coords_path = os.path.join(peak_folder, "voltage_coordinates.txt")
        CoordinateMapper.local_to_voltage(
            summary_file=summary_local_path,
            mapping_file=mapping_path,
            output_file=voltage_coords_path,
        )

        # Binary images — built from the DOUBLE-DOT data, i.e. the same exact
        # charge-state-change map as double_dot_stability_diagram.jpg and
        # ground_truth_labels.npy, on the full grid so the axes show [-1, 1].
        #
        # These used to be local maxima of the charge-sensing signal, which is a
        # measurement-derived approximation, not the ground truth: it misses
        # every interdot transition (they barely move the sensor amplitude), so
        # the binary images disagreed with the stability diagram.
        no_overlay_path = os.path.join(peak_folder, "binary_no_overlay.png")
        with_overlay_path = os.path.join(peak_folder, "binary_with_overlay.png")
        ChargeSensorBinarizer().convert(
            data_path=double_dot_npy_in_peak,
            data_is_ground_truth=True,
            output_no_overlay=no_overlay_path,
            output_with_overlay=with_overlay_path,
            center_row=center_row,
            center_col=center_col,
            voltage_coords_path=voltage_coords_path,
            axis_vxmin=vs["vx_min"],
            axis_vxmax=vs["vx_max"],
            axis_vymin=vs["vy_min"],
            axis_vymax=vs["vy_max"],
        )

        if os.path.exists(charge_sensing_npy_in_peak):
            OverlayRenderer.highlight_voltage_coordinates(
                npy_file=charge_sensing_npy_in_peak,
                voltage_coords_file=voltage_coords_path,
                output_file=os.path.join(
                    peak_folder, "charge_sensing_overlay_highlighted.png"
                ),
                is_double_dot=False,
            )


    # ------------------------------------------------------------------
    # Sample-level overlays
    # ------------------------------------------------------------------

    def _generate_sample_overlays(
        self,
        sample_dir: str,
        charge_sensing_npy: str,
        ground_truth_path: str,
        paired_dict: Dict,
        rays_dict: Optional[Dict] = None,
        peaks_dict: Optional[Dict] = None,
    ) -> None:
        vs = self.voltage_sweep

        OverlayRenderer.highlight_all_rays_in_sample(
            npy_file=charge_sensing_npy,
            paired_dict=paired_dict,
            angles=self.angles,
            output_file=os.path.join(sample_dir, "all_rays_peaks_overlay.png"),
            vxmin=vs["vx_min"],
            vxmax=vs["vx_max"],
            vymin=vs["vy_min"],
            vymax=vs["vy_max"],
            is_double_dot=False,
        )

        OverlayRenderer.highlight_all_scanned_and_peaks_in_sample_in_binary(
            sample_dir=sample_dir,
            output_file=os.path.join(
                sample_dir, "all_scanned_peaks_overlay_binary.png"
            ),
            is_double_dot=False,
        )

        if os.path.isfile(ground_truth_path):
            OverlayRenderer.visualize_grid_2d(
                file_path=ground_truth_path,
                vxmin=vs["vx_min"],
                vxmax=vs["vx_max"],
                vymin=vs["vy_min"],
                vymax=vs["vy_max"],
                ray_resolution=self.ray_resolution,
                output_file=os.path.join(sample_dir, "summary.png"),
            )
            if rays_dict is not None and peaks_dict is not None:
                _ray_kwargs = dict(
                    file_path=ground_truth_path,
                    vxmin=vs["vx_min"],
                    vxmax=vs["vx_max"],
                    vymin=vs["vy_min"],
                    vymax=vs["vy_max"],
                    ray_resolution=self.ray_resolution,
                    rays_dict=rays_dict,
                    peaks_dict=peaks_dict,
                )
                OverlayRenderer.visualize_grid_2d_with_rays(
                    output_file=os.path.join(sample_dir, "summary_total.png"),
                    **_ray_kwargs,
                )
                OverlayRenderer.visualize_grid_2d_with_rays(
                    output_file=os.path.join(sample_dir, "summary_peaks_only.png"),
                    show_measured_points=False,
                    **_ray_kwargs,
                )

                # Publication figure: the per-ray peak crosses all mean the same
                # thing (the start point of a directional sweep), so draw them
                # large, in ONE colour, with a single legend entry.
                _cross_kwargs = dict(
                    ray_peak_color="magenta",
                    ray_peak_marker="X",
                    ray_peak_size=420,
                    ray_peak_linewidths=1.6,
                    ray_peak_edgecolor="black",
                    ray_peak_label="Directional Sweep Start Points",
                    legend_fontsize=14,
                )
                OverlayRenderer.visualize_grid_2d_with_rays(
                    output_file=os.path.join(
                        sample_dir, "summary_total_all_crosses.png"
                    ),
                    **_cross_kwargs,
                    **_ray_kwargs,
                )

    # ------------------------------------------------------------------
    # Ray processing helper  (replaces process_and_plot + dqd_processor)
    # ------------------------------------------------------------------

    def _run_ray_processing(self, params: Dict) -> Dict:
        """
        Run pre_process (data loading + ray extraction) and the ray plots,
        then organise the sample directory.

        This replaces the old process_dqd_data / process_and_plot chain.
        Import from data_processing kept local to avoid circular deps.
        """
        from ..analysis.ray_processor import RayProcessor  # lazy import

        file_path_numpy = params["file_path_numpy"]
        save_dir = os.path.dirname(file_path_numpy)

        processor = RayProcessor(
            file_path_numpy=file_path_numpy,
            angles_deg=params["angles_deg"],
            resolution=params["resolution"],
            save_dir=save_dir,
        )
        results_dict = processor.run()

        # Organise the sample directory
        sample = Sample(params["target_directory"])
        sample.manage_images()

        return results_dict
