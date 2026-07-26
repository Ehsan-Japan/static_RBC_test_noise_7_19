"""
DQDSimulator — runs a charge-sensing DQD simulation, optionally adds white
noise, and saves all outputs.  Absorbs the old simulation.py + preprocessing.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from qarray import ChargeSensedDotArray, DotArray, NoNoise

from ..config.axis_labels import x_label, y_label
from ..config.figure_style import (
    apply_voltage_axes,
    draw_ground_truth_map,
    figure_size,
    new_map_figure,
    save_figure,
)


# ---------------------------------------------------------------------------
# Helper: double-dot charge-state-change detection (from old preprocessing.py)
# ---------------------------------------------------------------------------

class _Tetrad(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        assert obj.ndim == 3, f"Array not of rank 3 -\n{obj}"
        return obj


def _charge_state_changes(n: np.ndarray) -> np.ndarray:
    if not isinstance(n, _Tetrad):
        n = _Tetrad(n)
    change_x = np.logical_not(np.isclose(n[1:, :-1, :], n[:-1, :-1, :], atol=1e-3)).any(axis=-1)
    change_y = np.logical_not(np.isclose(n[:-1, 1:, :], n[:-1, :-1, :], atol=1e-3)).any(axis=-1)
    return np.logical_or(change_x, change_y)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DQDSimulator:
    """
    Runs a full DQD charge-sensing simulation.

    Parameters
    ----------
    params : dict
        Must contain:
          save_dir, capacitance, model_params, xlabel, ylabel,
          voltage_sweep, plot_options, noise_params (optional)
    """

    def __init__(self, params: Dict):
        self.params = params
        self.save_dir: str = params["save_dir"]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full simulation: charge-sensing + double-dot."""
        self._run_charge_sensing()
        self._run_double_dot()

    # ------------------------------------------------------------------
    # Charge-sensing simulation (from old preprocessing.py)
    # ------------------------------------------------------------------

    def _run_charge_sensing(self) -> None:
        p = self.params
        cap = p["capacitance"]
        mp = p["model_params"]
        vs = p["voltage_sweep"]
        po = p["plot_options"]

        vx_min, vx_max = vs["vx_min"], vs["vx_max"]
        vy_min, vy_max = vs["vy_min"], vs["vy_max"]
        nx, ny = vs["n_points_x"], vs["n_points_y"]
        # Axis labels come from the shared axis-label config so every figure
        # in the pipeline agrees; params may still override them explicitly.
        xlabel = p.get("xlabel") or x_label()
        ylabel = p.get("ylabel") or y_label()
        dpi = po.get("dpi", 300)

        cs_save = po.get("charge_sensing_save_path", os.path.join(self.save_dir, "charge_sensing.jpg"))
        grad_save = po.get("charge_sensing_grad_save_path", os.path.join(self.save_dir, "charge_sensing2.jpg"))

        # Ensure directories exist
        for path in [cs_save, grad_save]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        image_dir = os.path.join(self.save_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        # Output numpy paths
        npy_z = os.path.join(os.path.dirname(cs_save), "charge_sensing_data.npy")
        npy_dz = os.path.join(os.path.dirname(cs_save), "charge_sensing_grad_data.npy")
        npy_noisy = os.path.join(os.path.dirname(cs_save), "charge_sensing_noisy_z_data.npy")

        # hyperparameters.json is written by DatasetPipeline (which knows the
        # analysis hyperparameters too), not here.

        # Build model and sweep
        model = ChargeSensedDotArray(
            Cdd=cap["Cdd"],
            Cgd=cap["Cgd"],
            Cds=cap["Cds"],
            Cgs=cap["Cgs"],
            coulomb_peak_width=mp["coulomb_peak_width"],
            T=mp["T"],
        )
        vg = model.gate_voltage_composer.do2d(
            "P1", vy_min, vx_max, nx,
            "P2", vy_min, vy_max, ny,
        )
        Vx, Vy = vg[:, :, 0], vg[:, :, 1]

        model.noise_model = NoNoise()
        z, _ = model.charge_sensor_open(vg)
        dz = np.gradient(z, axis=0) + np.gradient(z, axis=1)

        # Add optional white noise
        noise_p = p.get("noise_params", {})
        white_amp = noise_p.get("white_amplitude", 0.0)
        white_mean = noise_p.get("white_mean", 0.0)
        white_seed = noise_p.get("white_seed", None)

        if white_amp > 0:
            if white_seed is not None:
                np.random.seed(white_seed)
            noisy_z = z + np.random.normal(loc=white_mean, scale=white_amp, size=z.shape)
        else:
            noisy_z = z.copy()

        # Save noisy-z plot
        noisy_plot = os.path.join(self.save_dir, "images", "noisy_z_output.jpg")
        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(noisy_z, extent=[vx_min, vx_max, vy_min, vy_max],
                       origin="lower", aspect="auto", cmap="hot")
        apply_voltage_axes(ax, vx_min, vx_max, vy_min, vy_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Charge Sensor Output with Additional White Noise")
        fig.colorbar(im, cax=cax)
        save_figure(fig, noisy_plot, dpi=dpi)

        # Flatten and stack
        Vx_f, Vy_f = Vx.flatten(), Vy.flatten()
        np.save(npy_z, np.stack((Vx_f, Vy_f, z.flatten()), axis=-1))
        np.save(npy_dz, np.stack((Vx_f, Vy_f, dz.flatten()), axis=-1))
        np.save(npy_noisy, np.stack((Vx_f, Vy_f, noisy_z.flatten()), axis=-1))

        # Save side-by-side charge-sensing plot
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                                 figsize=figure_size(ncols=2),
                                 constrained_layout=True)
        im0 = axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max],
                             origin="lower", aspect="auto", cmap="hot")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title("$z$")
        axes[0].set_aspect("equal", adjustable="box")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)


        im1 = axes[1].imshow(dz, extent=[vx_min, vx_max, vy_min, vy_max],
                             origin="lower", aspect="auto", cmap="hot")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_title(r"$\frac{dz}{dVx} + \frac{dz}{dVy}$")
        axes[1].set_aspect("equal", adjustable="box")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        save_figure(fig, cs_save, dpi=dpi)

        print(f"Charge sensing data saved: {npy_z}")

    # ------------------------------------------------------------------
    # Double-dot stability diagram (from old preprocessing.py)
    # ------------------------------------------------------------------

    def _run_double_dot(self) -> None:
        p = self.params
        cap = p["capacitance"]
        vs = p["voltage_sweep"]
        po = p["plot_options"]
        dpi = po.get("dpi", 300)
        cs_save = po.get("charge_sensing_save_path", os.path.join(self.save_dir, "charge_sensing.jpg"))
        out_dir = os.path.dirname(cs_save)

        x_min, x_max = vs["vx_min"], vs["vx_max"]
        y_min, y_max = vs["vy_min"], vs["vy_max"]
        x_res, y_res = vs["n_points_x"], vs["n_points_y"]

        model = DotArray(
            Cdd=cap["Cdd"],
            Cgd=cap["Cgd"],
            charge_carrier="hole",
        )
        n_open = model.do2d_open(
            x_gate="P1", x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate="P2", y_min=y_min, y_max=y_max, y_res=y_res,
        )
        n_open = np.nan_to_num(n_open, nan=0)
        z_open = _charge_state_changes(n_open)

        # Build the full-size double-dot array first: it is the array everything
        # else calls "ground truth" (double_dot_data.npy -> ground_truth_labels
        # .npy -> summary.png, summary_total.png, the per-peak binary_*.png), so
        # the stability diagram must plot exactly it, not the unpadded z_open.
        x = np.linspace(x_min, x_max, x_res)
        y = np.linspace(y_min, y_max, y_res)
        xv, yv = np.meshgrid(x, y)
        z_full = np.zeros((y_res, x_res))
        z_full[:-1, :-1] = z_open

        # House style, identical to summary.png: white background, black
        # transition cells, visible cell grid.  No colorbar — the map is binary.
        x_edges = np.linspace(x_min, x_max, x_res + 1)
        y_edges = np.linspace(y_min, y_max, y_res + 1)
        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(ax, x_edges, y_edges, z_full)
        apply_voltage_axes(ax, x_min, x_max, y_min, y_max)
        ax.set_title("Double dot stability diagram")
        save_figure(fig, os.path.join(out_dir, "double_dot_stability_diagram.jpg"),
                    dpi=dpi)

        dd_array = np.stack((xv.flatten(), yv.flatten(), z_full.flatten().astype(int)), axis=-1)
        np.save(os.path.join(out_dir, "double_dot_data.npy"), dd_array)
        print(f"Double dot data saved: {os.path.join(out_dir, 'double_dot_data.npy')}")
