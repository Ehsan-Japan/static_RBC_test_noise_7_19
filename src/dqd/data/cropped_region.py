"""
CroppedRegion — crops a rectangular region around a voltage peak, saves the
cropped .npy file, and builds the grid-mapping file.
Absorbs the old crop_image.py.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

from ..coordinates.coordinate_mapper import CoordinateMapper
from ..config.figure_style import (
    apply_voltage_axes,
    draw_cell_grid,
    new_map_figure,
    save_figure,
)


class CroppedRegion:
    """
    Manages the cropping of a charge-sensor (or double-dot) dataset around
    a voltage-space centre point and saves all artefacts.

    Parameters
    ----------
    output_dir : directory where cropped .npy and mapping files are saved
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main crop + save  (from old crop_image.plot_from_numpy_array_modified)
    # ------------------------------------------------------------------

    def crop_and_save(
        self,
        file_path: str,
        x_center: float,
        y_center: float,
        crop_size: float,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Load `file_path`, crop a square of side `crop_size` around
        (x_center, y_center) in voltage space, and save:
          - cropped .npy file
          - grid mapping .txt
          - PNG image

        Returns
        -------
        (center_i, center_j, local_i, local_j)
            center_i / center_j : 0-based row/col of the peak in the
                *original* (uncropped) grid — used for full-grid overlays.
            local_i / local_j   : the same peak's 0-based row/col INSIDE the
                cropped array — used by the sweeps, which run on the cropped
                data.  The two only coincide when the crop happens to start
                at index 0; mixing them up puts the sweep on the wrong cell,
                or out of bounds entirely.
            All four are None if the crop window is empty.
        """
        array = np.load(file_path)
        Vx_flat, Vy_flat, z_flat = array[:, 0], array[:, 1], array[:, 2]

        is_double_dot = "double_dot" in file_path
        base_name = "double_dot" if is_double_dot else "charge_sensing"
        cmap = "binary" if is_double_dot else "hot"

        if is_double_dot:
            z_flat = np.where(z_flat > 0.5, 1.0, 0.0).astype(np.float32)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = None, None

        Vx_values = np.unique(Vx_flat)
        Vy_values = np.unique(Vy_flat)
        num_x = len(Vx_values)
        num_y = len(Vy_values)
        original_extent = [Vx_values[0], Vx_values[-1], Vy_values[0], Vy_values[-1]]

        center_i, center_j = None, None
        local_i, local_j = None, None
        cropped_array = None

        if crop_size > 0 and x_center is not None and y_center is not None:
            x_min = max(x_center - crop_size / 2, Vx_values[0])
            x_max = min(x_center + crop_size / 2, Vx_values[-1])
            y_min = max(y_center - crop_size / 2, Vy_values[0])
            y_max = min(y_center + crop_size / 2, Vy_values[-1])

            x_mask = (Vx_values >= x_min) & (Vx_values <= x_max)
            y_mask = (Vy_values >= y_min) & (Vy_values <= y_max)

            if np.any(x_mask) and np.any(y_mask):
                z_2d = z_flat.reshape((num_y, num_x))[y_mask, :][:, x_mask]
                cropped_Vx = Vx_values[x_mask]
                cropped_Vy = Vy_values[y_mask]

                VX_g, VY_g = np.meshgrid(cropped_Vx, cropped_Vy)
                cropped_array = np.column_stack([VX_g.ravel(), VY_g.ravel(), z_2d.ravel()])

                # Save .npy
                npy_path = os.path.join(self.output_dir, f"{base_name}_cropped.npy")
                np.save(npy_path, cropped_array)

                # Save grid mapping
                CoordinateMapper.write_cropped_grid_mapping(
                    cropped_Vx, cropped_Vy, self.output_dir, base_name
                )

                # Record center in *original* grid …
                center_j = int(np.argmin(np.abs(Vx_values - x_center)))
                center_i = int(np.argmin(np.abs(Vy_values - y_center)))
                # … and in the CROPPED grid, which is what the sweeps index.
                local_j = int(np.argmin(np.abs(cropped_Vx - x_center)))
                local_i = int(np.argmin(np.abs(cropped_Vy - y_center)))

        # ------ Plot ------
        fig, ax, cax = new_map_figure(with_colorbar=True)

        if cropped_array is not None:
            z_plot = cropped_array[:, 2].reshape(
                len(np.unique(cropped_array[:, 1])),
                len(np.unique(cropped_array[:, 0])),
            )
            extent_plot = [
                cropped_array[:, 0].min(), cropped_array[:, 0].max(),
                cropped_array[:, 1].min(), cropped_array[:, 1].max(),
            ]
        else:
            z_plot = z_flat.reshape((num_y, num_x))
            extent_plot = original_extent

        im = ax.imshow(z_plot, extent=extent_plot, origin="lower", aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        # Same cell grid as the ground-truth figures, so the cropped view
        # shows the voltage grid the sweeps actually step over.
        draw_cell_grid(
            ax,
            np.linspace(extent_plot[0], extent_plot[1], z_plot.shape[1] + 1),
            np.linspace(extent_plot[2], extent_plot[3], z_plot.shape[0] + 1),
        )

        if x_center is not None and y_center is not None and crop_size > 0:
            rect = plt.Rectangle(
                (x_center - crop_size / 2, y_center - crop_size / 2),
                crop_size, crop_size, linewidth=2, edgecolor="r", facecolor="none",
            )
            ax.scatter(x_center, y_center, color="red", s=50, zorder=10)
            ax.add_patch(rect)

        if not is_double_dot:
            apply_voltage_axes(ax, extent_plot[0], extent_plot[1],
                               extent_plot[2], extent_plot[3])
            fig.colorbar(im, cax=cax, label="Sensor Signal")
            out_png = os.path.join(self.output_dir, f"{base_name}_cropped.png")
            save_figure(fig, out_png)
        else:
            plt.close(fig)

        print(f"Center indices: global (i={center_i}, j={center_j}), "
              f"local (i={local_i}, j={local_j})")
        return center_i, center_j, local_i, local_j

    # ------------------------------------------------------------------
    # Batch processing  (from old crop_image.process_data_with_hyperparameters)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        base_dir: str,
        x_center: float,
        y_center: float,
        crop_size: float,
    ) -> Dict:
        """
        Crop both charge_sensing_data.npy and double_dot_data.npy from
        `base_dir` and save results in self.output_dir.

        Returns a dict mapping dataset filename → centre indices.
        """
        results: Dict = {}
        for dataset in ["double_dot_data.npy", "charge_sensing_data.npy"]:
            fp = os.path.join(base_dir, dataset)
            if not os.path.isfile(fp):
                results[dataset] = {"cropped": None}
                continue
            ci, cj, li, lj = self.crop_and_save(fp, x_center, y_center, crop_size)
            results[dataset] = {
                "cropped": (
                    {
                        # indices in the full voltage grid
                        "row_index": ci,
                        "column_index": cj,
                        # the same point inside the cropped array
                        "local_row_index": li,
                        "local_column_index": lj,
                    }
                    if ci is not None else None
                )
            }

        # Save centre indices to JSON
        json_path = os.path.join(self.output_dir, "center_indices.json")
        try:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
        except TypeError as e:
            print(f"Error saving centre indices: {e}")
        return results
