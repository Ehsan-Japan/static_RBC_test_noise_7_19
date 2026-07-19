"""
Sample — represents one simulation sample on disk and handles file organisation.
Absorbs the old data_processing.py (move_and_organize_files, save_all_data,
manage_images).
"""
import os
import shutil
from pathlib import Path
from typing import Optional


class Sample:
    """
    Wraps the directory of a single simulation sample and provides helpers
    for saving, loading, and organising its files.

    Parameters
    ----------
    sample_dir : str  path to the sample_N folder
    """

    # Files that must be moved into numpy/simulation/
    _NUMPY_FILES = [
        "double_dot_data.npy",
        "charge_sensing_data.npy",
        "charge_sensing_grad_data.npy",
        "charge_sensing_noisy_z_data.npy",
    ]

    # Image files that must be moved into images/
    _IMAGE_FILES = [
        "rays_current_vs_distance.png",
        "rays_only.png",
        "rays_with_distance_and_peaks_original.png",
        "rays_with_distance_and_peaks_original_annotation.png",
        "rays_with_distance_original.png",
        "combined_plot.png",
        "original.png",
    ]

    def __init__(self, sample_dir: str):
        self.sample_dir = sample_dir

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    @property
    def numpy_simulation_dir(self) -> str:
        return os.path.join(self.sample_dir, "numpy", "simulation")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.sample_dir, "images")

    # ------------------------------------------------------------------
    # File organisation (from old data_processing.move_and_organize_files)
    # ------------------------------------------------------------------

    def organize(self) -> None:
        """
        Move raw output files into the canonical sub-directory layout:
          numpy/simulation/  ← .npy data files
          images/            ← generated plot images
        """
        directory = self.sample_dir
        numpy_path = os.path.join(directory, "numpy")
        sim_tmp = os.path.join(numpy_path, "simulation_numpy")
        sim_final = os.path.join(numpy_path, "simulation")
        images_path = os.path.join(directory, "images")

        for d in [numpy_path, sim_tmp, images_path]:
            os.makedirs(d, exist_ok=True)

        # Move image files
        for filename in os.listdir(directory):
            full = os.path.join(directory, filename)
            if ("charge_sensing" in filename and os.path.isfile(full)
                    and not filename.endswith(".npy")):
                try:
                    shutil.move(full, images_path)
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

        # Move .npy files into simulation_numpy
        for fname in self._NUMPY_FILES:
            src = os.path.join(directory, fname)
            if os.path.exists(src):
                try:
                    shutil.move(src, sim_tmp)
                except Exception as e:
                    print(f"Error moving {fname}: {e}")

        # Rename simulation_numpy → simulation
        if os.path.exists(sim_tmp) and not os.path.exists(sim_final):
            try:
                os.rename(sim_tmp, sim_final)
            except Exception as e:
                print(f"Error renaming simulation folder: {e}")
        elif os.path.exists(sim_final):
            print("simulation/ folder already exists — skipping rename.")

        # Clean up stray images sub-folder
        stale = os.path.join(images_path, "charge_sensing")
        if os.path.isdir(stale):
            shutil.rmtree(stale)

        print("Sample organisation complete.")

    # ------------------------------------------------------------------
    # Image management (from old data_processing.manage_images)
    # ------------------------------------------------------------------

    def manage_images(self) -> None:
        """
        Delete temporary gradient images and move ray plots into images/.
        """
        base = Path(self.sample_dir)
        if not base.is_dir():
            print(f"Error: {self.sample_dir} does not exist.")
            return

        for fname in ["gradient.png", "sobel_gradient.png"]:
            p = base / fname
            if p.exists():
                p.unlink()

        images_folder = base / "images"
        images_folder.mkdir(exist_ok=True)

        to_move = list(self._IMAGE_FILES)
        for item in base.iterdir():
            if item.is_file() and item.name.startswith("original_ray_") and item.name.endswith(".png"):
                to_move.append(item.name)

        for fname in to_move:
            src = base / fname
            dst = images_folder / fname
            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    print(f"Error moving {fname}: {e}")
