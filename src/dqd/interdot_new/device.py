"""
device.py — the measurement oracle.

The single most important design rule in this package:

    **No algorithm ever touches the sensor array directly.**

Every read goes through ``Device.measure(i, j)``, which records the cell that
was visited.  Measurement cost is therefore *instrumented*, not estimated, and
it is structurally impossible to forget to count a family of measurements
(e.g. the ray samples) when reporting grid coverage.

Two cost figures are tracked, because they answer different questions:

    n_unique : number of distinct grid cells ever visited.
               The right number if the experiment caches results.
    n_calls  : total number of measurement operations.
               The right number if every access costs wall-clock time,
               which is the case on real hardware.

Both are reported.  Coverage is quoted as ``n_unique / (Nx*Ny)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoltageGrid:
    """Rectangular gate-voltage grid.

    Convention used everywhere in this package:
        row index i  <->  V_P2  (y axis)
        col index j  <->  V_P1  (x axis)
    This matches ``qarray.do2d('P1', ..., 'P2', ...)``, whose output is
    indexed ``[P2, P1]``.
    """

    vx_min: float = -1.0
    vx_max: float = 1.0
    vy_min: float = -1.0
    vy_max: float = 1.0
    nx: int = 100
    ny: int = 100

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def vx(self) -> np.ndarray:
        return np.linspace(self.vx_min, self.vx_max, self.nx)

    @property
    def vy(self) -> np.ndarray:
        return np.linspace(self.vy_min, self.vy_max, self.ny)

    @property
    def dvx(self) -> float:
        return (self.vx_max - self.vx_min) / (self.nx - 1)

    @property
    def dvy(self) -> float:
        return (self.vy_max - self.vy_min) / (self.ny - 1)

    def to_pixel(self, vx: float, vy: float) -> Tuple[int, int]:
        """Voltage -> (row, col), clipped to the grid."""
        j = int(round((vx - self.vx_min) / (self.vx_max - self.vx_min) * (self.nx - 1)))
        i = int(round((vy - self.vy_min) / (self.vy_max - self.vy_min) * (self.ny - 1)))
        return (max(0, min(self.ny - 1, i)), max(0, min(self.nx - 1, j)))

    def to_voltage(self, i: int, j: int) -> Tuple[float, float]:
        """(row, col) -> (V_P1, V_P2)."""
        return (self.vx_min + j * self.dvx, self.vy_min + i * self.dvy)

    def pixels_to_voltages(self, pixels) -> np.ndarray:
        """(K,2) array of (row,col) -> (K,2) array of (vx,vy)."""
        p = np.asarray(list(pixels), dtype=float).reshape(-1, 2)
        vx = self.vx_min + p[:, 1] * self.dvx
        vy = self.vy_min + p[:, 0] * self.dvy
        return np.column_stack([vx, vy])


# ---------------------------------------------------------------------------
# Measurement budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Records which cells were measured and how many times."""

    grid: VoltageGrid
    counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _by_stage: Dict[str, set] = field(default_factory=dict)
    stage: str = "unassigned"

    def record(self, i: int, j: int) -> None:
        key = (i, j)
        self.counts[key] = self.counts.get(key, 0) + 1
        self._by_stage.setdefault(self.stage, set()).add(key)

    @property
    def n_unique(self) -> int:
        return len(self.counts)

    @property
    def n_calls(self) -> int:
        return sum(self.counts.values())

    @property
    def coverage(self) -> float:
        """Fraction of the grid measured at least once."""
        return self.n_unique / self.grid.size

    @property
    def cells(self) -> np.ndarray:
        """(K,2) array of measured (row, col)."""
        if not self.counts:
            return np.empty((0, 2), dtype=int)
        return np.array(sorted(self.counts.keys()), dtype=int)

    def breakdown(self) -> Dict[str, int]:
        """Unique cells attributable to each pipeline stage (stages overlap)."""
        return {k: len(v) for k, v in self._by_stage.items()}

    def summary(self) -> Dict[str, float]:
        d = {
            "unique_cells": self.n_unique,
            "total_calls": self.n_calls,
            "coverage": self.coverage,
            "redundancy": self.n_calls / max(self.n_unique, 1),
        }
        for k, v in self.breakdown().items():
            d[f"unique_cells[{k}]"] = v
        return d


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


class Device:
    """A simulated DQD charge sensor that can only be queried point-by-point.

    Parameters
    ----------
    sensor : (ny, nx) array
        Noise-free charge-sensor response.
    ground_truth : (ny, nx) bool array
        True charge-transition pixels (from the constant-capacitance
        charge-state map, so interdot lines are included).
    grid : VoltageGrid
    noise_std : float
        Standard deviation of additive Gaussian noise, in units of the
        *peak-to-peak span* of the noise-free sensor signal.  0.01 means 1 %
        of full scale.  This is a relative unit so that results transfer
        across samples whose absolute currents differ.
    noise_mode : {"static", "shot"}
        "static": one frozen noise field, drawn at construction.  Re-measuring
                  a cell returns the same value.  Models slow charge noise /
                  offset drift and is the conservative default because the
                  detector cannot average it away.
        "shot":   a fresh draw on every call.  Models white readout noise;
                  repeated measurement helps, so this is the optimistic case.
    seed : int
    """

    def __init__(
        self,
        sensor: np.ndarray,
        ground_truth: np.ndarray,
        grid: VoltageGrid,
        noise_std: float = 0.0,
        noise_mode: str = "static",
        seed: Optional[int] = None,
        meta: Optional[dict] = None,
    ):
        if sensor.shape != grid.shape:
            raise ValueError(f"sensor {sensor.shape} != grid {grid.shape}")
        if ground_truth.shape != grid.shape:
            raise ValueError(f"ground_truth {ground_truth.shape} != grid {grid.shape}")
        if noise_mode not in ("static", "shot"):
            raise ValueError("noise_mode must be 'static' or 'shot'")

        self.grid = grid
        self.ground_truth = ground_truth.astype(bool)
        self.meta = meta or {}
        self.noise_mode = noise_mode
        self._rng = np.random.default_rng(seed)

        # Normalise to [0, 1] so that noise_std is a fraction of full scale.
        lo, hi = float(sensor.min()), float(sensor.max())
        self._span = hi - lo
        self._clean = (sensor - lo) / self._span if hi > lo else np.zeros_like(sensor)
        self.noise_std = float(noise_std)

        if noise_mode == "static" and noise_std > 0:
            self._field = self._clean + self._rng.normal(
                0.0, noise_std, size=self._clean.shape
            )
        else:
            self._field = self._clean

        self.budget = Budget(grid=grid)

    # -- measurement ------------------------------------------------------

    def measure(self, i: int, j: int) -> float:
        """Measure one grid cell.  This is the ONLY way to read the device."""
        if not (0 <= i < self.grid.ny and 0 <= j < self.grid.nx):
            raise IndexError(f"cell ({i},{j}) outside grid {self.grid.shape}")
        self.budget.record(i, j)
        if self.noise_mode == "shot" and self.noise_std > 0:
            return float(self._field[i, j] + self._rng.normal(0.0, self.noise_std))
        return float(self._field[i, j])

    def measure_voltage(self, vx: float, vy: float) -> Tuple[float, Tuple[int, int]]:
        """Measure at the grid cell nearest to (vx, vy)."""
        i, j = self.grid.to_pixel(vx, vy)
        return self.measure(i, j), (i, j)

    def measure_many(self, cells) -> np.ndarray:
        return np.array([self.measure(int(i), int(j)) for i, j in cells])

    # -- bookkeeping ------------------------------------------------------

    def stage(self, name: str) -> "Device":
        """Tag subsequent measurements with a pipeline-stage label."""
        self.budget.stage = name
        return self

    def reset_budget(self) -> None:
        self.budget = Budget(grid=self.grid)

    # -- oracle access (evaluation only — never for detection) ------------

    def _full_field(self) -> np.ndarray:
        """Uncounted access to the whole array.

        Reserved for (a) computing ground truth and (b) the full-raster
        reference baseline, whose cost is N^2 by definition.  Calling this
        from a detector would silently invalidate the coverage figure, so it
        is named with a leading underscore and used in exactly two places.
        """
        return self._field

    @property
    def gt_pixels(self) -> np.ndarray:
        """(K,2) array of ground-truth transition pixels (row, col)."""
        return np.argwhere(self.ground_truth)
