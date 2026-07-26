"""
axis_labels.py — one place that decides what the x and y axes are called.

Every figure in the pipeline plots the same two gate voltages, so they must
all carry the same axis labels.  Previously each plotting module hard-coded
its own ("P1", "Vx (mV)", "Vx (V)", "X (mV)" …), which is why the images
disagreed with each other.

The names and units are hyperparameters: set them ONCE at the start of the
program (scripts/run_simulation.py passes them to DatasetPipeline, which
calls :func:`set_axis_labels`), and every plot picks them up.

    from ..config.axis_labels import x_label, y_label
    ax.set_xlabel(x_label())      # -> "P1 (mV)"
    ax.set_ylabel(y_label())      # -> "P2 (mV)"

Always CALL the functions at plot time rather than storing the string in a
default argument — default arguments are evaluated at import time, before
set_axis_labels() has run.
"""
from dataclasses import dataclass

DEFAULT_X_NAME = "P1"
DEFAULT_Y_NAME = "P2"
DEFAULT_X_UNIT = "mV"
DEFAULT_Y_UNIT = "mV"


@dataclass
class AxisLabels:
    """Axis names and units shared by every figure."""

    x_name: str = DEFAULT_X_NAME
    y_name: str = DEFAULT_Y_NAME
    x_unit: str = DEFAULT_X_UNIT
    y_unit: str = DEFAULT_Y_UNIT

    @property
    def x_label(self) -> str:
        return f"{self.x_name} ({self.x_unit})" if self.x_unit else self.x_name

    @property
    def y_label(self) -> str:
        return f"{self.y_name} ({self.y_unit})" if self.y_unit else self.y_name

    def as_dict(self) -> dict:
        return {
            "x_axis_name": self.x_name,
            "y_axis_name": self.y_name,
            "x_axis_unit": self.x_unit,
            "y_axis_unit": self.y_unit,
        }


_ACTIVE = AxisLabels()


def set_axis_labels(
    x_name: str = None,
    y_name: str = None,
    x_unit: str = None,
    y_unit: str = None,
) -> AxisLabels:
    """Set the axis names / units used by every plot. None = leave unchanged."""
    if x_name is not None:
        _ACTIVE.x_name = x_name
    if y_name is not None:
        _ACTIVE.y_name = y_name
    if x_unit is not None:
        _ACTIVE.x_unit = x_unit
    if y_unit is not None:
        _ACTIVE.y_unit = y_unit
    return _ACTIVE


def get_axis_labels() -> AxisLabels:
    """The active AxisLabels object."""
    return _ACTIVE


def x_label() -> str:
    """Full x-axis label, e.g. 'P1 (mV)'."""
    return _ACTIVE.x_label


def y_label() -> str:
    """Full y-axis label, e.g. 'P2 (mV)'."""
    return _ACTIVE.y_label
