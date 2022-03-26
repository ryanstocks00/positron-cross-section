"""Functions to assist with plotting cross sections."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from uncertainties import unumpy


def cross_section_plot() -> Any:
    """Create figure and axes for a cross section plot."""
    fig, ax = plt.subplots()
    # ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set(xlabel="Incident energy (eV)", ylabel="$\\sigma\\ \\ (Å^2)$")
    return fig, ax


def average_columns_with_uncertainty(array: NDArray[np.float64]) -> Any:
    """Average columns in np.ndarray and return their averages with statistical uncertainty.

    Args:
        array (NDArray): array

    Returns:
        NDArray:
    """
    return unumpy.uarray(
        np.mean(array, axis=0),
        np.std(array, axis=0) / np.sqrt(array.shape[0]),
    )
