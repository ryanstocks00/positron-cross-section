"""Functions to assist with plotting cross sections."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from uncertainties import unumpy

from positron_cross_section.matplotlib_importer import plt

plt.style.use("classic")
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def cross_section_plot() -> Any:
    """Create figure and axes for a cross section plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    # ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set(xlabel="Incident energy (eV)", ylabel="Cross section $\\sigma\\ \\ (Å^2)$")
    ax.set_xlim(0.9, 110)
    ax.set_ylim(-2.5, 40)
    return fig, ax


def save_plot(fig: Any, filename: Path, dpi: int = 300) -> None:
    """Save a plot.

    Args:
        fig (Any): fig
        filename (Path): filename
        dpi (int): dpi
    """
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)


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


def median_columns_with_uncertainty(array: NDArray[np.float64]) -> Any:
    """Find the median of the columns in np.ndarray and return with statistical uncertainty.

    Args:
        array (NDArray): array

    Returns:
        NDArray:
    """
    N = array.shape[0]
    n = (N - 1) / 2
    return unumpy.uarray(
        np.median(array, axis=0),
        np.std(array, axis=0) / np.sqrt(array.shape[0]) * np.sqrt(np.pi * N / (4 * n)),
    )
