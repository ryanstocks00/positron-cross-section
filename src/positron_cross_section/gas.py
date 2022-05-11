"""Module for calculating properties of gasses."""


from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas
from numpy.typing import NDArray

GAS_CONSTANT = 8.31446261815324
AVOGADROS_CONSTANT = 6.02214086e23
MTORR_TO_PASCALS = 0.13332237

FloatLike = TypeVar("FloatLike", float, NDArray[np.float64])


def numeric_density(pressure: FloatLike, temperature: float = 300) -> FloatLike:
    """Calculate numeric density of a gas.

    Args:
        pressure (float): pressure in mTorr
        temperature (float): temperature

    Returns:
        float: Numeric density (atoms per cubic meter)
    """
    return pressure * MTORR_TO_PASCALS / (temperature * GAS_CONSTANT) * AVOGADROS_CONSTANT


def plot_existing_GTCS_data(ax: Any, target: str) -> None:
    """Plot existing GTCS data."""
    filename = (
        Path(__file__).parent.parent.parent / "previous_results" / f"{target.lower()}_tcs.csv"
    )
    if filename.exists():
        previous_tcs = pandas.read_csv(filename)
        ax.errorbar(
            previous_tcs["Energy"],
            previous_tcs["TCS"],
            yerr=previous_tcs["Error"],
            fmt="-d",
            color="lightgray",
            ecolor="black",
            capsize=4,
            label="Chiari et. al.",
        )
    filename = (
        Path(__file__).parent.parent.parent
        / "previous_results"
        / f"{target.lower()}_tcs_floeder.csv"
    )
    if filename.exists():
        previous_tcs = pandas.read_csv(filename)
        ax.errorbar(
            previous_tcs["Energy"],
            previous_tcs["TCS"],
            yerr=previous_tcs["Error"],
            fmt="-^",
            color="lightpink",
            ecolor="white",
            capsize=0,
            label="Floeder et. al.",
        )
