"""Module for calculating properties of gasses."""


from typing import TypeVar

import numpy as np
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
