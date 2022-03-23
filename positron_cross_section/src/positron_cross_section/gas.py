"""Module for calculating properties of gasses."""


def numeric_density(pressure: float, temperature: float = 23) -> float:
    """Calculate numeric density of a gas.

    Args:
        pressure (float): pressure
        temperature (float): temperature

    Returns:
        float: Numeric density (atoms per cubic meter)
    """
    return pressure / temperature
