"""Functions to assist with plotting cross sections."""

from typing import Any

import matplotlib.pyplot as plt


def cross_section_plot() -> Any:
    """Create figure and axes for a cross section plot."""
    fig, ax = plt.subplots()
    # ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set(xlabel="Incident energy (eV)", ylabel="$\\sigma\\ \\ (Å^2)$")
    return fig, ax
