"""Main entry point for positron_cross_section."""

import logging
from pathlib import Path
from sys import stderr

import matplotlib.pyplot as plt
import numpy as np
import typer

from positron_cross_section.gtcs import GTCSData

app = typer.Typer()


def main() -> None:
    """Entry point for positron_cross_section."""
    logging.basicConfig(stream=stderr, level=logging.WARNING)
    app()


@app.command()
def grand_total(
    data_filename: Path = typer.Argument(..., help="Path to cross section data file."),
    output_path: Path = typer.Argument("output", help="Path to store output data."),
) -> None:
    """Calculate and plot grand total cross section."""
    output_path.mkdir(exist_ok=True, parents=True)

    gtcs_data = GTCSData.from_csv(data_filename)
    gtcs_data.plot_cross_sections(output_path)
    gtcs_data.plot_I_0_ratio(output_path)
    np.savetxt(
        output_path / "normalized_signal.csv",
        gtcs_data.normalized_signal_data,
        delimiter=",",
    )
    np.savetxt(
        output_path / "cross_sections.csv",
        gtcs_data.raw_total_cross_sections,
        delimiter=",",
    )
    fig, ax = plt.subplots()
    ax.scatter(list(range(gtcs_data.num_scans)), gtcs_data.raw_total_cross_sections[:, 0])
    z = np.polyfit(list(range(gtcs_data.num_scans)), gtcs_data.raw_total_cross_sections[:, 0], 1)
    p = np.poly1d(z)
    ax.plot(
        list(range(gtcs_data.num_scans)),
        p(list(range(gtcs_data.num_scans))),
        "r",
        label=f"$y={z[0]:0.3f} x{z[1]:+0.3f}$",
    )
    ax.set(xlabel="Scan #", ylabel="$\\sigma\\ \\ (Å^2)$")
    ax.set_title("Cross section measured per scan at 1eV")
    ax.legend()
    fig.savefig(output_path / "increasing-cross-section.png")

    fig, ax = plt.subplots()
    ax.scatter(gtcs_data.metadata.cross_section_energies, gtcs_data.delta_theta)
    ax.set(xlabel="Incident Energy (ev)", ylabel="$\\delta\\theta$")
    ax.set_title("$\\delta\\theta$ for positron incident energies")
    fig.savefig(output_path / "delta-theta.png")


if __name__ == "__main__":
    main()
