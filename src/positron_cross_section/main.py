"""Main entry point for positron_cross_section."""

import logging
from pathlib import Path
from sys import stderr
from typing import Optional

import numpy as np
import typer

from positron_cross_section.gtcs import GTCSData
from positron_cross_section.matplotlib_importer import plt
from positron_cross_section.plot import save_plot

app = typer.Typer()


def main() -> None:
    """Entry point for positron_cross_section."""
    logging.basicConfig(stream=stderr, level=logging.WARNING)
    app()


@app.command()
def grand_total(
    data_filename: Path = typer.Argument(..., help="Path to cross section data file."),
    output_path: Path = typer.Argument("output", help="Path to store output data."),
    n: Optional[int] = typer.Option(None, help="Path to store output data."),
) -> None:
    """Calculate and plot grand total cross section."""
    output_path.mkdir(exist_ok=True, parents=True)

    gtcs_data = GTCSData.from_csv(data_filename, num_scans=n)
    gtcs_data.plot_cross_sections(output_path)
    gtcs_data.systematic_checks(output_path)
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
    ax.scatter(gtcs_data.metadata.cross_section_energies, gtcs_data.delta_theta)
    ax.set(xlabel="Incident Energy (ev)", ylabel="$\\delta\\theta$")
    ax.set_title("$\\delta\\theta$ for positron incident energies")
    save_plot(fig, output_path / "delta-theta.png")

    gtcs_data.summary_to_csv(output_path / "summary.csv")


if __name__ == "__main__":
    main()
