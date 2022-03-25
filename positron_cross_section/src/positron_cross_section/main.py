"""Main entry point for positron_cross_section."""

import logging
from pathlib import Path
from sys import stderr

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
) -> None:
    """Calculate and plot grand total cross section."""
    gtcs_data = GTCSData.from_csv(data_filename)

    gtcs_data.plot_cross_sections()


if __name__ == "__main__":
    main()
