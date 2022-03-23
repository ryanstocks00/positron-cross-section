"""Main entry point for bushfire-drone-simulation."""

import logging
from pathlib import Path
from sys import stderr

import typer

app = typer.Typer()


def main() -> None:
    """Entry point for positron_cross_section."""
    logging.basicConfig(stream=stderr, level=logging.WARNING)
    app()


@app.command()
def grand_total(
    data_filename: Path = typer.Argument(..., help="Path to parameters file."),
) -> None:
    """Calculate and plot grand total cross section."""
    print(data_filename)


if __name__ == "__main__":
    main()
