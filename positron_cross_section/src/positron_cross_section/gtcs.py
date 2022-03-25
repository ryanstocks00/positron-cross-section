"""Data and processing for grand total cross section (GTCS)."""

import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, validator
from uncertainties import unumpy

from positron_cross_section.gas import numeric_density
from positron_cross_section.plot import cross_section_plot

VarType = TypeVar("VarType")

SQUARE_METRES_TO_ANGSTROMS = 1e20


def extract_from_csv_row(
    row: List[str], regex_format: str, var_type: Callable[[Any], VarType]
) -> VarType:
    """Extract value using regular expression from a list of strings (a csv row).

    Args:
        row (List[str]): row
        regex_format (str): regex_format
        var_type (Type[VarType]): var_type

    Returns:
        VarType:
    """
    regexp = re.compile(regex_format)
    for cell in row:
        search = regexp.search(cell)
        if search:
            try:
                return var_type(search.group(1))
            except ValueError as e:
                raise ValueError(
                    f"Cannot convert {search.group(1)} to {var_type} in '{cell}'"
                ) from e
    formatted_row = "\n".join([c for c in row if c])
    raise ValueError(f"Could not find regex '{regex_format}' in parameters: \n{formatted_row}")


class GTCSMetadata(BaseModel):
    """Data for GTCS."""

    target: str
    pre_positronium_energy: float
    load_time: float
    cool_time: float
    dump_time: float
    trigger_delay: float
    sample_window: float
    resolution: float
    SC_cutoff: float
    RPA2_cutoff: float
    RPA1_potential: float
    M_ratio: float
    dumps_per_point: int

    SC_energies: List[float]
    RPA2_potentials: List[float]

    SC_length: float = 0.05

    @validator("SC_energies")
    @classmethod
    def energies_expected(cls, SC_energies: List[float]) -> List[float]:
        """Check SC energies are in groups of 3."""
        if SC_energies[1:-1:3] != SC_energies[2:-1:3] or SC_energies[2:-1:3] != SC_energies[3:-1:3]:
            raise ValueError("SC_energies not in groups of 3")
        return SC_energies

    @validator("RPA2_potentials")
    @classmethod
    def cutoffs_expected(cls, RPA2_potentials: List[float], values: Dict[str, Any]) -> List[float]:
        """Check RPA cutoffs are as expected for GTCS."""
        if RPA2_potentials[-1] < values["RPA2_cutoff"]:
            raise ValueError("Final RPA 2 potential is greater than cutoff")
        if any(v != 0 for v in RPA2_potentials[1:-1:3]):
            raise ValueError("RPA2_potentials non-zero in first test per energy.")
        RPA2_energy_ranges = [
            round(RPA2_potentials[3:-1:3][i] - RPA2_potentials[2:-1:3][i], 5)
            for i in range(len(RPA2_potentials) // 3)
        ]
        if RPA2_energy_ranges != values["SC_energies"][1:-1:3]:
            raise ValueError(f"{RPA2_energy_ranges} does not equal {values['SC_energies'][1:-1:3]}")
        return RPA2_potentials

    @property
    def cross_section_energies(self) -> List[float]:
        """Get energies for GTCS.

        Returns:
            List[float]:
        """
        return self.SC_energies[3:-1:3]

    @property
    def I_m_indices(self) -> List[int]:
        """Get I_m indices for GTCS.

        Returns:
            List[int]:
        """
        return list(range(3, len(self.SC_energies), 3))

    @property
    def I_0_indices(self) -> List[int]:
        """Get I_0 indices for GTCS.

        Returns:
            List[int]:
        """
        return list(range(1, len(self.SC_energies) - 1, 3))


class GTCSData:
    """GTCSData."""

    def __init__(
        self,
        metadata: GTCSMetadata,
        pressures: NDArray[np.float64],
        signal_data: NDArray[np.float64],
    ) -> None:
        self.metadata = metadata
        self.pressures = pressures
        self.numeric_densities = np.transpose(numeric_density(pressures)).reshape(
            len(self.pressures), 1
        )
        self.signal_data = signal_data

        self.zeroed_signal_data = self.signal_data - self.signal_data[:, -1:]

        self.I_or = self.zeroed_signal_data[0:, :1]
        self.I_m = self.zeroed_signal_data[:, self.metadata.I_m_indices]
        self.I_0 = self.zeroed_signal_data[:, self.metadata.I_0_indices]

        self.raw_total_cross_sections = (
            np.log(self.I_or / self.I_m)
            / (self.numeric_densities * self.metadata.SC_length)
            * SQUARE_METRES_TO_ANGSTROMS
        )
        self.total_cross_sections = unumpy.uarray(
            np.mean(self.raw_total_cross_sections, axis=0),
            np.std(self.raw_total_cross_sections, axis=0) / np.sqrt(len(self.pressures)),
        )
        self.raw_ps_cross_sections = (
            np.log(self.I_or / self.I_0)
            / (self.numeric_densities * self.metadata.SC_length)
            * SQUARE_METRES_TO_ANGSTROMS
        )
        self.ps_cross_sections = unumpy.uarray(
            np.mean(self.raw_ps_cross_sections, axis=0),
            np.std(self.raw_ps_cross_sections, axis=0) / np.sqrt(len(self.pressures)),
        )
        self.scattering_cross_sections = self.total_cross_sections - self.ps_cross_sections
        self.scattering_cross_sections = self.total_cross_sections - self.ps_cross_sections

    def plot_total_cross_section(self, ax: Any) -> None:
        """Plot grand total cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.total_cross_sections),
            yerr=unumpy.std_devs(self.total_cross_sections),
            fmt="o",
            color="blue",
            ecolor="black",
            capsize=4,
            label="Grand Total",
        )

    def plot_ps_cross_section(self, ax: Any) -> None:
        """Plot positronium cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.ps_cross_sections),
            yerr=unumpy.std_devs(self.ps_cross_sections),
            fmt="o",
            color="red",
            ecolor="black",
            capsize=4,
            label="Positronium",
        )

    def plot_scattering_cross_section(self, ax: Any) -> None:
        """Plot scattering cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.scattering_cross_sections),
            yerr=unumpy.std_devs(self.scattering_cross_sections),
            fmt="o",
            color="green",
            ecolor="black",
            capsize=4,
            label="Scattering",
        )

    def plot_cross_sections(self) -> None:
        """Plot grand total, positronium, and scattering cross sections."""
        fig, ax = cross_section_plot()
        self.plot_total_cross_section(ax)
        ax.set_title(
            f"Grand total cross section for positron-{self.metadata.target.lower()} interaction"
        )
        fig.savefig(f"grand-total-cross-section-{self.metadata.target.lower()}.png")

        fig, ax = cross_section_plot()
        self.plot_ps_cross_section(ax)
        ax.set_title(
            f"Positronium formation cross section for positron-{self.metadata.target.lower()} "
            "interaction"
        )
        fig.savefig(f"ps-cross-section-{self.metadata.target.lower()}.png")

        fig, ax = cross_section_plot()
        self.plot_total_cross_section(ax)
        self.plot_ps_cross_section(ax)
        self.plot_scattering_cross_section(ax)
        ax.set_title(f"Cross section for positron-{self.metadata.target.lower()} interaction")
        ax.legend()
        fig.savefig(f"cross-sections-{self.metadata.target.lower()}.png")

    @classmethod
    def from_csv(cls, csv_filename: Path) -> "GTCSData":
        """Read GTCS data from CSV file.

        Args:
            csv_filename (Path): csv_filename

        Returns:
            GTCSData:
        """
        with open(csv_filename, "r", encoding="utf-8") as csv_file:
            csv_data = list(csv.reader(csv_file, delimiter=","))

        parameter_row = csv_data[1]
        RPA2_cutoff = extract_from_csv_row(parameter_row, "RPA 2 Cutoff: (.*) V", float)
        SC_cutoff = extract_from_csv_row(parameter_row, "Scattering Cell Cutoff: (.*) V", float)

        run_rows = csv_data[6:]
        pressures: NDArray[np.float64] = np.ndarray(len(run_rows) // 2, dtype="float64")
        signal_data: NDArray[np.float64] = np.ndarray(
            (len(run_rows) // 2, len(run_rows[0])), dtype="float64"
        )
        for i in range(0, len(run_rows) // 2):
            pressures[i] = run_rows[2 * i][0]
            signal_data[i] = run_rows[2 * i + 1]

        return cls(
            metadata=GTCSMetadata(
                target=extract_from_csv_row(parameter_row, "Target: (.*)$", str),
                pre_positronium_energy=extract_from_csv_row(
                    parameter_row, "Pre-positronium Energy: (.*) eV", float
                ),
                load_time=extract_from_csv_row(parameter_row, "Load Time: (.*) us", float),
                cool_time=extract_from_csv_row(parameter_row, "Cool Time: (.*) us", float),
                dump_time=extract_from_csv_row(parameter_row, "Dump Time: (.*) us", float),
                trigger_delay=extract_from_csv_row(parameter_row, "Trigger Delay: (.*) us", float),
                sample_window=extract_from_csv_row(parameter_row, "Sample Window: (.*) us", float),
                resolution=extract_from_csv_row(parameter_row, "Resolution: (.*) meV", float),
                SC_cutoff=SC_cutoff,
                RPA2_cutoff=RPA2_cutoff,
                RPA1_potential=extract_from_csv_row(
                    parameter_row, "RPA 1 Potential: (.*) V", float
                ),
                M_ratio=extract_from_csv_row(parameter_row, "Magnetic Beach Ratio: (.*)$", float),
                dumps_per_point=extract_from_csv_row(parameter_row, "Dumps per Point: (.*)$", int),
                SC_energies=[round(SC_cutoff - float(val), 2) for val in csv_data[3]],
                RPA2_potentials=[float(val) for val in csv_data[2]],
            ),
            pressures=pressures,
            signal_data=signal_data,
        )
