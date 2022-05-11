"""Data and processing for grand total cross section (GTCS)."""

import csv
import math
import re
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np
import pandas
from matplotlib.pyplot import close
from numpy.typing import NDArray
from pydantic import BaseModel, validator
from uncertainties import unumpy

from positron_cross_section.gas import numeric_density, plot_existing_GTCS_data
from positron_cross_section.matplotlib_importer import plt
from positron_cross_section.plot import (
    average_columns_with_uncertainty,
    cross_section_plot,
    median_columns_with_uncertainty,
    save_plot,
)

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

    @property
    def I_0_prime_indices(self) -> List[int]:
        """Get I_0' indices for GTCS.

        Returns:
            List[int]:
        """
        return list(range(2, len(self.SC_energies) - 1, 3))


class GTCSData:
    """GTCSData."""

    def __init__(
        self,
        metadata: GTCSMetadata,
        pressures: NDArray[np.float64],
        signal_data: NDArray[np.float64],
    ) -> None:
        """Initialise GTCS Data.

        Args:
            metadata (GTCSMetadata): metadata
            pressures (NDArray[np.float64]): pressures
            signal_data (NDArray[np.float64]): signal_data
        """
        self.metadata = metadata
        self.pressures = pressures
        self.signal_data = signal_data
        self.num_scans = self.signal_data.shape[0]

        self.zeroed_signal_data = self.signal_data - np.mean(self.signal_data[:, -1:])
        # self.zeroed_signal_data = self.signal_data - self.signal_data[:, -1:]
        self.normalized_signal_data = self.zeroed_signal_data / self.zeroed_signal_data[:, :1]
        self.numeric_gas_densities = numeric_density(pressures).T.reshape(len(self.pressures), 1)

        self.I_0R = self.normalized_signal_data[:, :1]
        self.I_0 = self.normalized_signal_data.T[self.metadata.I_0_indices].T
        self.I_0_prime = self.normalized_signal_data.T[self.metadata.I_0_prime_indices].T
        self.I_0_ratios = average_columns_with_uncertainty(self.I_0 / self.I_0_prime)
        self.I_0_differences = average_columns_with_uncertainty(self.I_0 - self.I_0_prime)
        # self.I_0_ratio = np.mean(unumpy.nominal_values(self.I_0_ratios[:10]))
        self.I_0_difference = np.mean(unumpy.nominal_values(self.I_0_differences[:1]))
        self.I_m = self.normalized_signal_data.T[self.metadata.I_m_indices].T + (
            self.I_0_difference
        )  # self.I_0_ratio
        self.I_st = self.I_0R - self.I_m
        self.I_s = self.I_0_prime - self.I_m
        self.average_I_0R = average_columns_with_uncertainty(self.I_0R)
        self.average_I_0 = average_columns_with_uncertainty(self.I_0)
        self.average_I_m = average_columns_with_uncertainty(self.I_m)

        self.raw_total_cross_sections = (
            -np.log(self.I_m / self.I_0R)
            / (self.numeric_gas_densities * self.metadata.SC_length)
            * SQUARE_METRES_TO_ANGSTROMS
        )
        self.total_cross_sections = average_columns_with_uncertainty(self.raw_total_cross_sections)
        self.median_total_cross_sections = median_columns_with_uncertainty(
            self.raw_total_cross_sections
        )
        self.total_cross_sections_by_I_average = (
            -unumpy.log(self.average_I_m / self.average_I_0R)  # pylint: disable=no-member
            / (
                average_columns_with_uncertainty(self.numeric_gas_densities)
                * self.metadata.SC_length
            )
            * SQUARE_METRES_TO_ANGSTROMS
        )
        self.ps_cross_sections = (
            self.total_cross_sections
            * (self.average_I_0R - self.average_I_0)
            / (self.average_I_0R - self.average_I_m)
        )
        self.scattering_cross_sections = self.total_cross_sections - self.ps_cross_sections

        self.raw_ps_cross_sections = (
            self.raw_total_cross_sections * (self.I_0R - self.I_0) / (self.I_0R - self.I_m)
        )

        self.delta_E = [
            (self.metadata.RPA2_cutoff - self.metadata.RPA2_potentials[3:-1:3][i], E)
            for i, E in enumerate(self.metadata.cross_section_energies)
        ]
        self.delta_theta = [
            math.degrees(np.arccos(sqrt((E - delta_E) / E))) for delta_E, E in self.delta_E
        ]

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

    def plot_total_cross_section_by_I(self, ax: Any) -> None:
        """Plot grand total cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.total_cross_sections_by_I_average),
            yerr=unumpy.std_devs(self.total_cross_sections_by_I_average),
            fmt="o",
            color="purple",
            ecolor="black",
            capsize=4,
            label="Grand Total",
            zorder=100,
        )

    def plot_total_cross_section_by_median(self, ax: Any) -> None:
        """Plot grand total cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.median_total_cross_sections),
            yerr=unumpy.std_devs(self.median_total_cross_sections),
            fmt="o",
            color="orange",
            ecolor="black",
            capsize=4,
            label="Grand Total (median)",
        )

    def plot_ps_cross_section(self, ax: Any) -> None:
        """Plot positronium cross section on axes."""
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.ps_cross_sections),
            yerr=unumpy.std_devs(self.ps_cross_sections),
            fmt="p",
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
            fmt="v",
            color="green",
            ecolor="black",
            capsize=4,
            label="Scattering",
            zorder=50,
        )

    def plot_cross_sections(self, output_path: Path) -> None:
        """Plot grand total, positronium, and scattering cross sections."""
        fig, ax = cross_section_plot()
        self.plot_total_cross_section_by_I(ax)
        # self.plot_total_cross_section(ax)
        # self.plot_total_cross_section_by_median(ax)
        ax.set_title(
            f"Grand total cross section for positron-{self.metadata.target.lower()} interaction"
        )
        ax.legend(prop={"size": 10}, numpoints=1)
        save_plot(
            fig, output_path / f"grand-total-cross-section-{self.metadata.target.lower()}.png"
        )

        fig, ax = cross_section_plot()
        self.plot_ps_cross_section(ax)
        ax.set_title(
            f"Positronium formation cross section for positron-{self.metadata.target.lower()} "
            "interaction"
        )
        save_plot(fig, output_path / f"ps-cross-section-{self.metadata.target.lower()}.png")

        fig, ax = cross_section_plot()
        self.plot_total_cross_section_by_I(ax)
        self.plot_ps_cross_section(ax)
        self.plot_scattering_cross_section(ax)
        plot_existing_GTCS_data(ax, self.metadata.target)
        # self.plot_total_cross_section(ax)
        # self.plot_total_cross_section_by_median(ax)
        ax.set_title(f"Cross section for positron-{self.metadata.target.lower()} interaction")
        ax.legend(prop={"size": 10}, numpoints=1)
        save_plot(fig, output_path / f"cross-sections-{self.metadata.target.lower()}.png")

    def systematic_checks(self, output_path: Path) -> None:
        """Run and plot systematic checks for GTCS.

        Args:
            output_path:
        """
        output_path = output_path / "systematic_checks"
        output_path.mkdir(parents=True, exist_ok=True)

        self.plot_I_0_ratio(output_path)

        for i, energy in enumerate(self.metadata.cross_section_energies):
            fig, ax = plt.subplots()
            ax.scatter(
                list(range(self.num_scans)),
                self.raw_total_cross_sections[:, i],
                marker=".",
                label="Total",
            )
            z = np.polyfit(list(range(self.num_scans)), self.raw_total_cross_sections[:, i], 1)
            p = np.poly1d(z)
            ax.plot(
                list(range(self.num_scans)),
                p(list(range(self.num_scans))),
                "r",
                label=f"$y={z[0]:0.3f} x{z[1]:+0.3f}$ (Total)",
            )

            ax.scatter(
                list(range(self.num_scans)),
                self.raw_ps_cross_sections[:, i],
                marker=".",
                color="skyblue",
                label="Positronium",
            )
            z = np.polyfit(list(range(self.num_scans)), self.raw_ps_cross_sections[:, i], 1)
            p = np.poly1d(z)
            ax.plot(
                list(range(self.num_scans)),
                p(list(range(self.num_scans))),
                "orange",
                label=f"$y={z[0]:0.3f} x{z[1]:+0.3f}$ (Positronium)",
            )

            ax.set(xlabel="Scan #", ylabel="$\\sigma\\ \\ (Å^2)$", ylim=(-50, 100))
            ax.set_title(f"Cross section measured per scan at {energy}eV")
            ax.legend(prop={"size": 10}, numpoints=1)
            save_plot(fig, output_path / f"cross-section-against-time-{energy}eV.png")
            close(fig)

        fig, ax = plt.subplots()
        ax.scatter(list(range(len(self.pressures))), self.pressures)
        ax.set(xlabel="Scan #", ylabel="Pressure (mTorr)")
        ax.set_title("Pressure over time")
        save_plot(fig, output_path / "pressure.png")

    def plot_I_0_ratio(self, output_path: Path) -> None:
        """Plot the ratio I_0/I_0' as a systematic check."""
        fig, ax = plt.subplots()
        ax.errorbar(
            self.metadata.cross_section_energies,
            unumpy.nominal_values(self.I_0_ratios),
            yerr=unumpy.std_devs(self.I_0_ratios),
            fmt="o",
            color="red",
            ecolor="black",
            capsize=4,
        )
        ax.set_title("Ratio $\\frac{I_0}{I_0'}$")
        save_plot(fig, output_path / f"I_0-ratio-{self.metadata.target.lower()}.png")

    @classmethod
    def from_csv(cls, csv_filename: Path, num_scans: Optional[int]) -> "GTCSData":
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

        scan_rows = csv_data[6:]
        if num_scans:
            num_scans = min(num_scans, len(scan_rows) // 2)
        pressures: NDArray[np.float64] = np.ndarray(
            num_scans or len(scan_rows) // 2, dtype="float64"
        )
        signal_data: NDArray[np.float64] = np.ndarray(
            (num_scans or len(scan_rows) // 2, len(scan_rows[1])), dtype="float64"
        )
        for i in range(0, num_scans or len(scan_rows) // 2):
            pressures[i] = scan_rows[2 * i][0]
            signal_data[i] = scan_rows[2 * i + 1]

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
                SC_energies=[round(SC_cutoff - float(val), 2) for val in csv_data[3] if val],
                RPA2_potentials=[float(val) for val in csv_data[2] if val],
            ),
            pressures=pressures,
            signal_data=signal_data,
        )

    def summary_to_csv(self, filename: Path) -> None:
        """Save cross section summary to csv file.

        Args:
            filename (Path): filename
        """
        summary = pandas.DataFrame(
            {
                "Energy (eV)": self.metadata.cross_section_energies,
                "Total cross section (Angstrom^2)": unumpy.nominal_values(
                    self.total_cross_sections_by_I_average
                ),
                "TCS uncertainty (Angstrom^2)": unumpy.std_devs(
                    self.total_cross_sections_by_I_average
                ),
                "Positronium cross section (Angstrom^2)": unumpy.nominal_values(
                    self.ps_cross_sections
                ),
                "PCS uncertainty (Angstrom^2)": unumpy.std_devs(self.ps_cross_sections),
            },
        )
        summary.to_csv(filename)
