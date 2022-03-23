"""Data and processing for grand total cross section (GTCS)."""

import csv
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np
from pydantic import BaseModel

VarType = TypeVar("VarType")


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

    @classmethod
    def from_csv(
        cls, csv_filename: Path
    ) -> Tuple["GTCSMetadata", np.ndarray[np.float64, (Any)], np.ndarray[np.float64, (Any)]]:
        """Read GTCS data from CSV file.

        Args:
            csv_filename (Path): csv_filename

        Returns:
            GTCSMetadata:
        """
        with open(csv_filename, "r", encoding="utf-8") as csv_file:
            csv_data = list(csv.reader(csv_file, delimiter=","))

        parameter_row = csv_data[1]
        RPA2_cutoff = extract_from_csv_row(parameter_row, "RPA 2 Cutoff: (.*) V", float)
        SC_cutoff = extract_from_csv_row(parameter_row, "Scattering Cell Cutoff: (.*) V", float)

        run_rows = csv_data[6:]
        pressures: np.ndarray[np.float64, (Any)] = np.ndarray(len(run_rows) // 2, dtype="float64")
        signal_data: np.ndarray[np.float64, (Any)] = np.ndarray(
            (len(run_rows) // 2, len(run_rows[0])), dtype="float64"
        )
        for i in range(0, len(run_rows) // 2):
            pressures[i] = run_rows[2 * i][0]
            signal_data[i] = run_rows[2 * i + 1]

        return (
            cls(
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
            pressures,
            signal_data,
        )
