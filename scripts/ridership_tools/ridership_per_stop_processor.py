"""Aggregate per-stop ridership data and outputs one Excel file per route.

This utility reads stop-level ridership data from a single Excel workbook,
optionally filters the records by ``STOP_ID``, aggregates boardings and/or
alightings per route, computes each stop’s share of the totals, and writes
**one Excel workbook per route**.

Typical use
Identify the key stops on each route by absolute volume and by percentage
share of ridership.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the input Excel file
INPUT_FILE_PATH: Path = Path(r"\\Your\File\Path\To\STOP_USAGE_(BY_STOP_NAME).XLSX")

# Path to the directory where output files will be saved
OUTPUT_DIR: Path = Path(r"\\Your\Folder\Path\To\Output")

# List of STOP_IDs to filter on. If empty, no filter is applied.
STOP_FILTER_LIST: Sequence[int] = [1107, 2816, 6548]  # Example default

# Decide which columns to use in the output (True/False)
USE_BOARDINGS: bool = True
USE_ALIGHTINGS: bool = True

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(excel_path: Path | str) -> pd.DataFrame:
    """Load stop-level ridership data from an Excel workbook.

    Args:
        excel_path: Absolute or relative path to the source workbook.

    Returns:
        A ``pandas.DataFrame`` containing the workbook contents.

    Raises:
        FileNotFoundError: If *excel_path* does not exist.
        ValueError: If *excel_path* cannot be parsed as an Excel file.
    """
    data_frame = pd.read_excel(excel_path)
    return data_frame


def filter_by_stops(
    data_frame: pd.DataFrame,
    stop_ids: Sequence[int] | None,
) -> pd.DataFrame:
    """Return rows whose ``STOP_ID`` is in *stop_ids*.

    Args:
        data_frame: DataFrame produced by :func:`load_data`.
        stop_ids: Sequence of ``STOP_ID`` values to retain; ``None`` or
            an empty sequence means *data_frame* is returned unchanged.

    Returns:
        Filtered copy of *data_frame* (may be the original object).
    """
    if stop_ids:
        return data_frame[data_frame["STOP_ID"].isin(stop_ids)]
    return data_frame


def get_route_names(data_frame: pd.DataFrame) -> list[str]:
    """Extract the distinct route names present in *data_frame*.

    Args:
        data_frame: DataFrame containing a ``ROUTE_NAME`` column.

    Returns:
        Unsorted list of unique route names.
    """
    return data_frame["ROUTE_NAME"].unique()


def aggregate_route_data(
    data_frame: pd.DataFrame,
    route_name: str,
    boardings_flag: bool,
    alightings_flag: bool,
) -> pd.DataFrame | None:
    """Aggregate ridership for a single route.

    Args:
        data_frame: Full stop-level ridership DataFrame.
        route_name: Name of the route to process.
        boardings_flag: ``True`` to include ``XBOARDINGS``.
        alightings_flag: ``True`` to include ``XALIGHTINGS``.

    Returns:
        Aggregated DataFrame or ``None`` if both flags are ``False``.

    Raises:
        ValueError: If *route_name* is absent from *data_frame*.
    """
    # Filter the route
    route_df = data_frame[data_frame["ROUTE_NAME"] == route_name].copy()

    # Determine which columns to aggregate
    agg_dict = {}
    if boardings_flag:
        agg_dict["XBOARDINGS"] = "sum"
    if alightings_flag:
        agg_dict["XALIGHTINGS"] = "sum"

    # If neither boardings nor alightings is selected, return None
    if not agg_dict:
        return None

    # Aggregate data by STOP_ID and STOP_NAME
    grouped = route_df.groupby(["STOP_ID", "STOP_NAME"], as_index=False).agg(agg_dict)

    # Calculate totals from raw sums
    total_boardings = grouped["XBOARDINGS"].sum() if boardings_flag else 0
    total_alightings = grouped["XALIGHTINGS"].sum() if alightings_flag else 0

    # Calculate individual percentages for boardings and alightings
    if boardings_flag:
        grouped["PCT_BOARDINGS"] = (
            grouped["XBOARDINGS"] / total_boardings if total_boardings != 0 else 0.0
        )
    if alightings_flag:
        grouped["PCT_ALIGHTINGS"] = (
            grouped["XALIGHTINGS"] / total_alightings if total_alightings != 0 else 0.0
        )

    # If both boardings and alightings are used, calculate a combined total and percentage
    if boardings_flag and alightings_flag:
        grouped["XTOTAL"] = grouped["XBOARDINGS"] + grouped["XALIGHTINGS"]
        total_combined = grouped["XTOTAL"].sum()
        grouped["PCT_TOTAL"] = grouped["XTOTAL"] / total_combined if total_combined != 0 else 0.0

    # ==========================
    # Rounding step (to 1 decimal place)
    # ==========================
    if boardings_flag:
        grouped["XBOARDINGS"] = grouped["XBOARDINGS"].round(1)
    if alightings_flag:
        grouped["XALIGHTINGS"] = grouped["XALIGHTINGS"].round(1)
    if boardings_flag and alightings_flag:
        grouped["XTOTAL"] = grouped["XTOTAL"].round(1)

    return grouped


def save_route_data(
    route_data: pd.DataFrame,
    route_name: str,
    output_directory: Path | str,
) -> None:
    """Write *route_data* to ``<output_directory>/<route_name>.xlsx``.

    Args:
        route_data: Aggregated DataFrame from
            :func:`aggregate_route_data`.
        route_name: Route identifier used to build the filename.
        output_directory: Directory that will receive the workbook.
    """
    filename = f"{route_name}.xlsx"
    filepath = os.path.join(output_directory, filename)
    route_data.to_excel(filepath, index=False)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Coordinate the end-to-end aggregation workflow.

    Steps
    -----
    1. Load the source workbook.
    2. Optionally filter to the stops listed in *STOP_FILTER_LIST*.
    3. Determine which routes to process.
    4. Aggregate ridership per route.
    5. Export each route’s results to an individual workbook.
    6. Log progress to stdout.

    Returns:
        None
    """
    # 1. Read the Excel file
    data_frame = load_data(INPUT_FILE_PATH)

    # 2. Optionally filter by stop IDs to identify the routes serving them
    if STOP_FILTER_LIST:
        filtered_data_frame = filter_by_stops(data_frame, STOP_FILTER_LIST)
        route_names = get_route_names(filtered_data_frame)
    else:
        route_names = get_route_names(data_frame)

    # 3. Process each route
    for route_name in route_names:
        route_data = aggregate_route_data(data_frame, route_name, USE_BOARDINGS, USE_ALIGHTINGS)
        if route_data is not None and not route_data.empty:
            save_route_data(route_data, route_name, OUTPUT_DIR)
            print(f"Saved data for route '{route_name}' to {OUTPUT_DIR}")
        else:
            print(f"No boardings/alightings selected or no data for route '{route_name}'")


if __name__ == "__main__":
    main()
