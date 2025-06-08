"""
Aggregates per-stop ridership data and outputs one Excel file per route.

Reads a stop-level ridership Excel file, optionally filters by STOP_ID,
aggregates boardings and/or alightings by route, and calculates percentage
contributions per stop. Outputs one Excel workbook per route with totals
and percentage columns for each metric.

Typical use: Identify key stops by volume and share of ridership.

Inputs:
    - Excel file with stop-level ridership fields (e.g., 'ROUTE_NAME', 'STOP_ID',
      'XBOARDINGS', 'XALIGHTINGS')
    - Configurable flags for filtering STOP_IDs and including boardings/alightings

Outputs:
    - One Excel file per route with aggregated stop data and calculated percentages
    - Console messages listing processed routes
"""

import os

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the input Excel file
INPUT_FILE_PATH = r"\\Your\File\Path\To\STOP_USAGE_(BY_STOP_NAME).XLSX"

# Path to the directory where output files will be saved
OUTPUT_DIR = r"\\Your\Folder\Path\To\Output"

# List of STOP_IDs to filter on. If empty, no filter is applied.
STOP_FILTER_LIST = [1107, 2816, 6548]  # Example default

# Decide which columns to use in the output (True/False)
USE_BOARDINGS = True
USE_ALIGHTINGS = True

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def load_data(excel_path):
    """
    Reads the Excel file into a pandas DataFrame.
    """
    data_frame = pd.read_excel(excel_path)
    return data_frame


def filter_by_stops(data_frame, stop_ids):
    """
    Filters the DataFrame by a list of stop_ids if the list is not empty.
    If stop_ids is empty, returns the original data_frame.
    """
    if stop_ids:
        return data_frame[data_frame["STOP_ID"].isin(stop_ids)]
    return data_frame


def get_route_names(data_frame):
    """
    Returns a list of unique route names in the given DataFrame.
    """
    return data_frame["ROUTE_NAME"].unique()


def aggregate_route_data(data_frame, route_name, boardings_flag, alightings_flag):
    """
    For a given route_name, creates a DataFrame of stops served by that route,
    and calculates totals and percentages for boardings and/or alightings.
    Rounded to 1 decimal place for XBOARDINGS, XALIGHTINGS, and XTOTAL.
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
        grouped["PCT_TOTAL"] = (
            grouped["XTOTAL"] / total_combined if total_combined != 0 else 0.0
        )

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


def save_route_data(route_data, route_name, output_directory):
    """
    Saves the route_data DataFrame to an Excel file named after the route.
    """
    filename = f"{route_name}.xlsx"
    filepath = os.path.join(output_directory, filename)
    route_data.to_excel(filepath, index=False)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main function to:
      1. Load data from Excel.
      2. Optionally filter for specific stops.
      3. Identify route names to process.
      4. Aggregate boardings/alightings for each route.
      5. Save each route's results to an individual Excel file.
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
        route_data = aggregate_route_data(
            data_frame, route_name, USE_BOARDINGS, USE_ALIGHTINGS
        )
        if route_data is not None and not route_data.empty:
            save_route_data(route_data, route_name, OUTPUT_DIR)
            print(f"Saved data for route '{route_name}' to {OUTPUT_DIR}")
        else:
            print(
                f"No boardings/alightings selected or no data for route '{route_name}'"
            )


if __name__ == "__main__":
    main()
