"""
Script Name:
    ridership_per_trip_processor.py

Purpose:
    Processes trip-level ridership data from an Excel file. It allows for
    optional filtering by a specified column (e.g., ROUTE_NAME), splits the
    data into separate sheets per route and direction, calculates the
    percentage of total route ridership for each trip, highlights trips with
    the highest ridership, and can optionally generate bar charts of ridership
    by trip start time and flag ultra-low ridership trips. Useful for normalizing
    adjusted ridership to the trip level.

Inputs:
    1. Excel file (INPUT_FILE): Containing trip-level statistics, including
       at least 'SERIAL_NUMBER', 'TRIP_START_TIME', 'ROUTE_NAME',
       'DIRECTION_NAME', and 'PASSENGERS_ON'.
    2. Configuration constants defined in the script:
        - OUTPUT_FOLDER: Path to save the output Excel files.
        - DATE_TYPE: Label for the type of day being processed (e.g., 'Weekday'), used in chart titles.
        - COLUMNS_CONFIG: List defining which columns from the input to retain and their optional display names in the output.
        - CREATE_CHARTS: Boolean, if True, generates bar charts in output Excel sheets.
        - FLAG_ULTRA_LOW: Boolean, if True, flags trips with ridership at or below ULTRA_LOW_THRESHOLD.
        - ULTRA_LOW_THRESHOLD: Numeric threshold for flagging ultra-low ridership.
        - FILTER_COLUMN_NAME: Name of the column to apply filters on (e.g., 'ROUTE_NAME').
        - FILTER_IN_LIST: List of values to keep for the FILTER_COLUMN_NAME.
        - FILTER_OUT_LIST: List of values to exclude for the FILTER_COLUMN_NAME.

Outputs:
    1. Excel files (.xlsx): One file per unique route name (after filtering).
       - Each file is named '{route_name}.xlsx' and saved in OUTPUT_FOLDER.
       - Each workbook contains separate sheets for each direction of the route.
       - Sheets include trip details, ridership counts ('PASSENGERS_ON' rounded to 1 decimal),
         and 'Percent of Route Ridership' (share of total route passengers for that trip, as a percentage rounded to 1 decimal).
       - The trip with the highest 'PASSENGERS_ON' in each direction sheet is bolded.
       - Optionally, trips with 'PASSENGERS_ON' at or below ULTRA_LOW_THRESHOLD are highlighted in red.
       - Optionally, a bar chart visualizing 'PASSENGERS_ON' by 'TRIP_START_TIME' is included in each sheet.
    2. Console output: Status messages indicating saved workbooks.

Dependencies:
    pandas, openpyxl, os (standard library)
"""

import datetime
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"\\Path\To\Your\STATISTICS_BY_ROUTE_AND_TRIP.XLSX"
OUTPUT_FOLDER = r"\\Path\To\Your\Output\Folder"

# Label for the type of day (appears in chart titles, e.g. “Weekday”).
DATE_TYPE = "Weekday"

# COLUMN CONFIGURATION --------------------------------------------------------

# List of columns to retain; each entry may optionally include “: Custom Header”.
COLUMNS_CONFIG = [
    "SERIAL_NUMBER: Trip ID",
    "TRIP_START_TIME: Start Time",
    "ROUTE_NAME: Route",
    "DIRECTION_NAME: Direction",
    "PASSENGERS_ON: Passengers",
]

# Process COLUMNS_CONFIG into two structures:
#   1) COLUMNS_TO_RETAIN = ["SERIAL_NUMBER", "TRIP_START_TIME", …]
#   2) COLUMN_RENAME_MAP = { "SERIAL_NUMBER": "Trip ID", "TRIP_START_TIME": "Start Time", … }
COLUMNS_TO_RETAIN = []
COLUMN_RENAME_MAP = {}
for col_entry in COLUMNS_CONFIG:
    if ":" in col_entry:
        original, new_name = col_entry.split(":", 1)
        original = original.strip()
        new_name = new_name.strip()
        COLUMNS_TO_RETAIN.append(original)
        COLUMN_RENAME_MAP[original] = new_name
    else:
        col_name = col_entry.strip()
        COLUMNS_TO_RETAIN.append(col_name)

# OTHER OPTIONS ---------------------------------------------------------------

# Whether to create bar charts (True/False).
CREATE_CHARTS = True

# Whether to highlight ultra‐low trips (True/False).
FLAG_ULTRA_LOW = False
ULTRA_LOW_THRESHOLD = 1.0  # trips with ≤1 passenger are “ultra-low”

# FILTERING OPTIONS -----------------------------------------------------------

# If filtering is desired, specify the column (e.g. "ROUTE_NAME"); otherwise, set to None or "".
FILTER_COLUMN_NAME = "ROUTE_NAME"

# Only keep rows whose FILTER_COLUMN_NAME is in this list (if non-empty).
FILTER_IN_LIST = ["101", "202"]

# Exclude rows whose FILTER_COLUMN_NAME is in this list (if non-empty).
FILTER_OUT_LIST = []

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(input_file: str, columns_to_retain: list[str]) -> pd.DataFrame:
    """
    1. Read the entire Excel sheet into a DataFrame.
    2. Normalize 'TRIP_START_TIME' to a Python datetime.time object whenever possible.
       - If pandas already reads it as datetime64[ns], extract .dt.time.
       - Otherwise, coerce strings like “05:20” or “05:20:00” into time.
    3. Keep only the requested columns (in order).
    """
    # 1. Read the Excel file
    df = pd.read_excel(input_file)

    # 2. Normalize 'TRIP_START_TIME' if it exists
    if "TRIP_START_TIME" in df.columns:
        series = df["TRIP_START_TIME"]

        # Case A: pandas recognized it as datetime64[ns]
        if pd.api.types.is_datetime64_any_dtype(series):
            df["TRIP_START_TIME"] = series.dt.time

        else:
            # Case B: parse strings / other representations into datetime
            parsed = pd.to_datetime(
                series.astype(str).str.strip(),
                errors="coerce",  # invalid rows → NaT
                infer_datetime_format=True,
            )
            df["TRIP_START_TIME"] = parsed.dt.time

    # 3. Retain only columns that actually exist
    existing_cols = [c for c in columns_to_retain if c in df.columns]
    df = df[existing_cols]

    return df


def create_output_folder(folder_path: str) -> None:
    """
    Create the output folder (and any necessary parent folders) if it doesn't exist.
    """
    os.makedirs(folder_path, exist_ok=True)


def write_direction_sheet(
    wb: Workbook,
    direction_df: pd.DataFrame,
    direction_name: str,
    date_type: str,
    create_charts: bool,
    flag_ultra_low: bool,
    ultra_low_threshold: float,
) -> None:
    """
    For a single route‐&‐direction subset (direction_df):
      - Sort by TRIP_START_TIME
      - Compute Percent‐of‐Route‐Ridership
      - Bold the row with maximum passengers
      - Optionally flag ultra‐low trips in red
      - Write everything to an Excel sheet
      - Optionally add a bar chart of 'Passengers' vs. 'Start Time'
    """
    # 1. Locally sort by TRIP_START_TIME (NaT last)
    direction_df = direction_df.sort_values(
        "TRIP_START_TIME", na_position="last"
    ).reset_index(drop=True)

    # 2. Create a new worksheet named after this direction
    ws = wb.create_sheet(title=direction_name)

    # 3. Calculate total ridership and percentage share for each trip
    total_ridership = direction_df["PASSENGERS_ON"].sum()
    if total_ridership != 0:
        direction_df["Percent of Route Ridership"] = (
            direction_df["PASSENGERS_ON"] / total_ridership
        ).apply(lambda x: round(x * 100, 1) / 100)
    else:
        direction_df["Percent of Route Ridership"] = 0.0

    # 4. Round PASSENGERS_ON to 1 decimal
    direction_df["PASSENGERS_ON"] = direction_df["PASSENGERS_ON"].round(1)

    # 5. Determine the maximum ridership value (for bolding)
    max_ridership = direction_df["PASSENGERS_ON"].max()

    # 6. Write headers (using COLUMN_RENAME_MAP where provided)
    headers = list(direction_df.columns)
    for col_idx, header in enumerate(headers, start=1):
        display = COLUMN_RENAME_MAP.get(header, header)
        ws.cell(row=1, column=col_idx, value=display).font = Font(bold=True)

    # 7. Write each data row
    for row_idx, (_, row_data) in enumerate(direction_df.iterrows(), start=2):
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)

            if header == "TRIP_START_TIME":
                val = row_data[header]
                # If val is a Python time object
                if isinstance(val, type(datetime.datetime.now().time())):
                    cell.value = val
                    cell.number_format = "hh:mm"
                # If val is a pandas Timestamp (just in case)
                elif isinstance(val, pd.Timestamp):
                    cell.value = val.time()
                    cell.number_format = "hh:mm"
                else:
                    # Blank if unparseable / NaT
                    cell.value = None
            else:
                cell.value = row_data[header]

            # Format percentage column
            if header == "Percent of Route Ridership":
                cell.number_format = "0.0%"

        # 8. Bold the row if this trip has the maximum ridership
        if row_data["PASSENGERS_ON"] == max_ridership:
            for c_idx in range(1, len(headers) + 1):
                ws.cell(row=row_idx, column=c_idx).font = Font(bold=True)

        # 9. Optionally flag ultra‐low trips (≤ ULTRA_LOW_THRESHOLD) in red
        if flag_ultra_low and row_data["PASSENGERS_ON"] <= ultra_low_threshold:
            for c_idx in range(1, len(headers) + 1):
                orig = ws.cell(row=row_idx, column=c_idx).font
                ws.cell(row=row_idx, column=c_idx).font = Font(
                    name=orig.name,
                    size=orig.size,
                    bold=orig.bold,
                    color="FF0000",  # red
                )

    # 10. Optionally create a bar chart of ‘Passengers’ vs. ‘Start Time’
    if create_charts:
        chart = BarChart()
        chart.title = f"Ridership by Time – {direction_name} ({date_type})"
        chart.x_axis.title = "Trip Start Time"
        chart.y_axis.title = "Ridership"

        pass_col_idx = headers.index("PASSENGERS_ON") + 1
        time_col_idx = headers.index("TRIP_START_TIME") + 1
        first_data_row = 2
        last_data_row = direction_df.shape[0] + 1

        data_series = Reference(
            ws,
            min_col=pass_col_idx,
            min_row=first_data_row,
            max_col=pass_col_idx,
            max_row=last_data_row,
        )
        chart.add_data(data_series, titles_from_data=False)

        categories = Reference(
            ws,
            min_col=time_col_idx,
            min_row=first_data_row,
            max_row=last_data_row,
        )
        chart.set_categories(categories)

        anchor_cell = f"{get_column_letter(len(headers) + 2)}2"
        ws.add_chart(chart, anchor_cell)


def create_route_workbook(
    route_name: str,
    route_df: pd.DataFrame,
    output_folder: str,
    date_type: str,
    create_charts: bool,
    flag_ultra_low: bool,
    ultra_low_threshold: float,
) -> None:
    """
    For one route (all directions):
    - Remove default sheet
    - Call write_direction_sheet() for each direction
    - Save as '{route_name}.xlsx' in OUTPUT_FOLDER
    """
    wb = Workbook()
    default = wb.active
    wb.remove(default)

    for direction_name, direction_df in route_df.groupby("DIRECTION_NAME", sort=False):
        write_direction_sheet(
            wb=wb,
            direction_df=direction_df,
            direction_name=str(direction_name),
            date_type=date_type,
            create_charts=create_charts,
            flag_ultra_low=flag_ultra_low,
            ultra_low_threshold=ultra_low_threshold,
        )

    output_path = os.path.join(output_folder, f"{route_name}.xlsx")
    wb.save(output_path)
    print(f"Saved workbook: {output_path}")


def main() -> None:
    """
    1. Load data from INPUT_FILE.
    2. Optionally filter rows.
    3. Sort globally by (ROUTE_NAME, DIRECTION_NAME, TRIP_START_TIME).
    4. Ensure OUTPUT_FOLDER exists.
    5. If FLAG_ULTRA_LOW is True, create a .txt log of all ultra-low trips.
    6. Create one Excel workbook per route.
    """
    # 1. Load
    df = load_data(INPUT_FILE, COLUMNS_TO_RETAIN)

    # 2. Optional filtering
    if FILTER_COLUMN_NAME:
        if FILTER_IN_LIST:
            df = df[df[FILTER_COLUMN_NAME].isin(FILTER_IN_LIST)]
        if FILTER_OUT_LIST:
            df = df[~df[FILTER_COLUMN_NAME].isin(FILTER_OUT_LIST)]

    # 3. Global sort – so that groupby("ROUTE_NAME") inherits chronological order
    df = df.sort_values(
        by=["ROUTE_NAME", "DIRECTION_NAME", "TRIP_START_TIME"],
        kind="mergesort",  # stable sort
    ).reset_index(drop=True)

    # 4. Ensure output folder exists
    create_output_folder(OUTPUT_FOLDER)

    # 5. If FLAG_ULTRA_LOW, write a log of all trips ≤ ULTRA_LOW_THRESHOLD
    if FLAG_ULTRA_LOW:
        low_df = df[df["PASSENGERS_ON"] <= ULTRA_LOW_THRESHOLD]
        log_path = os.path.join(OUTPUT_FOLDER, "ultra_low_ridership_log.txt")

        with open(log_path, "w", encoding="utf-8") as log_file:
            if low_df.empty:
                log_file.write(
                    "No trips found with ultra-low ridership (≤ "
                    f"{ULTRA_LOW_THRESHOLD}).\n"
                )
            else:
                log_file.write(
                    "Trips with ultra-low ridership " f"(≤ {ULTRA_LOW_THRESHOLD}):\n\n"
                )
                for _, row in low_df.iterrows():
                    trip_id = row.get("SERIAL_NUMBER", "")
                    route = row.get("ROUTE_NAME", "")
                    direction = row.get("DIRECTION_NAME", "")
                    passengers = row.get("PASSENGERS_ON", "")
                    start_val = row.get("TRIP_START_TIME", None)
                    if isinstance(start_val, type(datetime.datetime.now().time())):
                        start_str = start_val.strftime("%H:%M")
                    else:
                        start_str = "" if pd.isna(start_val) else str(start_val)
                    log_file.write(
                        f"Route: {route}, Direction: {direction}, "
                        f"Trip ID: {trip_id}, Start Time: {start_str}, "
                        f"Passengers: {passengers}\n"
                    )
        print(f"Exported ultra-low ridership log: {log_path}")

    # 6. Create one workbook per route
    for route_name, route_df in df.groupby("ROUTE_NAME", sort=False):
        create_route_workbook(
            route_name=str(route_name),
            route_df=route_df,
            output_folder=OUTPUT_FOLDER,
            date_type=DATE_TYPE,
            create_charts=CREATE_CHARTS,
            flag_ultra_low=FLAG_ULTRA_LOW,
            ultra_low_threshold=ULTRA_LOW_THRESHOLD,
        )


if __name__ == "__main__":
    main()
