"""
Script Name:
        gtfs_trips_hourly_reporter.py

Purpose:
        Processes GTFS data to count and report trips for specified routes
        and directions within defined time intervals, exporting results to
        Excel files. Allows filtering by calendar service id or processing
        all services separately.

Inputs:
        1. GTFS text files (e.g., trips.txt, stop_times.txt, routes.txt,
           calendar.txt) located in the `BASE_INPUT_PATH`.
        2. Configuration constants within the script:
           - `BASE_INPUT_PATH`: Path to the directory containing GTFS files.
           - `BASE_OUTPUT_PATH`: Path to the directory where Excel reports
             will be saved.
           - `GTFS_FILES`: List of required GTFS filenames.
           - `ROUTE_DIRECTIONS`: List of dictionaries defining routes and
             direction_ids to process.
           - `TIME_INTERVAL_MINUTES`: Integer defining the time interval
             for grouping trips (e.g., 60 for hourly).
           - `SERVICE_ID`: String defining the service_id value to filter
             active services. If empty, each service_id is processed
             separately.

Outputs:
        1. Excel (.xlsx) files generated in the `BASE_OUTPUT_PATH`.
            - Each file contains a report of trip counts per time interval for a
               specific route, direction, and, if applicable, service_id.

Dependencies:
        pandas, openpyxl, os, logging
"""

import logging
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input directory containing GTFS files
BASE_INPUT_PATH = r"\\your_file_path\here\\"

# Output directory for the Excel file
BASE_OUTPUT_PATH = r"\\your_file_path\here\\"

# GTFS files to load
GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "routes.txt",
    "calendar.txt",
    # 'stops.txt',  # Uncomment if needed
]

# Routes and directions to process
ROUTE_DIRECTIONS = [
    {
        "route_short_name": "101",
        "direction_id": 0,
    },  # Process only direction 0 for route 101
    {
        "route_short_name": "202",
        "direction_id": None,
    },  # Process all directions for route 202
]

# New Configuration: Time Interval in Minutes
TIME_INTERVAL_MINUTES = 60  # Users can change this to 30, 15, etc.

# Choose one service to analyse (None → run each service separately)
SERVICE_ID = "3"  # Replace with your desired service_id value from calendar.txt

# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------
def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from the specified directory.
    This function uses the logging module for output.

    Parameters:
        gtfs_folder_path (str): Path to the directory containing GTFS files.
        files (list[str], optional): GTFS filenames to load. Default is all
            standard GTFS files:
            [
                "agency.txt",
                "stops.txt",
                "routes.txt",
                "trips.txt",
                "stop_times.txt",
                "calendar.txt",
                "calendar_dates.txt",
                "fare_attributes.txt",
                "fare_rules.txt",
                "feed_info.txt",
                "frequencies.txt",
                "shapes.txt",
                "transfers.txt"
            ]
        dtype (str or dict, optional): Pandas dtype to use. Default is str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        OSError: If gtfs_folder_path doesn't exist or if any required file is missing.
        ValueError: If a file is empty or there's a parsing error.
        RuntimeError: For OS errors during file reading.
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = [
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        ]

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(
            f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}"
        )

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"File '{file_name}' in '{gtfs_folder_path}' is empty."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------
def fix_time_format(time_str):
    """
    Fix time formats by:
    - Adding leading zeros to single-digit hours
    - Converting hours greater than 23 by subtracting 24
    """
    if pd.isna(time_str):
        return time_str
    parts = time_str.split(":")

    # Add leading zero if the hour is a single digit
    if len(parts[0]) == 1:
        parts[0] = "0" + parts[0]

    # Correct times where the hour exceeds 23 (indicating next day service)
    if int(parts[0]) >= 24:
        parts[0] = str(int(parts[0]) - 24).zfill(2)

    return ":".join(parts)


def get_time_bin(t, interval):
    """
    Assigns a time object to a specific time bin based on the interval.

    Args:
        t (datetime.time): The time to bin.
        interval (int): The interval in minutes.

    Returns:
        str: A string representing the time bin (e.g., "08:00-08:59").
    """
    total_minutes = t.hour * 60 + t.minute
    bin_start = (total_minutes // interval) * interval
    bin_end = bin_start + interval - 1
    if bin_end >= 1440:
        bin_end -= 1440  # Wrap around if necessary
    start_hour, start_min = divmod(bin_start, 60)
    end_hour, end_min = divmod(bin_end, 60)
    return f"{str(start_hour).zfill(2)}:{str(start_min).zfill(2)}-{str(end_hour).zfill(2)}:{str(end_min).zfill(2)}"


def process_and_export(
    data: dict[str, pd.DataFrame],
    route_dirs: list[dict],
    output_path: str,
    interval_minutes: int,
    service_id: str | None = None,
) -> None:
    """
    Create “trips-per-time-bin” Excel workbooks from a GTFS feed.

    Parameters
    ----------
    data
        Dictionary returned by `load_gtfs_data()`.
    route_dirs
        List like [{"route_short_name": "101", "direction_id": 0}, …].
    output_path
        Directory where the Excel files will be written.
    interval_minutes
        Width of each time bin in minutes (e.g. 60 for hourly).
    service_id
        • str  → process only that service_id.
        • None → iterate over every service_id in the feed.
    """
    trips = data["trips"]
    stop_times = data["stop_times"]
    routes = data["routes"]

    # ── Apply service filter ────────────────────────────────────────────────
    if service_id is not None:
        trips_filtered = trips[trips["service_id"] == str(service_id)]
        logging.info(
            "Analysing only service_id=%s (%d trips).", service_id, len(trips_filtered)
        )
    else:
        trips_filtered = trips.copy()
        logging.info("No service_id specified – will iterate over each one.")

    # ── Merge the required tables ───────────────────────────────────────────
    merged = stop_times.merge(trips_filtered, on="trip_id", how="inner").merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )

    # Cast key fields to numeric so comparisons work (dtype=str in loader)
    merged["direction_id"] = pd.to_numeric(merged["direction_id"], errors="coerce")
    merged["stop_sequence"] = pd.to_numeric(merged["stop_sequence"], errors="coerce")

    # ── Fix & parse time columns ────────────────────────────────────────────
    for col in ("arrival_time", "departure_time"):
        merged[col] = (
            merged[col]
            .apply(fix_time_format)
            .str.strip()
            .pipe(pd.to_datetime, format="%H:%M:%S", errors="coerce")
            .dt.time
        )

    # ── Pre-compute full list of time-bin labels ────────────────────────────
    time_bins = [
        f"{h:02d}:{m:02d}-{(h + (m + interval_minutes - 1) // 60) % 24:02d}:"
        f"{(m + interval_minutes - 1) % 60:02d}"
        for h in range(24)
        for m in range(0, 60, interval_minutes)
    ]

    # ── Loop over requested routes/directions ───────────────────────────────
    for rd in route_dirs:
        r_short = rd["route_short_name"]
        d_id = rd["direction_id"]

        sel = merged[merged["route_short_name"] == r_short]
        if d_id is not None:
            sel = sel[sel["direction_id"] == d_id]

        # first stop of each trip
        starts = sel[sel["stop_sequence"] == 1].dropna(subset=["departure_time"])

        # Determine which service_ids to iterate over
        sid_iter = (
            [service_id] if service_id is not None else starts["service_id"].unique()
        )

        for sid in sid_iter:
            cur = (
                starts
                if service_id is not None
                else starts[starts["service_id"] == sid]
            )

            if cur.empty:
                logging.info(
                    "No trips for route=%s dir=%s service=%s – skipping.",
                    r_short,
                    d_id,
                    sid,
                )
                continue

            cur = cur.assign(
                time_bin=cur["departure_time"].apply(
                    lambda t: get_time_bin(t, interval_minutes)
                )
            )

            trips_per_bin = (
                cur.groupby("time_bin")
                .size()
                .reindex(time_bins, fill_value=0)
                .reset_index()
                .rename(columns={0: "trip_count"})
            )

            _export_to_excel(
                trips_per_bin,
                output_path,
                interval_minutes,
                r_short,
                d_id,
                service_id=sid if service_id is None else service_id,
            )


def _export_to_excel(
    df, output_path, interval_minutes, route_short, direction_id=None, service_id=None
):
    """
    Helper function to export a given DataFrame to an Excel file.
    """
    wb = Workbook()
    ws = wb.active

    # Create a title reflecting route/direction/service
    if service_id:
        if direction_id is not None:
            ws.title = f"Service_{service_id}_Route_{route_short}_Dir_{direction_id}"
            file_name = (
                f"Trips_Per_{interval_minutes}Min_"
                f"Service_{service_id}_Route_{route_short}_Dir_{direction_id}.xlsx"
            )
        else:
            ws.title = f"Service_{service_id}_Route_{route_short}_All_Dirs"
            file_name = (
                f"Trips_Per_{interval_minutes}Min_"
                f"Service_{service_id}_Route_{route_short}_All_Directions.xlsx"
            )
    else:
        if direction_id is not None:
            ws.title = f"Route_{route_short}_Dir_{direction_id}"
            file_name = (
                f"Trips_Per_{interval_minutes}Min_"
                f"Route_{route_short}_Dir_{direction_id}.xlsx"
            )
        else:
            ws.title = f"Route_{route_short}_All_Directions"
            file_name = (
                f"Trips_Per_{interval_minutes}Min_"
                f"Route_{route_short}_All_Directions.xlsx"
            )

    # Write headers
    ws.append(df.columns.tolist())

    # Write data rows
    for row in df.itertuples(index=False, name=None):
        ws.append(row)

    # Adjust column widths and alignments
    for col in ws.columns:
        max_length = max(len(str(cell.value)) for cell in col) + 2
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max_length
        for cell in col:
            cell.alignment = Alignment(horizontal="center")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the workbook
    output_file = os.path.join(output_path, file_name)
    wb.save(output_file)
    logging.info(
        "Trips per %d minutes for %s successfully exported!",
        interval_minutes,
        file_name,
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Entry-point."""
    # ─── Logging setup (moved here) ───────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # overrides any library defaults
    )
    # ─────────────────────────────────────────────────────────────────────────

    try:
        # Validate interval
        if TIME_INTERVAL_MINUTES <= 0 or 1440 % TIME_INTERVAL_MINUTES:
            raise ValueError("TIME_INTERVAL_MINUTES must divide evenly into 1,440.")

        # Use the *new* loader (it will raise if files/dir are missing)
        data = load_gtfs_data(
            BASE_INPUT_PATH,
            files=GTFS_FILES,
            dtype=str,  # keep everything as strings – avoids TZ parsing issues
        )

        # Run the core logic
        process_and_export(
            data,
            ROUTE_DIRECTIONS,
            BASE_OUTPUT_PATH,
            TIME_INTERVAL_MINUTES,
            service_id=SERVICE_ID,  # NEW ARG
        )

    except Exception as exc:  # catch-all so the stack-trace hits your log
        logging.exception("Fatal error: %s", exc)


if __name__ == "__main__":
    main()
