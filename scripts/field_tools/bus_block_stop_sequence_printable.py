"""
Script Name:
    bus_block_stop_sequence_printable.py

Purpose:
    Generates printable, stop-by-stop Excel schedules for each
    vehicle block defined in GTFS data.  Merges trip, stop-time,
    stop, and route information.  Supports optional filtering by
    service ID and route short name.  Primarily intended for field
    operations and audits.

Inputs:
    1. GTFS data files (requires at minimum trips.txt, stop_times.txt,
       stops.txt, routes.txt, calendar.txt) located in the directory
       specified by GTFS_FOLDER_PATH.

Outputs:
    1. Individual Excel (.xlsx) files, one per vehicle block after
       filtering.  Each file contains a formatted schedule with
       columns for field data entry (Actual Time, Boardings, etc.).
       Files are saved to the folder specified by BASE_OUTPUT_PATH.

Dependencies:
    pandas, openpyxl, logging
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER_PATH = r"C:\Path\To\Your\Input\Folder"  # <<< EDIT HERE
BASE_OUTPUT_PATH = r"C:\Path\To\Your\Output\Folder"  # <<< EDIT HERE

REQUIRED_GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "routes.txt",
    "calendar.txt",
]

# If you only want certain service IDs or route short names, specify them here:
FILTER_SERVICE_IDS: list[str] = []  # e.g. ["WKD", "SAT"]
FILTER_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["101", "202"]

# Placeholder values for printing:
MISSING_TIME = "________"
MISSING_VALUE = "_____"

# Maximum column width for neat Excel formatting:
MAX_COLUMN_WIDTH = 35

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
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def time_to_seconds(time_str):
    """
    Converts a 'HH:MM:SS' or 'HH:MM' string into total seconds.
    Handles hours >= 24 by rolling over (e.g., 25:10:00 -> 1:10:00).
    """
    if pd.isnull(time_str):
        return math.nan

    parts = time_str.strip().split(":")
    if len(parts) < 2:
        return math.nan

    try:
        hours = int(parts[0]) % 24  # Roll over hours >= 24
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return math.nan

    return hours * 3600 + minutes * 60 + seconds


def format_hhmm(total_seconds):
    """
    Given a time in total seconds, returns a 'HH:MM' string (24-hour).
    If invalid, returns an empty string.
    """
    if pd.isnull(total_seconds) or total_seconds < 0:
        return ""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------


def export_to_excel(data_frame, output_file):
    """
    Exports a DataFrame to an Excel file and applies basic formatting:
    - Left alignment
    - Sets column widths
    - Text wrapping for headers
    """
    if data_frame.empty:
        print(f"No data to export to {output_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write DataFrame to Excel and access the worksheet object for formatting
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data_frame.to_excel(writer, index=False, sheet_name="Schedule")
        worksheet = writer.sheets["Schedule"]

        # Adjust columns
        for col_i, col_name in enumerate(data_frame.columns, 1):
            col_letter = get_column_letter(col_i)

            # Header alignment and text wrap
            header_cell = worksheet[f"{col_letter}1"]
            header_cell.alignment = Alignment(horizontal="left", wrap_text=True)

            # Data alignment
            for row_i in range(2, worksheet.max_row + 1):
                cell = worksheet[f"{col_letter}{row_i}"]
                cell.alignment = Alignment(horizontal="left")

            # Set column width based on max content length, capped at MAX_COLUMN_WIDTH
            max_len = max(len(str(col_name)), 10)  # Minimum width
            for row_i in range(2, worksheet.max_row + 1):
                val = worksheet[f"{col_letter}{row_i}"].value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            worksheet.column_dimensions[col_letter].width = min(
                max_len + 2, MAX_COLUMN_WIDTH
            )

    print(f"Exported: {output_file}")


def filter_data(trips_df, stop_times_df, routes_df):
    """
    Merges and filters trips/routes, then filters stop_times accordingly.
    Returns updated trips_df, stop_times_df.
    """

    # Merge route_short_name into trips
    routes_subset = routes_df[["route_id", "route_short_name"]]
    trips_df = trips_df.merge(routes_subset, on="route_id", how="left")

    # Apply Route Filtering
    if FILTER_ROUTE_SHORT_NAMES:
        blocks_for_selected_routes = (
            trips_df[trips_df["route_short_name"].isin(FILTER_ROUTE_SHORT_NAMES)][
                "block_id"
            ]
            .dropna()
            .unique()
        )
        if len(blocks_for_selected_routes) == 0:
            print("No blocks found with the specified route short names.")
            return pd.DataFrame(), pd.DataFrame()

        trips_df = trips_df[trips_df["block_id"].isin(blocks_for_selected_routes)]

    # Apply Service ID Filtering
    if FILTER_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(FILTER_SERVICE_IDS)]

    # Filter stop_times to only include relevant trips
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    return trips_df, stop_times_df


def prepare_stop_times(trips_df, stop_times_df, stops_df):
    """
    Adds and formats columns in stop_times_df with scheduling and stop info.
    Returns the updated stop_times_df.
    """
    # If 'timepoint' does not exist, create a new column with 0.
    if "timepoint" not in stop_times_df.columns:
        stop_times_df["timepoint"] = 0
    else:
        # Convert to numeric, fill NaN with 0
        stop_times_df["timepoint"] = (
            pd.to_numeric(stop_times_df["timepoint"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # Merge essential trip columns into stop_times
    needed_trip_cols = ["trip_id", "block_id", "route_short_name", "direction_id"]
    stop_times_df = stop_times_df.merge(
        trips_df[needed_trip_cols], on="trip_id", how="left"
    )

    # Convert arrival/departure times to seconds and format
    stop_times_df["arrival_seconds"] = stop_times_df["arrival_time"].apply(
        time_to_seconds
    )
    stop_times_df["departure_seconds"] = stop_times_df["departure_time"].apply(
        time_to_seconds
    )
    stop_times_df["scheduled_time_hhmm"] = stop_times_df["departure_seconds"].apply(
        format_hhmm
    )

    # Merge in stop names
    stop_name_map = stops_df.set_index("stop_id")["stop_name"].to_dict()
    stop_times_df["stop_name"] = (
        stop_times_df["stop_id"].map(stop_name_map).fillna("Unknown Stop")
    )

    # Sort by block, trip, and stop_sequence
    stop_times_df = stop_times_df.dropna(subset=["block_id"])
    stop_times_df["stop_sequence"] = pd.to_numeric(
        stop_times_df["stop_sequence"], errors="coerce"
    )
    stop_times_df = stop_times_df.dropna(subset=["stop_sequence"])
    stop_times_df.sort_values(["block_id", "trip_id", "stop_sequence"], inplace=True)

    return stop_times_df


def export_blocks(stop_times_df):
    """
    Groups rows by block_id and exports each block to a separate Excel file.
    """
    all_blocks = stop_times_df["block_id"].unique()
    print(f"Found {len(all_blocks)} blocks to export.\n")

    for block_id in all_blocks:
        block_subset = stop_times_df[stop_times_df["block_id"] == block_id].copy()
        if block_subset.empty:
            continue

        # For each trip_id within this block, find earliest departure
        first_departures = (
            block_subset.groupby("trip_id")["departure_seconds"]
            .min()
            .reset_index(name="trip_start_seconds")
        )
        first_departures["trip_start_hhmm"] = first_departures[
            "trip_start_seconds"
        ].apply(format_hhmm)
        block_subset = block_subset.merge(first_departures, on="trip_id", how="left")

        block_subset["Trip Start Time"] = block_subset["trip_start_hhmm"]

        # Select and rename columns for clarity
        out_cols = [
            "block_id",
            "route_short_name",
            "direction_id",
            "trip_id",
            "Trip Start Time",
            "stop_sequence",
            "timepoint",
            "stop_id",
            "stop_name",
            "scheduled_time_hhmm",
        ]
        final_df = block_subset[out_cols].copy()
        final_df.rename(
            columns={
                "block_id": "Block ID",
                "route_short_name": "Route",
                "direction_id": "Direction",
                "trip_id": "Trip ID",
                "stop_sequence": "Stop Sequence",
                "timepoint": "Timepoint",
                "stop_id": "Stop ID",
                "stop_name": "Stop Name",
                "scheduled_time_hhmm": "Scheduled Time",
            },
            inplace=True,
        )

        # Insert placeholders
        final_df["Actual Time"] = MISSING_TIME
        final_df["Boardings"] = MISSING_VALUE
        final_df["Alightings"] = MISSING_VALUE
        final_df["Comments"] = MISSING_VALUE

        # Reorder columns to place 'Timepoint' after 'Stop Sequence'
        final_df = final_df[
            [
                "Block ID",
                "Route",
                "Direction",
                "Trip ID",
                "Trip Start Time",
                "Stop Sequence",
                "Timepoint",
                "Stop ID",
                "Stop Name",
                "Scheduled Time",
                "Actual Time",
                "Boardings",
                "Alightings",
                "Comments",
            ]
        ]

        final_df.sort_values(
            by=["Trip Start Time", "Trip ID", "Stop Sequence"], inplace=True
        )

        filename = f"block_{block_id}_schedule_printable.xlsx"
        output_path = os.path.join(BASE_OUTPUT_PATH, filename)
        export_to_excel(final_df, output_path)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point – orchestrates GTFS load, filter, prep, export."""
    # --------------------------------------------------------------
    # Configure logging *inside* main (repo style)
    # --------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logging.info("========================================================")
    logging.info("GTFS Block Schedule Printable Generator")
    logging.info("Input GTFS Folder: %s", GTFS_FOLDER_PATH)
    logging.info("Output Folder:     %s", BASE_OUTPUT_PATH)
    if FILTER_ROUTE_SHORT_NAMES:
        logging.info("Filtering for Routes: %s", FILTER_ROUTE_SHORT_NAMES)
    if FILTER_SERVICE_IDS:
        logging.info("Filtering for Service IDs: %s", FILTER_SERVICE_IDS)

    try:
        gtfs_data = load_gtfs_data(
            gtfs_folder_path=GTFS_FOLDER_PATH,
            files=REQUIRED_GTFS_FILES,
            dtype=str,
        )

        trips_df = gtfs_data["trips"]
        stop_times_df = gtfs_data["stop_times"]
        stops_df = gtfs_data["stops"]
        routes_df = gtfs_data["routes"]

        trips_df, stop_times_df = filter_data(trips_df, stop_times_df, routes_df)
        if trips_df.empty or stop_times_df.empty:
            logging.warning("No data remains after filtering – no files generated.")
            return

        prepared = prepare_stop_times(trips_df, stop_times_df, stops_df)
        if prepared.empty:
            logging.warning("No data remains after preparation – no files generated.")
            return

        export_blocks(prepared)
        logging.info("Script finished successfully.")

    except (OSError, ValueError, RuntimeError) as err:
        logging.error("%s", err)
    except Exception as err:  # catch-all for unforeseen issues
        logging.exception("Unexpected error: %s", err)
    finally:
        logging.info("Exiting script.")


if __name__ == "__main__":
    main()
