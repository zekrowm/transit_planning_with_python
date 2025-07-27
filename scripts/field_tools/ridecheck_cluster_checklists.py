"""Generates printable Excel checklists of GTFS stop times by custom stop clusters and schedule types.

The script is intended for field auditing and service verification. It:
    1. Validates the presence of required GTFS text files.
    2. Loads the data into `pandas` DataFrames.
    3. Optionally swaps `stop_id` for `stop_code` so downstream logic is agnostic.
    4. Slices trips by calendar service days and user-defined clusters.
    5. Writes one Excel workbook per *cluster × schedule × time-window* to
       `BASE_OUTPUT_PATH`, pre-formatted with placeholders for actual arrival /
       departure times, bus numbers, and comments.

Typical Usage
- ArcGIS Pro Python window
- Jupyter Notebook
- Plain command line

Outputs:
    - One Excel file per cluster × schedule × time window to BASE_OUTPUT_PATH
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
BASE_OUTPUT_PATH = r"\\your_file_path\here\\"

# Input file paths to load GTFS files with specified dtypes
BASE_INPUT_PATH = r"\\your_file_path\here\\"

# Which field to use for clustering filters, 'stop_id' or 'stop_code'
STOP_IDENTIFIER_FIELD = "stop_code"  # or 'stop_id'

# Define columns to read as strings
DTYPE_DICT = {
    "stop_id": str,
    "trip_id": str,
    "route_id": str,
    "service_id": str,
    # Add other ID fields as needed
}

# List of required GTFS files
GTFS_FILES = ["trips.txt", "stop_times.txt", "routes.txt", "stops.txt", "calendar.txt"]

# Define clusters with stop IDs or stop_codes (depending on STOP_IDENTIFIER_FIELD)
# Format: {'Cluster Name': ['identifier1', 'identifier2', ...]}
CLUSTERS = {
    "Your Cluster 1": ["1", "2", "3"],  # If using 'stop_id', these are stop_ids
    "Your Cluster 2": ["4", "5", "6"],  # If using 'stop_code', these must be stop_codes
    "Your Cluster 3": ["7", "8", "9", "10"],
}

# Define schedule types and corresponding days in the calendar
# Format: {'Schedule Type': ['day1', 'day2', ...]}
SCHEDULE_TYPES = {
    "Weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "Saturday": ["saturday"],
    "Sunday": ["sunday"],
    # 'Friday': ['friday'],  # Uncomment if you have a unique Friday schedule
}

# Time windows for filtering trips
# Format: {'Schedule Type': {'Time Window Name': ('Start Time', 'End Time')}}
# Times should be in 'HH:MM' 24-hour format
TIME_WINDOWS = {
    "Weekday": {
        "morning": ("06:00", "09:59"),
        "afternoon": ("14:00", "17:59"),
        # 'evening': ('18:00', '21:59'),  # Add as needed
    },
    "Saturday": {
        "midday": ("10:00", "13:59"),
        # Add more time windows for Saturday if needed
    },
    # 'Sunday': {  # Uncomment and customize for Sunday if needed
    #     'morning': ('08:00', '11:59'),
    #     'afternoon': ('12:00', '15:59'),
    # },
}

# Optional: List of route_short_name strings to be bolded in the Excel output.
# Ensure these are strings, matching the type in your routes.txt -> route_short_name.
SPECIAL_ROUTES = [
    str(x) for x in (101, 202, 303, 404, 505, 606, 707)
]  # Example list, customize as needed

# =============================================================================
# FUNCTIONS
# =============================================================================


def validate_input_directory(base_input_path: str, gtfs_files: List[str]) -> None:
    """Verify that *all* required GTFS files exist in the given directory.

    Args:
        base_input_path: Absolute or relative path that should contain the
            GTFS text files.
        gtfs_files: File names that **must** be found in
            ``base_input_path`` (e.g. ``["trips.txt", ...]``).

    Raises:
        FileNotFoundError: If the directory itself or any file in
            ``gtfs_files`` is missing.
    """
    if not os.path.exists(base_input_path):
        raise FileNotFoundError(f"The input directory {base_input_path} does not exist.")

    for file_name in gtfs_files:
        file_path = os.path.join(base_input_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The required GTFS file {file_name} does not exist in {base_input_path}."
            )


def create_output_directory(base_output_path: str) -> None:
    """Create ``base_output_path`` if it does not already exist.

    Args:
        base_output_path: Folder into which Excel files will be written.

    Notes:
        *The function is idempotent*: calling it repeatedly is safe.
    """
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)


def load_gtfs_data(
    base_input_path: str, dtype_dict: Dict[str, type]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read required GTFS tables into memory.

    Args:
        base_input_path: Directory containing the GTFS text files.
        dtype_dict: Mapping of column names to dtypes passed to
            :pymeth:`pandas.read_csv`.

    Returns:
        A 5-tuple of DataFrames in the order
        ``(trips, stop_times, routes, stops, calendar)``.

    Raises:
        FileNotFoundError: If any of the required files are not present.
        pandas.errors.ParserError: If a file cannot be parsed as CSV.
    """
    trips = pd.read_csv(os.path.join(base_input_path, "trips.txt"), dtype=dtype_dict)
    stop_times = pd.read_csv(os.path.join(base_input_path, "stop_times.txt"), dtype=dtype_dict)
    routes = pd.read_csv(os.path.join(base_input_path, "routes.txt"), dtype=dtype_dict)
    stops = pd.read_csv(os.path.join(base_input_path, "stops.txt"), dtype=dtype_dict)
    calendar = pd.read_csv(os.path.join(base_input_path, "calendar.txt"), dtype=dtype_dict)
    return trips, stop_times, routes, stops, calendar


def apply_stop_identifier_mode(stops_df: pd.DataFrame, stop_identifier_field: str) -> None:
    """Align `stops_df` so downstream code can always key on ``stop_id``.

    If the user elects to use ``stop_code`` as the primary identifier, this
    function overwrites (or creates) the ``stop_id`` column with the contents
    of ``stop_code``.

    Args:
        stops_df: The *stops.txt* table.
        stop_identifier_field: Either ``"stop_id"`` or ``"stop_code"``.

    Raises:
        ValueError: If ``stop_identifier_field`` is not one of the two allowed
            values **or** ``stop_code`` is missing when requested.
    """
    if stop_identifier_field not in ["stop_id", "stop_code"]:
        raise ValueError("STOP_IDENTIFIER_FIELD must be 'stop_id' or 'stop_code'.")

    if stop_identifier_field == "stop_code":
        if "stop_code" not in stops_df.columns:
            raise ValueError("No 'stop_code' column found in stops data.")
        # Overwrite stops['stop_id'] with the values from stop_code
        stops_df["stop_id"] = stops_df["stop_code"]


def fix_time_format(time_str: str) -> str:
    """Normalize GTFS HH:MM[:SS] strings to *24-hour* ``"HH:MM"``.

    The GTFS spec permits hours ≥ 24 for post-midnight trips; values in that
    range are wrapped back into 0–23.

    Args:
        time_str: A time string such as ``"27:15:00"`` or ``"05:07"``.

    Returns:
        The same instant expressed as ``"HH:MM"`` (two-digit padding).

    Raises:
        ValueError: If ``time_str`` is not parseable as ``H+:MM[:SS]``.
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    if hours >= 24:
        hours -= 24
    return f"{hours:02}:{minutes:02}"


def export_to_excel(df: pd.DataFrame, output_file: str) -> None:
    """Write a DataFrame to a formatted Excel workbook.

    Formatting rules
    ----------------
    * Header row left-aligned.
    * Entire row **bold** where ``route_short_name`` is in
      :pydata:`SPECIAL_ROUTES`.
    * Column widths auto-sized based on max cell length.

    Args:
        df: Data to export.
        output_file: Full path of the ``*.xlsx`` file to create.

    Raises:
        PermissionError: If the target file is open or write-protected.
    """
    bold_font = Font(bold=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")  # Explicit sheet_name
        ws = writer.sheets["Sheet1"]

        # left-align header row
        for cell in ws[1]:
            cell.alignment = Alignment(horizontal="left")

        # find 1-based column index for route_short_name
        # Ensure 'route_short_name' is in df.columns
        if "route_short_name" in df.columns:
            route_col_idx = df.columns.get_loc("route_short_name") + 1

            # bold entire row if route_short_name is in SPECIAL_ROUTES
            # Check if SPECIAL_ROUTES is defined and not empty
            if "SPECIAL_ROUTES" in globals() and SPECIAL_ROUTES:
                for row_idx in range(2, ws.max_row + 1):  # Iterate from the first data row
                    # Get the cell in the route_short_name column for the current row
                    route_cell = ws.cell(row=row_idx, column=route_col_idx)
                    if str(route_cell.value) in SPECIAL_ROUTES:
                        for cell_in_row in ws[row_idx]:  # ws[row_idx] gets all cells in that row
                            cell_in_row.font = bold_font
        else:
            print(
                "Warning: 'route_short_name' column not found. Cannot apply special route bolding."
            )

        # auto-fit column widths
        for idx, col_name in enumerate(df.columns, 1):  # Use col_name from df.columns
            # Get max length of data in the column
            max_len_data = 0
            if not df[col_name].empty:  # Check if column is not empty
                max_len_data = df[col_name].astype(str).map(len).max()

            # Max length is the greater of header length or data length
            max_len = max(max_len_data, len(str(col_name))) + 2  # Add a little padding
            ws.column_dimensions[get_column_letter(idx)].width = max_len


def process_cluster_data(
    cluster_data: pd.DataFrame,
    stops_df: pd.DataFrame,
    cluster_name: str,
    schedule_name: str,
    base_output_path: str,
    time_windows: Optional[Dict[str, Dict[str, Tuple[str, str]]]] = None,
) -> None:
    """Transform and export a *cluster × schedule* slice.

    Workflow
    --------
    1. Normalize and sort times.
    2. Inject placeholder columns needed for field audits.
    3. Join readable stop names from *stops.txt*.
    4. Drop internal GTFS columns not useful to inspectors.
    5. Emit:
        * A full-day workbook.
        * Optional time-window workbooks (e.g. *morning*, *afternoon*).

    Args:
        cluster_data: Subset of stop-times already filtered to the cluster.
        stops_df: The *stops.txt* table (for name lookup).
        cluster_name: Friendly cluster label used in filenames.
        schedule_name: Calendar schedule label used in filenames.
        base_output_path: Folder for output files.
        time_windows: Optional mapping
            ``{"Weekday": {"morning": ("06:00","09:59"), ...}, ...}``.

    Side Effects:
        Writes one or more ``*.xlsx`` files and logs progress to ``stdout``.
    """
    # ensure we have a copy to avoid SettingWithCopyWarning
    cluster_data = cluster_data.copy()

    # --- normalize times ---
    cluster_data["arrival_time"] = cluster_data["arrival_time"].apply(fix_time_format)
    cluster_data["departure_time"] = cluster_data["departure_time"].apply(fix_time_format)
    cluster_data["arrival_time"] = cluster_data["arrival_time"].astype(str)
    cluster_data["departure_time"] = cluster_data["departure_time"].astype(str)

    # --- sort ---
    # Create a temporary column for proper time sorting, handling potential errors if times are not valid
    try:
        cluster_data["arrival_sort"] = pd.to_datetime(
            cluster_data["arrival_time"], format="%H:%M", errors="raise"
        )
        cluster_data = cluster_data.sort_values("arrival_sort").drop(columns=["arrival_sort"])
    except ValueError as e:
        print(
            f"Warning: Could not sort by arrival_time due to invalid time format for {cluster_name} on {schedule_name}. Error: {e}"
        )
        # Proceed without sorting if conversion fails, or handle more gracefully

    # --- placeholders (no act_block) ---
    cluster_data.insert(cluster_data.columns.get_loc("arrival_time") + 1, "act_arrival", "________")
    cluster_data.insert(
        cluster_data.columns.get_loc("departure_time") + 1, "act_departure", "________"
    )
    # Ensure 'sequence_long' column exists before trying to use it for loc
    if "sequence_long" in cluster_data.columns:
        cluster_data.loc[cluster_data["sequence_long"] == "start", "act_arrival"] = "__XXXX__"
        cluster_data.loc[cluster_data["sequence_long"] == "last", "act_departure"] = "__XXXX__"
    else:
        print(
            f"Warning: 'sequence_long' column not found in cluster_data for {cluster_name}. Cannot set start/last placeholders."
        )

    cluster_data["bus_number"] = "________"
    cluster_data["comments"] = "________________"

    # --- merge stop names (using stop_id) ---
    # Ensure 'stop_id' is the correct merge key and present in both DataFrames
    if "stop_id" in cluster_data.columns and "stop_id" in stops_df.columns:
        cluster_data = pd.merge(
            cluster_data,
            stops_df[["stop_id", "stop_name"]],  # Only get stop_name
            on="stop_id",
            how="left",
        )
    else:
        print(
            f"Warning: 'stop_id' not found in cluster_data or stops_df for {cluster_name}. Cannot merge stop names."
        )

    # --- reorder columns, show stop_id ---
    first_cols = [
        "route_short_name",
        "trip_headsign",
        "stop_sequence",
        "sequence_long",
        "stop_id",  # Displaying stop_id as per revised request
        "stop_name",
        "arrival_time",
        "act_arrival",
        "departure_time",
        "act_departure",
        "block_id",  # block_id from GTFS is kept, but no act_block placeholder
        "bus_number",
        "comments",
    ]
    # Ensure all columns in first_cols exist in cluster_data before reordering
    existing_first_cols = [col for col in first_cols if col in cluster_data.columns]
    missing_cols = [col for col in first_cols if col not in cluster_data.columns]
    if missing_cols:
        print(
            f"Warning: The following expected columns are missing for reordering: {missing_cols} in {cluster_name}"
        )

    other_cols = [c for c in cluster_data.columns if c not in existing_first_cols]
    cluster_data = cluster_data[existing_first_cols + other_cols]

    # --- drop internal/unnecessary columns (similar to old script, ensuring stop_code is dropped if present) ---
    columns_to_drop = [
        "shape_dist_traveled",
        "shape_id",
        "route_id",
        "service_id",
        "trip_id",
        "timepoint",
        "direction_id",
        "stop_headsign",
        "pickup_type",
        "drop_off_type",
        "wheelchair_accessible",
        "bikes_allowed",
        "trip_short_name",
        "stop_code",  # Add stop_code here to drop it, if it exists and stop_id is preferred
    ]
    # Drop only columns that actually exist in the DataFrame
    actual_columns_to_drop = [col for col in columns_to_drop if col in cluster_data.columns]
    cluster_data = cluster_data.drop(columns=actual_columns_to_drop, errors="ignore")

    # --- helper to prepend sample row 5 min before first trip ---
    def prepend_sample(df_to_sample: pd.DataFrame) -> pd.DataFrame:
        if df_to_sample.empty:
            print(
                f"Warning: Dataframe for {cluster_name} on {schedule_name} is empty. Cannot prepend sample row."
            )
            return df_to_sample  # Return empty df if no data to sample from

        sample = {col: "" for col in df_to_sample.columns}
        try:
            # use iloc to grab the first row by position
            first_arr_str = df_to_sample["arrival_time"].iloc[0]
            first_dep_str = df_to_sample["departure_time"].iloc[0]

            first_arr = pd.to_datetime(first_arr_str, format="%H:%M")
            first_dep = pd.to_datetime(first_dep_str, format="%H:%M")

            sample_arr_dt = first_arr - pd.Timedelta(minutes=5)
            sample_dep_dt = first_dep - pd.Timedelta(minutes=5)

            # Handle potential day rollover for sample time if first trip is close to midnight
            sample_arr = sample_arr_dt.strftime("%H:%M")
            sample_dep = sample_dep_dt.strftime("%H:%M")

            update_dict = {
                "route_short_name": "SAMPLE",
                "trip_headsign": "Sample Trip",
                "arrival_time": sample_arr,
                "act_arrival": sample_arr,  # Fill in actuals for sample
                "departure_time": sample_dep,
                "act_departure": sample_dep,  # Fill in actuals for sample
                "comments": "Please use 24-hour HH:MM format",
            }
            # Ensure keys in update_dict exist as columns in sample before updating
            for key, value in update_dict.items():
                if key in sample:
                    sample[key] = value
                else:
                    print(
                        f"Warning: Column '{key}' not found in sample row for prepending. Skipping update for this key."
                    )

            return pd.concat([pd.DataFrame([sample]), df_to_sample], ignore_index=True)

        except (
            IndexError,
            TypeError,
            ValueError,
        ) as e:  # Catch errors if times are bad or df is unexpectedly empty after check
            print(
                f"Error creating sample row for {cluster_name} on {schedule_name}: {e}. Proceeding without sample row."
            )
            return df_to_sample

    # --- export full dataset ---
    # Ensure base_output_path is a directory that exists (or created by create_output_directory)
    if not os.path.exists(base_output_path):
        print(f"Error: Output directory {base_output_path} does not exist. Cannot save Excel file.")
        return  # Exit if path is invalid

    output_file_full_path = os.path.join(
        base_output_path, f"{cluster_name}_{schedule_name}_data.xlsx"
    )
    if not cluster_data.empty:
        df_full_with_sample = prepend_sample(
            cluster_data.copy()
        )  # Use .copy() for safety before prepending
        export_to_excel(df_full_with_sample, output_file_full_path)
        print(
            f"Processed and exported {cluster_name} ({schedule_name}) to {output_file_full_path}."
        )
    else:
        print(
            f"No data to export for {cluster_name} ({schedule_name}). Skipping full dataset export."
        )

    # --- export each time window subset ---
    if time_windows and schedule_name in time_windows and not cluster_data.empty:
        for win_name, (start_s, end_s) in time_windows[schedule_name].items():
            try:
                st = pd.to_datetime(start_s, format="%H:%M").time()
                et = pd.to_datetime(end_s, format="%H:%M").time()
                # Ensure 'arrival_time' is in correct format for conversion
                atimes = pd.to_datetime(
                    cluster_data["arrival_time"], format="%H:%M", errors="coerce"
                ).dt.time

                # Filter out NaT from atimes if errors were coerced
                valid_times_mask = pd.notnull(atimes)
                subset = cluster_data[
                    valid_times_mask
                    & (atimes[valid_times_mask] >= st)
                    & (atimes[valid_times_mask] <= et)
                ]

                if subset.empty:
                    print(f"  No {win_name} data for {cluster_name} ({schedule_name}).")
                    continue

                output_file_window_path = os.path.join(
                    base_output_path,
                    f"{cluster_name}_{schedule_name}_{win_name}_data.xlsx",
                )
                df_sub_with_sample = prepend_sample(subset.copy())  # Use .copy() for safety
                export_to_excel(df_sub_with_sample, output_file_window_path)
                print(
                    f"  Exported {win_name} window for {cluster_name} ({schedule_name}) to {output_file_window_path}."
                )
            except ValueError as e:
                print(
                    f"  Error processing time window {win_name} for {cluster_name} ({schedule_name}): {e}. Skipping."
                )
    elif cluster_data.empty:
        print(
            f"Skipping time window processing for {cluster_name} ({schedule_name}) as main dataset is empty."
        )


def generate_gtfs_checklists() -> None:
    """Orchestrate the end-to-end checklist generation process.

    High-level steps:
        1. Validate inputs and create outputs folder.
        2. Load GTFS data.
        3. Loop over each *schedule* and *cluster*,
           delegating to :pyfunc:`process_cluster_data`.

    The function prints status messages but returns nothing.
    """
    # 1) Validate input directory
    validate_input_directory(BASE_INPUT_PATH, GTFS_FILES)

    # 2) Create output directory
    create_output_directory(BASE_OUTPUT_PATH)

    # 3) Load GTFS data
    trips, stop_times, routes, stops, calendar = load_gtfs_data(BASE_INPUT_PATH, DTYPE_DICT)

    # 4) Potentially replace stop_id with stop_code
    apply_stop_identifier_mode(stops, STOP_IDENTIFIER_FIELD)

    # Ensure stop_id is a string in stops
    stops["stop_id"] = stops["stop_id"].astype(str)

    # 5) Process each schedule type
    for schedule_name, days in SCHEDULE_TYPES.items():
        print(f"Processing schedule: {schedule_name}")

        # Filter calendar by days
        service_mask = calendar[days].astype(bool).all(axis=1)
        relevant_service_ids = calendar.loc[service_mask, "service_id"]

        # Filter trips by relevant service IDs
        trips_filtered = trips[trips["service_id"].isin(relevant_service_ids)]

        if trips_filtered.empty:
            print(f"No trips found for {schedule_name} schedule. Skipping.")
            continue

        # Merge with stop_times and routes
        merged_data = pd.merge(stop_times, trips_filtered, on="trip_id")
        merged_data = pd.merge(merged_data, routes[["route_id", "route_short_name"]], on="route_id")

        # Ensure stop_id is string in merged_data
        merged_data["stop_id"] = merged_data["stop_id"].astype(str)

        # Create sequence_long column
        merged_data["sequence_long"] = "middle"
        merged_data.loc[merged_data["stop_sequence"] == 1, "sequence_long"] = "start"
        max_sequence = merged_data.groupby("trip_id")["stop_sequence"].transform("max")
        merged_data.loc[merged_data["stop_sequence"] == max_sequence, "sequence_long"] = "last"

        # 6) Process each cluster
        for cluster_name, cluster_stop_ids in CLUSTERS.items():
            print(f"Processing cluster: {cluster_name} for {schedule_name} schedule")

            # Ensure cluster_stop_ids are strings
            cluster_stop_ids = [str(sid) for sid in cluster_stop_ids]

            # Filter merged data by cluster's stops
            cluster_data = merged_data[merged_data["stop_id"].isin(cluster_stop_ids)]

            if cluster_data.empty:
                print(f"No data found for {cluster_name} on {schedule_name} schedule. Skipping.")
                continue

            # Transform and export data
            process_cluster_data(
                cluster_data,
                stops,
                cluster_name,
                schedule_name,
                BASE_OUTPUT_PATH,
                time_windows=TIME_WINDOWS,
            )

    print("All clusters and schedules have been processed and exported.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Script entry point: simply calls :pyfunc:`generate_gtfs_checklists`."""
    generate_gtfs_checklists()


if __name__ == "__main__":
    main()
