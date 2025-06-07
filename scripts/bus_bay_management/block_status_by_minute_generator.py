"""
Generates minute-by-minute transit block status timelines from GTFS data.

Processes GTFS files to determine operational statuses (e.g., DWELL, LAYOVER, TRAVELING)
of transit blocks at one-minute intervals, producing Excel reports for further analysis.

Typically used interactively within a Jupyter notebook or ArcGIS Pro environment,
though direct execution via the command line is also supported.
"""

import logging
import os

import pandas as pd

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

GTFS_FOLDER_PATH = r"\\your_GTFS_folder_path\here\\"
BLOCK_OUTPUT_FOLDER = r"\\your_output_folder_path\here\\"

DEFAULT_HOURS = 26
TIME_INTERVAL_MINUTES = 1

DWELL_THRESHOLD = 3  # minutes
LAYOVER_THRESHOLD = 20  # minutes
MAX_TRIPS_PER_BLOCK = 150

CALENDAR_SERVICE_IDS = ["3"]

ROUTE_SHORTNAME_FILTER = []
STOP_ID_FILTER = []
STOP_CODE_FILTER = []

CLUSTER_DEFINITIONS = {
    "Metro": {
        "stops": ["2956", "2955", "65", "3295", "2957", "66", "3296", "64", "63"],
        "overflow_bays": ["OverflowA", "OverflowB", "OverflowC"],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
    "Park & Ride": {
        "stops": ["3882", "3881", "1880"],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
    "Metro North": {
        "stops": ["2832", "2373"],
        "overflow_bays": [],
        "two_bay_stops": [],
        "three_bay_stops": [],
    },
}

BUS_STOP_CLUSTERS_STEP1 = [
    {"name": name, "stops": info["stops"]} for name, info in CLUSTER_DEFINITIONS.items()
]

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------------------
# VALIDATION UTILITY
# --------------------------------------------------------------------------------------------------


def validate_folders(input_path, output_path):
    """
    Check that the input folder exists, and ensure the output folder
    is created if it does not already exist.
    """
    if not os.path.isdir(input_path):
        raise NotADirectoryError(
            f"Input path does not exist or is not a directory: {input_path}"
        )
    os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------


def time_to_minutes(time_str):
    """
    Convert 'HH:MM:SS' or 'HH:MM' to integer minutes (e.g. "26:30:00" -> 1590).
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 60 + minutes + (seconds // 60)


def minutes_to_hhmm(total_minutes):
    """
    Convert integer minutes into 'HH:MM' (e.g. 1590 -> "26:30").
    """
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def mark_first_and_last_stops(df_in):
    """
    Mark each stop in the trip as the first or last using boolean columns.
    """
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    df_out["is_first_stop"] = False
    df_out["is_last_stop"] = False

    for _, group in df_out.groupby("trip_id"):
        min_seq_idx = group["stop_sequence"].idxmin()
        max_seq_idx = group["stop_sequence"].idxmax()
        df_out.loc[min_seq_idx, "is_first_stop"] = True
        df_out.loc[max_seq_idx, "is_last_stop"] = True

    return df_out


def find_cluster(stop_id, bus_stop_clusters):
    """
    Given a stop_id, return the named cluster (if any) or None if not found.
    """
    for cluster_item in bus_stop_clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# --------------------------------------------------------------------------------------------------
# BRIDGING LOGIC REFACTOR
# --------------------------------------------------------------------------------------------------


def _status_for_same_trip(minute, stop_info):
    """
    Handle same-trip logic. stop_info is a tuple:
      (arrival, departure, stop_id, stop_name, trip_id,
       is_first, is_last, stop_seq, timepoint)
    """
    (arr, dep, s_id, s_name, t_id, is_first, is_last, s_seq, t_val) = stop_info

    # Check for arrive/depart/dwell
    if minute == arr and is_last:
        return (
            "ARRIVE",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if minute == dep and is_first:
        return (
            "DEPART",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if arr == dep and minute == arr:
        return (
            "ARRIVE/DEPART",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if arr < minute < dep:
        return (
            "DWELL",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    return None


def _status_for_different_trip(
    dep, next_arr, current_stop_id, current_stop_name, next_stop_id, bus_stop_clusters
):
    """
    Sub-logic for bridging between two different trips,
    determining LAYOVER, DEADHEAD, etc.
    """
    gap = next_arr - dep
    current_cluster = find_cluster(current_stop_id, bus_stop_clusters)
    next_cluster = find_cluster(next_stop_id, bus_stop_clusters)

    same_cluster = current_cluster and next_cluster and current_cluster == next_cluster
    same_stop = current_stop_id == next_stop_id

    if same_stop or same_cluster:
        # Use threshold-based logic
        if gap <= DWELL_THRESHOLD:
            return ("DWELL", current_stop_id, current_stop_name)
        if gap > LAYOVER_THRESHOLD:
            return ("LONG BREAK", current_stop_id, current_stop_name)
        return ("LAYOVER", current_stop_id, current_stop_name)

    # Otherwise, different cluster => "DEADHEAD"
    return ("DEADHEAD", current_stop_id, current_stop_name)


def get_status_for_minute(minute, stop_times_sequence, bus_stop_clusters):
    """
    Determine the block's status at a specific 'minute'.

    Returns tuple:
       (status, stop_id, stop_name, arrival_str, departure_str,
        trip_id_for_status, stop_sequence, timepoint_value)
    """
    if not stop_times_sequence:
        return ("EMPTY", None, None, None, None, None, None, 0)

    for i, item in enumerate(stop_times_sequence):
        same_trip_result = _status_for_same_trip(minute, item)
        if same_trip_result:
            return same_trip_result

        # If no same-trip match, check bridging
        arr, dep = item[0], item[1]
        if i < len(stop_times_sequence) - 1:
            next_item = stop_times_sequence[i + 1]
            next_arr = next_item[0]
            next_trip_id = next_item[4]  # we do need to return it
            next_stop_id = next_item[2]
            stop_id = item[2]
            stop_name = item[3]
            trip_id = item[4]
            stop_seq = item[7]
            t_val = item[8]

            if dep < minute < next_arr:
                # Check if same or different trip
                if trip_id == next_trip_id:
                    return (
                        "TRAVELING BETWEEN STOPS",
                        None,
                        None,
                        None,
                        None,
                        trip_id,
                        None,
                        0,
                    )

                # Different trip => dwell/layover/deadhead
                new_status, fill_stop_id, fill_stop_name = _status_for_different_trip(
                    dep, next_arr, stop_id, stop_name, next_stop_id, bus_stop_clusters
                )
                return (
                    new_status,
                    fill_stop_id,
                    fill_stop_name,
                    minutes_to_hhmm(arr),
                    minutes_to_hhmm(dep),
                    next_trip_id,
                    stop_seq,
                    t_val,
                )

    return ("EMPTY", None, None, None, None, None, None, 0)


# --------------------------------------------------------------------------------------------------
# MAIN BLOCK PROCESSING
# --------------------------------------------------------------------------------------------------


def check_for_overlapping_trips(block_subset, block_id):
    """
    Print a warning if any trips within this block overlap in time.
    """
    trip_times = []
    for trip_id, group in block_subset.groupby("trip_id"):
        start_val = group["arrival_min"].min()
        end_val = group["departure_min"].max()
        trip_times.append((trip_id, start_val, end_val))

    trip_times.sort(key=lambda x: x[1])
    for i in range(len(trip_times) - 1):
        trip1_id, start1, end1 = trip_times[i]
        for j in range(i + 1, len(trip_times)):
            trip2_id, start2, end2 = trip_times[j]
            # If they intersect in any way
            if start2 <= end1 and start1 <= end2:
                print(
                    f"WARNING: Overlapping trips in block {block_id}: "
                    f"Trip {trip1_id} ({start1}–{end1}) overlaps with "
                    f"{trip2_id} ({start2}–{end2})"
                )


def fill_stop_ids_for_dwell_layover_loading(df_in):
    """
    For rows where status is DWELL, LAYOVER, or LOADING (Stop ID is empty),
    fill in the last known stop_id from the same block to make final
    spreadsheet more readable.
    """
    df_out = df_in.copy()
    last_stop_id = None
    last_stop_name = None
    last_stop_seq = None
    last_arr = None
    last_dep = None
    last_trip_id = None

    for idx in df_out.index:
        status = df_out.at[idx, "Status"]
        stop_id = df_out.at[idx, "Stop ID"]
        if stop_id:
            # Update "last known" stop info
            last_stop_id = stop_id
            last_stop_name = df_out.at[idx, "Stop Name"]
            last_stop_seq = df_out.at[idx, "Stop Sequence"]
            last_arr = df_out.at[idx, "Arrival Time"]
            last_dep = df_out.at[idx, "Departure Time"]
            last_trip_id = df_out.at[idx, "Trip ID"]
        else:
            if status in ["DWELL", "LAYOVER", "LOADING"]:
                if last_stop_id is not None:
                    df_out.at[idx, "Stop ID"] = last_stop_id
                if last_stop_name is not None:
                    df_out.at[idx, "Stop Name"] = last_stop_name
                if last_stop_seq is not None:
                    df_out.at[idx, "Stop Sequence"] = last_stop_seq
                if last_arr is not None:
                    df_out.at[idx, "Arrival Time"] = last_arr
                if last_dep is not None:
                    df_out.at[idx, "Departure Time"] = last_dep
                if last_trip_id is not None:
                    df_out.at[idx, "Trip ID"] = last_trip_id
    return df_out


def _create_trips_summary(block_subset):
    """
    Helper to build trip summaries for a block.
    Returns a list of dictionaries with trip info and sorted stop times.
    """
    trips_summary = []
    for trip_id, trip_df in block_subset.groupby("trip_id"):
        trip_df_sorted = trip_df.sort_values("stop_sequence")
        start_time = trip_df_sorted["arrival_min"].min()
        end_time = trip_df_sorted["departure_min"].max()

        stop_times_sequence = []
        for _, row in trip_df_sorted.iterrows():
            t_val = row.get("timepoint", 0)
            stop_times_sequence.append(
                (
                    row["arrival_min"],
                    row["departure_min"],
                    row["stop_id"],
                    row["stop_name"],
                    row["trip_id"],
                    row["is_first_stop"],
                    row["is_last_stop"],
                    row["stop_sequence"],
                    t_val,
                )
            )

        route_id = trip_df_sorted.iloc[0]["route_id"]
        direction_id = trip_df_sorted.iloc[0]["direction_id"]
        trips_summary.append(
            {
                "trip_id": trip_id,
                "start": start_time,
                "end": end_time,
                "stop_times_sequence": stop_times_sequence,
                "route_id": route_id,
                "direction_id": direction_id,
            }
        )

    trips_summary.sort(key=lambda x: x["start"])
    return trips_summary


def _status_for_active_trips(minute, active_trips, bus_stop_clusters):
    """
    Among all 'active' trips for the given minute, determine the single chosen status.
    """
    candidate_info = []
    for trip_obj in active_trips:
        status_tuple = get_status_for_minute(
            minute, trip_obj["stop_times_sequence"], bus_stop_clusters
        )
        candidate_info.append((trip_obj, status_tuple))

    # Filter out actual "EMPTY" statuses
    valid_candidates = [
        (trip_obj, stat) for (trip_obj, stat) in candidate_info if stat[0] != "EMPTY"
    ]
    if not valid_candidates:
        # All were "EMPTY"
        return None, ("EMPTY", None, None, None, None, None, None, 0)

    if len(valid_candidates) == 1:
        return valid_candidates[0]

    # Tie-break if multiple
    def candidate_sort_key(item):
        stat = item[1]
        stop_seq = stat[6] if stat[6] else 999999
        timepoint_val = stat[7] if len(stat) == 8 else 0
        # Prefer timepoints, then lower stop_sequence
        return (timepoint_val == 0, stop_seq)

    valid_candidates.sort(key=candidate_sort_key)
    return valid_candidates[0]


def _row_for_inactive(minute, block_id, all_trips):
    """
    Return the dictionary row for an 'inactive' minute (or bridging).
    """
    # Identify trips just before and just after this minute
    prev_trip_info = None
    next_trip_info = None
    for trip_obj in all_trips:
        if trip_obj["end"] < minute:
            if (prev_trip_info is None) or (trip_obj["end"] > prev_trip_info["end"]):
                prev_trip_info = trip_obj
        if trip_obj["start"] > minute:
            if (next_trip_info is None) or (
                trip_obj["start"] < next_trip_info["start"]
            ):
                next_trip_info = trip_obj

    # Distinguish dwell, layover, or inactive
    if prev_trip_info and next_trip_info:
        gap = next_trip_info["start"] - prev_trip_info["end"]
        if gap <= DWELL_THRESHOLD:
            status = "DWELL"
        elif gap <= LAYOVER_THRESHOLD:
            status = "LAYOVER"
        else:
            status = "LONG BREAK"
    else:
        status = "INACTIVE"

    # If the very next minute is a start, label as LOADING
    if (
        next_trip_info
        and next_trip_info["start"] == (minute + 1)
        and status in ["DWELL", "LAYOVER"]
    ):
        status = "LOADING"

    return {
        "Timestamp": minutes_to_hhmm(minute),
        "Block": block_id,
        "Route": "",
        "Direction": "",
        "Trip ID": "",
        "Stop ID": "",
        "Stop Name": "",
        "Stop Sequence": "",
        "Arrival Time": "",
        "Departure Time": "",
        "Status": status,
        "Timepoint": 0,
    }


def _build_schedule_rows(trips_summary, timeline, block_id, bus_stop_clusters):
    """
    Build the final minute-by-minute schedule rows for one block.
    """
    rows = []
    for minute in timeline:
        # Which trips are active at this minute?
        possible_trips = [t for t in trips_summary if t["start"] <= minute <= t["end"]]
        if possible_trips:
            chosen_trip, chosen_status = _status_for_active_trips(
                minute, possible_trips, bus_stop_clusters
            )
            if chosen_trip:
                (
                    status,
                    stop_id,
                    stop_name,
                    arr_str,
                    dep_str,
                    trip_id_for_status,
                    stop_seq,
                    t_val,
                ) = chosen_status

                if status == "EMPTY":
                    # If an active trip is "EMPTY", consider it traveling
                    status = "TRAVELING BETWEEN STOPS"

                # Convert dwell/layover to LOADING if the next minute is DEPART
                if status in ["DWELL", "LAYOVER"]:
                    next_minute = minute + TIME_INTERVAL_MINUTES
                    if next_minute <= chosen_trip["end"]:
                        next_status = get_status_for_minute(
                            next_minute,
                            chosen_trip["stop_times_sequence"],
                            bus_stop_clusters,
                        )[0]
                        if next_status == "DEPART":
                            status = "LOADING"

                row = {
                    "Timestamp": minutes_to_hhmm(minute),
                    "Block": block_id,
                    "Route": chosen_trip["route_id"],
                    "Direction": chosen_trip["direction_id"],
                    "Trip ID": trip_id_for_status if trip_id_for_status else "",
                    "Stop ID": stop_id if stop_id else "",
                    "Stop Name": stop_name if stop_name else "",
                    "Stop Sequence": stop_seq if stop_seq else "",
                    "Arrival Time": arr_str if arr_str else "",
                    "Departure Time": dep_str if dep_str else "",
                    "Status": status,
                    "Timepoint": t_val,
                }
            else:
                # Means possible_trips existed, but all statuses were "EMPTY"
                # => treat as traveling
                row = {
                    "Timestamp": minutes_to_hhmm(minute),
                    "Block": block_id,
                    "Route": "",
                    "Direction": "",
                    "Trip ID": "",
                    "Stop ID": "",
                    "Stop Name": "",
                    "Stop Sequence": "",
                    "Arrival Time": "",
                    "Departure Time": "",
                    "Status": "TRAVELING BETWEEN STOPS",
                    "Timepoint": 0,
                }
        else:
            # Truly no active trips => bridging or inactive
            row = _row_for_inactive(minute, block_id, trips_summary)
        rows.append(row)
    return rows


def process_block(block_subset, block_id, timeline, bus_stop_clusters):
    """
    Generate a minute-by-minute schedule DataFrame for a single block.
    """
    trips_summary = _create_trips_summary(block_subset)
    rows = _build_schedule_rows(trips_summary, timeline, block_id, bus_stop_clusters)
    df = pd.DataFrame(rows)
    return fill_stop_ids_for_dwell_layover_loading(df)


# --------------------------------------------------------------------------------------------------
# STEP 1: GTFS -> Block Spreadsheets
# --------------------------------------------------------------------------------------------------


def _merge_and_filter_data(trips_df, stop_times_df, stops_df):
    """
    Merge trips and stops, filter by service_id, route, etc.
    Return a single merged DataFrame with arrival_min/departure_min.
    """
    # Filter by service_id if set
    if CALENDAR_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    # Convert times to minutes
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(
        time_to_minutes
    )
    # Keep only stop_times for the trips that survived
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    # Make sure stops_df has stop_code
    if "stop_code" not in stops_df.columns:
        stops_df["stop_code"] = None

    # Merge stop_times + trips
    merged_df = pd.merge(stop_times_df, trips_df, on="trip_id", how="left")

    # Merge with stops to get stop_name, stop_code, timepoint
    stops_merge_cols = ["stop_id", "stop_name", "stop_code"]
    if "timepoint" in stops_df.columns:
        stops_merge_cols.append("timepoint")
    merged_df = pd.merge(
        merged_df, stops_df[stops_merge_cols], on="stop_id", how="left"
    )

    # Mark first/last stops
    merged_df = mark_first_and_last_stops(merged_df)

    # Ensure 'timepoint' is numeric
    if "timepoint" not in merged_df.columns:
        merged_df["timepoint"] = 0
    else:
        merged_df["timepoint"] = (
            pd.to_numeric(merged_df["timepoint"], errors="coerce").fillna(0).astype(int)
        )

    # Force first/last to timepoint=2 if it was 0
    merged_df.loc[
        (merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0), "timepoint"
    ] = 2
    merged_df.loc[
        (merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0), "timepoint"
    ] = 2

    # Block-level filtering based on route_short_name, stop_id, stop_code
    if ROUTE_SHORTNAME_FILTER or STOP_ID_FILTER or STOP_CODE_FILTER:
        route_match = (
            merged_df["route_short_name"].isin(ROUTE_SHORTNAME_FILTER)
            if ROUTE_SHORTNAME_FILTER
            else pd.Series(False, index=merged_df.index)
        )
        stopid_match = (
            merged_df["stop_id"].astype(str).isin(STOP_ID_FILTER)
            if STOP_ID_FILTER
            else pd.Series(False, index=merged_df.index)
        )
        stopcode_match = (
            merged_df["stop_code"].astype(str).isin(STOP_CODE_FILTER)
            if STOP_CODE_FILTER
            else pd.Series(False, index=merged_df.index)
        )

        row_match = route_match | stopid_match | stopcode_match
        blocks_that_qualify = merged_df.loc[row_match, "block_id"].unique()
        blocks_that_qualify = [b for b in blocks_that_qualify if pd.notna(b)]

        block_filter_mask = merged_df["block_id"].isin(blocks_that_qualify)
        merged_df = merged_df[block_filter_mask]
        print(f"After filtering, total rows = {len(merged_df)}")
    else:
        print("No block-level filters applied.")

    return merged_df


def run_step1_gtfs_to_blocks():
    """
    Step 1: Generate block-level schedules from GTFS.
    """
    print("=== Step 1: Reading GTFS and generating block-level schedules ===")
    validate_folders(GTFS_FOLDER_PATH, BLOCK_OUTPUT_FOLDER)

    # Use the standardized loader
    print("Loading GTFS data using standardized function ...")
    gtfs_data = load_gtfs_data(GTFS_FOLDER_PATH, dtype=str)

    # Pull out frames you'll need
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    stops_df = gtfs_data["stops"]

    # Optionally handle routes
    routes_df = gtfs_data.get("routes", None)
    if routes_df is not None and "route_short_name" in routes_df.columns:
        print("Merging routes.txt with trips ...")
        if "route_short_name" not in trips_df.columns:
            trips_df = pd.merge(
                trips_df,
                routes_df[["route_id", "route_short_name"]],
                on="route_id",
                how="left",
            )
    else:
        print("WARNING: No routes.txt found or missing route_short_name column.")
        if "route_short_name" not in trips_df.columns:
            trips_df["route_short_name"] = None

    # Now merge & filter to get the final dataset
    merged_df = _merge_and_filter_data(trips_df, stop_times_df, stops_df)

    all_blocks = merged_df["block_id"].dropna().unique()
    print(f"Identified {len(all_blocks)} block(s) to process.")

    max_minutes = DEFAULT_HOURS * 60
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    for blk_id in all_blocks:
        print(f"\nProcessing block {blk_id}...")
        block_data = merged_df[merged_df["block_id"] == blk_id].copy()
        trip_ids = block_data["trip_id"].unique()

        check_for_overlapping_trips(block_data, blk_id)

        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            print(
                f"Block {blk_id} has {len(trip_ids)} trips > limit "
                f"{MAX_TRIPS_PER_BLOCK}. Skipped."
            )
            continue

        block_schedule_df = process_block(
            block_data, blk_id, timeline, BUS_STOP_CLUSTERS_STEP1
        )
        block_schedule_df.sort_values("Timestamp", inplace=True)

        block_route_ids = block_data["route_id"].dropna().unique()
        if len(block_route_ids) > 0:
            block_route_str = "_".join(str(rte) for rte in block_route_ids)
        else:
            block_route_str = "NA"

        out_name = f"block_{blk_id}_{block_route_str}.xlsx"
        out_path = os.path.join(BLOCK_OUTPUT_FOLDER, out_name)
        block_schedule_df.to_excel(out_path, index=False)
        print(f"Finished block {blk_id}; saved to {out_path}")

    print("\nStep 1 complete: All block-level spreadsheets generated.")


# ==================================================================================================
# MAIN
# ==================================================================================================


def main():
    """
    Master entry point.
    """
    run_step1_gtfs_to_blocks()


if __name__ == "__main__":
    main()
