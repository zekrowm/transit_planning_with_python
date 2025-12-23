"""Generates minute-by-minute transit block status timelines from GTFS data.

Processes GTFS files to determine operational statuses (e.g., DWELL, LAYOVER, TRAVELING)
of transit blocks at one-minute intervals, producing Excel reports for further analysis.

Typically used interactively within a Jupyter notebook or ArcGIS Pro environment,
though direct execution via the command line is also supported.
"""

import logging
import os
from typing import Any, Mapping, Optional, cast

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

ROUTE_SHORTNAME_FILTER: list[str] = []
STOP_ID_FILTER: list[str] = []
STOP_CODE_FILTER: list[str] = []

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


def validate_folders(input_path: str, output_path: str) -> None:
    """Check that the input folder exists, and ensure the output folder is created if not."""
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Input path does not exist or is not a directory: {input_path}")
    os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------


def time_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM:SS' or 'HH:MM' to integer minutes (e.g. "26:30:00" -> 1590)."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 60 + minutes + (seconds // 60)


def minutes_to_hhmm(total_minutes: int) -> str:
    """Convert integer minutes into 'HH:MM' (e.g. 1590 -> "26:30")."""
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def mark_first_and_last_stops(df_in: pd.DataFrame) -> pd.DataFrame:
    """Mark each stop in the trip as the first or last using boolean columns."""
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    df_out["is_first_stop"] = False
    df_out["is_last_stop"] = False

    for _, group in df_out.groupby("trip_id"):
        min_seq_idx = group["stop_sequence"].idxmin()
        max_seq_idx = group["stop_sequence"].idxmax()
        df_out.loc[min_seq_idx, "is_first_stop"] = True
        df_out.loc[max_seq_idx, "is_last_stop"] = True

    return df_out


def find_cluster(stop_id: str, bus_stop_clusters: list[dict[str, Any]]) -> Optional[str]:
    """Given a stop_id, return the named cluster (if any) or None if not found."""
    for cluster_item in bus_stop_clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# --------------------------------------------------------------------------------------------------
# BRIDGING LOGIC REFACTOR
# --------------------------------------------------------------------------------------------------


def _status_for_same_trip(minute: int, stop_info: tuple) -> Optional[tuple]:
    """Handle same-trip logic. stop_info is a tuple."""
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
    dep: int,
    next_arr: int,
    current_stop_id: str,
    current_stop_name: str,
    next_stop_id: str,
    bus_stop_clusters: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """Sub-logic for bridging between two different trips, determining LAYOVER, DEADHEAD, etc."""
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


def get_status_for_minute(
    minute: int, stop_times_sequence: list[tuple], bus_stop_clusters: list[dict[str, Any]]
) -> tuple[
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    int,
]:
    """Determine the block's status at a specific 'minute'.

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


def check_for_overlapping_trips(
    block_subset: pd.DataFrame, block_id: str
) -> None:  # Added return type annotation
    """Print a warning if any trips within this block overlap in time."""
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
                logging.warning(
                    "WARNING: Overlapping trips in block %s: "
                    "Trip %s (%s–%s) overlaps with "
                    "%s (%s–%s)",
                    block_id,
                    trip1_id,
                    start1,
                    end1,
                    trip2_id,
                    start2,
                    end2,
                )


def fill_stop_ids_for_dwell_layover_loading(
    df_in: pd.DataFrame,
) -> pd.DataFrame:  # Added return type annotation
    """Fills in missing stop information for certain vehicle statuses.

    For rows with a status of 'DWELL', 'LAYOVER', or 'LOADING', where the
    'Stop ID' is typically empty, this function populates the 'Stop ID',
    'Stop Name', 'Stop Sequence', 'Arrival Time', 'Departure Time', and
    'Trip ID' fields. The data used for filling is based on the last known
    stop information from the same vehicle block, enhancing the readability
    of the final output.

    Args:
        df_in (pd.DataFrame): The input DataFrame containing vehicle status and
            stop information.

    Returns:
        pd.DataFrame: A new DataFrame with the missing stop information filled in.
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


def _create_trips_summary(
    block_subset: pd.DataFrame,
) -> list[dict[str, Any]]:  # Added return type annotation
    """Helper to build trip summaries for a block.

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


def _status_for_active_trips(
    minute: int,
    active_trips: list[dict[str, Any]],
    bus_stop_clusters: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], tuple]:  # Added return type annotation
    """Among all 'active' trips for the given minute, determine the single chosen status."""
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
    def candidate_sort_key(item: tuple[dict[str, Any], tuple]) -> tuple[bool, int]:
        stat = item[1]
        stop_seq = stat[6] if stat[6] is not None else 999999
        timepoint_val = stat[7] if len(stat) == 8 else 0
        # Prefer timepoints, then lower stop_sequence
        return (timepoint_val == 0, stop_seq)

    valid_candidates.sort(key=candidate_sort_key)
    return valid_candidates[0]


def _row_for_inactive(
    minute: int,
    block_id: str,
    all_trips: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a dictionary row describing the vehicle status when **no trip is active**.

    The function looks at the closest finished trip and the next upcoming trip to
    decide whether the vehicle is *DWELL*, *LAYOVER*, *LONG BREAK*, *LOADING*, or truly
    *INACTIVE* at the requested minute.
    """
    prev_trip_info: Optional[dict[str, Any]] = None
    next_trip_info: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ locate bounding trips
    for trip_obj in all_trips:
        # The most recent trip that has already finished
        if trip_obj["end"] < minute:
            if prev_trip_info is None or trip_obj["end"] > prev_trip_info["end"]:
                prev_trip_info = trip_obj

        # The very next trip that has not yet started
        elif trip_obj["start"] > minute:
            if next_trip_info is None or trip_obj["start"] < next_trip_info["start"]:
                next_trip_info = trip_obj

    # ------------------------------------------------------------------ status decision
    if prev_trip_info is not None and next_trip_info is not None:
        # Cast after the None-check so static analysers know the objects are dicts
        prev_trip = cast("dict[str, Any]", prev_trip_info)
        next_trip = cast("dict[str, Any]", next_trip_info)

        gap: int = next_trip["start"] - prev_trip["end"]
        if gap <= DWELL_THRESHOLD:
            status = "DWELL"
        elif gap <= LAYOVER_THRESHOLD:
            status = "LAYOVER"
        else:
            status = "LONG BREAK"
    else:
        status = "INACTIVE"

    # If the very next minute is the first minute of a trip, mark this minute “LOADING”.
    if (
        next_trip_info is not None
        and cast("dict[str, Any]", next_trip_info)["start"] == minute + 1
        and status in {"DWELL", "LAYOVER"}
    ):
        status = "LOADING"

    # ------------------------------------------------------------------ final record
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


def _build_schedule_rows(
    trips_summary: list[dict[str, Any]],
    timeline: range,
    block_id: str,
    bus_stop_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the minute-by-minute schedule rows for a single block.

    Args:
        trips_summary: Output of :func:`_create_trips_summary`.
        timeline: Range object representing every minute to be evaluated.
        block_id: Identifier of the vehicle block.
        bus_stop_clusters: Cluster definitions used for dwell/layover logic.

    Returns:
        A list of dictionaries (one per minute) suitable for `pd.DataFrame`.
    """
    rows: list[dict[str, Any]] = []

    for minute in timeline:
        # --------------------------------------------------- identify active trips
        possible_trips = [t for t in trips_summary if t["start"] <= minute <= t["end"]]

        if possible_trips:
            chosen_trip, chosen_status = _status_for_active_trips(
                minute, possible_trips, bus_stop_clusters
            )

            if chosen_trip is not None:
                chosen_trip = cast("dict[str, Any]", chosen_trip)

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

                # Treat “EMPTY” as travelling between stops.
                if status == "EMPTY":
                    status = "TRAVELING BETWEEN STOPS"

                # Convert DWELL/LAYOVER to LOADING if departure is in the next minute.
                if status in {"DWELL", "LAYOVER"}:
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
                    "Trip ID": trip_id_for_status or "",
                    "Stop ID": stop_id or "",
                    "Stop Name": stop_name or "",
                    "Stop Sequence": stop_seq or "",
                    "Arrival Time": arr_str or "",
                    "Departure Time": dep_str or "",
                    "Status": status,
                    "Timepoint": t_val,
                }
            else:
                # All active trips returned “EMPTY”.
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
            # No active trips ⇒ bridging or fully inactive.
            row = _row_for_inactive(minute, block_id, trips_summary)

        rows.append(row)

    return rows


def process_block(
    block_subset: pd.DataFrame,
    block_id: str,
    timeline: range,
    bus_stop_clusters: list[dict[str, Any]],
) -> pd.DataFrame:
    """Generate a minute-by-minute schedule DataFrame for a single block."""
    trips_summary = _create_trips_summary(block_subset)
    rows = _build_schedule_rows(trips_summary, timeline, block_id, bus_stop_clusters)
    df = pd.DataFrame(rows)
    return fill_stop_ids_for_dwell_layover_loading(df)


# --------------------------------------------------------------------------------------------------
# STEP 1: GTFS -> Block Spreadsheets
# --------------------------------------------------------------------------------------------------


def _merge_and_filter_data(
    trips_df: pd.DataFrame, stop_times_df: pd.DataFrame, stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge trips and stops, filter by service_id, route, etc.

    Return a single merged DataFrame with arrival_min/departure_min.
    """
    # Filter by service_id if set
    if CALENDAR_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    # Convert times to minutes
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(time_to_minutes)
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
    merged_df = pd.merge(merged_df, stops_df[stops_merge_cols], on="stop_id", how="left")

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
    merged_df.loc[(merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2
    merged_df.loc[(merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2

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
        logging.info("After filtering, total rows = %d", len(merged_df))
    else:
        logging.info("No block-level filters applied.")

    return merged_df


def run_step1_gtfs_to_blocks() -> None:
    """Step 1: Generate block-level schedules from GTFS."""
    logging.info("=== Step 1: Reading GTFS and generating block-level schedules ===")
    validate_folders(GTFS_FOLDER_PATH, BLOCK_OUTPUT_FOLDER)

    # Use the standardized loader
    logging.info("Loading GTFS data using standardized function ...")
    gtfs_data = load_gtfs_data(GTFS_FOLDER_PATH, dtype=str)

    # Pull out frames you'll need
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    stops_df = gtfs_data["stops"]

    # Optionally handle routes
    routes_df = gtfs_data.get("routes", None)
    if routes_df is not None and "route_short_name" in routes_df.columns:
        logging.info("Merging routes.txt with trips ...")
        if "route_short_name" not in trips_df.columns:
            trips_df = pd.merge(
                trips_df,
                routes_df[["route_id", "route_short_name"]],
                on="route_id",
                how="left",
            )
    else:
        logging.warning("WARNING: No routes.txt found or missing route_short_name column.")
        if "route_short_name" not in trips_df.columns:
            trips_df["route_short_name"] = None

    # Now merge & filter to get the final dataset
    merged_df = _merge_and_filter_data(trips_df, stop_times_df, stops_df)

    all_blocks = merged_df["block_id"].dropna().unique()
    logging.info("Identified %d block(s) to process.", len(all_blocks))

    max_minutes = DEFAULT_HOURS * 60
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    for blk_id in all_blocks:
        logging.info("\nProcessing block %s...", blk_id)
        block_data = merged_df[merged_df["block_id"] == blk_id].copy()
        trip_ids = block_data["trip_id"].unique()

        check_for_overlapping_trips(block_data, blk_id)

        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            logging.info(
                "Block %s has %d trips > limit %d. Skipped.",
                blk_id,
                len(trip_ids),
                MAX_TRIPS_PER_BLOCK,
            )
            continue

        block_schedule_df = process_block(block_data, blk_id, timeline, BUS_STOP_CLUSTERS_STEP1)
        block_schedule_df.sort_values("Timestamp", inplace=True)

        block_route_ids = block_data["route_id"].dropna().unique()
        if len(block_route_ids) > 0:
            block_route_str = "_".join(str(rte) for rte in block_route_ids)
        else:
            block_route_str = "NA"

        out_name = f"block_{blk_id}_{block_route_str}.xlsx"
        out_path = os.path.join(BLOCK_OUTPUT_FOLDER, out_name)
        block_schedule_df.to_excel(out_path, index=False)
        logging.info("Finished block %s; saved to %s", blk_id, out_path)

    logging.info("\nStep 1 complete: All block-level spreadsheets generated.")


# --------------------------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[list[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into a dictionary of DataFrames.

    Args:
        gtfs_folder_path (str): Absolute or relative path to the directory
            containing GTFS text files.
        files (list[str] | None): Explicit list of GTFS filenames to load.
            If ``None``, the full standard GTFS set is read.
        dtype (str | Mapping[str, Any]): Value forwarded to
            :pyfunc:`pandas.read_csv` to control column dtypes;
            defaults to ``str``.

    Returns:
        dict[str, pandas.DataFrame]: Mapping of file stem → DataFrame.
        For example, ``data["trips"]`` contains *trips.txt*.

    Raises:
        OSError: The folder does not exist or a required file is missing.
        ValueError: A file is empty or malformed.
        RuntimeError: An OS-level error occurs while reading.
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
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc
    return data


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:
    """Master entry point."""
    logging.basicConfig(level=logging.INFO)
    run_step1_gtfs_to_blocks()


if __name__ == "__main__":
    main()
