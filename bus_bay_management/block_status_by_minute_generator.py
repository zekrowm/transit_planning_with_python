"""
GTFS Block Timeline Generator

This script processes General Transit Feed Specification (GTFS) data to create
minute-by-minute block-level spreadsheets (one file per block). It is designed
as "Step 1" in a multi-step pipeline, but you can also run it independently.

CHANGE NOTE:
  - The optional filter logic has been revised so that an entire block is kept
    if it has at least one row matching the filter criteria. All rows for that
    block are retained. 
"""

import os
import pandas as pd

###############################################################################
#                       UNIFIED CONFIGURATION SECTION
###############################################################################

GTFS_FOLDER = r"Path\To\Your\GTFS_Folder"
BLOCK_OUTPUT_FOLDER = r"Path\To\Your\Output_Folder"

# For how many hours (starting at 00:00) do you want to generate the minute-by-minute timeline?
DEFAULT_HOURS = 26
TIME_INTERVAL_MINUTES = 1

# Thresholds for deciding DWELL / LAYOVER vs. longer breaks
DWELL_THRESHOLD = 3     # minutes
LAYOVER_THRESHOLD = 20  # minutes

# If a block has more than this many trips, we skip it (to avoid huge files)
MAX_TRIPS_PER_BLOCK = 150

# Only include trips whose service_id is in this list (if not empty)
CALENDAR_SERVICE_IDS = [3]

# ------------------------------------------------------------------------------
# Optional filter lists. 
# If any of these lists are non-empty, we only keep blocks (in their entirety)
# for which the block has at least one row matching the filter criteria.
# ------------------------------------------------------------------------------
ROUTE_SHORTNAME_FILTER = []  # e.g. ["10", "105"]
STOP_ID_FILTER = []          # e.g. ["2956", "2955"]
STOP_CODE_FILTER = []        # e.g. ["X1234", "B9876"]
# ------------------------------------------------------------------------------

# For Step 1, cluster definitions can help decide whether bridging between two
# trip segments is "LAYOVER" vs. "DEADHEAD". If you don't want cluster logic,
# you can simplify or remove CLUSTER_DEFINITIONS entirely.
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

# For Step 1, the old structures remain:
BUS_STOP_CLUSTERS_STEP1 = [
    {"name": name, "stops": cluster_info["stops"]}
    for name, cluster_info in CLUSTER_DEFINITIONS.items()
]


###############################################################################
#                            VALIDATION UTILITY
###############################################################################

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


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def time_to_minutes(time_str):
    """
    Convert 'HH:MM:SS' (or 'HH:MM') strings into integer minutes from 0.

    Example:
        "00:30:00" -> 30
        "01:00:00" -> 60
        "26:30:00" -> 1590
    """
    parts = time_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = int(parts[2]) if len(parts) == 3 else 0
    return hh * 60 + mm + (ss // 60)


def minutes_to_hhmm(minutes_in):
    """
    Convert an integer minute count into 'HH:MM' format.

    Example:
        0 -> "00:00"
        30 -> "00:30"
        1590 -> "26:30"
    """
    hh = minutes_in // 60
    mm = minutes_in % 60
    return f"{hh:02d}:{mm:02d}"


def mark_first_and_last_stops(df_in):
    """
    Mark each stop in the trip as the first or last using boolean columns
    'is_first_stop' and 'is_last_stop'.

    Returns a new DataFrame with 'is_first_stop' and 'is_last_stop' assigned.
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
    Given a stop_id, return which named cluster it belongs to (if any).
    If not found in any cluster, returns None.
    """
    for cluster_item in bus_stop_clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


def get_status_for_minute(minute, stop_times_sequence, bus_stop_clusters):
    """
    Determine the block's status at a specific 'minute' by scanning
    the trip's stop-time sequence.

    'stop_times_sequence' is a list of tuples:
        (arrival_min, departure_min, stop_id, stop_name, trip_id,
         is_first_stop, is_last_stop, stop_sequence, timepoint_value)

    Returns a tuple:
       (status, stop_id, stop_name, arrival_str, departure_str,
        trip_id_for_status, stop_sequence, timepoint_value)

    Possible status results include:
        - "ARRIVE", "DEPART", "ARRIVE/DEPART"
        - "DWELL", "LOADING"
        - "LAYOVER", "LONG BREAK"
        - "TRAVELING BETWEEN STOPS", "DEADHEAD", "EMPTY", "INACTIVE"
    """
    if not stop_times_sequence:
        return ("EMPTY", None, None, None, None, None, None, 0)

    for i, item in enumerate(stop_times_sequence):
        (
            arr,
            dep,
            s_id,
            s_name,
            trip_id,
            is_first,
            is_last,
            stop_seq,
            t_val,
        ) = item

        # Exact arrival/departure checks:
        if minute == arr and is_last:
            return (
                "ARRIVE",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val,
            )
        if minute == dep and is_first:
            return (
                "DEPART",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val,
            )
        if minute == arr == dep:
            return (
                "ARRIVE/DEPART",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val,
            )

        # Midpoint arrival/departure checks:
        if arr < minute < dep:
            return (
                "DWELL",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val,
            )

        # Between stops (check next):
        if i < len(stop_times_sequence) - 1:
            (
                next_arr,
                next_dep,
                next_stop_id,
                next_stop_name,
                next_trip_id,
                next_is_first,
                next_is_last,
                next_stop_seq,
                next_t_val,
            ) = stop_times_sequence[i + 1]

            if dep < minute < next_arr:
                # If it is the same trip => traveling
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

                # Different trip => dwell, layover, or deadhead
                gap = next_arr - dep
                same_stop = (s_id == next_stop_id)

                # If cluster logic is defined, see if the current stop & next stop share a cluster
                same_cluster = False
                if bus_stop_clusters:
                    current_cluster = find_cluster(s_id, bus_stop_clusters)
                    next_cluster = find_cluster(next_stop_id, bus_stop_clusters)
                    same_cluster = (
                        current_cluster and next_cluster and current_cluster == next_cluster
                    )

                if same_stop or same_cluster:
                    if gap <= DWELL_THRESHOLD:
                        return (
                            "DWELL",
                            s_id,
                            s_name,
                            minutes_to_hhmm(arr),
                            minutes_to_hhmm(dep),
                            next_trip_id,
                            stop_seq,
                            t_val,
                        )
                    if gap > LAYOVER_THRESHOLD:
                        return (
                            "LONG BREAK",
                            s_id,
                            s_name,
                            minutes_to_hhmm(arr),
                            minutes_to_hhmm(dep),
                            next_trip_id,
                            stop_seq,
                            t_val,
                        )
                    return (
                        "LAYOVER",
                        s_id,
                        s_name,
                        minutes_to_hhmm(arr),
                        minutes_to_hhmm(dep),
                        next_trip_id,
                        stop_seq,
                        t_val,
                    )

                # If different cluster => "DEADHEAD"
                if bus_stop_clusters:
                    return ("DEADHEAD", None, None, None, None, next_trip_id, None, 0)
                return ("LAYOVER/DEADHEAD", None, None, None, None, next_trip_id, None, 0)

    # If no condition matches, treat as "EMPTY"
    return ("EMPTY", None, None, None, None, None, None, 0)


def process_block(block_subset, block_id, timeline, bus_stop_clusters):
    """
    Generate a minute-by-minute schedule for a single block.

    Args:
        block_subset (DataFrame): Rows from the GTFS DataFrame for the given block_id.
        block_id (str or int): The block identifier.
        timeline (range): A range of minutes [0..N].
        bus_stop_clusters (list): If defined, used for deciding LAYOVER vs. DEADHEAD.

    Returns:
        DataFrame: One row per minute with columns:
            [
                "Timestamp", "Block", "Route", "Direction", "Trip ID",
                "Stop ID", "Stop Name", "Stop Sequence",
                "Arrival Time", "Departure Time", "Status", "Timepoint"
            ]
    """
    # Organize all trips in this block
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
    rows = []

    # For each minute in the timeline:
    for minute in timeline:
        possible_trips = [
            trip for trip in trips_summary
            if trip["start"] <= minute <= trip["end"]
        ]
        if possible_trips:
            candidate_info = []
            for trip_obj in possible_trips:
                status_tuple = get_status_for_minute(
                    minute,
                    trip_obj["stop_times_sequence"],
                    bus_stop_clusters
                )
                candidate_info.append((trip_obj, status_tuple))

            valid_candidates = [
                (trip_obj, stat)
                for (trip_obj, stat) in candidate_info
                if stat[0] != "EMPTY"
            ]
            if not valid_candidates:
                # All were "EMPTY" => bridging time
                chosen_trip = None
                chosen_status = ("EMPTY", None, None, None, None, None, None, 0)
            elif len(valid_candidates) == 1:
                chosen_trip, chosen_status = valid_candidates[0]
            else:
                # Tie-break if multiple trips claim a status
                def candidate_sort_key(item):
                    stat = item[1]
                    stop_seq = stat[6] if stat[6] else 999999
                    t_val = stat[7] if len(stat) == 8 else 0
                    is_timepoint = t_val in [1, 2]  # prefer timepoints over non-timepoints
                    return (not is_timepoint, stop_seq)

                valid_candidates.sort(key=candidate_sort_key)
                chosen_trip, chosen_status = valid_candidates[0]
        else:
            # No active trips => "EMPTY"
            chosen_trip = None
            chosen_status = ("EMPTY", None, None, None, None, None, None, 0)

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

        # Build final row
        if chosen_trip:
            if status == "EMPTY":
                # If an active trip is "EMPTY", we consider it traveling
                status = "TRAVELING BETWEEN STOPS"

            # Convert a minute of dwell/layover into "LOADING" if next minute is a DEPART
            if status in ["DWELL", "LAYOVER"]:
                next_minute = minute + TIME_INTERVAL_MINUTES
                if next_minute <= chosen_trip["end"]:
                    next_status = get_status_for_minute(
                        next_minute, chosen_trip["stop_times_sequence"], bus_stop_clusters
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
            # Inactive or bridging logic
            prev_trip = None
            next_trip = None
            for trip_obj in trips_summary:
                if trip_obj["end"] < minute:
                    if (prev_trip is None) or (trip_obj["end"] > prev_trip["end"]):
                        prev_trip = trip_obj
                if trip_obj["start"] > minute:
                    if (next_trip is None) or (trip_obj["start"] < next_trip["start"]):
                        next_trip = trip_obj

            # Distinguish "DWELL", "LAYOVER", "LONG BREAK", or "INACTIVE"
            if prev_trip and next_trip:
                gap = next_trip["start"] - prev_trip["end"]
                if gap <= DWELL_THRESHOLD:
                    status = "DWELL"
                elif gap <= LAYOVER_THRESHOLD:
                    status = "LAYOVER"
                else:
                    status = "LONG BREAK"
            else:
                status = "INACTIVE"

            # If we are exactly 1 minute away from next trip’s start => "LOADING"
            if (
                next_trip
                and next_trip["start"] == minute + 1
                and status in ["DWELL", "LAYOVER"]
            ):
                status = "LOADING"

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
                "Status": status,
                "Timepoint": 0,
            }

        rows.append(row)

    return pd.DataFrame(rows)


def check_for_overlapping_trips(block_subset, block_id):
    """
    Print a warning if any trips within this block overlap in time.
    Overlap means the time ranges [start..end] intersect.
    """
    trip_times = []
    for trip_id, group in block_subset.groupby("trip_id"):
        start = group["arrival_min"].min()
        end = group["departure_min"].max()
        trip_times.append((trip_id, start, end))

    trip_times.sort(key=lambda x: x[1])
    for i in range(len(trip_times) - 1):
        t1, s1, e1 = trip_times[i]
        for j in range(i + 1, len(trip_times)):
            t2, s2, e2 = trip_times[j]
            if s2 <= e1 and s1 <= e2:
                print(f"WARNING: Overlapping trips in block {block_id}:")
                print(f"  Trip {t1} {s1} to {e1} overlaps with {t2} {s2} to {e2}")


def fill_stop_ids_for_dwell_layover_loading(df_in):
    """
    For rows where status is DWELL, LAYOVER, or LOADING (but the Stop ID is empty),
    fill in the last known stop_id, stop_name, etc. from the same block.

    This is purely cosmetic for easier reading in the final spreadsheet.
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

        if stop_id not in (None, "", float("nan")):
            # Update the "last known" stop info
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


###############################################################################
#                   STEP 1: GTFS -> Block Spreadsheets
###############################################################################

def run_step1_gtfs_to_blocks():
    """
    Step 1:
      1) Validate input/output directories.
      2) Read in trips.txt, stop_times.txt, routes.txt (optional), blocks.txt (optional), stops.txt.
      3) Filter by CALENDAR_SERVICE_IDS (if non-empty).
      4) If any route/stop filter lists are non-empty, we find all blocks that
         have at least one row matching those filters, and retain those blocks in full.
      5) Create per-block minute-by-minute schedules from 0..DEFAULT_HOURS.
      6) Save each block’s schedule to Excel in BLOCK_OUTPUT_FOLDER.
    """
    print("=== Step 1: Reading GTFS and generating block-level schedules ===")
    validate_folders(GTFS_FOLDER, BLOCK_OUTPUT_FOLDER)

    trips_path = os.path.join(GTFS_FOLDER, "trips.txt")
    stop_times_path = os.path.join(GTFS_FOLDER, "stop_times.txt")
    stops_path = os.path.join(GTFS_FOLDER, "stops.txt")
    blocks_path = os.path.join(GTFS_FOLDER, "blocks.txt")  # optional
    routes_path = os.path.join(GTFS_FOLDER, "routes.txt")  # optional, for route_short_name

    print("Reading GTFS files...")
    trips_df = pd.read_csv(trips_path)
    stop_times_df = pd.read_csv(stop_times_path)
    stops_df = pd.read_csv(stops_path)

    # Check if routes.txt exists; if so, merge route_short_name into trips_df
    if os.path.exists(routes_path):
        routes_df = pd.read_csv(routes_path)
        # We assume standard GTFS with 'route_id' and 'route_short_name'
        # Merge route_short_name into trips_df
        trips_df = pd.merge(
            trips_df, 
            routes_df[["route_id", "route_short_name"]], 
            on="route_id", 
            how="left"
        )
    else:
        print("WARNING: No routes.txt found; route_short_name filters will not be possible.")
        trips_df["route_short_name"] = None

    if os.path.exists(blocks_path):
        blocks_df = pd.read_csv(blocks_path)
        print("Blocks file found and read.")
    else:
        blocks_df = pd.DataFrame()
        print("No blocks file found; proceeding without it.")

    # Filter by service_id if desired
    if CALENDAR_SERVICE_IDS:
        print(f"Filtering trips to service_ids in {CALENDAR_SERVICE_IDS} ...")
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    print("Merging data and converting times to minutes...")
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(time_to_minutes)

    # Keep only stop_times for the trips that survived the service_id filter
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    # Make sure stops_df has stop_code; if not, create a blank column
    if "stop_code" not in stops_df.columns:
        stops_df["stop_code"] = None

    # Merge stop_times + trips
    merged_df = pd.merge(stop_times_df, trips_df, on="trip_id", how="left")

    # Merge with stops to get stop_name, stop_code, timepoint if it exists
    stops_merge_cols = ["stop_id", "stop_name", "stop_code"]
    if "timepoint" in stops_df.columns:
        stops_merge_cols.append("timepoint")
    merged_df = pd.merge(merged_df, stops_df[stops_merge_cols], on="stop_id", how="left")

    print("Marking first and last stops...")
    merged_df = mark_first_and_last_stops(merged_df)

    # Ensure 'timepoint' is numeric
    if "timepoint" not in merged_df.columns:
        merged_df["timepoint"] = 0
    else:
        merged_df["timepoint"] = pd.to_numeric(merged_df["timepoint"], errors="coerce").fillna(0).astype(int)

    # Force first/last stops to timepoint=2 if they were 0
    merged_df.loc[
        (merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0),
        "timepoint"
    ] = 2
    merged_df.loc[
        (merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0),
        "timepoint"
    ] = 2

    # =====================================================
    # BLOCK-LEVEL FILTERING BASED ON USER-SPECIFIED CRITERIA
    # =====================================================
    if ROUTE_SHORTNAME_FILTER or STOP_ID_FILTER or STOP_CODE_FILTER:
        print("Applying block-level filter based on route_short_name, stop_id, stop_code...")

        # Build a per-row match for the filter
        if ROUTE_SHORTNAME_FILTER:
            route_match = merged_df["route_short_name"].isin(ROUTE_SHORTNAME_FILTER)
        else:
            route_match = pd.Series(False, index=merged_df.index)

        if STOP_ID_FILTER:
            stopid_match = merged_df["stop_id"].astype(str).isin(STOP_ID_FILTER)
        else:
            stopid_match = pd.Series(False, index=merged_df.index)

        if STOP_CODE_FILTER:
            stopcode_match = merged_df["stop_code"].astype(str).isin(STOP_CODE_FILTER)
        else:
            stopcode_match = pd.Series(False, index=merged_df.index)

        # True if this row matches ANY filter
        row_match = route_match | stopid_match | stopcode_match

        # Identify all blocks that appear in at least one matching row
        blocks_that_qualify = merged_df.loc[row_match, "block_id"].unique()
        blocks_that_qualify = [b for b in blocks_that_qualify if pd.notna(b)]

        # Keep only rows whose block_id is in blocks_that_qualify
        block_filter_mask = merged_df["block_id"].isin(blocks_that_qualify)
        filtered_count = block_filter_mask.sum()
        original_count = len(merged_df)
        merged_df = merged_df[block_filter_mask]
        print(f"Filtered from {original_count} down to {filtered_count} rows, across {len(blocks_that_qualify)} blocks.")
    else:
        print("No block-level filters applied; using entire dataset.")

    # Identify unique block_ids (after filter)
    all_blocks = merged_df["block_id"].dropna().unique()
    print(f"Identified {len(all_blocks)} block(s) to process after filtering.")

    # Define the minute timeline from 0 to (DEFAULT_HOURS * 60)
    max_minutes = DEFAULT_HOURS * 60
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    for blk_id in all_blocks:
        print(f"\nProcessing block {blk_id}...")
        block_subset = merged_df[merged_df["block_id"] == blk_id].copy()
        trip_ids = block_subset["trip_id"].unique()
        print(f"Found {len(trip_ids)} trip(s) in block {blk_id}.")

        # Overlap check
        check_for_overlapping_trips(block_subset, blk_id)

        # Skip extremely large blocks if desired
        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            print(
                f"Block {blk_id} has {len(trip_ids)} trips > limit {MAX_TRIPS_PER_BLOCK}, skipping."
            )
            continue

        # Build the minute-by-minute schedule
        block_schedule_df = process_block(block_subset, blk_id, timeline, BUS_STOP_CLUSTERS_STEP1)
        block_schedule_df = fill_stop_ids_for_dwell_layover_loading(block_schedule_df)
        block_schedule_df.sort_values("Timestamp", inplace=True)

        # Build a route string for the file name
        block_route_ids = block_subset["route_id"].dropna().unique()
        if len(block_route_ids) > 0:
            block_route_str = "_".join(str(rte) for rte in block_route_ids)
        else:
            block_route_str = "NA"

        out_name = f"block_{blk_id}_{block_route_str}.xlsx"
        out_path = os.path.join(BLOCK_OUTPUT_FOLDER, out_name)
        block_schedule_df.to_excel(out_path, index=False)
        print(f"Finished block {blk_id}; saved to {out_path}")

    print("\nStep 1 complete: All block-level spreadsheets generated.")


###############################################################################
#                        MASTER MAIN FUNCTION
###############################################################################

def main():
    """
    Runs the step1_gtfs_to_blocks function directly if script is run as __main__.
    """
    run_step1_gtfs_to_blocks()


if __name__ == "__main__":
    main()
