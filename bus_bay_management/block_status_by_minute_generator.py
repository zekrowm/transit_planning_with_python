"""
Script to export minute-by-minute status of blocks .xlsx (i.e. buses) from GTFS data.
Feeds into script that uses block sheets to construct conflict matrix and
long format spreadsheet for defined bus stop clusters of interest.
"""
import os
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
GTFS_FOLDER = r"Path\To\Your\GTFS_Folder"  # Folder containing trips.txt, stop_times.txt, stops.txt, blocks.txt (optional)
OUTPUT_FOLDER = r"Path\To\Your\Output_Folder"

DEFAULT_HOURS = 26  # Generate minute-by-minute for 26 hours by default
TIME_INTERVAL_MINUTES = 1  # Granularity: 1 minute

DWELL_THRESHOLD = 3   # Set desired maximum time, in minutes, for DWELL
LAYOVER_THRESHOLD = 20  # Set desired maximum time, in minutes, for LAYOVER

# Configurable limit on the number of trips in a block
MAX_TRIPS_PER_BLOCK = 150

# New: Filter trips by these service_id(s)
# By default, keep only service_id == "3" (typical for 'weekday' in some feeds)
CALENDAR_SERVICE_IDS = ["3"]

BUS_STOP_CLUSTERS = [
    {
        "name": "Heavy Rail Station",
        "stops": ["1001", "1002", "1003"],
    },
    {"name": "Bus Transfer Station", "stops": ["2001", "2002"]},
]
# If you don't want cluster logic, simply set BUS_STOP_CLUSTERS = []

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def time_to_minutes(t_str):
    """Convert HH:MM[:SS] string to total minutes."""
    parts = t_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = int(parts[2]) if len(parts) == 3 else 0
    return hh * 60 + mm + ss // 60


def minutes_to_hhmm(m):
    """Convert minutes since midnight to a HH:MM string."""
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


def find_cluster(stop_id, bus_stop_clusters):
    """Return the cluster name for a given stop_id if it exists."""
    for cluster in bus_stop_clusters:
        if stop_id in cluster["stops"]:
            return cluster["name"]
    return None


def mark_first_and_last_stops(df):
    """
    For each trip in the DataFrame, mark the first and last stops.
    Adds two Boolean columns: 'is_first_stop' and 'is_last_stop'.
    """
    df = df.sort_values(["trip_id", "stop_sequence"]).copy()
    df["is_first_stop"] = False
    df["is_last_stop"] = False

    for trip_id, grp in df.groupby("trip_id"):
        min_seq_idx = grp["stop_sequence"].idxmin()
        max_seq_idx = grp["stop_sequence"].idxmax()
        df.loc[min_seq_idx, "is_first_stop"] = True
        df.loc[max_seq_idx, "is_last_stop"] = True

    return df


def get_status_for_minute(minute, stop_times_sequence, bus_stop_clusters):
    """
    Determine the bus status at a given minute by examining stop_times_sequence.
    Each element is a tuple:
        (arrival_min, departure_min, stop_id, stop_name, trip_id,
         is_first_stop, is_last_stop, stop_sequence, timepoint_value).

    Returns a tuple:
        (status, stop_id, stop_name, arr_str, dep_str, trip_id_for_status, stop_seq, timepoint_value)
    """
    if not stop_times_sequence:
        return ("EMPTY", None, None, None, None, None, None, 0)

    for i in range(len(stop_times_sequence)):
        (
            arr,
            dep,
            s_id,
            s_name,
            trip_id,
            is_first,
            is_last,
            stop_seq,
            t_val
        ) = stop_times_sequence[i]

        # 1) Exact arrival or departure times
        if minute == arr and is_last:
            return (
                "ARRIVE",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val
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
                t_val
            )

        # Arrival == departure
        if minute == arr == dep:
            return (
                "ARRIVE/DEPART",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val
            )

        # Normal arrive/depart for intermediate stops
        if minute == arr:
            return (
                "ARRIVE",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val
            )
        if minute == dep:
            return (
                "DEPART",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val
            )

        # 2) Dwell at a stop (arr < minute < dep)
        if arr < minute < dep:
            return (
                "DWELL",
                s_id,
                s_name,
                minutes_to_hhmm(arr),
                minutes_to_hhmm(dep),
                trip_id,
                stop_seq,
                t_val
            )

        # 3) Between stops within the same trip
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
                next_t_val
            ) = stop_times_sequence[i + 1]

            if dep < minute < next_arr:
                # If it's the same trip => traveling
                if trip_id == next_trip_id:
                    return (
                        "TRAVELING BETWEEN STOPS",
                        None,
                        None,
                        None,
                        None,
                        trip_id,
                        None,
                        0
                    )
                else:
                    # Different trip => check dwell or deadhead/layover logic
                    gap = next_arr - dep
                    same_stop = s_id == next_stop_id
                    same_cluster = False
                    if bus_stop_clusters:
                        current_cluster = find_cluster(s_id, bus_stop_clusters)
                        next_cluster = find_cluster(next_stop_id, bus_stop_clusters)
                        same_cluster = (
                            current_cluster
                            and next_cluster
                            and current_cluster == next_cluster
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
                                t_val
                            )
                        elif gap > LAYOVER_THRESHOLD:
                            return (
                                "LONG BREAK",
                                s_id,
                                s_name,
                                minutes_to_hhmm(arr),
                                minutes_to_hhmm(dep),
                                next_trip_id,
                                stop_seq,
                                t_val
                            )
                        else:
                            return (
                                "LAYOVER",
                                s_id,
                                s_name,
                                minutes_to_hhmm(arr),
                                minutes_to_hhmm(dep),
                                next_trip_id,
                                stop_seq,
                                t_val
                            )
                    else:
                        if bus_stop_clusters:
                            return ("DEADHEAD", None, None, None, None, next_trip_id, None, 0)
                        else:
                            return (
                                "LAYOVER/DEADHEAD",
                                None,
                                None,
                                None,
                                None,
                                next_trip_id,
                                None,
                                0
                            )

    # Before the first or after the last stop in the sequence
    return ("EMPTY", None, None, None, None, None, None, 0)


def process_block(block_subset, block_id, timeline, bus_stop_clusters):
    """
    Process a block to produce one row per minute.

    1) Summarize each trip in the block (start/end times, stop_times_sequence).
    2) For each minute in the timeline:
       - Collect *all* trips that are active at this minute.
       - Run get_status_for_minute for each trip.
       - If multiple trips return non-"EMPTY" statuses, pick the one with the
         *lowest stop_sequence* but prefer timepoints (1 or 2) over non-timepoints (0).
       - If all are "EMPTY," treat as no active trip.
       - Apply your existing logic (DWELL vs. LAYOVER vs. LONG BREAK vs. INACTIVE)
         for minutes outside any trip.
       - Convert the final status to "LOADING" if the next minute is "DEPART"
         (within the same trip) or if the next trip starts at minute+1.
    """
    # Summarize trips for the block
    trips_summary = []
    for trip_id, trip_df in block_subset.groupby("trip_id"):
        trip_df = trip_df.sort_values("stop_sequence")
        start_time = trip_df["arrival_min"].min()
        end_time = trip_df["departure_min"].max()
        # Build the stop_times_sequence for this trip; include timepoint info
        stop_times_sequence = []
        for _, row in trip_df.iterrows():
            t_val = row.get("timepoint", 0)  # can be 0,1,2
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
        route_id = trip_df.iloc[0]["route_id"]
        direction_id = trip_df.iloc[0]["direction_id"]
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

    # Sort trips by start time
    trips_summary.sort(key=lambda x: x["start"])

    rows = []
    for minute in timeline:
        # -----------------------------------------------------------
        # 1) Gather all "active" trips for this minute
        # -----------------------------------------------------------
        possible_trips = [t for t in trips_summary if t["start"] <= minute <= t["end"]]

        if possible_trips:
            # Evaluate get_status_for_minute for each trip
            candidate_info = []
            for t in possible_trips:
                status_tuple = get_status_for_minute(
                    minute, t["stop_times_sequence"], bus_stop_clusters
                )
                # status_tuple is:
                # (status, stop_id, stop_name, arr_str, dep_str, trip_id_for_status, stop_seq, timepoint_val)
                candidate_info.append((t, status_tuple))

            # Filter out purely "EMPTY" statuses
            valid_candidates = [(t, s) for (t, s) in candidate_info if s[0] != "EMPTY"]

            if not valid_candidates:
                # Everything was "EMPTY" => effectively no active trip
                chosen_trip = None
                chosen_status = ("EMPTY", None, None, None, None, None, None, 0)
            elif len(valid_candidates) == 1:
                chosen_trip, chosen_status = valid_candidates[0]
            else:
                # Multiple valid statuses => tie-break:
                #  1) prefer any timepoint (1 or 2) over 0
                #  2) among those, pick lowest stop_sequence
                def candidate_sort_key(item):
                    st = item[1]
                    stop_seq = st[6] if st[6] is not None else 999999
                    t_val = st[7] if st[7] is not None else 0
                    is_timepoint = (t_val in [1, 2])
                    return (not is_timepoint, stop_seq)

                valid_candidates.sort(key=candidate_sort_key)
                chosen_trip, chosen_status = valid_candidates[0]
        else:
            # No active trips => chosen_trip is None
            chosen_trip = None
            chosen_status = ("EMPTY", None, None, None, None, None, None, 0)

        # -----------------------------------------------------------
        # 2) Build the row from chosen_trip / chosen_status
        # -----------------------------------------------------------
        (
            status,
            stop_id,
            stop_name,
            arr_str,
            dep_str,
            trip_id_for_status,
            stop_seq,
            t_val
        ) = chosen_status

        if chosen_trip:
            # We have an active trip. Possibly "EMPTY" if all were empty, but let's see:
            if status == "EMPTY":
                status = "TRAVELING BETWEEN STOPS"

            # If the current minute is DWELL or LAYOVER, check if next minute in the same
            # trip is DEPART => LOADING
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
                "Timepoint": t_val
            }
        else:
            # No active trip => check gap logic
            prev_trip = None
            next_trip = None
            for t in trips_summary:
                if t["end"] < minute:
                    if (prev_trip is None) or (t["end"] > prev_trip["end"]):
                        prev_trip = t
                if t["start"] > minute:
                    if (next_trip is None) or (t["start"] < next_trip["start"]):
                        next_trip = t

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

            # If the next trip starts exactly at minute+1, and our status is DWELL or LAYOVER =>
            # LOADING
            if next_trip and next_trip["start"] == minute + 1 and status in ["DWELL", "LAYOVER"]:
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
                "Timepoint": 0
            }

        rows.append(row)

    return pd.DataFrame(rows)


def check_for_overlapping_trips(block_subset, block_id):
    """
    Check if any two trips within this block have overlapping start/end times.
    If so, print a warning. Overlap means they share at least one minute in time.
    """
    # Gather trip start/end times
    trip_times = []
    for trip_id, grp in block_subset.groupby("trip_id"):
        start = grp["arrival_min"].min()
        end = grp["departure_min"].max()
        trip_times.append((trip_id, start, end))

    # Sort by start time
    trip_times.sort(key=lambda x: x[1])

    # Compare each trip to every trip after it
    for i in range(len(trip_times) - 1):
        t1, s1, e1 = trip_times[i]
        for j in range(i + 1, len(trip_times)):
            t2, s2, e2 = trip_times[j]
            # Overlap if start2 <= end1 and start1 <= end2
            if s2 <= e1 and s1 <= e2:
                print(f"WARNING: Overlapping trips in block {block_id}:")
                print(f"  Trip {t1} from {s1} to {e1} overlaps with Trip {t2} from {s2} to {e2}")


def fill_stop_ids_for_dwell_layover_loading(df):
    """
    Forward-fill the last known Stop ID (and related columns) if the current row 
    is DWELL, LAYOVER, or LOADING but has no Stop ID. 
    We do NOT fill for statuses like TRAVELING BETWEEN STOPS, DEADHEAD, etc.
    """
    last_stop_id = None
    last_stop_name = None
    last_stop_seq = None
    last_arr = None
    last_dep = None
    last_trip_id = None
    
    for idx in df.index:
        status = df.at[idx, "Status"]
        stop_id = df.at[idx, "Stop ID"]
        
        # If we have a new non-empty Stop ID, remember it
        if stop_id not in (None, "", float("nan")):
            last_stop_id = stop_id
            last_stop_name = df.at[idx, "Stop Name"]
            last_stop_seq = df.at[idx, "Stop Sequence"]
            last_arr = df.at[idx, "Arrival Time"]
            last_dep = df.at[idx, "Departure Time"]
            last_trip_id = df.at[idx, "Trip ID"]
        else:
            # If status is one of the "fillable" statuses, copy from the last known
            if status in ["DWELL", "LAYOVER", "LOADING"]:
                df.at[idx, "Stop ID"] = last_stop_id if last_stop_id else ""
                df.at[idx, "Stop Name"] = last_stop_name if last_stop_name else ""
                df.at[idx, "Stop Sequence"] = last_stop_seq if last_stop_seq else ""
                df.at[idx, "Arrival Time"] = last_arr if last_arr else ""
                df.at[idx, "Departure Time"] = last_dep if last_dep else ""
                df.at[idx, "Trip ID"] = last_trip_id if last_trip_id else ""
            # If status is TRAVELING BETWEEN STOPS, DEADHEAD, INACTIVE, etc., do nothing
    
    return df

# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------
def main():
    print("Starting GTFS processing...")

    # 1. Read GTFS files
    print("Reading GTFS files...")
    trips_path = os.path.join(GTFS_FOLDER, "trips.txt")
    stop_times_path = os.path.join(GTFS_FOLDER, "stop_times.txt")
    stops_path = os.path.join(GTFS_FOLDER, "stops.txt")
    blocks_path = os.path.join(GTFS_FOLDER, "blocks.txt")  # optional

    trips_df = pd.read_csv(trips_path)
    stop_times_df = pd.read_csv(stop_times_path)
    stops_df = pd.read_csv(stops_path)

    if os.path.exists(blocks_path):
        blocks_df = pd.read_csv(blocks_path)
        print("Blocks file found and read.")
    else:
        blocks_df = pd.DataFrame()
        print("No blocks file found; proceeding without it.")

    # 2. Filter trips by service_id if desired
    if CALENDAR_SERVICE_IDS:
        print(f"Filtering trips to service_ids in {CALENDAR_SERVICE_IDS} ...")
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    # 3. Merge data to form a master schedule
    print("Merging data and converting times to minutes...")
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(time_to_minutes)

    # Because we've filtered trips_df, let's also remove stop_times that don't match
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    merged_df = pd.merge(
        stop_times_df,
        trips_df,
        on="trip_id",
        how="left",
        suffixes=("_st", "_trip"),
    )

    # Include the "timepoint" column if available in stops.txt
    if "timepoint" in stops_df.columns:
        stops_merge_cols = ["stop_id", "stop_name", "timepoint"]
    else:
        stops_merge_cols = ["stop_id", "stop_name"]

    merged_df = pd.merge(
        merged_df,
        stops_df[stops_merge_cols],
        on="stop_id",
        how="left",
    )

    # Mark first/last stops
    print("Marking first and last stops for each trip...")
    merged_df = mark_first_and_last_stops(merged_df)

    # Ensure we have a numeric timepoint column
    if "timepoint" not in merged_df.columns:
        merged_df["timepoint"] = 0
    else:
        # If it exists but might be boolean or string, ensure it's numeric
        merged_df["timepoint"] = pd.to_numeric(merged_df["timepoint"], errors="coerce").fillna(0).astype(int)

    # Force first/last stops to timepoint=2 if they were 0
    merged_df.loc[(merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2
    merged_df.loc[(merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0), "timepoint"] = 2

    # 4. Identify unique blocks
    all_blocks = merged_df["block_id"].dropna().unique()
    print(f"Identified {len(all_blocks)} block(s) to process.")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Define timeline (one row per minute for DEFAULT_HOURS)
    max_minutes = DEFAULT_HOURS * 60  # e.g., 1560 for 26 hours
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    # 5. For each block, process its timeline into one row per minute
    for block_id in all_blocks:
        print(f"\nProcessing block {block_id}...")
        block_subset = merged_df[merged_df["block_id"] == block_id].copy()

        trip_ids = block_subset["trip_id"].unique()
        print(f"Found {len(trip_ids)} trip(s) in block {block_id}.")

        # Check for overlapping trips
        check_for_overlapping_trips(block_subset, block_id)

        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            print(
                f"Block {block_id} has {len(trip_ids)} trips which exceeds the limit of "
                f"{MAX_TRIPS_PER_BLOCK}. Skipping this block."
            )
            continue

        block_schedule_df = process_block(
            block_subset, block_id, timeline, BUS_STOP_CLUSTERS
        )

        # NEW: Fill Stop ID for DWELL / LAYOVER / LOADING
        block_schedule_df = fill_stop_ids_for_dwell_layover_loading(block_schedule_df)

        # Sort and save
        block_schedule_df.sort_values("Timestamp", inplace=True)

        block_route_ids = block_subset["route_id"].dropna().unique()
        block_route_str = (
            "-".join(str(r) for r in block_route_ids)
            if len(block_route_ids) > 0
            else "NA"
        )
        output_file = os.path.join(
            OUTPUT_FOLDER, f"block_{block_id}_{block_route_str}.xlsx"
        )
        block_schedule_df.to_excel(output_file, index=False)
        print(f"Finished processing block {block_id}. Schedule saved to {output_file}")

    print("All block-level spreadsheets have been generated.")


if __name__ == "__main__":
    main()
