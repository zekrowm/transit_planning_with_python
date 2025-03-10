"""
Script to process GTFS data in three phases:
1. GTFS to minute-by-minute block status spreadsheets
2. Minute-by-minute block status spreadsheets to event/conflict spreadsheets.
3. Route/stop id assignment optimization and final assessment.
"""

import os
import pandas as pd

###############################################################################
#                       UNIFIED CONFIGURATION SECTION
###############################################################################

GTFS_FOLDER = r"Path\To\Your\GTFS_Folder"
BLOCK_OUTPUT_FOLDER = (
    r"Path\To\Your\Block_Output_Folder"
)
DEFAULT_HOURS = 26
TIME_INTERVAL_MINUTES = 1

DWELL_THRESHOLD = 3
LAYOVER_THRESHOLD = 20
MAX_TRIPS_PER_BLOCK = 150
CALENDAR_SERVICE_IDS = [3]

CLUSTER_DEFINITIONS = {
    "Metro": {
        "stops": ["2956", "2955", "65", "3295", "2957", "66", "3296", "64", "63"],
        "overflow_bays": ["OverflowA", "OverflowB", "OverflowC"],
        "two_bay_stops": [],    # not used by the simplified conflict logic
        "three_bay_stops": [],  # not used by the simplified conflict logic
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
BUS_BAY_CLUSTERS_STEP2 = {
    name: cluster_info["stops"] for name, cluster_info in CLUSTER_DEFINITIONS.items()
}

CONFLICT_OUTPUT_FOLDER = (
    r"Path\To\Your\Conflict_Output_Folder"
)
FINAL_SOLVER_OUTPUT_FOLDER = (
    r"Path\To\Your\Solver_Output_Folder"
)

NEW_ROUTE_FILE = r""  # optional new route minute-by-minute file

WEIGHT_CONFLICT = 1000
WEIGHT_ASSIGNMENT_CHANGE = 1

ALLOW_DIFFERENT_DIRECTIONS = False
PAIRED_ROUTES = []
ROUTE_BAY_PREFERENCES = {}

###############################################################################
#             SIMPLE CONFLICT CAPACITY DEFINITIONS & STATUSES
###############################################################################
# Cluster capacity = (# of official stops) + (# of overflow bays)
# Stop capacity = 1 (each official stop and each overflow bay is capacity 1)
# We'll define presence statuses (for cluster conflicts) and
# passenger-service statuses (for official stop conflicts).

PRESENCE_STATUSES = {
    "ARRIVE", "DEPART", "ARRIVE/DEPART", "DWELL", "LOADING", "LAYOVER", "LONG BREAK"
}
PASSENGER_SERVICE_STATUSES = {
    "ARRIVE", "DEPART", "ARRIVE/DEPART", "LOADING"
}

def build_cluster_capacities():
    """
    Returns a dict: {cluster_name: cluster_capacity},
    where cluster_capacity = (#stops + #overflow_bays).
    """
    capacities = {}
    for cname, cinfo in CLUSTER_DEFINITIONS.items():
        n_official = len(cinfo["stops"])
        n_overflow = len(cinfo["overflow_bays"])
        capacities[cname] = n_official + n_overflow
    return capacities

def build_stop_capacities():
    """
    Returns {stop_id: 1} for official stops and overflow bays alike,
    because we treat each one as capacity=1 in simpler logic.
    """
    stop_caps = {}
    for cname, cinfo in CLUSTER_DEFINITIONS.items():
        # all official stops => capacity 1
        for s in cinfo["stops"]:
            stop_caps[str(s)] = 1
        # overflow => capacity 1
        for s in cinfo["overflow_bays"]:
            stop_caps[str(s)] = 1
    return stop_caps

###############################################################################
# NEW UTILS FOR CONFLICT DETECTION
###############################################################################

def normalize_stop_id(stop_id):
    """Convert stop IDs like '2956.0' to '2956'."""
    if pd.isna(stop_id):
        return None
    sid_str = str(stop_id).strip()
    if sid_str.endswith(".0"):
        sid_str = sid_str[:-2]
    return sid_str

def assign_cluster_name(df_in):
    """
    For each row, check if the 'Stop ID' is in a cluster's official stops
    or overflow. Assign that cluster name in a 'ClusterName' column.
    """
    df_out = df_in.copy()
    df_out["ClusterName"] = None
    for cname, cinfo in CLUSTER_DEFINITIONS.items():
        these_stop_ids = set(map(str, cinfo["stops"] + cinfo["overflow_bays"]))
        mask = df_out["Stop ID"].isin(these_stop_ids)
        df_out.loc[mask, "ClusterName"] = cname
    return df_out

def find_cluster_conflicts(df_in):
    """
    Return set of (cluster_name, timestamp) with more vehicles present
    than the cluster capacity.
    """
    cluster_caps = build_cluster_capacities()
    conflict_set = set()

    # Only consider rows with presence statuses
    present_df = df_in[df_in["Status"].isin(PRESENCE_STATUSES)].copy()
    # Drop any that have no cluster name
    present_df = present_df.dropna(subset=["ClusterName"])

    group = present_df.groupby(["ClusterName", "Timestamp"])
    for (cname, ts), grp in group:
        cap = cluster_caps.get(cname, 1)
        if len(grp) > cap:
            conflict_set.add((cname, ts))
    return conflict_set

def find_stop_conflicts(df_in):
    """
    Return set of (stop_id, timestamp) with more vehicles in
    passenger-service statuses than that stop_id's capacity.
    Here capacity=1 for each official stop or overflow bay.
    """
    stop_caps = build_stop_capacities()
    conflict_set = set()

    pass_df = df_in[df_in["Status"].isin(PASSENGER_SERVICE_STATUSES)].copy()
    # Only consider rows that actually have a recognized stop_id
    pass_df = pass_df[pass_df["Stop ID"].notna()]

    group = pass_df.groupby(["Stop ID", "Timestamp"])
    for (sid, ts), grp in group:
        cap = stop_caps.get(sid, 1)
        if len(grp) > cap:
            conflict_set.add((sid, ts))
    return conflict_set

def annotate_conflicts(df_in, cluster_conflicts, stop_conflicts):
    """
    Add a 'ConflictType' column with "NONE","CLUSTER","STOP","BOTH".
    """
    df_out = df_in.copy()
    conflict_types = []
    for idx, row in df_out.iterrows():
        cname = row["ClusterName"]
        ts = row["Timestamp"]
        sid = row["Stop ID"]

        has_cluster_conf = False
        if pd.notna(cname):
            if (cname, ts) in cluster_conflicts:
                has_cluster_conf = True

        has_stop_conf = False
        if sid is not None and (sid, ts) in stop_conflicts:
            has_stop_conf = True

        if has_cluster_conf and has_stop_conf:
            conflict_types.append("BOTH")
        elif has_cluster_conf:
            conflict_types.append("CLUSTER")
        elif has_stop_conf:
            conflict_types.append("STOP")
        else:
            conflict_types.append("NONE")

    df_out["ConflictType"] = conflict_types
    return df_out


###############################################################################
# NEW: Just before we build the solver-friendly tables in Step 2,
# we annotate a "ConflictType" column in the final DataFrame.
###############################################################################

def time_to_minutes(time_str):
    parts = time_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = int(parts[2]) if len(parts) == 3 else 0
    return hh * 60 + mm + ss // 60

def minutes_to_hhmm(minutes_in):
    hh = minutes_in // 60
    mm = minutes_in % 60
    return f"{hh:02d}:{mm:02d}"

def find_cluster(stop_id, bus_stop_clusters):
    for cluster_item in bus_stop_clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None

def mark_first_and_last_stops(df_in):
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    df_out["is_first_stop"] = False
    df_out["is_last_stop"] = False
    for _, group in df_out.groupby("trip_id"):
        min_seq_idx = group["stop_sequence"].idxmin()
        max_seq_idx = group["stop_sequence"].idxmax()
        df_out.loc[min_seq_idx, "is_first_stop"] = True
        df_out.loc[max_seq_idx, "is_last_stop"] = True
    return df_out

def get_status_for_minute(minute, stop_times_sequence, bus_stop_clusters):
    """
    The same logic from your existing script...
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

        # (Identical to your code snippet)
        if minute == arr and is_last:
            return ("ARRIVE", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)
        if minute == dep and is_first:
            return ("DEPART", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)
        if minute == arr == dep:
            return ("ARRIVE/DEPART", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)
        if minute == arr:
            return ("ARRIVE", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)
        if minute == dep:
            return ("DEPART", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)
        if arr < minute < dep:
            return ("DWELL", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                    trip_id, stop_seq, t_val)

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
                # same trip => traveling
                if trip_id == next_trip_id:
                    return ("TRAVELING BETWEEN STOPS", None, None, None, None, trip_id, None, 0)
                # different trip => dwell/layover
                gap = next_arr - dep
                same_stop = (s_id == next_stop_id)

                # check cluster membership
                same_cluster = False
                if bus_stop_clusters:
                    current_cluster = find_cluster(s_id, bus_stop_clusters)
                    next_cluster = find_cluster(next_stop_id, bus_stop_clusters)
                    same_cluster = (current_cluster and next_cluster and current_cluster == next_cluster)

                if same_stop or same_cluster:
                    if gap <= DWELL_THRESHOLD:
                        return ("DWELL", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                                next_trip_id, stop_seq, t_val)
                    if gap > LAYOVER_THRESHOLD:
                        return ("LONG BREAK", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                                next_trip_id, stop_seq, t_val)
                    return ("LAYOVER", s_id, s_name, minutes_to_hhmm(arr), minutes_to_hhmm(dep),
                            next_trip_id, stop_seq, t_val)

                if bus_stop_clusters:
                    return ("DEADHEAD", None, None, None, None, next_trip_id, None, 0)
                return ("LAYOVER/DEADHEAD", None, None, None, None, next_trip_id, None, 0)

    return ("EMPTY", None, None, None, None, None, None, 0)

def process_block(block_subset, block_id, timeline, bus_stop_clusters):
    """
    Identical to your original block expansion logic...
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
    rows = []

    for minute in timeline:
        possible_trips = [trip for trip in trips_summary if trip["start"] <= minute <= trip["end"]]
        if possible_trips:
            candidate_info = []
            for trip_obj in possible_trips:
                status_tuple = get_status_for_minute(minute, trip_obj["stop_times_sequence"], bus_stop_clusters)
                candidate_info.append((trip_obj, status_tuple))

            valid_candidates = [(trip_obj, stat) for (trip_obj, stat) in candidate_info if stat[0] != "EMPTY"]
            if not valid_candidates:
                chosen_trip = None
                chosen_status = ("EMPTY", None, None, None, None, None, None, 0)
            elif len(valid_candidates) == 1:
                chosen_trip, chosen_status = valid_candidates[0]
            else:
                def candidate_sort_key(item):
                    stat = item[1]
                    stop_seq = stat[6] if stat[6] else 999999
                    t_val = stat[7] if stat[7] else 0
                    is_timepoint = t_val in [1, 2]
                    return (not is_timepoint, stop_seq)

                valid_candidates.sort(key=candidate_sort_key)
                chosen_trip, chosen_status = valid_candidates[0]
        else:
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

        if chosen_trip:
            if status == "EMPTY":
                status = "TRAVELING BETWEEN STOPS"
            if status in ["DWELL", "LAYOVER"]:
                # check next minute
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
            # No chosen trip => fill as "INACTIVE", "DWELL", or "LAYOVER" etc.
            prev_trip = None
            next_trip = None
            for trip_obj in trips_summary:
                if trip_obj["end"] < minute:
                    if (prev_trip is None) or (trip_obj["end"] > prev_trip["end"]):
                        prev_trip = trip_obj
                if trip_obj["start"] > minute:
                    if (next_trip is None) or (trip_obj["start"] < next_trip["start"]):
                        next_trip = trip_obj

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

            if (next_trip
                and next_trip["start"] == minute + 1
                and status in ["DWELL", "LAYOVER"]):
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

    df_outcome = pd.DataFrame(rows)
    return df_outcome

def check_for_overlapping_trips(block_subset, block_id):
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
    print("=== Step 1: Reading GTFS and generating block-level schedules ===")
    os.makedirs(BLOCK_OUTPUT_FOLDER, exist_ok=True)

    trips_path = os.path.join(GTFS_FOLDER, "trips.txt")
    stop_times_path = os.path.join(GTFS_FOLDER, "stop_times.txt")
    stops_path = os.path.join(GTFS_FOLDER, "stops.txt")
    blocks_path = os.path.join(GTFS_FOLDER, "blocks.txt")  # optional

    print("Reading GTFS files...")
    trips_df = pd.read_csv(trips_path)
    stop_times_df = pd.read_csv(stop_times_path)
    stops_df = pd.read_csv(stops_path)

    if os.path.exists(blocks_path):
        blocks_df = pd.read_csv(blocks_path)
        print("Blocks file found and read.")
    else:
        blocks_df = pd.DataFrame()
        print("No blocks file found; proceeding without it.")

    if CALENDAR_SERVICE_IDS:
        print(f"Filtering trips to service_ids in {CALENDAR_SERVICE_IDS} ...")
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    print("Merging data and converting times to minutes...")
    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(time_to_minutes)

    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]
    merged_df = pd.merge(stop_times_df, trips_df, on="trip_id", how="left")

    if "timepoint" in stops_df.columns:
        stops_merge_cols = ["stop_id", "stop_name", "timepoint"]
    else:
        stops_merge_cols = ["stop_id", "stop_name"]
    merged_df = pd.merge(merged_df, stops_df[stops_merge_cols], on="stop_id", how="left")

    print("Marking first/last stops...")
    merged_df = mark_first_and_last_stops(merged_df)

    if "timepoint" not in merged_df.columns:
        merged_df["timepoint"] = 0
    else:
        merged_df["timepoint"] = pd.to_numeric(merged_df["timepoint"], errors="coerce").fillna(0).astype(int)

    merged_df.loc[(merged_df["is_first_stop"]) & (merged_df["timepoint"] == 0),"timepoint"] = 2
    merged_df.loc[(merged_df["is_last_stop"]) & (merged_df["timepoint"] == 0),"timepoint"] = 2

    all_blocks = merged_df["block_id"].dropna().unique()
    print(f"Identified {len(all_blocks)} block(s) to process.")

    max_minutes = DEFAULT_HOURS * 60
    timeline = range(0, max_minutes, TIME_INTERVAL_MINUTES)

    for blk_id in all_blocks:
        print(f"\nProcessing block {blk_id}...")
        block_subset = merged_df[merged_df["block_id"] == blk_id].copy()
        trip_ids = block_subset["trip_id"].unique()
        print(f"Found {len(trip_ids)} trip(s) in block {blk_id}.")

        check_for_overlapping_trips(block_subset, blk_id)
        if len(trip_ids) > MAX_TRIPS_PER_BLOCK:
            print(
                f"Block {blk_id} has {len(trip_ids)} trips > limit {MAX_TRIPS_PER_BLOCK}, skipping."
            )
            continue

        block_schedule_df = process_block(block_subset, blk_id, timeline, BUS_STOP_CLUSTERS_STEP1)
        block_schedule_df = fill_stop_ids_for_dwell_layover_loading(block_schedule_df)
        block_schedule_df.sort_values("Timestamp", inplace=True)

        block_route_ids = block_subset["route_id"].dropna().unique()
        if len(block_route_ids) > 0:
            block_route_str = "_".join(str(rte) for rte in block_route_ids)
        else:
            block_route_str = "NA"

        out_path = os.path.join(BLOCK_OUTPUT_FOLDER, f"block_{blk_id}_{block_route_str}.xlsx")
        block_schedule_df.to_excel(out_path, index=False)
        print(f"Finished block {blk_id}; saved to {out_path}")

    print("\nStep 1 complete: All block-level spreadsheets generated.")


###############################################################################
#                   STEP 2: Conflict Detection & Solver
###############################################################################
def normalize_route(route):
    if pd.isna(route):
        return ""
    route_str = str(route).strip()
    if route_str.endswith(".0"):
        route_str = route_str[:-2]
    return route_str


def gather_block_spreadsheets(block_folder):
    all_files = [
        f_name
        for f_name in os.listdir(block_folder)
        if f_name.lower().endswith(".xlsx") and f_name.startswith("block_")
    ]
    if not all_files:
        raise ValueError(f"No block-level XLSX files found in {block_folder}")
    big_df_list = []
    for fname in all_files:
        fpath = os.path.join(block_folder, fname)
        block_df = pd.read_excel(fpath)
        block_df["FileName"] = fname
        big_df_list.append(block_df)
    df_combined = pd.concat(big_df_list, ignore_index=True)
    print("Loaded total rows from block XLSX files:", len(df_combined))
    return df_combined

def run_step2_conflict_detection():
    """
    We produce a minute-by-minute conflict check for the original stop usage
    *and* generate the solver-friendly tables as before.
    """
    print("\n=== Step 2: Conflict detection & solver file creation ===")
    os.makedirs(CONFLICT_OUTPUT_FOLDER, exist_ok=True)

    df = gather_block_spreadsheets(BLOCK_OUTPUT_FOLDER)

    expected_cols = [
        "Timestamp",
        "Trip ID",
        "Block",
        "Route",
        "Direction",
        "Stop ID",
        "Stop Name",
        "Stop Sequence",
        "Arrival Time",
        "Departure Time",
        "Status",
        "Timepoint",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in block-level DataFrame.")

    # Clean up Stop ID and Route
    df["Stop ID"] = df["Stop ID"].apply(normalize_stop_id)
    df["Timestamp"] = df["Timestamp"].astype(str).str.strip()
    df["Route"] = df["Route"].apply(normalize_route)
    print("Rows read:", len(df))

    # -------------------- NEW: Conflict detection logic ---------------------
    # 1) Assign clusters
    df = assign_cluster_name(df)

    # 2) Find cluster conflicts and stop conflicts
    cluster_conflicts = find_cluster_conflicts(df)
    stop_conflicts = find_stop_conflicts(df)

    # 3) Annotate each row with "ConflictType"
    df_annotated = annotate_conflicts(df, cluster_conflicts, stop_conflicts)

    # 4) Optionally save a combined conflict-annotated file
    combined_out = os.path.join(CONFLICT_OUTPUT_FOLDER, "AllClusters_Full_LongForm_WithConflicts.xlsx")
    df_annotated.to_excel(combined_out, index=False)
    print(f"Saved conflict-annotated file to {combined_out}")

    # If you want per-cluster files:
    for cname in CLUSTER_DEFINITIONS.keys():
        sub = df_annotated[df_annotated["ClusterName"] == cname].copy()
        if sub.empty:
            continue
        safe_name = cname.replace(" ", "_")
        c_path = os.path.join(CONFLICT_OUTPUT_FOLDER, f"{safe_name}_Full_LongForm_WithConflicts.xlsx")
        sub.to_excel(c_path, index=False)
        print(f"Saved conflict-annotated file for cluster '{cname}' to {c_path}")

    # Count distinct conflict time points
    print(f"Distinct cluster-conflict points: {len(cluster_conflicts)}")
    print(f"Distinct stop-conflict points: {len(stop_conflicts)}")

    # ------------------ Existing code to build solver-friendly tables -------
    # We'll continue to use df_annotated (which has "ConflictType" but that doesn't
    # hurt the solver tables).
    occupancy_events = [
        "ARRIVE", "DEPART", "ARRIVE/DEPART", "DWELL",
        "LOADING", "LAYOVER", "LONG BREAK"
    ]
    for cluster_name, _c_info in CLUSTER_DEFINITIONS.items():
        cluster_df = df_annotated[df_annotated["ClusterName"] == cluster_name].copy()
        if cluster_df.empty:
            print(f"No data for cluster {cluster_name}, skipping.")
            continue

        occ_mask = cluster_df["Status"].isin(occupancy_events)
        solver_df = cluster_df[occ_mask].copy()
        if solver_df.empty:
            print(f"No occupancy events for solver table in {cluster_name}, skipping.")
        else:
            solver_cols = [
                "Timestamp",
                "ClusterName",
                "Stop ID",
                "Stop Name",
                "Trip ID",
                "Block",
                "Route",
                "Direction",
                "Status",
                "ConflictType",  # keep conflict info if you like
            ]
            solver_df = solver_df[solver_cols].drop_duplicates().copy()

            solver_df.rename(
                columns={
                    "Stop ID": "stop_id",
                    "Stop Name": "stop_name",
                    "Trip ID": "trip_id",
                    "Status": "Event",
                },
                inplace=True,
            )

            solver_df.sort_values(["Timestamp"], inplace=True)

            solver_out = os.path.join(
                CONFLICT_OUTPUT_FOLDER,
                f"{cluster_name}_LongFormat_Solver.xlsx",
            )
            solver_df.to_excel(solver_out, index=False)
            print(f"Solver-friendly long format table saved to {solver_out}")

    print("\nStep 2 complete: conflict detection and solver files generated.")


###############################################################################
#                STEP 3: Solver with Route->Slot Assignment
###############################################################################
def load_long_format_excels(folder):
    files = [
        f_name for f_name in os.listdir(folder)
        if f_name.lower().endswith("_longformat_solver.xlsx")
    ]
    if not files:
        raise FileNotFoundError(
            f"No *_LongFormat_Solver.xlsx files found in {folder}."
        )
    df_list = []
    for f_name in files:
        fpath = os.path.join(folder, f_name)
        temp = pd.read_excel(fpath)
        df_list.append(temp)
    big_df = pd.concat(df_list, ignore_index=True)
    print("Loaded long-format solver data. Rows:", len(big_df))
    return big_df

def preprocess_solver_df(df_in):
    df_out = df_in.copy()
    columns_to_clean = [
        "stop_id",
        "stop_name",
        "trip_id",
        "Block",
        "Route",
        "Direction",
        "ClusterName",
    ]
    for col in columns_to_clean:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).str.strip()
        else:
            df_out[col] = ""

    df_out["Timestamp"] = df_out["Timestamp"].astype(str).str.strip()

    if ALLOW_DIFFERENT_DIRECTIONS:
        df_out["RouteKey"] = df_out["Route"] + "_" + df_out["Direction"].replace("", "NA")
    else:
        df_out["RouteKey"] = df_out["Route"]

    df_out["OriginalStopID"] = df_out["stop_id"]
    return df_out

def load_new_route(new_route_path):
    if not new_route_path:
        return pd.DataFrame()
    if not os.path.exists(new_route_path):
        raise FileNotFoundError(f"New route file not found: {new_route_path}")
    new_df = pd.read_excel(new_route_path)
    if "ClusterName" not in new_df.columns:
        raise ValueError("New route file must include a 'ClusterName' column.")
    return new_df

###############################################################################
# Build a dictionary of "slots" for each cluster
###############################################################################
def build_cluster_slot_definitions():
    """
    Returns a dict:
      cluster_name -> [slot1, slot2, ...]
    where each official stop in 'two_bay_stops' or 'three_bay_stops' gets multiple
    slots. Overflow bays are each single-slot, etc.
    """
    cluster_slots = {}
    for cname, cinfo in CLUSTER_DEFINITIONS.items():
        slots = []
        for stop_id in cinfo["stops"]:
            # Simplify if you like, or handle multi-bay if populated
            if stop_id in cinfo["three_bay_stops"]:
                cap = 3
            elif stop_id in cinfo["two_bay_stops"]:
                cap = 2
            else:
                cap = 1
            for b_num in range(1, cap + 1):
                slot_id = f"{stop_id}_bay{b_num}"
                slots.append(slot_id)

        # overflow each as single
        for ovf in cinfo["overflow_bays"]:
            slots.append(ovf)
        cluster_slots[cname] = slots
    return cluster_slots

###############################################################################
# Compute conflicts, build solver, etc. (identical to your code)
###############################################################################
def compute_conflicts_for_assignment(sub_df, route_to_slot):
    """
    ...
    (unchanged from your original approach)
    """
    # your existing occupancy event logic
    occupancy_events = {
        "ARRIVE","DEPART","ARRIVE/DEPART","DWELL","LOADING","LAYOVER","LONG BREAK"
    }
    occ = sub_df[sub_df["Event"].isin(occupancy_events)].copy()
    if occ.empty:
        return 0, 0, 0

    occ["AssignedSlot"] = occ["RouteKey"].map(route_to_slot)

    conflict_count = 0
    for _, group in occ.groupby(["Timestamp"]):
        by_slot = group.groupby("AssignedSlot")
        for slot_id, subg in by_slot:
            n_rows = len(subg)
            if n_rows > 1:
                # simple count of pairwise conflicts
                conflict_count += n_rows * (n_rows - 1) // 2

    # Count route assignment changes vs. their originalStopID mode
    route_changes = 0
    for rkey, rg in occ.groupby("RouteKey"):
        original_stops = rg["OriginalStopID"].dropna().unique().tolist()
        if not original_stops:
            continue
        mode_stop = pd.Series(original_stops).mode()[0]
        final_slot = route_to_slot.get(rkey, "")
        if final_slot != mode_stop:
            route_changes += 1

    conflict_cost = conflict_count * WEIGHT_CONFLICT
    change_cost = route_changes * WEIGHT_ASSIGNMENT_CHANGE
    total_cost = conflict_cost + change_cost
    return total_cost, conflict_count, route_changes

def build_initial_slot_assignment(sub_df, cluster_slots):
    """
    ...
    """
    occupancy_events = {
        "ARRIVE","DEPART","ARRIVE/DEPART","DWELL","LOADING","LAYOVER","LONG BREAK"
    }
    occ = sub_df[sub_df["Event"].isin(occupancy_events)].copy()
    assignment = {}

    for route_key, group in occ.groupby("RouteKey"):
        original_stops = group["OriginalStopID"].dropna().tolist()
        if not original_stops:
            if cluster_slots:
                assignment[route_key] = cluster_slots[0]
            else:
                assignment[route_key] = None
            continue
        mode_stop = pd.Series(original_stops).mode()[0]
        candidates = [s for s in cluster_slots if s.startswith(mode_stop + "_bay")]
        if candidates:
            possible_slot = candidates[0]
        else:
            possible_slot = cluster_slots[0] if cluster_slots else None
        assignment[route_key] = possible_slot

    return assignment

def apply_preferences_and_pairings(assignment, cluster_slots):
    """
    ...
    """
    # do nothing by default
    return assignment

def attempt_greedy_slot_reassign(sub_df, cluster_slots, initial_assignment):
    """
    ...
    """
    current_assignment = initial_assignment.copy()
    best_cost, best_conflicts, best_changes = compute_conflicts_for_assignment(sub_df, current_assignment)
    improved = True

    while improved:
        improved = False
        route_keys = sorted(current_assignment.keys())

        for rkey in route_keys:
            orig_slot = current_assignment[rkey]
            local_best_slot = orig_slot
            local_best_cost = best_cost

            for slot_id in cluster_slots:
                if slot_id == orig_slot:
                    continue
                current_assignment[rkey] = slot_id
                test_cost, _, _ = compute_conflicts_for_assignment(sub_df, current_assignment)
                if test_cost < local_best_cost:
                    local_best_cost = test_cost
                    local_best_slot = slot_id

            if local_best_slot != orig_slot:
                current_assignment[rkey] = local_best_slot
                best_cost = local_best_cost
                improved = True
                break

    return current_assignment

def rebuild_minute_by_minute_with_slots(sub_df, final_assignment):
    sub_out = sub_df.copy()
    sub_out["AssignedSlot"] = sub_out["RouteKey"].map(final_assignment)
    return sub_out

def build_wide_table_slot_based(sub_df, cluster_name, cluster_slots):
    """
    ...
    """
    occupancy_events = {
        "ARRIVE","DEPART","ARRIVE/DEPART","DWELL","LOADING","LAYOVER","LONG BREAK"
    }
    occ = sub_df[sub_df["Event"].isin(occupancy_events)].copy()
    if occ.empty:
        return pd.DataFrame()

    occ["OccupantString"] = occ.apply(
        lambda r: f"{r['Block']}({r['Route']})/{r['Event']}",
        axis=1,
    )
    occupant_map = occ.groupby(["Timestamp","AssignedSlot"])["OccupantString"].apply(list).to_dict()

    def to_total_minutes(ts_str):
        hh_val, mm_val = map(int, ts_str.split(":"))
        return hh_val * 60 + mm_val

    min_ts = occ["Timestamp"].min()
    max_ts = occ["Timestamp"].max()
    start_min = to_total_minutes(min_ts)
    end_min = to_total_minutes(max_ts)

    all_minutes = []
    for t_val in range(start_min, end_min + 1):
        hh_val = t_val // 60
        mm_val = t_val % 60
        all_minutes.append(f"{hh_val:02d}:{mm_val:02d}")

    out_df = pd.DataFrame(index=all_minutes)
    out_df.index.name = "Timestamp"

    for slot_id in cluster_slots:
        out_df[slot_id] = ""

    out_df["conflict"] = False
    out_df["ConflictBlocks"] = ""

    for current_ts in all_minutes:
        row_conflicts = []
        for slot_id in cluster_slots:
            occupant_list = occupant_map.get((current_ts, slot_id), [])
            if len(occupant_list) > 1:
                row_conflicts.extend(occupant_list)
                combined = " & ".join(occupant_list)
                out_df.at[current_ts, slot_id] = combined
        if row_conflicts:
            out_df.at[current_ts, "conflict"] = True
            out_df.at[current_ts, "ConflictBlocks"] = "; ".join(row_conflicts)

    out_df.reset_index(inplace=True)
    out_df["conflict"] = out_df["conflict"].replace({True: "TRUE", False: "FALSE"})
    return out_df

def run_step3_solver():
    print("\n=== Step 3: Slot-based solver ===")
    os.makedirs(FINAL_SOLVER_OUTPUT_FOLDER, exist_ok=True)

    solver_df = load_long_format_excels(CONFLICT_OUTPUT_FOLDER)
    solver_df = preprocess_solver_df(solver_df)

    new_df = load_new_route(NEW_ROUTE_FILE)
    if not new_df.empty:
        print("Appending new route data to solver DataFrame...")
        new_df = preprocess_solver_df(new_df)
        solver_df = pd.concat([solver_df, new_df], ignore_index=True)

    cluster_slot_map = build_cluster_slot_definitions()

    for cluster_name in cluster_slot_map.keys():
        print(f"\n--- Solving for cluster: {cluster_name} ---")
        sub_df = solver_df[solver_df["ClusterName"] == cluster_name].copy()
        if sub_df.empty:
            print(f"No data for {cluster_name}, skipping.")
            continue

        cluster_slots = cluster_slot_map[cluster_name]
        if not cluster_slots:
            print(f"No slot definitions for {cluster_name}, skipping.")
            continue

        init_assign = build_initial_slot_assignment(sub_df, cluster_slots)
        original_assignment = init_assign.copy()

        init_assign = apply_preferences_and_pairings(init_assign, cluster_slots)
        final_assign = attempt_greedy_slot_reassign(sub_df, cluster_slots, init_assign)

        final_cost, final_conflicts, final_changes = compute_conflicts_for_assignment(sub_df, final_assign)
        if final_conflicts > 0:
            print(f"WARNING: {final_conflicts} conflict-pairs remain in {cluster_name}.")
        else:
            print(f"No conflicts remain in {cluster_name}.")

        print(f"Weighted cost = {final_cost}, route assignment changes vs. original = {final_changes}")

        # Save route->slot assignments
        assignment_rows = []
        for rkey in sorted(final_assign.keys()):
            rep_row = sub_df[sub_df["RouteKey"] == rkey].iloc[0]
            route = rep_row["Route"]
            direction = ""
            if ALLOW_DIFFERENT_DIRECTIONS:
                direction = rep_row["Direction"]

            orig_stop = original_assignment.get(rkey, "")
            new_slot = final_assign[rkey]

            assignment_rows.append({
                "ClusterName": cluster_name,
                "RouteKey": rkey,
                "Route": route,
                "Direction": direction,
                "OriginalStopID_Mode": orig_stop,
                "FinalSlotID": new_slot,
            })
        df_assignment = pd.DataFrame(assignment_rows)
        assign_path = os.path.join(
            FINAL_SOLVER_OUTPUT_FOLDER, f"{cluster_name}_RouteSlotAssignments.xlsx"
        )
        df_assignment.to_excel(assign_path, index=False)
        print(f"Route->Slot assignments saved to {assign_path}")

        # Revised minute-by-minute
        revised_sub = rebuild_minute_by_minute_with_slots(sub_df, final_assign)
        revised_path = os.path.join(
            FINAL_SOLVER_OUTPUT_FOLDER, f"{cluster_name}_RevisedMinuteByMinute.xlsx"
        )
        revised_sub.to_excel(revised_path, index=False)
        print(f"Revised long-format with slot assignment saved to {revised_path}")

        # Wide format
        final_wide_df = build_wide_table_slot_based(revised_sub, cluster_name, cluster_slots)
        if not final_wide_df.empty:
            final_wide_path = os.path.join(
                FINAL_SOLVER_OUTPUT_FOLDER, f"{cluster_name}_FinalSlotMinuteByMinute.xlsx"
            )
            final_wide_df.to_excel(final_wide_path, index=False)
            print(f"Final wide table (slot-based) saved to {final_wide_path}")
            if final_wide_df["conflict"].eq("TRUE").any():
                print("Note: Some final slot conflicts remain for this cluster.")
        else:
            print("No occupancy for final wide table. Possibly no events?")

    print("\nStep 3 complete: route->slot assignments and final schedules generated.")

###############################################################################
#                        MASTER MAIN FUNCTION
###############################################################################
def main():
    run_step1_gtfs_to_blocks()
    run_step2_conflict_detection()  # now includes conflict annotation
    run_step3_solver()

if __name__ == "__main__":
    main()
