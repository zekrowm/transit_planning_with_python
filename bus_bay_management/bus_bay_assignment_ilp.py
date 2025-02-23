#!/usr/bin/env python
# coding: utf-8
"""
This script optimizes bus bay assignments by analyzing GTFS data.
It supports two optimization approaches:
  1) ILP (integer linear programming) using pulp
  2) Greedy (assign busiest routes first)

A note for users: ILP can become very slow when the number of bays
and routes becomes large (e.g., >4 bays, >12 routes). For large cases,
consider using the "greedy" approach for faster run times.
"""

import os
import re
from collections import defaultdict

from openpyxl import Workbook
import pandas as pd
import pulp  # Required for ILP
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpBinary, LpInteger, LpStatus, value

###############################################################################
# CONFIGURATION
###############################################################################

CONFIG = {
    "gtfs_folder": r"C:\Your\Path\To\GTFS_data_folder",
    "whitelisted_service_id": '1',
    "stops_of_interest": ['1001', '1002'],
    "num_bays_per_stop": {
        "1001": 1,
        "1002": 1
    },
    "output_folder": r"C:\Your\Folder\Path\for\Output",
    "comparison_output_filename": "BayAssignment_Comparison.xlsx",
    "extended_offservice_threshold": 60,  # in minutes
    "allow_splitting_by_direction": True, # True => route-direction separate, False => same bay
    "optimization_approach": "ilp"       # "ilp" or "greedy"
}

LAYOVER_THRESHOLD = 15  # in minutes

###############################################################################
# SNIPPET-STYLE HELPER FUNCTIONS
###############################################################################

def time_to_seconds(time_str):
    """
    Convert HH:MM:SS format to total seconds, even beyond 24:00:00.
    """
    parts = time_str.split(":")
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def get_trip_ranges_and_ends(block_segments):
    """
    Return a list of tuples with trip-related info:
    (trip_id, trip_start, trip_end, route_short_name, direction_id, start_stop, end_stop).
    """
    trips_info = []
    unique_trips = block_segments['trip_id'].unique()
    for trip_id in unique_trips:
        tsub = block_segments[block_segments['trip_id'] == trip_id].sort_values('arrival_seconds')
        trip_start = tsub['arrival_seconds'].min()
        trip_end = tsub['departure_seconds'].max()
        route_short_name = tsub['route_short_name'].iloc[0]
        direction_id = tsub['direction_id'].iloc[0]
        start_stop = tsub.iloc[0]['stop_id']
        end_stop = tsub.iloc[-1]['stop_id']
        trips_info.append((
            trip_id,
            trip_start,
            trip_end,
            route_short_name,
            direction_id,
            start_stop,
            end_stop
        ))
    trips_info.sort(key=lambda x: x[1])  # sort by trip_start
    return trips_info

def get_minute_status_location_complex(
    minute: int,
    block_segments: pd.DataFrame,
    trips_info: list,
    layover_threshold: int = LAYOVER_THRESHOLD,
    extended_offservice_threshold: int = CONFIG["extended_offservice_threshold"]
):
    """
    Determine the bus's status and location at a given minute for the given block.
    Returns (status, location, route_short_name, direction_id, stop_id).
    """
    current_sec = minute * 60

    if not trips_info:
        # No trips in this block => always inactive
        return ("inactive", "inactive", "", "", "")

    # Identify earliest start and latest end among all trips
    earliest_start = min(trp[1] for trp in trips_info)
    latest_end = max(trp[2] for trp in trips_info)

    # If we’re before the first trip or after the last trip
    if current_sec < earliest_start or current_sec > latest_end:
        return ("inactive", "inactive", "", "", "")

    # Check if we are in the window of a specific trip
    active_trip_idx = None
    for idx, (tid, tstart, tend, rname, dirid, stp_st, stp_end) in enumerate(trips_info):
        if tstart <= current_sec <= tend:
            active_trip_idx = idx
            break

    if active_trip_idx is not None:
        (tid, tstart, tend, rname, dirid, start_stp, end_stp) = trips_info[active_trip_idx]
        tsub = block_segments[block_segments['trip_id'] == tid].sort_values('arrival_seconds')

        # Step through each stop in this trip
        for _, row in tsub.iterrows():
            arr_sec = row['arrival_seconds']
            dep_sec = row['departure_seconds']
            next_stp = row['next_stop_id']
            next_arr = row['next_arrival_seconds']

            # Dwelling at the current stop
            if arr_sec <= current_sec <= dep_sec:
                return ("dwelling at stop", row['stop_id'], rname, dirid, row['stop_id'])

            # traveling vs. layover
            if pd.notnull(next_arr):
                narr_sec = int(next_arr)
                if dep_sec < current_sec < narr_sec:
                    # Could be traveling or short layover
                    if next_stp == row['stop_id']:
                        gap = narr_sec - dep_sec
                        if gap > layover_threshold * 60:
                            return ("laying over", row['stop_id'], rname, dirid, row['stop_id'])
                        return ("running route", "traveling between stops", rname, dirid, "")
                    return ("running route", "traveling between stops", rname, dirid, "")

        # If we reach here, we might be beyond the last stop's departure_seconds
        return ("laying over", end_stp, rname, dirid, end_stp)

    # We’re between trips in this block. Determine short layover vs off-service
    prev_trip = None
    next_trip = None
    for idx, (tid, tstart, tend, rname, dirid, stp_st, stp_end) in enumerate(trips_info):
        if tend < current_sec:
            prev_trip = (tid, tstart, tend, rname, dirid, stp_st, stp_end)
        if current_sec < tstart and next_trip is None:
            next_trip = (tid, tstart, tend, rname, dirid, stp_st, stp_end)
            break

    if not prev_trip:
        # We haven't reached the first trip's start yet
        return ("inactive", "inactive", "", "", "")

    (ptid, ptstart, ptend, prname, pdirid, pstart_stp, pend_stp) = prev_trip
    gap_to_next_trip = None
    if next_trip:
        (ntid, ntstart, ntend, nrname, ndirid, nstart_stp, nend_stp) = next_trip
        gap_to_next_trip = ntstart - ptend

    if gap_to_next_trip is not None:
        if gap_to_next_trip <= layover_threshold * 60:
            return ("laying over", pend_stp, prname, pdirid, pend_stp)
        return ("off-service", "inactive", "", "", "")

    # No next trip => after the last trip but inside earliest..latest => off-service
    return ("off-service", "inactive", "", "", "")

###############################################################################
# LOADING & PROCESSING GTFS
###############################################################################

def load_gtfs_data(gtfs_folder):
    """
    Load essential GTFS CSVs into DataFrames.
    """
    required_files = ["trips.txt", "stop_times.txt", "routes.txt", "stops.txt", "calendar.txt"]
    data = {}
    for file in required_files:
        path = os.path.join(gtfs_folder, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required GTFS file not found: {path}")
        print(f"Loading {file}...")
        data[file.split('.', maxsplit=1)[0]] = pd.read_csv(path, dtype=str)
    return data

###############################################################################
# BUILD OCCUPANCY
###############################################################################

def build_occupancy(block_segments, stops_of_interest, split_by_direction=False):
    """
    Build a dictionary for occupancy:
        - Key = route_short_name  OR  (route_short_name, direction_id)
        - Value = {minute: occupant_count}

    If split_by_direction=True, we use (route_short_name, direction_id) as the key,
    otherwise just the route_short_name alone.
    """
    occupancy = defaultdict(lambda: defaultdict(int))

    blocks = block_segments['block_id'].unique()
    for blk in blocks:
        block_data = block_segments[block_segments['block_id'] == blk].copy()
        if block_data.empty:
            continue

        trips_info = get_trip_ranges_and_ends(block_data)
        if not trips_info:
            continue

        min_sec = min(t[1] for t in trips_info)  # earliest start
        max_sec = max(t[2] for t in trips_info)  # latest end
        start_min = min_sec // 60
        end_min = max_sec // 60

        for minute in range(start_min, end_min + 1):
            status, location, rname, dirid, stop_id = get_minute_status_location_complex(
                minute, block_data, trips_info
            )
            if status in ["dwelling at stop", "laying over"] and location in stops_of_interest:
                if rname:
                    if split_by_direction and dirid in ['0', '1', 0, 1]:
                        direction_str = str(dirid)
                        occupancy[(rname, direction_str)][minute] += 1
                    else:
                        occupancy[rname][minute] += 1

    # Convert to normal dicts
    occupancy = {k: dict(v) for k, v in occupancy.items()}
    return occupancy

###############################################################################
# DEFAULT ASSIGNMENT (FROM GTFS)
###############################################################################

def build_time_based_assignments(occupancy_dict, bay_labels):
    """
    A time-based "default" assignment approach, using actual schedule occupancy:

    1) For each key (route), gather its earliest minute of occupancy.
    2) Sort routes by that earliest minute.
    3) For each route in that order:
       - Find a bay that causes no (or minimal) conflict with already-assigned routes.
       - Assign the route to that bay.

    Returns:
      assignments: dict of { key -> bay_label }
    """
    # 1) Identify earliest minute of occupancy for each key
    #    If a route never occupies the stop, set it to large sentinel so it goes last
    route_earliest_min = {}
    for k, minute_dict in occupancy_dict.items():
        if minute_dict:
            route_earliest_min[k] = min(minute_dict.keys())
        else:
            # If this route doesn't occupy the stop at all
            route_earliest_min[k] = float('inf')

    # 2) Sort keys by earliest usage time
    all_keys_sorted = sorted(occupancy_dict.keys(), key=lambda x: route_earliest_min[x])

    # Keep track of assignments
    assignments = {}

    # Track which keys are in each bay (for conflict checks)
    # bay_to_keys = {bay_label: set_of_keys_assigned}
    bay_to_keys = {bay: set() for bay in bay_labels}

    # Function to compute incremental conflicts if we add 'new_key' into 'candidate_bay'
    def compute_conflict_if_assigned(new_key, candidate_bay):
        # Build a temporary assignment for conflict checking
        temp_assignments = assignments.copy()
        temp_assignments[new_key] = candidate_bay

        # Evaluate total conflicts (or just for new_key) using existing evaluate_conflicts
        conflicts_dict = evaluate_conflicts(occupancy_dict, temp_assignments)
        # Return the added conflict minutes specifically for new_key
        return conflicts_dict.get(new_key, 0)

    # 3) Assign each route to the best bay in ascending time order
    for key in all_keys_sorted:
        best_bay = None
        best_conflict = None

        for bay in bay_labels:
            conflict_here = compute_conflict_if_assigned(key, bay)

            if best_conflict is None or conflict_here < best_conflict:
                best_conflict = conflict_here
                best_bay = bay

        # Finalize the best bay
        assignments[key] = best_bay
        bay_to_keys[best_bay].add(key)

    return assignments

###############################################################################
# ILP: ASSIGN ONE BAY PER KEY
###############################################################################

def build_and_solve_ilp(occupancy_dict, bay_labels, stop_ids, problem_name="BayAssignment"):
    """
    ILP: Assign exactly one bay to each key in occupancy_dict.
    Minimize total conflicts across minutes.

    occupancy_dict:
      { key -> { minute -> occupant_count } }
      where key is either route_short_name or (route_short_name, direction_id)
    """
    print(f"\n=== ILP: {problem_name} ===")
    print(f"Stops of interest = {stop_ids}")

    # 1) Gather all keys and minutes
    all_keys = sorted(occupancy_dict.keys(), key=lambda x: str(x))
    all_minutes = sorted({m for occ in occupancy_dict.values() for m in occ.keys()})

    # 2) Create the model
    model = LpProblem(problem_name, LpMinimize)

    # Binary var x_{k,bay} = 1 if key k is assigned to bay
    x = {}
    for k in all_keys:
        safe_k = re.sub(r'[^A-Za-z0-9]', '_', str(k))
        for bay in bay_labels:
            safe_bay = re.sub(r'[^A-Za-z0-9]', '_', bay)
            var_name = f"x_{safe_k}_{safe_bay}"
            x[(k, bay)] = LpVariable(var_name, cat=LpBinary)

    # Each key must be assigned to exactly one bay
    for k in all_keys:
        model += (
            lpSum(x[(k, bay)] for bay in bay_labels) == 1,
            f"OneBayFor_{k}"
        )

    # Conflict var z_{bay,minute} >= occupantCount_{bay,minute} - 1
    z = {}
    for bay in bay_labels:
        safe_bay = re.sub(r'[^A-Za-z0-9]', '_', bay)
        for minute in all_minutes:
            var_name = f"z_{safe_bay}_{minute}"
            z[(bay, minute)] = LpVariable(var_name, lowBound=0, cat=LpInteger)

            occupant_expr = lpSum(
                occupancy_dict[k].get(minute, 0) * x[(k, bay)]
                for k in all_keys
            )

            model += (
                z[(bay, minute)] >= occupant_expr - 1,
                f"ConflictMin_{bay}_{minute}"
            )

    # Objective: minimize sum of z_{bay,minute}
    model += lpSum(z.values()), "TotalConflicts"

    # 3) Solve
    solver = pulp.PULP_CBC_CMD(msg=True)
    result = model.solve(solver)
    print(f" Solve status: {LpStatus[result]}")
    if LpStatus[result] != "Optimal":
        print(" Optimization was not successful (not optimal or infeasible).")
        return {}, None

    total_conflicts = value(model.objective)
    print(f" Total conflict minutes = {total_conflicts}")

    # 4) Extract solution
    assignments = {}
    for k in all_keys:
        chosen_bay = None
        for bay in bay_labels:
            if value(x[(k, bay)]) > 0.5:
                chosen_bay = bay
                break
        assignments[k] = chosen_bay

    return assignments, total_conflicts

###############################################################################
# GREEDY: ASSIGN ONE BAY PER KEY (LARGEST FIRST)
###############################################################################

def build_and_solve_greedy(occupancy_dict, bay_labels):
    """
    A simple greedy approach to assign one bay per key.
    Strategy:
      1) Sort all keys by total occupant-minutes (descending).
      2) For each key in that order, assign it to the bay that yields the smallest
         incremental conflict.
      3) Return the final assignments & total conflict minutes.
    """
    all_keys = list(occupancy_dict.keys())

    # Sort "largest first" by sum of occupant-minutes
    def total_occupancy(k):
        return sum(occupancy_dict[k].values())
    all_keys.sort(key=total_occupancy, reverse=True)

    assignments = {}
    best_total_conflicts = 0

    for key in all_keys:
        best_bay = None
        best_conflicts_sum = None

        for bay in bay_labels:
            # Try assigning current key to this bay
            temp_assignments = assignments.copy()
            temp_assignments[key] = bay

            # Evaluate conflicts
            conflicts_dict = evaluate_conflicts(occupancy_dict, temp_assignments)
            conflict_sum = sum(conflicts_dict.values())

            if (best_conflicts_sum is None) or (conflict_sum < best_conflicts_sum):
                best_conflicts_sum = conflict_sum
                best_bay = bay

        # Lock in the best bay for this key
        assignments[key] = best_bay
        best_total_conflicts = best_conflicts_sum

    return assignments, best_total_conflicts

###############################################################################
# EVALUATE CONFLICTS
###############################################################################

def evaluate_conflicts(occupancy_dict, assignments):
    """
    From the occupancy perspective, count conflicts for each key.

    If a key k is in bay B at minute M, and at least one other key is also
    in B at minute M, that is a conflict for each bus occupant in that minute.

    Return: { key -> conflict_minutes_count }
    """
    # Build bay->minute->list-of-keys
    bay_to_minute_keys = defaultdict(lambda: defaultdict(list))
    for k, bay_assigned in assignments.items():
        for minute, count in occupancy_dict[k].items():
            for _ in range(count):
                bay_to_minute_keys[bay_assigned][minute].append(k)

    # For each key, count how many occupant-minutes are in conflict
    key_conflicts = defaultdict(int)
    for k, bay_assigned in assignments.items():
        for minute, count in occupancy_dict[k].items():
            total_buses = len(bay_to_minute_keys[bay_assigned][minute])
            if total_buses > 1:
                # conflict for each occupant
                key_conflicts[k] += count

    return dict(key_conflicts)

###############################################################################
# MINUTE-BY-MINUTE BAY SCHEDULES
###############################################################################

def rebuild_bay_schedules(occupancy_dict, assignments, bay_labels):
    """
    Construct minute-by-minute schedules for each bay: which keys are present?
    Return {bay_label: DataFrame} for debugging/export.
    """
    if not occupancy_dict:
        return {}

    max_minute = max(m for occ in occupancy_dict.values() for m in occ.keys())
    bay_schedules = {}

    for bay in bay_labels:
        # Which keys are assigned to this bay
        keys_in_bay = [k for k, a in assignments.items() if a == bay]

        # Minute -> which keys are present
        minute_to_keys = defaultdict(list)
        for k in keys_in_bay:
            for minute, count in occupancy_dict[k].items():
                for _ in range(count):
                    minute_to_keys[minute].append(k)

        records = []
        for minute in range(0, max_minute + 1):
            present_keys = minute_to_keys.get(minute, [])
            conflict_count = max(0, len(present_keys) - 1)
            keys_str = ", ".join(str(ky) for ky in sorted(present_keys)) if present_keys else ""
            time_str = f"{(minute // 60):02d}:{(minute % 60):02d}"

            records.append({
                "minute": minute,
                "time_str": time_str,
                "conflict_count": conflict_count,
                "keys_present_str": keys_str
            })

        bay_df = pd.DataFrame(records)
        bay_schedules[bay] = bay_df

    return bay_schedules

###############################################################################
# EXPORT RESULTS
###############################################################################

def export_comparison_results(
    occupancy_dict,
    default_assignments,
    default_conflicts,
    optimized_assignments,
    optimized_conflicts,
    bay_labels,
    output_folder,
    output_filename
):
    """
    Exports an Excel file comparing default and optimized assignments, plus
    minute-by-minute schedules for both.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Prepare comparison DataFrame
    all_keys = sorted(occupancy_dict.keys(), key=lambda x: str(x))
    comparison_data = []
    for k in all_keys:
        # If keys are (route, direction), parse them
        if isinstance(k, tuple):
            route_id, dir_id = k
        else:
            route_id = k
            dir_id = None

        comparison_data.append({
            "key": str(k),
            "route": route_id,
            "direction": dir_id if dir_id is not None else "",
            "default_bay": default_assignments.get(k, "Unassigned"),
            "default_conflict_minutes": default_conflicts.get(k, 0),
            "optimized_bay": optimized_assignments.get(k, "Unassigned"),
            "optimized_conflict_minutes": optimized_conflicts.get(k, 0)
        })
    df_comparison = pd.DataFrame(comparison_data)

    # Rebuild schedules
    default_schedules = rebuild_bay_schedules(occupancy_dict, default_assignments, bay_labels)
    optimized_schedules = rebuild_bay_schedules(occupancy_dict, optimized_assignments, bay_labels)

    # Write out to Excel
    output_path = os.path.join(output_folder, output_filename)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 1) Comparison sheet
        df_comparison.to_excel(writer, sheet_name="Assignment_Comparison", index=False)

        # 2) Default schedules (one sheet per bay)
        for bay_label, bay_df in default_schedules.items():
            sheet_name = f"Default_{bay_label}"
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            bay_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 3) Optimized schedules (one sheet per bay)
        for bay_label, bay_df in optimized_schedules.items():
            sheet_name = f"Optimized_{bay_label}"
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            bay_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nComparison results exported to: {output_path}\n")

###############################################################################
# MAIN FUNCTION
###############################################################################

def main():
    """
    Main entry point: load GTFS, process data, build default & optimized assignments,
    and export comparison results.
    """
    # 1. Load GTFS
    gtfs_data = load_gtfs_data(CONFIG["gtfs_folder"])

    # 2. Process data -> occupancy dict
    stop_times = gtfs_data['stop_times']
    trips = gtfs_data['trips']
    routes = gtfs_data['routes'][['route_id', 'route_short_name']]

    # Filter trips by service_id
    trips = trips[trips['service_id'] == CONFIG['whitelisted_service_id']]
    print(f"Filtered trips to service_id '{CONFIG['whitelisted_service_id']}': {len(trips)} trips")

    # Merge route info into trips
    trips = trips.merge(routes, on='route_id', how='left')

    # Merge trips info into stop_times
    stop_times = stop_times[stop_times['trip_id'].isin(trips['trip_id'])]
    stop_times = stop_times.merge(
        trips[['trip_id', 'route_short_name', 'block_id', 'direction_id']],
        on='trip_id',
        how='left'
    )

    # Convert arrival/departure times to seconds
    stop_times['arrival_seconds'] = stop_times['arrival_time'].apply(time_to_seconds)
    stop_times['departure_seconds'] = stop_times['departure_time'].apply(time_to_seconds)

    # Sort stop_times
    stop_times.sort_values(['block_id', 'trip_id', 'stop_sequence'], inplace=True)

    # Create next_stop_id and next_arrival_seconds for each trip
    stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times['next_arrival_seconds'] = stop_times.groupby('trip_id')['arrival_seconds'].shift(-1)

    # 3. Build occupancy dictionary (split or not, based on config)
    occupancy_dict = build_occupancy(
        stop_times,
        CONFIG["stops_of_interest"],
        split_by_direction=CONFIG["allow_splitting_by_direction"]
    )
    print("Built occupancy dictionary with block-level presence.")

    # 4. Build the list of bays
    bay_labels = []
    for stop_id in CONFIG["stops_of_interest"]:
        num_bays = CONFIG["num_bays_per_stop"].get(stop_id, 1)
        for idx in range(num_bays):
            bay_label = f"{stop_id}_Bay{idx+1}"
            bay_labels.append(bay_label)

    print(f"\nTotal bays to assign: {len(bay_labels)}")
    print(f"Bays: {bay_labels}")

    # -------------------------------------------------------------------------
    # (A) DEFAULT ASSIGNMENT
    # -------------------------------------------------------------------------
    default_assignments = build_time_based_assignments(occupancy_dict, bay_labels)
    default_conflicts = evaluate_conflicts(occupancy_dict, default_assignments)
    default_total_conflict = sum(default_conflicts.values())

    print("\nDEFAULT ASSIGNMENT - CONFLICTS PER KEY:")
    for k in sorted(default_conflicts.keys(), key=lambda x: str(x)):
        print(f"  {k} => {default_conflicts[k]} conflict minutes")
    print(f"Total conflict minutes (Default) = {default_total_conflict}")

    # -------------------------------------------------------------------------
    # (B) OPTIMIZED ASSIGNMENT: ILP or GREEDY
    # -------------------------------------------------------------------------
    approach = CONFIG["optimization_approach"].lower()
    if approach == "ilp":
        print("\n=== Using ILP Optimization ===")
        print("Note: ILP is only recommended for up to ~4 bays / 12 routes. "
              "Larger instances may be quite slow!")
        optimized_assignments, total_conflicts_bay_perspective = build_and_solve_ilp(
            occupancy_dict,
            bay_labels,
            CONFIG["stops_of_interest"],
            problem_name="OneBayPerKey_BlocksConflict"
        )

        if total_conflicts_bay_perspective is None:
            print("ILP was infeasible or not optimal. Consider adjusting the number of bays.")
            return
    else:
        print("\n=== Using Greedy Optimization (Largest First) ===")
        optimized_assignments, total_conflicts_bay_perspective = build_and_solve_greedy(
            occupancy_dict,
            bay_labels
        )

    # Evaluate from the perspective of each key
    optimized_conflicts = evaluate_conflicts(occupancy_dict, optimized_assignments)
    optimized_total_conflict = sum(optimized_conflicts.values())

    print("\nOPTIMIZED ASSIGNMENT - CONFLICTS PER KEY:")
    for k in sorted(optimized_conflicts.keys(), key=lambda x: str(x)):
        print(f"  {k} => {optimized_conflicts[k]} conflict minutes")
    print(f"Total conflict minutes (Optimized) = {optimized_total_conflict}")
    print(f"(Bay perspective: {total_conflicts_bay_perspective})")

    # -------------------------------------------------------------------------
    # (C) EXPORT COMPARISON TO EXCEL
    # -------------------------------------------------------------------------
    export_comparison_results(
        occupancy_dict,
        default_assignments,
        default_conflicts,
        optimized_assignments,
        optimized_conflicts,
        bay_labels,
        CONFIG["output_folder"],
        CONFIG["comparison_output_filename"]
    )

    # -------------------------------------------------------------------------
    # (D) PRINT FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\nFINAL (OPTIMIZED) ASSIGNMENTS:")
    sorted_keys = sorted(optimized_assignments.keys(), key=lambda x: str(x))
    for k in sorted_keys:
        print(f"  {k} => {optimized_assignments[k]}")

    print("\nDone.")

if __name__ == "__main__":
    main()
