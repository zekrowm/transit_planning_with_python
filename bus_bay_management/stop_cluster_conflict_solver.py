import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Configuration / User Inputs
# -------------------------------------------------------------------------
SOLVER_INPUT_FOLDER = r"Path\To\Your\Long_Format_Solver_Output_Folder"
# Folder that has the *Script 2* outputs, e.g. "SomeCluster_LongFormat_Solver.xlsx", etc.

NEW_ROUTE_FILE = r""  # If you have a new route Excel file in the same block-minute format, put path here.
# e.g. r"C:\Users\zach\Desktop\Zach\python_stuff\MyNewRoute.xlsx"

OUTPUT_FOLDER = r"Path\To\Your\Output\Folder"

# For each cluster, we know which stops (bays) belong to it:
CLUSTER_DEFINITIONS = {
    "Bus Station Cluster": ["1001", "1002", "1003", "1004"],
    "Train Station Cluster": ["2002", "2003", "2004"]
}

# Optional route pairing or constraints.
PAIRED_ROUTES = [
    # Example: ("101", "102"),
]
ROUTE_BAY_PREFERENCES = {
    # Example: "400": ["1003", "1004"]
}

# If False, then both directions for a given route (which share the same route name) are treated as one.
ALLOW_DIFFERENT_DIRECTIONS = False

# Weighted priorities
WEIGHT_CONFLICT = 1000  # large cost per conflict-minute
WEIGHT_ASSIGNMENT_CHANGE = 1  # small cost for changing from original assignment

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_long_format_excels(folder):
    """
    Loads all *_LongFormat_Solver.xlsx files from the given folder.
    Returns a single DataFrame with a column 'ClusterName' deduced from the filename.
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith("_longformat_solver.xlsx")]
    if not files:
        raise FileNotFoundError(f"No *_LongFormat_Solver.xlsx files found in {folder}.")

    df_list = []
    for f in files:
        fpath = os.path.join(folder, f)
        # Derive the cluster name from the filename (e.g., "TrainStation_LongFormat_Solver.xlsx" becomes "Train Station Cluster")
        cluster_name = f.replace("_LongFormat_Solver.xlsx", "")
        temp = pd.read_excel(fpath)
        temp["ClusterName"] = cluster_name
        df_list.append(temp)

    bigdf = pd.concat(df_list, ignore_index=True)
    print("Loaded long-format solver data. Rows:", len(bigdf))
    return bigdf

def preprocess_solver_df(df):
    """
    Clean up the DataFrame to ensure consistent columns for solving.
    Expected columns include:
        [Timestamp, Bay, stop_id, trip_id, Block, Route, Direction, Event, ClusterName]
    """
    # Standardize key columns as strings and strip whitespace.
    for col in ["stop_id", "trip_id", "Block", "Route", "Direction", "ClusterName"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""
    
    # For convenience, ensure Timestamp is a string "HH:MM"
    df["Timestamp"] = df["Timestamp"].astype(str).str.strip()

    # Create RouteKey:
    # If ALLOW_DIFFERENT_DIRECTIONS is True, use Route + "_" + Direction.
    # Otherwise, treat all directions for the same route as one.
    if ALLOW_DIFFERENT_DIRECTIONS:
        df["RouteKey"] = df["Route"] + "_" + df["Direction"].replace("", "NA")
    else:
        df["RouteKey"] = df["Route"]
    
    # Store original assignment from stop_id.
    df["OriginalStopID"] = df["stop_id"]

    return df

def load_new_route(new_route_path):
    """
    If provided, load the new route minute-by-minute data in the same format.
    The file must include a 'ClusterName' column to indicate its cluster.
    """
    if not new_route_path:
        return pd.DataFrame()  # empty DataFrame if not provided
    if not os.path.exists(new_route_path):
        raise FileNotFoundError(f"New route file not found: {new_route_path}")
    new_df = pd.read_excel(new_route_path)
    if "ClusterName" not in new_df.columns:
        raise ValueError("New route file must include a 'ClusterName' column to indicate which cluster it belongs to.")
    return new_df

def get_all_timestamps(df):
    """
    Extract all timestamps present in the data and sort them in ascending order.
    """
    def to_minutes(tstr):
        hh, mm = tstr.split(":")
        return int(hh) * 60 + int(mm)
    unique_ts = df["Timestamp"].unique().tolist()
    unique_ts_minutes = sorted(unique_ts, key=lambda x: to_minutes(x))
    return unique_ts_minutes

def compute_conflicts_for_assignment(subdf, route_to_stop):
    """
    Compute total conflict cost plus assignment-change cost for the given sub-DataFrame
    (corresponding to one cluster) under a route->stop assignment dictionary.
    A conflict is when two different routes are assigned the same stop at the same Timestamp.
    Assignment change cost is incurred if a route’s new assignment differs from its mode of OriginalStopID.
    """
    occupancy_events = {"ARRIVE", "DEPART", "ARRIVE/DEPART", "DWELL", "LOADING", "LAYOVER"}
    occ_mask = subdf["Event"].isin(occupancy_events)
    occ = subdf[occ_mask].copy()
    if occ.empty:
        return 0, 0, 0

    occ["AssignedStop"] = occ["RouteKey"].map(route_to_stop)

    conflict_count = 0
    for ts, group in occ.groupby("Timestamp"):
        stop_groups = group.groupby("AssignedStop")
        for stp, subg in stop_groups:
            n = len(subg)
            if n > 1:
                conflict_count += n * (n - 1) // 2

    # Count route-level assignment changes.
    route_changes = 0
    route_groups = occ.groupby("RouteKey")
    for rkey, rg in route_groups:
        original_stops = rg["OriginalStopID"].dropna().unique().tolist()
        if not original_stops:
            continue
        mode_stop = rg["OriginalStopID"].mode()[0]
        new_stop = route_to_stop.get(rkey, None)
        if new_stop is not None and new_stop != mode_stop:
            route_changes += 1

    conflict_cost = conflict_count * WEIGHT_CONFLICT
    change_cost = route_changes * WEIGHT_ASSIGNMENT_CHANGE
    total_cost = conflict_cost + change_cost
    return total_cost, conflict_count, route_changes

def build_initial_assignment(df, cluster_stops):
    """
    For each RouteKey in this cluster, choose the most common 'OriginalStopID' that is within cluster_stops.
    If none match, fall back to the first stop in cluster_stops.
    Returns a dictionary: RouteKey -> assigned stop.
    """
    occupancy_events = {"ARRIVE", "DEPART", "ARRIVE/DEPART", "DWELL", "LOADING", "LAYOVER"}
    occ_mask = df["Event"].isin(occupancy_events)
    occ = df[occ_mask].copy()

    assignment = {}
    for route_key, grp in occ.groupby("RouteKey"):
        original_candidates = grp["OriginalStopID"][grp["OriginalStopID"].isin(cluster_stops)]
        if not original_candidates.empty:
            assigned = original_candidates.mode()[0]
        else:
            assigned = cluster_stops[0]
        assignment[route_key] = assigned
    return assignment

def apply_preferences_and_pairings(assignment, cluster_stops):
    """
    Adjust the initial assignment based on any route-bay preferences or paired routes.
    """
    # Route→Bay preference.
    for route, possible_stops in ROUTE_BAY_PREFERENCES.items():
        for rkey in assignment.keys():
            # Check if rkey starts with the route.
            if rkey == route or rkey.startswith(route + "_"):
                if assignment[rkey] not in possible_stops:
                    valid_options = [stp for stp in possible_stops if stp in cluster_stops]
                    if valid_options:
                        assignment[rkey] = valid_options[0]
    # Paired routes logic.
    for (rA, rB) in PAIRED_ROUTES:
        for rkeyA in assignment:
            if rkeyA == rA or rkeyA.startswith(rA + "_"):
                for rkeyB in assignment:
                    if rkeyB == rB or rkeyB.startswith(rB + "_"):
                        assignment[rkeyB] = assignment[rkeyA]
    return assignment

def attempt_greedy_reassign(df, cluster_stops, initial_assignment):
    """
    Greedily reassign routes one at a time to a different stop in cluster_stops if it reduces the cost.
    Returns the final assignment dictionary: RouteKey -> assigned stop.
    """
    current_assignment = initial_assignment.copy()
    best_cost, best_conflicts, best_changes = compute_conflicts_for_assignment(df, current_assignment)
    improved = True
    while improved:
        improved = False
        for rkey in sorted(current_assignment.keys()):
            orig_stop = current_assignment[rkey]
            local_best_stop = orig_stop
            local_best_cost = best_cost
            for stp in cluster_stops:
                if stp == orig_stop:
                    continue
                current_assignment[rkey] = stp
                test_cost, test_conflicts, test_changes = compute_conflicts_for_assignment(df, current_assignment)
                if test_cost < local_best_cost:
                    local_best_cost = test_cost
                    local_best_stop = stp
            if local_best_stop != orig_stop:
                current_assignment[rkey] = local_best_stop
                best_cost = local_best_cost
                improved = True
                break  # Restart loop after an improvement.
    return current_assignment

def rebuild_minute_by_minute(subdf, route_to_stop):
    """
    Create a revised minute-by-minute DataFrame with an updated 'RevisedStopID' column based on route_to_stop.
    """
    subdf = subdf.copy()
    subdf["RevisedStopID"] = subdf["RouteKey"].map(route_to_stop)
    return subdf

# -------------------------------------------------------------------------
# Main solve() function
# -------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("1) Loading long-format solver files from Step 2...")
    solver_df = load_long_format_excels(SOLVER_INPUT_FOLDER)
    solver_df = preprocess_solver_df(solver_df)

    # Optionally load new route data.
    new_df = load_new_route(NEW_ROUTE_FILE)
    if not new_df.empty:
        print("Loading new route data and appending to solver DataFrame...")
        new_df = preprocess_solver_df(new_df)
        solver_df = pd.concat([solver_df, new_df], ignore_index=True)

    # Process each cluster separately.
    for cluster_name, cluster_stops in CLUSTER_DEFINITIONS.items():
        print(f"\n--- Solving for cluster: {cluster_name} ---")
        subdf = solver_df[solver_df["ClusterName"] == cluster_name].copy()
        if subdf.empty:
            print(f"    No data for cluster {cluster_name}, skipping.")
            continue

        # Build a lookup for stop names.
        stop_lookup = subdf[['stop_id', 'Stop Name']].drop_duplicates().set_index('stop_id')['Stop Name'].to_dict()

        # 3a) Build an initial assignment (using mode of OriginalStopID).
        init_assign = build_initial_assignment(subdf, cluster_stops)
        # Save the original assignment for export.
        original_assignment = init_assign.copy()

        # 3b) Apply route preferences/pairings.
        init_assign = apply_preferences_and_pairings(init_assign, cluster_stops)

        # 3c) Greedy reassign to reduce conflicts.
        final_assign = attempt_greedy_reassign(subdf, cluster_stops, init_assign)
        final_cost, final_conflict_count, final_changes = compute_conflicts_for_assignment(subdf, final_assign)

        if final_conflict_count > 0:
            print(f"    WARNING (yellow): {final_conflict_count} conflict-pairs remain.")
        else:
            print(f"    Check OK: No conflicts remain in {cluster_name}.")
        print(f"    {final_changes} route assignment changes vs. original. Weighted cost = {final_cost}")

        # 3d) Build output DataFrame including Route, Direction, original and new assignments with stop names.
        assignment_rows = []
        for rkey in sorted(final_assign.keys()):
            # Extract a representative row for this RouteKey.
            rep = subdf[subdf["RouteKey"] == rkey].iloc[0]
            route = rep["Route"]
            # If ALLOW_DIFFERENT_DIRECTIONS is True, include Direction; otherwise leave blank.
            direction = rep["Direction"] if ALLOW_DIFFERENT_DIRECTIONS else ""
            orig_stop = original_assignment.get(rkey, "")
            new_stop = final_assign[rkey]
            orig_stop_name = stop_lookup.get(orig_stop, "")
            new_stop_name = stop_lookup.get(new_stop, "")
            assignment_rows.append({
                "ClusterName": cluster_name,
                "Route": route,
                "Direction": direction,
                "OriginalStopID": orig_stop,
                "OriginalStopName": orig_stop_name,
                "NewStopID": new_stop,
                "NewStopName": new_stop_name
            })
        cluster_assignment_df = pd.DataFrame(assignment_rows)

        # 3e) Export the route assignment table for this cluster.
        assignment_outpath = os.path.join(OUTPUT_FOLDER, f"{cluster_name}_RouteAssignments.xlsx")
        cluster_assignment_df.to_excel(assignment_outpath, index=False)
        print(f"    Route assignments for {cluster_name} saved to: {assignment_outpath}")

        # 3f) Optionally export a revised minute-by-minute schedule.
        revised_sub = rebuild_minute_by_minute(subdf, final_assign)
        minute_outpath = os.path.join(OUTPUT_FOLDER, f"{cluster_name}_RevisedMinuteByMinute.xlsx")
        revised_sub.to_excel(minute_outpath, index=False)
        print(f"    Revised minute-by-minute schedule for {cluster_name} saved to: {minute_outpath}")

if __name__ == "__main__":
    main()
