"""
Script to process bus block minute-by-minute spreadsheets. Processes blocks using
stops of interest and identifies conflicts, where two buses are scheduled (or implied)
to be at the same stop at the same time.
"""
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLOCK_LEVEL_FOLDER = r"Path\To\Your\Block_Status_By_Minute_Files"  # Where your per-block Step 1 outputs live
OUTPUT_FOLDER = r"Path\To\Your\Output_Folder"

# Map bus-bay "labels" to the list of GTFS stop_ids that belong to each bay
BUS_BAY_CLUSTERS = {
    "Bus Station Cluster": ["1001", "1002", "1003", "1004"],
    "Train Station Cluster": ["2002", "2003", "2004"]
    # ... add as needed
}

# Any statuses that mean “the bus is physically occupying this bay”
OCCUPANCY_STATUSES = {"ARRIVE", "DEPART", "ARRIVE/DEPART", "DWELL", "LOADING", "LAYOVER"}

# Optional configurations for stops with multiple bays and for overflow bays:
TWO_BAY_STOPS = ["1001"]      # stops that get two columns
THREE_BAY_STOPS = ["2002"]    # stops that get three columns
OVERFLOW_BAYS = ["OverflowA", "OverflowB"]  # additional overflow bay columns

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_stop_id(stop_id):
    """
    Normalize a stop ID to a string.
    If stop_id is numeric (or a numeric string with trailing '.0'),
    convert to a string without the trailing .0.
    """
    if pd.isna(stop_id):
        return None
    s = str(stop_id).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

def normalize_route(route):
    """
    Normalize the Route column.
    If the route is numeric and ends with ".0", remove the trailing .0.
    Otherwise, just return a stripped string.
    """
    if pd.isna(route):
        return ""
    r = str(route).strip()
    if r.endswith('.0'):
        r = r[:-2]
    return r

def extract_bay_label(stop_id, bay_clusters):
    """
    Return the bay label (cluster name) for this stop_id if found,
    otherwise None.
    """
    norm_stop = normalize_stop_id(stop_id)
    for bay_label, stops in bay_clusters.items():
        norm_stops = [normalize_stop_id(s) for s in stops]
        if norm_stop in norm_stops:
            return bay_label
    return None

def gather_block_spreadsheets(block_folder):
    """
    Reads all XLSX files (one per block) from Step 1 and concatenates them.
    """
    all_files = [
        f for f in os.listdir(block_folder)
        if f.lower().endswith(".xlsx") and f.startswith("block_")
    ]
    big_df_list = []
    for fname in all_files:
        fpath = os.path.join(block_folder, fname)
        block_df = pd.read_excel(fpath)
        block_df["FileName"] = fname
        big_df_list.append(block_df)
    if not big_df_list:
        raise ValueError(f"No block-level XLSX files found in {block_folder}")
    df = pd.concat(big_df_list, ignore_index=True)
    print("Loaded total rows from all block XLSX files:", len(df))
    return df

def get_stop_capacity(stop_id):
    """
    Returns how many 'native' columns (bays) this stop has,
    based on the TWO_BAY_STOPS / THREE_BAY_STOPS configuration.
    """
    sid = normalize_stop_id(stop_id)
    if sid in THREE_BAY_STOPS:
        return 3
    elif sid in TWO_BAY_STOPS:
        return 2
    else:
        return 1

def generate_minute_range(start_str, end_str):
    """
    Given start and end times as "HH:MM" strings (e.g. "00:00" to "25:59"),
    generate a list of minute strings in "HH:MM" format covering the full range.
    """
    def to_total_minutes(s):
        h, m = map(int, s.split(':'))
        return h * 60 + m

    def from_total_minutes(total):
        h = total // 60
        m = total % 60
        return f"{h:02d}:{m:02d}"

    start_total = to_total_minutes(start_str)
    end_total = to_total_minutes(end_str)
    return [from_total_minutes(t) for t in range(start_total, end_total + 1)]

# ---------------------------------------------------------------------------
# Minute-by-minute assignment logic (using string Timestamps)
# ---------------------------------------------------------------------------
def build_minute_by_minute_table_str(cluster_df, cluster_stops):
    """
    For the given cluster (already filtered to relevant stops),
    build a minute-by-minute wide table using the string Timestamps.
    """
    # Restrict to rows where the bus is physically occupying the stop
    occupancy_mask = cluster_df["Status"].isin(OCCUPANCY_STATUSES)
    occ = cluster_df[occupancy_mask].copy()
    if occ.empty:
        return pd.DataFrame()  # no occupancy events

    # Ensure Timestamp is treated as a stripped string (e.g. "07:02", "24:10")
    occ["Timestamp"] = occ["Timestamp"].astype(str).str.strip()

    # Determine the overall time window based on the minimum and maximum Timestamp values
    min_time = occ["Timestamp"].min()
    max_time = occ["Timestamp"].max()
    all_minutes = generate_minute_range(min_time, max_time)
    out_df = pd.DataFrame(index=all_minutes)

    # Create an occupant string for each row (for display purposes)
    occ["OccupantString"] = occ.apply(lambda r: f"{r['Block']}({r['Route']})/{r['Status']}", axis=1)
    # Group by the Timestamp and Stop ID (using the string timestamp)
    occupant_map = occ.groupby(["Timestamp", "Stop ID"])["OccupantString"].apply(list).to_dict()

    # Create columns for each stop based on its capacity (native bays)
    for s in cluster_stops:
        cap = get_stop_capacity(s)
        for bay_num in range(1, cap + 1):
            col_name = f"{s}_{bay_num}"
            out_df[col_name] = ""

    # Create columns for each overflow bay
    for ovf in OVERFLOW_BAYS:
        col_name = f"Overflow_{ovf}"
        out_df[col_name] = ""

    # Create columns to track if any occupant is bumped (overflow) or not placed (conflict)
    out_df["overflow"] = False
    out_df["conflict"] = False

    # Loop over each minute in the generated range and assign occupant events
    for current_minute in all_minutes:
        minute_overflow = False
        minute_conflict = False

        # For each stop in the cluster, assign occupant events to the appropriate columns
        for s in cluster_stops:
            occupant_list = occupant_map.get((current_minute, s), [])
            capacity = get_stop_capacity(s)
            assigned_strings = [""] * capacity

            for occ_str in occupant_list:
                placed = False
                # Try to place in one of the native bay columns
                for col_idx in range(capacity):
                    if assigned_strings[col_idx] == "":
                        assigned_strings[col_idx] = occ_str
                        if col_idx >= 1:
                            minute_overflow = True
                        placed = True
                        break
                if not placed:
                    # If all native columns are occupied, try overflow columns
                    placed_in_ovf = False
                    for ovf in OVERFLOW_BAYS:
                        col_name = f"Overflow_{ovf}"
                        if out_df.at[current_minute, col_name] == "":
                            out_df.at[current_minute, col_name] = occ_str
                            minute_overflow = True
                            placed_in_ovf = True
                            break
                    if not placed_in_ovf:
                        minute_conflict = True

            # Write the assigned occupant strings to the appropriate stop columns
            for col_idx in range(capacity):
                col_name = f"{s}_{col_idx+1}"
                out_df.at[current_minute, col_name] = assigned_strings[col_idx]

        # Mark overflow and conflict status for the current minute
        out_df.at[current_minute, "overflow"] = minute_overflow
        out_df.at[current_minute, "conflict"] = minute_conflict

    # Reset index so that Timestamp becomes a column
    out_df.reset_index(inplace=True)
    out_df.rename(columns={"index": "Timestamp"}, inplace=True)
    return out_df

# ---------------------------------------------------------------------------
# Main Script to Generate Outputs (Minute-by-Minute Table & Long Format Solver Table)
# ---------------------------------------------------------------------------
def main():
    print("Gathering all block-level spreadsheets from Step 1...")
    df = gather_block_spreadsheets(BLOCK_LEVEL_FOLDER)

    # Ensure expected columns exist (adjust if needed)
    expected_cols = [
        "Timestamp", "Trip ID", "Block", "Route", "Direction",
        "Stop ID", "Stop Name", "Stop Sequence",
        "Arrival Time", "Departure Time", "Status", "Timepoint"
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in block-level DataFrame.")

    # Normalize Stop IDs and ensure Timestamp is treated as a string
    df["Stop ID"] = df["Stop ID"].apply(normalize_stop_id)
    df["Timestamp"] = df["Timestamp"].astype(str).str.strip()
    # Normalize Route IDs to remove trailing .0 if needed
    df["Route"] = df["Route"].apply(normalize_route)

    print("Unique normalized Stop IDs:", df["Stop ID"].unique())

    # Filter rows to include only those whose Stop ID belongs to a designated bus bay cluster.
    df["Bay"] = df["Stop ID"].apply(lambda sid: extract_bay_label(sid, BUS_BAY_CLUSTERS))
    bay_df = df[df["Bay"].notnull()].copy()
    print("Rows that matched a bus bay cluster:", len(bay_df))

    # Process each bus bay cluster separately
    for bay_label, bay_stop_ids in BUS_BAY_CLUSTERS.items():
        print(f"\nProcessing bus bay cluster: {bay_label}")
        cluster_df = bay_df[bay_df["Bay"] == bay_label].copy()
        if cluster_df.empty:
            print(f"No data for {bay_label}, skipping.")
            continue

        # 1) Build the minute-by-minute table using string timestamps
        minute_table = build_minute_by_minute_table_str(cluster_df, bay_stop_ids)
        if minute_table.empty:
            print(f"    No occupancy found for {bay_label}, skipping minute-by-minute table.")
        else:
            # Check if any conflict exists
            any_conflict = minute_table["conflict"].any()
            if any_conflict:
                print(f"    WARNING (yellow): Conflicts found in cluster {bay_label}!")
            else:
                print(f"    Check OK: No conflicts in {bay_label}.")

            # Force overflow/conflict columns to be consistent booleans
            minute_table["overflow"] = minute_table["overflow"].astype(bool)
            minute_table["conflict"] = minute_table["conflict"].astype(bool)

            minute_output_path = os.path.join(OUTPUT_FOLDER, f"{bay_label}_MinuteByMinute.xlsx")
            minute_table.to_excel(minute_output_path, index=False)
            print(f"    Minute-by-minute table saved to {minute_output_path}")

        # 2) Build the solver-friendly long format table (including Trip ID)
        occupancy_mask = cluster_df["Status"].isin(OCCUPANCY_STATUSES)
        solver_df = cluster_df[occupancy_mask].copy()
        if solver_df.empty:
            print(f"    No occupancy events for solver table in {bay_label}, skipping.")
        else:
            # Select and rename columns appropriately (including Trip ID)
            solver_cols = ["Timestamp", "Bay", "Stop ID", "Trip ID", "Block", "Route", "Direction", "Status"]
            solver_df = solver_df[solver_cols].copy()
            solver_df.rename(columns={"Status": "Event", "Stop ID": "stop_id", "Trip ID": "trip_id"}, inplace=True)
            solver_df.sort_values(["Timestamp", "Bay"], inplace=True)

            solver_output_path = os.path.join(OUTPUT_FOLDER, f"{bay_label}_LongFormat_Solver.xlsx")
            solver_df.to_excel(solver_output_path, index=False)
            print(f"    Solver-friendly long format table for {bay_label} saved to {solver_output_path}")

    print("\nAll per-bus-bay-cluster outputs generated successfully.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    main()
