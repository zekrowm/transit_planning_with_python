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

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_stop_id(stop_id):
    """
    Normalize a stop ID to a string.
    - If stop_id is numeric (or a numeric string with a trailing '.0'), it will be
      converted to an integer string.
    - Otherwise, returns the trimmed string.
    """
    if pd.isna(stop_id):
        return None
    s = str(stop_id).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

def extract_bay_label(stop_id, bay_clusters):
    """
    Given a stop_id (in any format), return the bay label (cluster name) it belongs to,
    by comparing against the normalized list of stop IDs defined in bay_clusters.
    """
    norm_stop = normalize_stop_id(stop_id)
    for bay_label, stops in bay_clusters.items():
        norm_stops = [normalize_stop_id(s) for s in stops]
        if norm_stop in norm_stops:
            return bay_label
    return None

def gather_block_spreadsheets(block_folder):
    """
    Reads all XLSX files produced in Step 1 (one per block) and
    concatenates them into a single DataFrame with an extra column 'FileName'
    so we know which block file it came from.
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

# ---------------------------------------------------------------------------
# Main Script to Generate Step 2 Outputs (Per Bus Bay Cluster)
# ---------------------------------------------------------------------------
def main():
    print("Gathering all block-level spreadsheets from Step 1...")
    df = gather_block_spreadsheets(BLOCK_LEVEL_FOLDER)

    # Ensure expected columns exist (adjust if needed)
    expected_cols = [
        "Timestamp", "Block", "Route", "Direction",
        "Stop ID", "Stop Name", "Stop Sequence",
        "Arrival Time", "Departure Time", "Status", "Timepoint"
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in block-level DataFrame.")

    # Normalize Stop IDs in the DataFrame
    df["Stop ID"] = df["Stop ID"].apply(normalize_stop_id)
    print("Unique normalized Stop IDs:", df["Stop ID"].unique())

    # Filter rows to include only those whose Stop ID belongs to a designated bus bay cluster.
    print("Filtering rows to bus bay clusters only...")
    df["Bay"] = df["Stop ID"].apply(lambda sid: extract_bay_label(sid, BUS_BAY_CLUSTERS))
    bay_df = df[df["Bay"].notnull()].copy()
    print("Rows that matched a bus bay cluster:", len(bay_df))

    # Process each bus bay cluster separately.
    for bay_label in BUS_BAY_CLUSTERS:
        print(f"\nProcessing bus bay cluster: {bay_label}")
        cluster_df = bay_df[bay_df["Bay"] == bay_label].copy()
        if cluster_df.empty:
            print(f"No data for {bay_label}, skipping.")
            continue

        # -----------------------------------
        # Build Conflict Matrix for the Cluster
        # -----------------------------------
        occupancy_mask = cluster_df["Status"].isin(OCCUPANCY_STATUSES)
        cluster_conflict_df = cluster_df[occupancy_mask].copy()
        # Create a helper column "BlockRoute" for display purposes.
        cluster_conflict_df["BlockRoute"] = cluster_conflict_df.apply(
            lambda row: f"{row['Block']} ({row['Route']})", axis=1
        )
        # Group by Timestamp (Bay is constant here) and join BlockRoute values.
        grouped_cluster = cluster_conflict_df.groupby(["Timestamp", "Bay"])["BlockRoute"].apply(list).reset_index()
        grouped_cluster["BlockRoute"] = grouped_cluster["BlockRoute"].apply(lambda x: ", ".join(x))
        # Pivot to wide format (for a single bay, this results in one column)
        conflict_matrix_cluster = grouped_cluster.pivot(index="Timestamp", columns="Bay", values="BlockRoute").fillna("-")
        conflict_matrix_cluster.sort_index(inplace=True)

        cm_output_path_cluster = os.path.join(OUTPUT_FOLDER, f"{bay_label}_ConflictMatrix.xlsx")
        conflict_matrix_cluster.to_excel(cm_output_path_cluster)
        print(f"Conflict matrix for {bay_label} saved to {cm_output_path_cluster}")

        # -----------------------------------
        # Build Solver-Friendly Long Format Table for the Cluster
        # -----------------------------------
        # Now include "Stop ID" as "stop_id" in the output.
        solver_cols = ["Timestamp", "Bay", "Stop ID", "Block", "Route", "Direction", "Status"]
        solver_cluster_df = cluster_conflict_df[solver_cols].copy()
        solver_cluster_df.rename(columns={"Status": "Event", "Stop ID": "stop_id"}, inplace=True)
        solver_cluster_df.sort_values(["Timestamp", "Bay"], inplace=True)

        solver_output_path_cluster = os.path.join(OUTPUT_FOLDER, f"{bay_label}_LongFormat_Solver.xlsx")
        solver_cluster_df.to_excel(solver_output_path_cluster, index=False)
        print(f"Solver-friendly long format table for {bay_label} saved to {solver_output_path_cluster}")

    print("\nAll per-bus bay cluster outputs generated successfully.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    main()
