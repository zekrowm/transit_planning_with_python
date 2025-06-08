"""
Extracts the most common GTFS stop pattern by direction for each route and exports a summary to CSV.

The output includes each stop in the most common pattern(s) for direction 0 and 1,
with its sequence number in each pattern (if applicable). Useful for visualizing or
auditing directional stop sequences in a route network.

Typical use case: run in ArcGIS Pro or Jupyter environment.

Inputs:
    GTFS files (routes.txt, trips.txt, stop_times.txt, stops.txt)

Outputs:
    CSV with columns:
        route_short_name, stop_id, stop_code, stop_name, seq_dir_0, seq_dir_1
"""

import os
from collections import Counter

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Folder containing the GTFS text files
INPUT_DIR = r"Path\To\Your\GTFS\Folder"

# Output CSV file path
OUTPUT_CSV_PATH = r"File\Path\To\Your\most_common_patterns_both_dirs.csv"

# Optional filters on route_short_name
FILTER_IN_ROUTE_SHORT_NAMES = []  # e.g. ["306", "50A"]
FILTER_OUT_ROUTE_SHORT_NAMES = []  # e.g. ["999"]

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_gtfs_file(filename: str) -> pd.DataFrame:
    """
    Load a GTFS text file (CSV‐formatted) from INPUT_DIR into a pandas DataFrame.
    Raises OSError if INPUT_DIR or the file does not exist.
    Raises ValueError if the file is empty or malformed.
    """
    if not os.path.isdir(INPUT_DIR):
        raise OSError(f"GTFS folder not found: {INPUT_DIR}")

    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise OSError(f"Missing GTFS file: {filename}")

    try:
        df = pd.read_csv(filepath, dtype=str, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"GTFS file is empty: {filename}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Parse error in {filename}: {e}")

    return df


def compute_most_common_pattern(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    route_id: str,
    direction_id: str,
) -> list[str]:
    """
    For the given route_id and direction_id, collect all trips,
    build each trip’s stop_id sequence (sorted by stop_sequence),
    then return the most frequent tuple of stop_ids as a list.
    If there are no trips in that direction, returns an empty list.
    """
    # 1. Filter trips for this route_id & direction_id
    mask = (trips_df["route_id"] == route_id) & (
        trips_df["direction_id"] == direction_id
    )
    route_trips = trips_df[mask]
    if route_trips.empty:
        return []

    # 2. Subset stop_times to only those trip_ids, ensure stop_sequence is numeric
    relevant_stop_times = stop_times_df[
        stop_times_df["trip_id"].isin(route_trips["trip_id"])
    ].copy()
    if relevant_stop_times.empty:
        return []

    if not pd.api.types.is_numeric_dtype(relevant_stop_times["stop_sequence"]):
        relevant_stop_times["stop_sequence"] = pd.to_numeric(
            relevant_stop_times["stop_sequence"], errors="raise"
        )

    # 3. Build a Counter of stop_id tuples (one tuple per trip)
    pattern_counter = Counter()
    for trip_id, group in relevant_stop_times.groupby("trip_id"):
        group_sorted = group.sort_values("stop_sequence")
        stop_seq = tuple(group_sorted["stop_id"].tolist())
        if stop_seq:
            pattern_counter[stop_seq] += 1

    if not pattern_counter:
        return []

    # 4. Return the most common sequence as a list of stop_ids
    most_common_sequence, _ = pattern_counter.most_common(1)[0]
    return list(most_common_sequence)


# =============================================================================
# MAIN
# =============================================================================


def main():
    # 1. Load GTFS files
    try:
        routes_df = load_gtfs_file("routes.txt")  # expects: route_id, route_short_name
        trips_df = load_gtfs_file(
            "trips.txt"
        )  # expects: trip_id, route_id, direction_id
        stop_times_df = load_gtfs_file(
            "stop_times.txt"
        )  # expects: trip_id, stop_id, stop_sequence
        stops_df = load_gtfs_file("stops.txt")  # expects: stop_id, stop_code, stop_name
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return

    # 2. Filter routes by route_short_name (if either FILTER_IN or FILTER_OUT is set)
    routes_df["route_short_name"] = routes_df["route_short_name"].astype(str)
    if FILTER_IN_ROUTE_SHORT_NAMES:
        routes_df = routes_df[
            routes_df["route_short_name"].isin(FILTER_IN_ROUTE_SHORT_NAMES)
        ]
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        routes_df = routes_df[
            ~routes_df["route_short_name"].isin(FILTER_OUT_ROUTE_SHORT_NAMES)
        ]

    # If no routes remain after filtering, exit
    if routes_df.empty:
        print("[INFO] No routes to process after applying filters.")
        return

    # 3. Prepare a lookup for stop_code and stop_name by stop_id
    stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    stops_df["stop_code"] = stops_df["stop_code"].fillna("").astype(str)
    stops_df["stop_name"] = stops_df["stop_name"].fillna("").astype(str)
    stop_info = stops_df.set_index("stop_id")[["stop_code", "stop_name"]]

    # 4. Iterate through each filtered route
    output_rows = []
    for _, route_row in routes_df.iterrows():
        route_id = route_row["route_id"]
        route_short = route_row["route_short_name"]

        # Determine which directions exist for this route
        route_trips = trips_df[trips_df["route_id"] == route_id]
        if route_trips.empty:
            continue

        directions = route_trips["direction_id"].dropna().unique()
        # Convert direction_id to string for safe comparison
        directions = [str(d) for d in directions]

        # Compute most common patterns for both directions
        pattern_0 = compute_most_common_pattern(
            trips_df, stop_times_df, route_id, direction_id="0"
        )
        pattern_1 = compute_most_common_pattern(
            trips_df, stop_times_df, route_id, direction_id="1"
        )

        # Build the union of stop_ids from both patterns
        all_stop_ids = set(pattern_0) | set(pattern_1)
        if not all_stop_ids:
            # No pattern found in either direction
            continue

        # Build sequence maps: stop_id → sequence index (1-based)
        seq_map_0 = {sid: idx + 1 for idx, sid in enumerate(pattern_0)}
        seq_map_1 = {sid: idx + 1 for idx, sid in enumerate(pattern_1)}

        # For each stop_id in the union, gather data
        for stop_id in sorted(
            all_stop_ids,
            key=lambda x: (
                seq_map_0.get(x, float("inf")),
                seq_map_1.get(x, float("inf")),
            ),
        ):
            stop_code = (
                stop_info.at[stop_id, "stop_code"] if stop_id in stop_info.index else ""
            )
            stop_name = (
                stop_info.at[stop_id, "stop_name"] if stop_id in stop_info.index else ""
            )
            output_rows.append(
                {
                    "route_short_name": route_short,
                    "stop_id": stop_id,
                    "stop_code": stop_code,
                    "stop_name": stop_name,
                    "seq_dir_0": seq_map_0.get(stop_id, pd.NA),
                    "seq_dir_1": seq_map_1.get(stop_id, pd.NA),
                }
            )

    # 5. Build a DataFrame from output_rows and export to CSV
    if not output_rows:
        print("[INFO] No stops/patterns found for any route/direction.")
        return

    result_df = pd.DataFrame(
        output_rows,
        columns=[
            "route_short_name",
            "stop_id",
            "stop_code",
            "stop_name",
            "seq_dir_0",
            "seq_dir_1",
        ],
    )

    # Ensure the parent folder of OUTPUT_CSV_PATH exists
    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            print(f"[INFO] Created folder for output: {out_dir}")
        except OSError as exc:
            print(f"[ERROR] Could not create output folder '{out_dir}': {exc}")
            return

    try:
        result_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"[INFO] Exported most‐common patterns to CSV: {OUTPUT_CSV_PATH}")
    except Exception as exc:
        print(f"[ERROR] Could not write CSV to '{OUTPUT_CSV_PATH}': {exc}")


if __name__ == "__main__":
    main()
