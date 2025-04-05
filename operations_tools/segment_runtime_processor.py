"""
Script to pivot data from a CLEVER CSV for runtime analysis, converting time columns
from seconds to minutes, and saving each route to a separate Excel file.

- If a time column doesn't exist in the DataFrame, that pivot is skipped.
- If no valid pivots exist for a route, a single "NoData" sheet is created instead.

Integrated:
1) check_route_validity function to warn if the route has multiple Variation values,
   multiple starts, or loops.
2) sort_route_segments function to produce a single continuous path of segments if valid.
"""

import os
import pandas as pd

# -------------------------------------------------------------------------
#                           CONFIGURATION
# -------------------------------------------------------------------------
CSV_PATH = (
    r"\\Your\File\Path\To\CLEVER_Runtime_by_Segment_by_Trip.csv"
)

# Optionally specify routes to EXCLUDE. If empty, no routes are excluded.
ROUTES_TO_EXCLUDE = [
    "101",  # Example: replace with your route numbers or leave blank
]

# Optionally specify routes to INCLUDE. If empty, all routes are included.
ROUTES_TO_INCLUDE = [
    "202", "303" # Example: replace with your route numbers or leave blank
]

# Columns to convert from seconds to minutes, mapped to sheet suffixes.
TIME_COLUMNS = {
    "Average Actual Running Time": "AvgActual(min)",
    "Average Deviation": "AvgDeviation(min)",
    "Average StartTPSScheduleDeviation": "StartTPSDev(min)",
    "Average StartTPScheduleDeviation": "StartTPSchedDev(min)",
}

# Directory to save output Excel files (optional). If None or empty, saves to current dir.
OUTPUT_DIR = (
    r"\\Folder\Path\To\Your\Output"
)
# -------------------------------------------------------------------------


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV file into a pandas DataFrame.

    :param csv_path: Path to the CSV file.
    :return: DataFrame containing the CSV data.
    """
    df = pd.read_csv(csv_path)

    # Drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def filter_routes(df: pd.DataFrame,
                  routes_to_exclude=None,
                  routes_to_include=None) -> pd.DataFrame:
    """
    Filters out routes if specified in 'routes_to_exclude', or keeps only
    routes listed in 'routes_to_include'.

    :param df: The original DataFrame.
    :param routes_to_exclude: A list of route IDs/Branches to exclude.
    :param routes_to_include: A list of route IDs/Branches to include exclusively.
    :return: Filtered DataFrame.
    """
    routes_to_exclude = routes_to_exclude or []
    routes_to_include = routes_to_include or []

    # Exclude certain routes
    if routes_to_exclude:
        df = df[~df['Branch'].isin(routes_to_exclude)]

    # Include only certain routes
    if routes_to_include:
        df = df[df['Branch'].isin(routes_to_include)]

    return df


def convert_time_columns(df: pd.DataFrame, time_columns=None) -> None:
    """
    Converts specified time columns from seconds to minutes (in-place).

    :param df: The DataFrame containing time columns.
    :param time_columns: A dict or list of column names to convert.
    """
    # If it's a dict, we only need the keys
    if isinstance(time_columns, dict):
        cols = list(time_columns.keys())
    else:
        cols = time_columns or []

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # handle non-numeric
            df[col] = df[col] / 60.0  # convert seconds to minutes


def check_route_validity(segments, variation_values):
    """
    Checks if a list of 'START - END' segments forms a single path and if there's only
    one Variation value. Returns True if valid, or False if it looks invalid.

    * Variation check: Warn if there are multiple unique Variation values.
    * Start/End check: Warn if we can’t find exactly one start stop.
    * Branching check: Warn if a start node appears more than once, indicating multiple
      possible ways out of the same stop.
    * Loop check: Implicitly done by trying to walk a single continuous path covering
      all edges. If we cannot do so, we warn.
    """
    is_valid = True

    # -----------------------------------
    # 1) Check Variation uniqueness
    # -----------------------------------
    if len(set(variation_values)) > 1:
        print(f"WARNING: Multiple Variation values encountered: {set(variation_values)}")
        is_valid = False

    # -----------------------------------
    # 2) Parse segments into edges
    # -----------------------------------
    edges = []
    for seg in segments:
        if " - " not in seg:
            print(f"WARNING: Malformed segment (missing ' - '): {seg}")
            is_valid = False
            continue
        start, end = seg.split(" - ")
        edges.append((start.strip(), end.strip()))

    if not edges:
        # No valid edges means there's nothing to sort or check
        return False

    # -----------------------------------
    # 3) Check for branching (any start repeated)
    # -----------------------------------
    start_counts = {}
    for s, e in edges:
        start_counts[s] = start_counts.get(s, 0) + 1

    for node, count in start_counts.items():
        if count > 1:
            print(f"WARNING: Branching detected at start '{node}' (appears {count} times).")
            is_valid = False

    # -----------------------------------
    # 4) Identify unique start node
    #    (i.e., a node that appears in starts but not ends)
    # -----------------------------------
    all_starts = [s for (s, _) in edges]
    all_ends = [e for (_, e) in edges]
    possible_starts = set(all_starts) - set(all_ends)

    # If we don't get exactly 1, we can still attempt to pick one. For a purely
    # linear route, though, we expect exactly 1 unique start node.
    if len(possible_starts) != 1:
        print(f"WARNING: Could not identify a unique start stop. Found: {possible_starts}")
        is_valid = False

    # We'll pick a start node to try chaining from (even if we have 0 or >1)
    if possible_starts:
        chain_start = list(possible_starts)[0]
    else:
        # fallback: pick the first in edges if no unique start
        chain_start = edges[0][0]

    # -----------------------------------
    # 5) Attempt a single-chain walk
    #    Similar to the sorting script
    # -----------------------------------
    # Build a direct map of start->end, if exactly one end per start
    adjacency = {}
    for s, e in edges:
        # We already flagged branching above, but just in case:
        if s in adjacency and adjacency[s] != e:
            # This means we have multiple different ends for the same start
            # so we can't do a single chain
            continue
        adjacency[s] = e

    # Walk the route
    chain_stops = [chain_start]
    used_edges_count = 0
    while chain_stops[-1] in adjacency:
        next_stop = adjacency[chain_stops[-1]]
        chain_stops.append(next_stop)
        used_edges_count += 1
        if used_edges_count > len(edges):
            # If we've used more edges than exist, there's definitely a loop
            print("WARNING: Route walk exceeded number of segments (loop detected).")
            is_valid = False
            break

    # If we didn't use all edges, something is off (maybe disjoint or branching)
    if used_edges_count < len(edges):
        print(
            f"WARNING: Not all segments were used in the chain walk "
            f"(used {used_edges_count}, total {len(edges)}). "
            "Possible loop/branching/disconnected segments."
        )
        is_valid = False

    return is_valid


def sort_route_segments(segments):
    """
    Sorts a list of segments of the form 'START - END' so that they form
    one continuous route path, purely by matching the end of one segment
    to the start of the next.

    Assumes there is exactly one continuous chain without branching.

    :param segments: list of strings, e.g. ['FHSH - WAPO', 'MVES - VVBU', ...]
    :return: list of strings in the correct travel sequence.
    """
    # 1) Parse each segment into a (start, end) tuple
    edges = []
    for seg in segments:
        start, end = seg.split(" - ")
        edges.append((start.strip(), end.strip()))

    # 2) Build a lookup from start -> end
    next_lookup = {}
    for start, end in edges:
        next_lookup[start] = end  # assumes each start appears only once

    # 3) Identify the unique "first" stop:
    all_starts = [start for (start, _) in edges]
    all_ends = [end for (_, end) in edges]
    possible_starts = set(all_starts) - set(all_ends)

    if len(possible_starts) != 1:
        # If we can't find a single unique start, return segments unsorted or raise error
        return segments  # or raise ValueError, depending on preference

    first_stop = possible_starts.pop()

    # 4) Walk the chain from the first stop until we can’t continue
    chain_stops = [first_stop]
    while chain_stops[-1] in next_lookup:
        chain_stops.append(next_lookup[chain_stops[-1]])

    # 5) Reconstruct segments in order
    sorted_segments = []
    for i in range(len(chain_stops) - 1):
        s = chain_stops[i]
        e = chain_stops[i + 1]
        sorted_segments.append(f"{s} - {e}")

    return sorted_segments


def create_and_save_pivots(df: pd.DataFrame,
                           output_dir: str = "",
                           file_prefix: str = "runtime_analysis_") -> None:
    """
    For each unique Branch in df, create a separate Excel file. Inside each file:
    - Separate data by unique Direction
    - For each time column (if it exists), pivot and write to a separate sheet

    Before pivoting, we:
      1) Check route validity (multiple Variation values, branching, loops, etc.).
      2) If valid, sort the segments. Then pivot in that sorted order if desired.

    :param df: The DataFrame with all routes.
    :param output_dir: Directory to save Excel files.
    :param file_prefix: Prefix for the output Excel file names.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pivot_configs = TIME_COLUMNS

    for route in df['Branch'].unique():
        route_df = df[df['Branch'] == route].copy()

        excel_file_name = f"{file_prefix}{route}.xlsx"
        if output_dir:
            excel_file_name = os.path.join(output_dir, excel_file_name)

        route_wrote_sheets = False

        with pd.ExcelWriter(excel_file_name, engine='openpyxl') as writer:
            for direction in route_df['Direction'].unique():
                direction_df = route_df[route_df['Direction'] == direction].copy()

                # 1) Check validity of this route slice
                # We'll look at all the segment strings in 'SegmentName' plus Variation
                segment_list = direction_df['SegmentName'].dropna().unique().tolist()
                variation_list = direction_df['Variation'].dropna().unique().tolist()

                is_single_path = check_route_validity(segment_list, variation_list)

                # 2) If it looks valid, you can attempt to sort the segments
                if is_single_path:
                    sorted_list = sort_route_segments(segment_list)
                else:
                    # If not valid, we can either skip sorting or keep them as-is
                    sorted_list = segment_list
                    print(f"Segments for route={route}, direction={direction} left unsorted.")

                # 3) Pivot each time_col to a separate sheet
                direction_wrote_sheets = False
                for time_col, sheet_suffix in pivot_configs.items():
                    if time_col not in direction_df.columns:
                        continue  # skip if column doesn't exist

                    # Create pivot
                    pivot_table = direction_df.pivot_table(
                        index=['Branch', 'Direction', 'TripNo', 'Variation', 'Trip'],
                        columns='SegmentName',
                        values=time_col,
                        aggfunc='mean'
                    )

                    # If pivot is empty, skip
                    if pivot_table.empty:
                        continue

                    # If we want the columns in sorted order:
                    # Filter any segments that might not appear as columns
                    pivot_cols = [seg for seg in sorted_list if seg in pivot_table.columns]
                    pivot_table = pivot_table.reindex(columns=pivot_cols)

                    sheet_name = f"{direction}_{sheet_suffix}"
                    pivot_table.to_excel(writer, sheet_name=sheet_name)
                    direction_wrote_sheets = True

                if direction_wrote_sheets:
                    route_wrote_sheets = True

            # If no sheets got created for this route, make a "NoData" sheet
            if not route_wrote_sheets:
                no_data_df = pd.DataFrame(
                    {"Info": [f"No valid data or columns found for route {route}."]}
                )
                no_data_df.to_excel(writer, sheet_name="NoData")

        print(f"Created {excel_file_name}")


def main():
    """
    Main function that orchestrates the steps:
    - Load data
    - Filter routes
    - Convert time columns
    - Create and save pivots
    """
    # Load the data
    df = load_data(CSV_PATH)

    # Filter routes if needed
    df = filter_routes(
        df,
        routes_to_exclude=ROUTES_TO_EXCLUDE,
        routes_to_include=ROUTES_TO_INCLUDE
    )

    # Convert time columns (from seconds to minutes)
    convert_time_columns(df, time_columns=TIME_COLUMNS)

    # Create pivot tables and save to Excel
    create_and_save_pivots(df, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
