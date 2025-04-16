"""
Module for processing CLEVER runtime datasets to analyze transit running times by segment and trip.

This script supports multiple datasets (weekday, Saturday, Sunday, and other), reading data from CSV or Excel files,
and exports pivot tables as CSV files organized by route, direction, and specified time metrics. It performs the following:

- Loads and filters input datasets based on inclusion/exclusion criteria for specific routes.
- Converts runtime metrics from seconds to minutes.
- Validates route segments for continuity, variation consistency, and absence of loops or branching.
- Sorts segments into sequential order when valid.
- Sorts individual trips chronologically by actual start times.
- Creates pivot tables summarizing segment-level runtime metrics, rounded for readability.
- Saves output CSVs into structured subdirectories by dataset type.

Configuration options allow users to specify file paths, output directories, included/excluded routes, and time columns for processing.
"""

import os
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

WKDY_FILE_PATH = r"\\Your\Path\CLEVER_Runtime_by_Segment_by_Trip_Weekday.csv"
SAT_FILE_PATH  = r""
SUN_FILE_PATH  = r""
OTHER_FILE_PATH = r""

PARENT_OUTPUT_DIR = r"C:\Path\To\Outputs"  # or "" to save in current folder

ROUTES_TO_EXCLUDE = []
ROUTES_TO_INCLUDE = []

TIME_COLUMNS = {
    "Average Actual Running Time": "AvgActual(min)",
    "Average Deviation": "AvgDeviation(min)",
    "Average StartTPSScheduleDeviation": "StartTPSDev(min)",
    "Average StartTPScheduleDeviation": "StartTPSchedDev(min)",
}

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def load_data(file_path: str) -> pd.DataFrame:
    """
    Read a CSV or Excel file, depending on extension.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for file: {file_path}")

    df = df.loc[:, ~df.columns.duplicated()]
    return df


def filter_routes(df: pd.DataFrame,
                  routes_to_exclude=None,
                  routes_to_include=None) -> pd.DataFrame:
    routes_to_exclude = routes_to_exclude or []
    routes_to_include = routes_to_include or []

    if routes_to_exclude:
        df = df[~df['Branch'].isin(routes_to_exclude)]
    if routes_to_include:
        df = df[df['Branch'].isin(routes_to_include)]
    return df


def convert_time_columns(df: pd.DataFrame, time_columns=None) -> None:
    """
    Convert certain columns from seconds to minutes in-place.
    """
    if isinstance(time_columns, dict):
        cols = list(time_columns.keys())
    else:
        cols = time_columns or []

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col] / 60.0


def parse_trip_time(trip_str: str):
    """
    Extract the HH:MM portion from something like '04:56 1419958'
    and convert to integer minutes after midnight.
    Returns None if parsing fails (goes to bottom in sort).
    """
    if not isinstance(trip_str, str):
        return None

    parts = trip_str.split(maxsplit=1)  # ['04:56', '1419958']
    if not parts:
        return None
    time_part = parts[0]  # '04:56'
    try:
        t = pd.to_datetime(time_part, format="%H:%M").time()
        return t.hour * 60 + t.minute
    except ValueError:
        return None


def check_route_validity(segments, variation_values):
    """
    Checks for single Variation, valid 'START - END' strings, no branching, etc.
    """
    is_valid = True
    if len(set(variation_values)) > 1:
        print(f"WARNING: Multiple Variation values: {set(variation_values)}")
        is_valid = False

    edges = []
    for seg in segments:
        if " - " not in seg:
            print(f"WARNING: Malformed segment: {seg}")
            is_valid = False
            continue
        start, end = seg.split(" - ")
        edges.append((start.strip(), end.strip()))

    if not edges:
        return False

    start_counts = {}
    for s, _ in edges:
        start_counts[s] = start_counts.get(s, 0) + 1
    for node, count in start_counts.items():
        if count > 1:
            print(f"WARNING: Branching at '{node}' ({count} times).")
            is_valid = False

    all_starts = [s for (s, _) in edges]
    all_ends = [e for (_, e) in edges]
    possible_starts = set(all_starts) - set(all_ends)
    if len(possible_starts) != 1:
        print(f"WARNING: No unique start. Found: {possible_starts}")
        is_valid = False

    if possible_starts:
        chain_start = list(possible_starts)[0]
    else:
        chain_start = edges[0][0]

    adjacency = {}
    for s, e in edges:
        if s in adjacency and adjacency[s] != e:
            continue
        adjacency[s] = e

    chain_stops = [chain_start]
    used_edges_count = 0
    while chain_stops[-1] in adjacency:
        next_stop = adjacency[chain_stops[-1]]
        chain_stops.append(next_stop)
        used_edges_count += 1
        if used_edges_count > len(edges):
            print("WARNING: Loop detected.")
            is_valid = False
            break

    if used_edges_count < len(edges):
        print(f"WARNING: Not all segments used ({used_edges_count} of {len(edges)}).")
        is_valid = False

    return is_valid


def sort_route_segments(segments):
    """
    Sorts the list of 'START - END' segments into a linear path if possible.
    """
    edges = []
    for seg in segments:
        start, end = seg.split(" - ")
        edges.append((start.strip(), end.strip()))

    next_lookup = {}
    for start, end in edges:
        next_lookup[start] = end

    all_starts = [s for (s, _) in edges]
    all_ends = [e for (_, e) in edges]
    possible_starts = set(all_starts) - set(all_ends)
    if len(possible_starts) != 1:
        return segments

    first_stop = possible_starts.pop()
    chain_stops = [first_stop]
    while chain_stops[-1] in next_lookup:
        chain_stops.append(next_lookup[chain_stops[-1]])

    sorted_segs = []
    for i in range(len(chain_stops) - 1):
        sorted_segs.append(f"{chain_stops[i]} - {chain_stops[i+1]}")
    return sorted_segs


def create_and_save_pivots(
    df: pd.DataFrame,
    output_subdir: str,
    dataset_label: str,
    time_columns_map: dict
) -> None:
    """
    Exports one CSV per (route, direction, time_col).
    - Sort each direction's rows by a temporary 'time_sort' column derived from Trip.
    - Round numeric pivoted columns to 1 decimal place.
    - Reindex the pivot table to preserve the sorted order of (Branch, Direction, TripNo, Variation, Trip).
    """
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for route in df['Branch'].unique():
        route_df = df[df['Branch'] == route].copy()

        route_wrote_any_csv = False

        for direction in route_df['Direction'].unique():
            direction_df = route_df[route_df['Direction'] == direction].copy()

            # 1) Create temp time_sort column, sort, then drop
            if 'Trip' in direction_df.columns:
                direction_df['time_sort'] = direction_df['Trip'].apply(parse_trip_time)
                direction_df.sort_values('time_sort', inplace=True, na_position='last')
                direction_df.drop(columns='time_sort', inplace=True)

            # 2) Keep track of the exact sorted order for reindexing
            #    so pivot_table doesn't reorder the rows.
            sorted_index_df = direction_df[['Branch','Direction','TripNo','Variation','Trip']].drop_duplicates()
            sorted_index_df = sorted_index_df.set_index(['Branch','Direction','TripNo','Variation','Trip'])
            sorted_index_list = sorted_index_df.index

            # 3) Check route validity
            segments = direction_df['SegmentName'].dropna().unique().tolist()
            variations = direction_df['Variation'].dropna().unique().tolist()
            is_single_path = check_route_validity(segments, variations)
            if is_single_path:
                segments = sort_route_segments(segments)

            direction_wrote_something = False

            # 4) Pivot each time column
            for time_col, sheet_suffix in time_columns_map.items():
                if time_col not in direction_df.columns:
                    continue

                pivot_table = direction_df.pivot_table(
                    index=['Branch', 'Direction', 'TripNo', 'Variation', 'Trip'],
                    columns='SegmentName',
                    values=time_col,
                    aggfunc='mean'
                )

                if pivot_table.empty:
                    continue

                # Reorder pivot columns by sorted segments
                pivot_cols = [seg for seg in segments if seg in pivot_table.columns]
                pivot_table = pivot_table.reindex(columns=pivot_cols)

                # 5) Reindex rows to preserve the original sorted order
                pivot_table = pivot_table.reindex(index=sorted_index_list)

                # 6) Round to 1 decimal place
                pivot_table = pivot_table.round(1)

                # 7) Build CSV filename
                csv_filename = f"{dataset_label}_Route{route}_Dir{direction}_{sheet_suffix}.csv"
                csv_path = os.path.join(output_subdir, csv_filename)

                pivot_table.to_csv(csv_path, index=True, float_format="%.1f")
                print(f"Created {csv_path}")

                direction_wrote_something = True
                route_wrote_any_csv = True

            # End of time_col loop
            # If you'd like a "NoData" per direction, handle here.

        # End of direction loop
        if not route_wrote_any_csv:
            no_data_filename = f"{dataset_label}_Route{route}_NoData.csv"
            no_data_path = os.path.join(output_subdir, no_data_filename)
            no_data_df = pd.DataFrame({"Info": [f"No valid data for route {route}."]})
            no_data_df.to_csv(no_data_path, index=False)
            print(f"Created {no_data_path}")


def process_file(file_path: str, dataset_label: str):
    """
    Processes a single dataset by performing data loading, filtering, converting time columns,
    and generating pivot CSV files for runtime analysis.

    :param file_path: Path to the input CSV or Excel file.
    :param dataset_label: Label identifying the dataset type (e.g., 'wkdy', 'sat', 'sun', 'other').
    """
    print(f"\nProcessing file: {file_path} (label='{dataset_label}')")
    df = load_data(file_path)
    df = filter_routes(df, ROUTES_TO_EXCLUDE, ROUTES_TO_INCLUDE)
    convert_time_columns(df, TIME_COLUMNS)

    if PARENT_OUTPUT_DIR:
        output_subdir = os.path.join(PARENT_OUTPUT_DIR, dataset_label)
    else:
        output_subdir = dataset_label

    create_and_save_pivots(df, output_subdir, dataset_label, TIME_COLUMNS)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main function that orchestrates the processing of runtime datasets for different service periods (weekday, Saturday, Sunday, and other).

    It checks for configured file paths and initiates processing for each available dataset, skipping any with empty paths.
    """
    if WKDY_FILE_PATH:
        process_file(WKDY_FILE_PATH, 'wkdy')
    else:
        print("Skipping 'wkdy' (blank path).")

    if SAT_FILE_PATH:
        process_file(SAT_FILE_PATH, 'sat')
    else:
        print("Skipping 'sat' (blank path).")

    if SUN_FILE_PATH:
        process_file(SUN_FILE_PATH, 'sun')
    else:
        print("Skipping 'sun' (blank path).")

    if OTHER_FILE_PATH:
        process_file(OTHER_FILE_PATH, 'other')
    else:
        print("Skipping 'other' (blank path).")


if __name__ == "__main__":
    main()
