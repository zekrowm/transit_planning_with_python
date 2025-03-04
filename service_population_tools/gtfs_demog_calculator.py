"""
Combined GTFS and Demographics Analysis Script with INCLUSION/EXCLUSION Filters
for Routes and Stops, Variable Buffer Distances, and Three Analysis Modes:
- network
- route
- stop

This script processes GTFS data and a demographic shapefile to produce
buffers around transit stops and compute estimated demographic measures.

Analysis modes:
1) "network": Dissolves buffers for all (final) included routes and stops combined.
2) "route": Performs a separate buffer-and-clip analysis per route.
3) "stop": Performs a separate buffer-and-clip analysis per individual stop.

Inclusion/Exclusion:
- ROUTES_TO_INCLUDE / ROUTES_TO_EXCLUDE: filters routes by route_short_name.
- STOP_IDS_TO_INCLUDE / STOP_IDS_TO_EXCLUDE: further filter stops after route filtering.

If both route lists are empty, all routes are analyzed.
If both stop lists are empty, all stops for the final set of routes are analyzed.

Usage:
    - Adjust the CONFIGURATION variables below.
    - Run the script (e.g., `python combined_analysis.py`).
    - Check console output, shapefile exports, and optional plots.
"""

import os

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# =============================================================================

# Select analysis mode: "network", "route", or "stop"
ANALYSIS_MODE = "network"  # Options: "network", "route", "stop"

# Paths
GTFS_DATA_PATH = r"C:\Path\To\GTFS_data_folder"
DEMOGRAPHICS_SHP_PATH = r"C:\Path\To\census_blocks.shp"
OUTPUT_DIRECTORY = r"C:\Path\To\Output"

# Route filters:
# 1) ROUTES_TO_INCLUDE: If non-empty, only these routes are considered.
# 2) ROUTES_TO_EXCLUDE: If non-empty, these routes are removed.
# If both are empty, all routes in routes.txt are used.
ROUTES_TO_INCLUDE = ["101", "102"]  # e.g. [] for no include filter
ROUTES_TO_EXCLUDE = ["104"]         # e.g. [] for no exclude filter

# Stop filters:
# 1) STOP_IDS_TO_INCLUDE: If non-empty, only these stops are considered (after route filter).
# 2) STOP_IDS_TO_EXCLUDE: If non-empty, these stops are removed (after route filter).
# If both are empty, all stops belonging to final routes are used.
STOP_IDS_TO_INCLUDE = []  # e.g. [] for no include filter or [1005, 1007] for include filter
STOP_IDS_TO_EXCLUDE = []  # e.g. [] for no include filter or [1010, 1011] for exclude filter

# Buffer distances in miles
BUFFER_DISTANCE = 0.25     # Standard buffer distance
LARGE_BUFFER_DISTANCE = 2.0  # Larger buffer distance for specified stops

# If a stop_id is in this list, use LARGE_BUFFER_DISTANCE instead.
STOP_IDS_LARGE_BUFFER = [
    1001,
    1002,
    1003
]

# Optional FIPS filter (list of codes). Empty list = no filter.
FIPS_FILTER = ["11001"] # Replace with FIPS code(s) for desired jurisdictions

# Fields in demographics shapefile to multiply by area ratio
SYNTHETIC_FIELDS = [
    "total_pop", "total_hh", "tot_empl", "low_wage", "mid_wage", "high_wage",
    "est_minori", "est_lep", "est_lo_veh", "est_lo_v_1", "est_youth",
    "est_elderl", "est_low_in"
]

# EPSG code for projected coordinate system used in area calculations
CRS_EPSG_CODE = 3395  # Replace with EPSG for your study area

# GTFS files expected
REQUIRED_GTFS_FILES = [
    "trips.txt", "stop_times.txt", "routes.txt", "stops.txt", "calendar.txt"
]

# =============================================================================
# END OF CONFIGURATION SECTION
# =============================================================================


def load_gtfs_data(gtfs_path: str) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load required GTFS files into DataFrames. Raises FileNotFoundError if missing.

    :param gtfs_path: Path to the folder containing GTFS .txt files.
    :return: (trips, stop_times, routes_df, stops_df, calendar) DataFrames.
    """
    for filename in REQUIRED_GTFS_FILES:
        full_path = os.path.join(gtfs_path, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Missing file: {filename} in {gtfs_path}")

    trips = pd.read_csv(os.path.join(gtfs_path, "trips.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"))
    routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"))
    stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"))
    calendar = pd.read_csv(os.path.join(gtfs_path, "calendar.txt"))

    return trips, stop_times, routes_df, stops_df, calendar


def filter_weekday_service(calendar_df: pd.DataFrame) -> pd.Series:
    """
    Return service_ids for routes that run Monday through Friday.

    :param calendar_df: DataFrame from calendar.txt.
    :return: Series of service_id values available on all weekdays.
    """
    weekday_filter = (
        (calendar_df["monday"] == 1)
        & (calendar_df["tuesday"] == 1)
        & (calendar_df["wednesday"] == 1)
        & (calendar_df["thursday"] == 1)
        & (calendar_df["friday"] == 1)
    )
    return calendar_df[weekday_filter]["service_id"]


def apply_fips_filter(
    demog_gdf: gpd.GeoDataFrame, fips_filter: list[str]
) -> gpd.GeoDataFrame:
    """
    Filter a demographics GeoDataFrame by a list of FIPS codes (optional).

    :param demog_gdf: A GeoDataFrame of demographic data with column 'FIPS'.
    :param fips_filter: List of FIPS codes to keep. If empty, no filter is applied.
    :return: Filtered or unfiltered GeoDataFrame.
    """
    if fips_filter:
        before_count = len(demog_gdf)
        demog_gdf = demog_gdf[demog_gdf["FIPS"].isin(fips_filter)]
        after_count = len(demog_gdf)
        print(
            f"Applied FIPS filter: {fips_filter} "
            f"(reduced from {before_count} to {after_count} records)"
        )
    else:
        print("No FIPS filter applied; processing all FIPS codes.")
    return demog_gdf


def get_included_routes(
    routes_df: pd.DataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str]
) -> pd.DataFrame:
    """
    Determine which routes to keep by applying inclusion/exclusion lists.

    1) Start with all routes in routes_df.
    2) If routes_to_include is non-empty, keep only those in that list.
    3) If routes_to_exclude is non-empty, remove those from the result.

    :param routes_df: DataFrame from routes.txt.
    :param routes_to_include: List of route_short_names to include.
    :param routes_to_exclude: List of route_short_names to exclude.
    :return: DataFrame containing only the final included routes.
    """
    filtered = routes_df.copy()

    if routes_to_include:
        filtered = filtered[filtered["route_short_name"].isin(routes_to_include)]

    if routes_to_exclude:
        filtered = filtered[
            ~filtered["route_short_name"].isin(routes_to_exclude)
        ]

    final_count = len(filtered)
    print(f"Including {final_count} routes after applying include/exclude lists.")
    included_names = ", ".join(sorted(filtered["route_short_name"].unique()))
    if included_names:
        print(f"  Included Routes: {included_names}")
    else:
        print("  Included Routes: None")
    return filtered


def get_included_stops(
    stops_df: pd.DataFrame,
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str]
) -> pd.DataFrame:
    """
    Determine which stops to keep by applying inclusion/exclusion lists.

    1) Start with all stops in stops_df.
    2) If stop_ids_to_include is non-empty, keep only those IDs.
    3) If stop_ids_to_exclude is non-empty, remove those from the result.

    :param stops_df: DataFrame from stops.txt (or an already merged subset).
    :param stop_ids_to_include: List of stop_ids to include (strings or ints).
    :param stop_ids_to_exclude: List of stop_ids to exclude (strings or ints).
    :return: DataFrame containing only the final included stops.
    """
    filtered = stops_df.copy()

    # Convert to string if necessary, or ensure consistent type
    # if original GTFS has them as strings. Adjust as needed.
    if stops_df["stop_id"].dtype == "O":
        # If it's a string/object type, make sure our lists are also strings
        stop_ids_to_include = [str(s) for s in stop_ids_to_include]
        stop_ids_to_exclude = [str(s) for s in stop_ids_to_exclude]
    else:
        # Otherwise, cast the DataFrame column to int if they are numeric
        filtered["stop_id"] = filtered["stop_id"].astype(int)
        stop_ids_to_include = [int(s) for s in stop_ids_to_include]
        stop_ids_to_exclude = [int(s) for s in stop_ids_to_exclude]

    if stop_ids_to_include:
        filtered = filtered[filtered["stop_id"].isin(stop_ids_to_include)]

    if stop_ids_to_exclude:
        filtered = filtered[~filtered["stop_id"].isin(stop_ids_to_exclude)]

    final_count = len(filtered)
    print(f"Including {final_count} stops after applying stop include/exclude lists.")
    return filtered


def pick_buffer_distance(stop_id: str, normal_buffer: float, large_buffer: float, large_buffer_ids: list[str]) -> float:
    """
    Determine the buffer distance for a given stop_id.

    :param stop_id: The stop_id to check.
    :param normal_buffer: The standard buffer distance in miles.
    :param large_buffer: The larger buffer distance in miles.
    :param large_buffer_ids: List of stop_ids that require the larger buffer.
    :return: Buffer distance in miles.
    """
    # Convert as needed to match what large_buffer_ids contain
    # for consistent comparison
    str_stop_id = str(stop_id)
    large_buffer_str_ids = [str(s) for s in large_buffer_ids]

    if str_stop_id in large_buffer_str_ids:
        return large_buffer
    else:
        return normal_buffer


def clip_and_calculate_synthetic_fields(
    demographics_gdf: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    synthetic_fields: list[str]
) -> gpd.GeoDataFrame:
    """
    Clip demographics_gdf with the buffer geometry and calculate synthetic fields.
    Correctly computes the area percentage based on original polygon areas.
    """

    # Step 1: Ensure we have an "original area" column
    if "area_ac_og" not in demographics_gdf.columns:
        demographics_gdf["area_ac_og"] = (
            demographics_gdf.geometry.area / 4046.86
        )  # Convert to acres

    # Step 2: Clip the demographics GeoDataFrame with the buffer GeoDataFrame
    clipped_gdf = gpd.clip(demographics_gdf, buffer_gdf)

    # Step 3: Compute clipped area and area percentage
    clipped_gdf["area_ac_cl"] = clipped_gdf.geometry.area / 4046.86  # Clipped area in acres
    clipped_gdf["area_perc"] = clipped_gdf["area_ac_cl"] / clipped_gdf["area_ac_og"]

    # Handle cases where original area is zero to avoid division by zero
    clipped_gdf["area_perc"].replace([float('inf'), -float('inf')], 0, inplace=True)
    clipped_gdf["area_perc"].fillna(0, inplace=True)

    # Step 4: Apply partial weighting to synthetic fields
    for field in synthetic_fields:
        # Ensure the field is numeric; non-numeric values are set to 0
        clipped_gdf[field] = pd.to_numeric(clipped_gdf[field], errors="coerce").fillna(0)
        # Calculate synthetic field based on area percentage
        clipped_gdf[f"synthetic_{field}"] = clipped_gdf["area_perc"] * clipped_gdf[field]

    return clipped_gdf


def export_summary_to_excel(
    totals_dict: dict,
    output_path: str,
    label_prefix: str = ""
) -> None:
    """
    Write a dictionary of aggregated synthetic fields to a single-row Excel file.

    :param totals_dict: A dictionary of {synthetic_field_name: numeric_total}.
    :param output_path: File path for the .xlsx output.
    :param label_prefix: An optional prefix to apply in column naming or titles.
    """
    # Convert the dictionary to a single-row DataFrame
    summary_data = {k: [v] for k, v in totals_dict.items()}
    summary_df = pd.DataFrame(summary_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_excel(output_path, index=False)
    print(f"Exported Excel summary: {output_path}")


def do_network_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str]
) -> None:
    """
    Perform a single "network-wide" buffer analysis across the final included routes
    and final included stops, applying variable buffer distances for specified stops.

    Exports:
      - A single shapefile (all_routes_service_buffer_data.shp)
      - A single Excel summary (all_routes_service_buffer_data.xlsx)
    """
    print("\n=== Network-wide Analysis ===")

    # 1) Filter routes
    final_routes_df = get_included_routes(
        routes_df, routes_to_include, routes_to_exclude
    )
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting network analysis.")
        return

    # 2) Subset trips to only final routes
    trips_merged = pd.merge(
        trips,
        final_routes_df[["route_id", "route_short_name"]],
        on="route_id"
    )
    # 3) Merge trips with stop_times
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    # 4) Merge with stops
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # 5) Filter final stops
    final_stops_df = get_included_stops(
        merged_data,
        stop_ids_to_include,
        stop_ids_to_exclude
    )
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting network analysis.")
        return

    # 6) Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(
        final_stops_df, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=CRS_EPSG_CODE)

    # 7) Compute variable buffer distances
    stops_gdf["buffer_distance_meters"] = stops_gdf["stop_id"].apply(
        lambda sid: pick_buffer_distance(
            sid,
            normal_buffer=buffer_distance_mi,
            large_buffer=large_buffer_distance_mi,
            large_buffer_ids=stop_ids_large_buffer
        ) * 1609.34
    )
    stops_gdf["geometry"] = stops_gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_distance_meters"]), axis=1
    )

    # 8) Dissolve all buffers to create a single “network” buffer
    network_buffer_gdf = stops_gdf.dissolve().reset_index(drop=True)

    # 9) Clip and export
    clipped_result = clip_and_calculate_synthetic_fields(
        demographics_gdf, network_buffer_gdf, synthetic_fields
    )
    synthetic_cols = [f"synthetic_{fld}" for fld in synthetic_fields]
    totals = clipped_result[synthetic_cols].sum().round(0)

    print("Network-wide totals:")
    for col, value in totals.items():
        display_col = col.replace("synthetic_", "").replace("_", " ").title()
        print(f"  Total Synthetic {display_col}: {int(value)}")

    os.makedirs(output_dir, exist_ok=True)
    shp_path = os.path.join(output_dir, "all_routes_service_buffer_data.shp")
    clipped_result.to_file(shp_path)
    print(f"Exported network shapefile: {shp_path}")

    # Also export the summary to Excel
    xlsx_path = os.path.join(output_dir, "all_routes_service_buffer_data.xlsx")
    final_dict = {col: int(val) for col, val in totals.items()}
    export_summary_to_excel(final_dict, xlsx_path)

    # Optional plot
    fig, ax = plt.subplots(figsize=(10, 10))
    network_buffer_gdf.plot(ax=ax, alpha=0.5, label="Network Buffer")
    stops_gdf.boundary.plot(ax=ax, linewidth=0.5, label="Stop Buffers")
    plt.title("Network Buffer")
    plt.legend()
    plt.show()


def do_route_by_route_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str]
) -> None:
    """
    Perform a buffer/clip analysis separately for each route in the final route set,
    applying variable buffer distances for specified stops and also filtering stops.

    Exports, for each route_short_name R:
      - A shapefile named R_service_buffer_data.shp
      - A summary Excel named R_service_buffer_data.xlsx
    """
    print("\n=== Route-by-Route Analysis ===")

    final_routes_df = get_included_routes(
        routes_df, routes_to_include, routes_to_exclude
    )
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting route-by-route analysis.")
        return

    # Merge the relevant GTFS data
    trips_merged = pd.merge(
        trips,
        final_routes_df[["route_id", "route_short_name"]],
        on="route_id"
    )
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # Filter stops per user-specified ID filters
    final_stops_df = get_included_stops(
        merged_data,
        stop_ids_to_include,
        stop_ids_to_exclude
    )
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting route-by-route analysis.")
        return

    # Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(
        final_stops_df, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=CRS_EPSG_CODE)

    # Apply variable buffer logic
    stops_gdf["buffer_distance_meters"] = stops_gdf["stop_id"].apply(
        lambda sid: pick_buffer_distance(
            sid,
            normal_buffer=buffer_distance_mi,
            large_buffer=large_buffer_distance_mi,
            large_buffer_ids=stop_ids_large_buffer
        ) * 1609.34
    )
    stops_gdf["geometry"] = stops_gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_distance_meters"]), axis=1
    )

    # Keep only necessary columns and remove duplicates
    stops_gdf = stops_gdf[["route_short_name", "stop_id", "geometry"]].drop_duplicates()

    # Dissolve buffers by route
    dissolved_by_route_gdf = stops_gdf.dissolve(
        by="route_short_name"
    ).reset_index()
    unique_route_names = dissolved_by_route_gdf["route_short_name"].unique()

    for route_name in unique_route_names:
        print(f"\nProcessing route: {route_name}")
        route_buffer_gdf = dissolved_by_route_gdf[
            dissolved_by_route_gdf["route_short_name"] == route_name
        ]
        if route_buffer_gdf.empty:
            print(f"No stops found for route '{route_name}' - skipping.")
            continue

        clipped_result = clip_and_calculate_synthetic_fields(
            demographics_gdf, route_buffer_gdf, synthetic_fields
        )

        synthetic_cols = [f"synthetic_{f}" for f in synthetic_fields]
        totals = clipped_result[synthetic_cols].sum().round(0)
        for col, val in totals.items():
            display_col = col.replace("synthetic_", "").replace("_", " ").title()
            print(f"  Total Synthetic {display_col} for route {route_name}: {int(val)}")

        # Shapefile export
        os.makedirs(output_dir, exist_ok=True)
        shp_path = os.path.join(
            output_dir, f"{route_name}_service_buffer_data.shp"
        )
        clipped_result.to_file(shp_path)
        print(f"Exported shapefile for route {route_name}: {shp_path}")

        # Export summary to Excel
        xlsx_path = os.path.join(output_dir, f"{route_name}_service_buffer_data.xlsx")
        final_dict = {col: int(val) for col, val in totals.items()}
        export_summary_to_excel(final_dict, xlsx_path)

        # Optional plot
        fig, ax = plt.subplots(figsize=(10, 10))
        route_buffer_gdf.plot(
            ax=ax, alpha=0.5, label=f"Route {route_name} Buffer"
        )
        plt.title(f"Route {route_name} Buffer Overlay")
        plt.legend()
        plt.show()


def do_stop_by_stop_analysis(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    demographics_gdf: gpd.GeoDataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
    buffer_distance_mi: float,
    large_buffer_distance_mi: float,
    stop_ids_large_buffer: list[str],
    output_dir: str,
    synthetic_fields: list[str]
) -> None:
    """
    Perform a buffer/clip analysis for each individual stop in the final set,
    applying variable buffer distances for specified stops.

    The final set of stops is determined by:
      1) route filters
      2) GTFS merges (only stops actually used by the final routes)
      3) stop include/exclude lists

    Exports, for each stop_id S:
      - A shapefile named stop_S_service_buffer_data.shp
      - A summary Excel named stop_S_service_buffer_data.xlsx
    """
    print("\n=== Stop-by-Stop Analysis ===")

    final_routes_df = get_included_routes(
        routes_df, routes_to_include, routes_to_exclude
    )
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting stop-by-stop analysis.")
        return

    # Merge the relevant GTFS data
    trips_merged = pd.merge(
        trips,
        final_routes_df[["route_id", "route_short_name"]],
        on="route_id"
    )
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # Filter stops per user-specified ID filters
    final_stops_df = get_included_stops(
        merged_data,
        stop_ids_to_include,
        stop_ids_to_exclude
    )
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting stop-by-stop analysis.")
        return

    # Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(
        final_stops_df, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=CRS_EPSG_CODE)

    # Apply variable buffer logic
    stops_gdf["buffer_distance_meters"] = stops_gdf["stop_id"].apply(
        lambda sid: pick_buffer_distance(
            sid,
            normal_buffer=buffer_distance_mi,
            large_buffer=large_buffer_distance_mi,
            large_buffer_ids=stop_ids_large_buffer
        ) * 1609.34
    )

    # For each stop, create a buffer, clip, and store results
    unique_stops = stops_gdf["stop_id"].unique()
    os.makedirs(output_dir, exist_ok=True)

    for sid in unique_stops:
        single_stop_gdf = stops_gdf[stops_gdf["stop_id"] == sid]
        if single_stop_gdf.empty:
            continue

        stop_id_str = str(sid)
        # Buffer
        single_stop_gdf["geometry"] = single_stop_gdf.apply(
            lambda row: row.geometry.buffer(row["buffer_distance_meters"]), axis=1
        )
        # Dissolve in case stop appears in multiple trips
        single_stop_buffer = single_stop_gdf.dissolve().reset_index(drop=True)

        # Clip
        clipped_result = clip_and_calculate_synthetic_fields(
            demographics_gdf, single_stop_buffer, synthetic_fields
        )
        synthetic_cols = [f"synthetic_{f}" for f in synthetic_fields]
        totals = clipped_result[synthetic_cols].sum().round(0)

        print(f"\nStop {stop_id_str} totals:")
        for col, val in totals.items():
            display_col = col.replace("synthetic_", "").replace("_", " ").title()
            print(f"  Total Synthetic {display_col}: {int(val)}")

        # Export shapefile
        shp_path = os.path.join(
            output_dir, f"stop_{stop_id_str}_service_buffer_data.shp"
        )
        clipped_result.to_file(shp_path)
        print(f"Exported shapefile for stop {stop_id_str}: {shp_path}")

        # Export summary to Excel
        xlsx_path = os.path.join(output_dir, f"stop_{stop_id_str}_service_buffer_data.xlsx")
        final_dict = {col: int(val) for col, val in totals.items()}
        export_summary_to_excel(final_dict, xlsx_path)

        # Optional plot
        fig, ax = plt.subplots(figsize=(8, 8))
        single_stop_buffer.plot(
            ax=ax, alpha=0.5, label=f"Stop {stop_id_str} Buffer"
        )
        plt.title(f"Stop {stop_id_str} Buffer Overlay")
        plt.legend()
        plt.show()


def main():
    """
    Main driver function. Adjust ANALYSIS_MODE, route filter variables
    (ROUTES_TO_INCLUDE, ROUTES_TO_EXCLUDE), and stop filter variables
    (STOP_IDS_TO_INCLUDE, STOP_IDS_TO_EXCLUDE) in the configuration section.
    Now also exports .xlsx summaries corresponding to each .shp.
    """
    try:
        trips, stop_times, routes_df, stops_df, calendar = load_gtfs_data(
            GTFS_DATA_PATH
        )
        relevant_service_ids = filter_weekday_service(calendar)
        trips = trips[trips["service_id"].isin(relevant_service_ids)]

        if not os.path.isfile(DEMOGRAPHICS_SHP_PATH):
            raise FileNotFoundError(
                f"Demographics shapefile not found: {DEMOGRAPHICS_SHP_PATH}"
            )
        demographics_gdf = gpd.read_file(DEMOGRAPHICS_SHP_PATH)
        demographics_gdf = apply_fips_filter(demographics_gdf, FIPS_FILTER)
        demographics_gdf = demographics_gdf.to_crs(epsg=CRS_EPSG_CODE)

        mode = ANALYSIS_MODE.lower()
        if mode == "network":
            do_network_analysis(
                trips=trips,
                stop_times=stop_times,
                routes_df=routes_df,
                stops_df=stops_df,
                demographics_gdf=demographics_gdf,
                routes_to_include=ROUTES_TO_INCLUDE,
                routes_to_exclude=ROUTES_TO_EXCLUDE,
                stop_ids_to_include=STOP_IDS_TO_INCLUDE,
                stop_ids_to_exclude=STOP_IDS_TO_EXCLUDE,
                buffer_distance_mi=BUFFER_DISTANCE,
                large_buffer_distance_mi=LARGE_BUFFER_DISTANCE,
                stop_ids_large_buffer=STOP_IDS_LARGE_BUFFER,
                output_dir=OUTPUT_DIRECTORY,
                synthetic_fields=SYNTHETIC_FIELDS
            )
        elif mode == "route":
            do_route_by_route_analysis(
                trips=trips,
                stop_times=stop_times,
                routes_df=routes_df,
                stops_df=stops_df,
                demographics_gdf=demographics_gdf,
                routes_to_include=ROUTES_TO_INCLUDE,
                routes_to_exclude=ROUTES_TO_EXCLUDE,
                stop_ids_to_include=STOP_IDS_TO_INCLUDE,
                stop_ids_to_exclude=STOP_IDS_TO_EXCLUDE,
                buffer_distance_mi=BUFFER_DISTANCE,
                large_buffer_distance_mi=LARGE_BUFFER_DISTANCE,
                stop_ids_large_buffer=STOP_IDS_LARGE_BUFFER,
                output_dir=OUTPUT_DIRECTORY,
                synthetic_fields=SYNTHETIC_FIELDS
            )
        elif mode == "stop":
            do_stop_by_stop_analysis(
                trips=trips,
                stop_times=stop_times,
                routes_df=routes_df,
                stops_df=stops_df,
                demographics_gdf=demographics_gdf,
                routes_to_include=ROUTES_TO_INCLUDE,
                routes_to_exclude=ROUTES_TO_EXCLUDE,
                stop_ids_to_include=STOP_IDS_TO_INCLUDE,
                stop_ids_to_exclude=STOP_IDS_TO_EXCLUDE,
                buffer_distance_mi=BUFFER_DISTANCE,
                large_buffer_distance_mi=LARGE_BUFFER_DISTANCE,
                stop_ids_large_buffer=STOP_IDS_LARGE_BUFFER,
                output_dir=OUTPUT_DIRECTORY,
                synthetic_fields=SYNTHETIC_FIELDS
            )
        else:
            raise ValueError(f"Invalid ANALYSIS_MODE: {ANALYSIS_MODE}")

        print("\nAnalysis completed successfully.")

    except FileNotFoundError as fnf_err:
        print(f"File not found error: {fnf_err}")
    except Exception as err:
        print(f"Unexpected error occurred: {err}")


if __name__ == "__main__":
    main()
