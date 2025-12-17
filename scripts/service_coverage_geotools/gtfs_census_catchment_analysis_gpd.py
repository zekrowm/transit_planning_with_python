"""Performs spatial analysis on GTFS transit data and demographic shapefiles.

Generates service area buffers around transit stops and estimates population,
household, and employment characteristics within those areas. Supports three
analysis modes: 'network', 'route', and 'stop'. Buffer sizes and stop filters
are configurable.

Intended for use in Jupyter notebooks with appropriate EPSG settings.

Typical inputs:
    - GTFS folder containing: trips.txt, stop_times.txt, routes.txt,
      stops.txt, calendar.txt.
    - Demographic shapefile with fields to estimate.
    - Configurable filter lists and buffer settings in the script.

Outputs:
    - Shapefiles (.shp) and Excel summaries (.xlsx) for each analysis unit.
    - Optional matplotlib plots for visual inspection.
"""

import logging
import os
from pathlib import Path
from typing import Final, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================

# Select analysis mode: "network", "route", or "stop"
ANALYSIS_MODE = "network"  # Options: "network", "route", "stop"

# Paths
GTFS_DATA_PATH = r"C:\Path\To\GTFS_data_folder"
DEMOGRAPHICS_SHP_PATH = r"C:\Path\To\census_blocks.shp"
OUTPUT_DIRECTORY = r"C:\Path\To\Output"

# Calendar / service-pattern filter
SERVICE_IDS_TO_INCLUDE: Final[list[str]] = ["3"]  # ← NEW
# e.g. ["1", "2", "3"] for your weekday patterns, [] for “no calendar filter”

# Route filters:
# 1) ROUTES_TO_INCLUDE: If non-empty, only these routes are considered.
# 2) ROUTES_TO_EXCLUDE: If non-empty, these routes are removed.
# If both are empty, all routes in routes.txt are used.
ROUTES_TO_INCLUDE: list[str] = ["101", "202"]  # e.g. [] for no include filter
ROUTES_TO_EXCLUDE: list[str] = []  # e.g. [] for no exclude filter

# Stop filters:
# 1) STOP_IDS_TO_INCLUDE: If non-empty, only these stops are considered (after route filter).
# 2) STOP_IDS_TO_EXCLUDE: If non-empty, these stops are removed (after route filter).
# If both are empty, all stops belonging to final routes are used.
STOP_IDS_TO_INCLUDE: list[
    str
] = []  # e.g. [] for no include filter or [1005, 1007] for include filter
STOP_IDS_TO_EXCLUDE: list[
    str
] = []  # e.g. [] for no include filter or [1010, 1011] for exclude filter

# Buffer distances in miles
BUFFER_DISTANCE = 0.25  # Standard buffer distance
LARGE_BUFFER_DISTANCE = 2.0  # Larger buffer distance for specified stops

# If a stop_id is in this list, use LARGE_BUFFER_DISTANCE instead.
STOP_IDS_LARGE_BUFFER: list[str] = []

# Optional FIPS filter (list of codes). Empty list = no filter.
FIPS_FILTER: list[str] = []  # Replace with FIPS code(s) for desired jurisdictions (e.g. "11001")

# Fields in demographics shapefile to multiply by area ratio
SYNTHETIC_FIELDS = [
    "total_pop",
    "total_hh",
    "tot_empl",
    "low_wage",
    "mid_wage",
    "high_wage",
    #    "minority",
    #    "est_lep",
    #    "est_lo_veh",
    #    "est_lo_v_1",
    #    "est_youth",
    #    "est_elderl",
    #    "est_low_in",
]

# EPSG code for projected coordinate system used in area calculations
CRS_EPSG_CODE = 3395  # Replace with EPSG for your study area

# GTFS files expected
REQUIRED_GTFS_FILES = [
    "trips.txt",
    "stop_times.txt",
    "routes.txt",
    "stops.txt",
    "calendar.txt",
]

# =============================================================================
# FUNCTIONS
# =============================================================================


def filter_weekday_service(calendar_df: pd.DataFrame) -> pd.Series:
    """Return service_ids for routes that run Monday through Friday.

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


def get_included_stops(
    stops_df: pd.DataFrame,
    stop_ids_to_include: list[str],
    stop_ids_to_exclude: list[str],
) -> pd.DataFrame:
    """Determine which stops to keep by applying inclusion/exclusion lists.

    Args:
        stops_df: DataFrame from stops.txt (or an already merged subset).
        stop_ids_to_include: Stop IDs to include. If non-empty, only these remain.
        stop_ids_to_exclude: Stop IDs to exclude. If non-empty, these are removed.

    Returns:
        DataFrame containing only the final included stops.
    """
    filtered = stops_df.copy()

    filtered["stop_id"] = filtered["stop_id"].astype(str)
    include = [str(s) for s in stop_ids_to_include]
    exclude = [str(s) for s in stop_ids_to_exclude]

    if include:
        filtered = filtered[filtered["stop_id"].isin(include)]

    if exclude:
        filtered = filtered[~filtered["stop_id"].isin(exclude)]

    logging.info(
        "Including %d stops after applying stop include/exclude lists.",
        len(filtered),
    )
    return filtered


def get_included_routes(
    routes_df: pd.DataFrame,
    routes_to_include: list[str],
    routes_to_exclude: list[str],
) -> pd.DataFrame:
    """Filter routes by route_short_name include/exclude lists."""
    filtered = routes_df.copy()

    if "route_short_name" not in filtered.columns:
        raise KeyError("routes_df is missing required column: 'route_short_name'")

    filtered["route_short_name"] = filtered["route_short_name"].astype(str)
    include = [str(r) for r in routes_to_include]
    exclude = [str(r) for r in routes_to_exclude]

    if include:
        filtered = filtered[filtered["route_short_name"].isin(include)]

    if exclude:
        filtered = filtered[~filtered["route_short_name"].isin(exclude)]

    logging.info("Including %d routes after route include/exclude lists.", len(filtered))
    return filtered


def pick_buffer_distance(
    stop_id: str, normal_buffer: float, large_buffer: float, large_buffer_ids: list[str]
) -> float:
    """Determine the buffer distance for a given stop_id.

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
    synthetic_fields: list[str],
) -> gpd.GeoDataFrame:
    """Clip *demographics_gdf* with *buffer_gdf* and compute synthetic totals.

    Steps
    -----
    1.  Ensure an original-area column exists (acres).
    2.  Clip polygons to the buffer.
    3.  Compute clipped-area and area-percentage.
    4.  For each requested field that exists, multiply by area percentage
        to create ``synthetic_<field>`` columns.
       * Missing fields are reported once and silently skipped.
    """
    # ---------------------------------------------------------------
    # 1. Original area (acres) — if not already present
    # ---------------------------------------------------------------
    if "area_ac_og" not in demographics_gdf.columns:
        demographics_gdf["area_ac_og"] = demographics_gdf.geometry.area / 4046.86

    # ---------------------------------------------------------------
    # 2. Clip to buffer
    # ---------------------------------------------------------------
    clipped_gdf = gpd.clip(demographics_gdf, buffer_gdf)

    # ---------------------------------------------------------------
    # 3. Clipped area + percentage
    # ---------------------------------------------------------------
    clipped_gdf["area_ac_cl"] = clipped_gdf.geometry.area / 4046.86
    clipped_gdf["area_perc"] = clipped_gdf["area_ac_cl"] / clipped_gdf["area_ac_og"]

    # Handle divide-by-zero and NaN without chained-assignment warnings
    clipped_gdf["area_perc"] = (
        clipped_gdf["area_perc"].replace([float("inf"), -float("inf")], 0).fillna(0)
    )

    # ---------------------------------------------------------------
    # 4. Synthetic fields — skip any that are missing
    # ---------------------------------------------------------------
    missing = [f for f in synthetic_fields if f not in clipped_gdf.columns]
    if missing:
        logging.warning("Synthetic field(s) not found and will be skipped: %s", missing)

    for field in synthetic_fields:
        if field not in clipped_gdf.columns:
            continue  # silently skip after the single warning above

        numeric = pd.to_numeric(clipped_gdf[field], errors="coerce").fillna(0)
        clipped_gdf[f"synthetic_{field}"] = clipped_gdf["area_perc"] * numeric

    return clipped_gdf


def export_summary_to_excel(totals_dict: dict, output_path: str) -> None:
    """Write a dictionary of aggregated synthetic fields to a single-row Excel file.

    :param totals_dict: A dictionary of {synthetic_field_name: numeric_total}.
    :param output_path: File path for the .xlsx output.
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
    synthetic_fields: list[str],
) -> None:
    """Run a single network-wide buffer/clip analysis.

    The function filters routes and stops, applies variable buffer radii,
    dissolves individual buffers into a single surface, clips demographic
    polygons, and exports both geometry and Excel summaries.

    Args:
        trips: DataFrame from *trips.txt*.
        stop_times: DataFrame from *stop_times.txt*.
        routes_df: DataFrame from *routes.txt*.
        stops_df: DataFrame from *stops.txt*.
        demographics_gdf: GeoDataFrame containing demographic data.
        routes_to_include: List of route_short_names to include.
        routes_to_exclude: List of route_short_names to exclude.
        stop_ids_to_include: List of stop_ids to include.
        stop_ids_to_exclude: List of stop_ids to exclude.
        buffer_distance_mi: Standard buffer distance in miles.
        large_buffer_distance_mi: Larger buffer distance in miles for specific stops.
        stop_ids_large_buffer: List of stop_ids that should use the large buffer distance.
        output_dir: Directory to save output files.
        synthetic_fields: List of demographic fields to synthesize.

    Returns:
        - A single shapefile (all_routes_service_buffer_data.shp)
        - A single Excel summary (all_routes_service_buffer_data.xlsx)
    """
    print("\n=== Network-wide Analysis ===")

    # 1) Filter routes
    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting network analysis.")
        return

    # 2) Subset trips to only final routes
    trips_merged = pd.merge(
        trips,
        final_routes_df[["route_id", "route_short_name"]],
        on="route_id",
    )

    # 3) Merge trips with stop_times
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")

    # 4) Merge with stops
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # 5) Filter final stops
    final_stops_df = get_included_stops(merged_data, stop_ids_to_include, stop_ids_to_exclude)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting network analysis.")
        return

    # 6) Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]),
        axis=1,
    )
    stops_gdf = gpd.GeoDataFrame(final_stops_df, geometry="geometry", crs="EPSG:4326").to_crs(
        epsg=CRS_EPSG_CODE
    )

    # 7) Compute variable buffer distances
    buffer_m = (
        stops_gdf["stop_id"].map(
            lambda sid: pick_buffer_distance(
                sid,
                normal_buffer=buffer_distance_mi,
                large_buffer=large_buffer_distance_mi,
                large_buffer_ids=stop_ids_large_buffer,
            )
        )
        * 1609.34
    )

    stops_gdf = stops_gdf.assign(buffer_distance_meters=buffer_m).set_geometry(
        stops_gdf.geometry.buffer(buffer_m)
    )

    # 8) Dissolve all buffers to create a single “network” buffer
    network_buffer_gdf = stops_gdf.dissolve().reset_index(drop=True)

    # 9) Clip and export
    clipped_result = clip_and_calculate_synthetic_fields(
        demographics_gdf,
        network_buffer_gdf,
        synthetic_fields,
    )
    synthetic_cols = [f"synthetic_{fld}" for fld in synthetic_fields]
    totals = clipped_result[synthetic_cols].sum().round(0)

    print("Network-wide totals:")
    for col, value in totals.items():
        display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
        print(f"  Total Synthetic {display_col}: {int(value)}")

    os.makedirs(output_dir, exist_ok=True)
    shp_path = os.path.join(output_dir, "all_routes_service_buffer_data.shp")
    clipped_result.to_file(shp_path)
    print(f"Exported network shapefile: {shp_path}")

    xlsx_path = os.path.join(output_dir, "all_routes_service_buffer_data.xlsx")
    final_dict = {col: int(val) for col, val in totals.items()}
    export_summary_to_excel(final_dict, xlsx_path)

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
    synthetic_fields: list[str],
) -> None:
    """Perform buffer/clip analysis for each individual route.

    The procedure repeats the buffer workflow for every `route_short_name`
    in the filtered set, exporting per-route shapefiles and Excel totals.

    Exports, for each route_short_name R:
      - A shapefile named R_service_buffer_data.shp
      - A summary Excel named R_service_buffer_data.xlsx
    """
    print("\n=== Route-by-Route Analysis ===")

    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting route-by-route analysis.")
        return

    # Merge the relevant GTFS data
    trips_merged = pd.merge(trips, final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # Filter stops per user-specified ID filters
    final_stops_df = get_included_stops(merged_data, stop_ids_to_include, stop_ids_to_exclude)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting route-by-route analysis.")
        return

    # Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(final_stops_df, geometry="geometry", crs="EPSG:4326").to_crs(
        epsg=CRS_EPSG_CODE
    )

    # Apply variable buffer logic
    stops_gdf["buffer_distance_meters"] = stops_gdf["stop_id"].apply(
        lambda sid: pick_buffer_distance(
            sid,
            normal_buffer=buffer_distance_mi,
            large_buffer=large_buffer_distance_mi,
            large_buffer_ids=stop_ids_large_buffer,
        )
        * 1609.34
    )
    stops_gdf["geometry"] = stops_gdf.apply(
        lambda row: row.geometry.buffer(row["buffer_distance_meters"]), axis=1
    )

    # Keep only necessary columns and remove duplicates
    stops_gdf = stops_gdf[["route_short_name", "stop_id", "geometry"]].drop_duplicates()

    # Dissolve buffers by route
    dissolved_by_route_gdf = stops_gdf.dissolve(by="route_short_name").reset_index()
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
            display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
            print(f"  Total Synthetic {display_col} for route {route_name}: {int(val)}")

        # Shapefile export
        os.makedirs(output_dir, exist_ok=True)
        shp_path = os.path.join(output_dir, f"{route_name}_service_buffer_data.shp")
        clipped_result.to_file(shp_path)
        print(f"Exported shapefile for route {route_name}: {shp_path}")

        # Export summary to Excel
        xlsx_path = os.path.join(output_dir, f"{route_name}_service_buffer_data.xlsx")
        final_dict = {col: int(val) for col, val in totals.items()}
        export_summary_to_excel(final_dict, xlsx_path)

        # Optional plot
        fig, ax = plt.subplots(figsize=(10, 10))
        route_buffer_gdf.plot(ax=ax, alpha=0.5, label=f"Route {route_name} Buffer")
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
    synthetic_fields: list[str],
) -> None:
    """Compute buffers and demographic catchments for each stop.

    Each GTFS stop that survives the route, trip, and stop filters is
    buffered (variable radius), clipped against the demographic layer, and
    written to individual shapefile/Excel pairs.

    Exports, for each stop_id S:
      - A shapefile named stop_S_service_buffer_data.shp
      - A summary Excel named stop_S_service_buffer_data.xlsx
    """
    print("\n=== Stop-by-Stop Analysis ===")

    final_routes_df = get_included_routes(routes_df, routes_to_include, routes_to_exclude)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting stop-by-stop analysis.")
        return

    # Merge the relevant GTFS data
    trips_merged = pd.merge(trips, final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    # Filter stops per user-specified ID filters
    final_stops_df = get_included_stops(merged_data, stop_ids_to_include, stop_ids_to_exclude)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting stop-by-stop analysis.")
        return

    # Convert to GeoDataFrame in projected CRS
    final_stops_df["geometry"] = final_stops_df.apply(
        lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(final_stops_df, geometry="geometry", crs="EPSG:4326").to_crs(
        epsg=CRS_EPSG_CODE
    )

    # Apply variable buffer logic
    stops_gdf["buffer_distance_meters"] = stops_gdf["stop_id"].apply(
        lambda sid: pick_buffer_distance(
            sid,
            normal_buffer=buffer_distance_mi,
            large_buffer=large_buffer_distance_mi,
            large_buffer_ids=stop_ids_large_buffer,
        )
        * 1609.34
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
            display_col = str(col).replace("synthetic_", "").replace("_", " ").title()
            print(f"  Total Synthetic {display_col}: {int(val)}")

        # Export shapefile
        shp_path = os.path.join(output_dir, f"stop_{stop_id_str}_service_buffer_data.shp")
        clipped_result.to_file(shp_path)
        print(f"Exported shapefile for stop {stop_id_str}: {shp_path}")

        # Export summary to Excel
        xlsx_path = os.path.join(output_dir, f"stop_{stop_id_str}_service_buffer_data.xlsx")
        final_dict = {col: int(val) for col, val in totals.items()}
        export_summary_to_excel(final_dict, xlsx_path)

        # Optional plot
        fig, ax = plt.subplots(figsize=(8, 8))
        single_stop_buffer.plot(ax=ax, alpha=0.5, label=f"Stop {stop_id_str} Buffer")
        plt.title(f"Stop {stop_id_str} Buffer Overlay")
        plt.legend()
        plt.show()


def apply_fips_filter(
    demog_gdf: gpd.GeoDataFrame,
    fips_filter: list[str],
    fips_col: str = "FIPS",
) -> gpd.GeoDataFrame:
    """Filter *demog_gdf* by county FIPS codes.

    If *fips_col* is absent the function tries to derive it from the first
    column whose name starts with ``GEOID`` (block, tract, etc.), slicing the
    first 5 characters.  If that also fails, the filter is skipped with a
    warning.
    """
    if not fips_filter:
        logging.info("No FIPS filter provided; processing all features.")
        return demog_gdf

    if fips_col not in demog_gdf.columns:
        # attempt automatic derivation
        geo_cols = [c for c in demog_gdf.columns if c.lower().startswith("geoid")]
        if geo_cols:
            src = geo_cols[0]
            demog_gdf[fips_col] = demog_gdf[src].str[:5]
            logging.info("Derived %s from %s (first 5 chars) for FIPS filtering.", fips_col, src)
        else:
            logging.warning(
                "FIPS filter requested (%s) but no '%s' column or GEOID-like "
                "field found.  Skipping the filter.",
                fips_filter,
                fips_col,
            )
            return demog_gdf

    before = len(demog_gdf)
    demog_gdf = demog_gdf[demog_gdf[fips_col].isin(fips_filter)]
    logging.info(
        "Applied FIPS filter %s — %d → %d features.",
        fips_filter,
        before,
        len(demog_gdf),
    )
    return demog_gdf


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: str, files: Optional[list[str]] = None, dtype: type = str
) -> dict[str, pd.DataFrame]:
    """Load GTFS text files from *gtfs_folder_path*.

    The function validates the presence of each requested file, reads it
    into a :class:`pandas.DataFrame`, and returns a dictionary keyed by file
    stem.

    Args:
        gtfs_folder_path: Absolute or relative path to the directory that
            contains GTFS text files.
        files: Specific GTFS file names to load.  If *None*, the canonical
            set defined in the function body is used.
        dtype: Either a single pandas dtype applied to every column or a
            mapping of column names to dtypes passed verbatim to
            :func:`pandas.read_csv`.

    Returns:
        A dictionary whose keys are file stems (e.g. ``"trips"``) and whose
        values are DataFrames with the raw GTFS contents.

    Raises:
        OSError: *gtfs_folder_path* does not exist or one or more files are
            missing.
        ValueError: A file is empty or cannot be parsed.
        RuntimeError: An :pyexc:`OSError` occurs while reading a file.
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = [
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        ]

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the catchment-area analysis."""
    # ------------------------------------------------------------------
    # Logging (leave it here if you didn't configure logging earlier)
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        # --------------------------------------------------------------
        # 1) LOAD GTFS
        # --------------------------------------------------------------
        gtfs_raw = load_gtfs_data(
            str(GTFS_DATA_PATH),
            files=REQUIRED_GTFS_FILES,
            dtype=str,  # keep everything as strings
        )
        trips = gtfs_raw["trips"]
        stop_times = gtfs_raw["stop_times"]
        routes_df = gtfs_raw["routes"]
        stops_df = gtfs_raw["stops"]

        # --------------------------------------------------------------
        # 2) OPTIONAL CALENDAR FILTER
        # --------------------------------------------------------------
        if SERVICE_IDS_TO_INCLUDE:  # e.g. ["1", "2", "3"]
            before = len(trips)
            trips = trips[trips["service_id"].isin(SERVICE_IDS_TO_INCLUDE)]
            logging.info(
                "Applied calendar filter %s — trips: %d → %d",
                SERVICE_IDS_TO_INCLUDE,
                before,
                len(trips),
            )
        else:
            logging.info("No calendar filter applied; using all %d trips.", len(trips))

        # --------------------------------------------------------------
        # 3) DEMOGRAPHICS LAYER
        # --------------------------------------------------------------
        demographics_path = Path(DEMOGRAPHICS_SHP_PATH)
        if not demographics_path.is_file():
            raise FileNotFoundError(f"Demographics shapefile not found: {demographics_path}")

        demographics_gdf = gpd.read_file(demographics_path)
        demographics_gdf = apply_fips_filter(demographics_gdf, FIPS_FILTER)
        demographics_gdf = demographics_gdf.to_crs(epsg=CRS_EPSG_CODE)

        # --------------------------------------------------------------
        # 4) ANALYSIS DISPATCH
        # --------------------------------------------------------------
        mode = ANALYSIS_MODE.lower()
        if mode == "network":
            do_network_analysis(
                trips,
                stop_times,
                routes_df,
                stops_df,
                demographics_gdf,
                ROUTES_TO_INCLUDE,
                ROUTES_TO_EXCLUDE,
                STOP_IDS_TO_INCLUDE,
                STOP_IDS_TO_EXCLUDE,
                BUFFER_DISTANCE,
                LARGE_BUFFER_DISTANCE,
                STOP_IDS_LARGE_BUFFER,
                str(OUTPUT_DIRECTORY),
                SYNTHETIC_FIELDS,
            )
        elif mode == "route":
            do_route_by_route_analysis(
                trips,
                stop_times,
                routes_df,
                stops_df,
                demographics_gdf,
                ROUTES_TO_INCLUDE,
                ROUTES_TO_EXCLUDE,
                STOP_IDS_TO_INCLUDE,
                STOP_IDS_TO_EXCLUDE,
                BUFFER_DISTANCE,
                LARGE_BUFFER_DISTANCE,
                STOP_IDS_LARGE_BUFFER,
                str(OUTPUT_DIRECTORY),
                SYNTHETIC_FIELDS,
            )
        elif mode == "stop":
            do_stop_by_stop_analysis(
                trips,
                stop_times,
                routes_df,
                stops_df,
                demographics_gdf,
                ROUTES_TO_INCLUDE,
                ROUTES_TO_EXCLUDE,
                STOP_IDS_TO_INCLUDE,
                STOP_IDS_TO_EXCLUDE,
                BUFFER_DISTANCE,
                LARGE_BUFFER_DISTANCE,
                STOP_IDS_LARGE_BUFFER,
                str(OUTPUT_DIRECTORY),
                SYNTHETIC_FIELDS,
            )
        else:
            raise ValueError(f"Invalid ANALYSIS_MODE: {ANALYSIS_MODE}")

        print("\nAnalysis completed successfully.")

    except Exception as exc:  # catch and log any error
        logging.error("Analysis terminated due to an error: %s", exc, exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    main()
