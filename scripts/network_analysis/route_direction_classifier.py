"""Classifies GTFS route directions and identifies dominant shapes per route and direction.

Determines whether GTFS shapes are cardinal (NB, SB, EB, WB) or looped (CW, CCW, LOOP),
flags suspicious direction inconsistencies, and optionally exports Excel summaries and
JPEG maps of dominant route shapes.

Typical use: ArcGIS Pro workflows or Jupyter notebooks with GTFS datasets.

Inputs:
    - GTFS files: routes.txt, trips.txt, stop_times.txt, shapes.txt, stops.txt

Outputs:
    - Directions_Summary.xlsx
    - Route_<short_name>_Dir_<direction_id>_departures.xlsx
    - Route_<short_name>_Dir_<direction_id>_DominantShape.jpeg (optional)
    - Suspicious_RouteDirections.xlsx (if inconsistencies are found)
"""

import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER = r"/path/to/your/gtfs_folder"
OUTPUT_FOLDER = r"/path/to/your/output_folder"

# Optional: Include only these route_short_name values
ROUTE_FILTER_IN: list[str] = []
# Routes to exclude
ROUTE_FILTER_OUT = ["9999A", "9999B", "9999C"]

# NAD83 / Maryland (meters)
PROJECTED_CRS = "EPSG:26985"
LOOP_THRESHOLD = 200

# Toggles
EXPORT_XLSX = True
EXPORT_JPEG = True

# NEW CODE: Boolean to analyze only the most common (modal) shape
ANALYZE_ONLY_DOMINANT_SHAPE = True

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
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
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=cast("Any", dtype), low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

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


def classify_direction(
    line_4326: LineString,
    line_projected: LineString,
    loop_threshold: int = LOOP_THRESHOLD,
) -> str:
    """Classify the direction of a GTFS shape between its start and end points.

    If the start/end points are close (less than loop_threshold), the shape
    is treated as a loop and classified based on the polygon's signed area.
    Otherwise, we compare latitude and longitude changes to decide NB, SB,
    EB, or WB.
    """
    start_lon, start_lat = line_4326.coords[0]
    end_lon, end_lat = line_4326.coords[-1]

    start_p = line_projected.coords[0]
    end_p = line_projected.coords[-1]

    dist_start_end = math.sqrt((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2)

    if dist_start_end < loop_threshold:
        coords = list(line_projected.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        area = sum(
            (x1 * y2 - x2 * y1) for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=True)
        )

        if area > 0:
            return "CCW"
        if area < 0:
            return "CW"
        return "LOOP"

    lat_diff = end_lat - start_lat
    lon_diff = end_lon - start_lon
    if abs(lat_diff) > abs(lon_diff):
        return "NB" if lat_diff > 0 else "SB"
    return "EB" if lon_diff > 0 else "WB"


def plot_route_shape(
    gdf_shape: gpd.GeoDataFrame, route: str, direction: str, output_path: str
) -> None:
    """Plot the given shape geometry and save as a .jpeg.

    Highlights the start and end of the route shape, and includes a simple
    'N' arrow for orientation in the top-left corner of the map.
    """
    if not isinstance(gdf_shape, gpd.GeoDataFrame):
        gdf_shape = cast("Any", gpd.GeoDataFrame)(
            data=[gdf_shape], columns=gdf_shape.index, crs="EPSG:4326"
        )

    gdf_shape_4326 = gdf_shape.to_crs(epsg=4326)
    shape_line = gdf_shape_4326.iloc[0].geometry
    start_lon, start_lat = shape_line.coords[0]
    end_lon, end_lat = shape_line.coords[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf_shape_4326.plot(ax=ax, color="blue", linewidth=2, alpha=0.8)

    ax.plot(start_lon, start_lat, "go", label="Start")
    ax.plot(end_lon, end_lat, "ro", label="End")

    ax.text(
        0.05,
        0.95,
        "N",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="center",
        rotation=0,
    )
    ax.annotate(
        "",
        xy=(0.05, 0.94),
        xytext=(0.05, 0.90),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=1, headwidth=6),
    )

    ax.set_title(f"Route {route}, Direction {direction} - Most Common Shape")
    ax.legend()
    plt.axis("equal")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def read_gtfs_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads GTFS files and returns the dataframes as a tuple."""
    files = ["routes.txt", "trips.txt", "stop_times.txt", "shapes.txt", "stops.txt"]
    gtfs_data = load_gtfs_data(GTFS_FOLDER, files=files, dtype=str)

    routes = gtfs_data["routes"]
    trips = gtfs_data["trips"]
    stop_times = gtfs_data["stop_times"]
    shapes = gtfs_data["shapes"]
    stops = gtfs_data["stops"]

    return routes, trips, stop_times, shapes, stops


def filter_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """Applies the route filters to the routes DataFrame and returns the filtered routes."""
    if ROUTE_FILTER_IN:
        routes_filtered = routes[routes["route_short_name"].isin(ROUTE_FILTER_IN)]
    else:
        routes_filtered = routes.copy()

    if ROUTE_FILTER_OUT:
        routes_filtered = routes_filtered[
            ~routes_filtered["route_short_name"].isin(ROUTE_FILTER_OUT)
        ]
    return routes_filtered


def create_lines_from_shapes(shapes: pd.DataFrame) -> gpd.GeoDataFrame:
    """Sorts shapes, builds LineString geometries, and returns a GeoDataFrame (EPSG:4326)."""
    shapes_grouped = shapes.sort_values(["shape_id", "shape_pt_sequence"]).groupby("shape_id")

    lines = []
    for sid, group in shapes_grouped:
        line = LineString(zip(group["shape_pt_lon"], group["shape_pt_lat"], strict=True))
        lines.append((sid, line))

    gdf = cast("Any", gpd.GeoDataFrame)(
        data=lines, columns=["shape_id", "geometry"], crs="EPSG:4326"
    )
    return gdf


def merge_and_classify_shapes(
    trips: pd.DataFrame, routes_filtered: pd.DataFrame, gdf_shapes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Filter trips by routes and merge with shape direction classification.

    Returns a DataFrame of trips joined to route metadata and a derived
    `shape_direction` for each `shape_id`.
    """
    trips_filtered = trips[trips["route_id"].isin(routes_filtered["route_id"])]
    trips_merged = trips_filtered.merge(
        routes_filtered[["route_id", "route_short_name", "route_long_name"]],
        on="route_id",
    )

    gdf_shapes_proj = gdf_shapes.to_crs(PROJECTED_CRS)

    directions: list[tuple[str, str]] = []
    for row_4326, row_proj in zip(
        gdf_shapes.itertuples(index=False),
        gdf_shapes_proj.itertuples(index=False),
        strict=True,
    ):
        directions.append(
            (row_4326.shape_id, classify_direction(row_4326.geometry, row_proj.geometry))
        )

    direction_df = pd.DataFrame(directions, columns=["shape_id", "shape_direction"])
    return trips_merged.merge(direction_df, on="shape_id")


def identify_first_last_stops(
    trips_merged: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame
) -> pd.DataFrame:
    """Identify first/last stops (with names) and merge into trips.

    Returns the augmented DataFrame with first/last stop IDs and names,
    plus `departure_time` from the first stop.
    """
    stop_times_sorted = stop_times.sort_values(["trip_id", "stop_sequence"])
    first_stops = stop_times_sorted.groupby("trip_id").first().reset_index()
    last_stops = stop_times_sorted.groupby("trip_id").last().reset_index()

    first_stops = first_stops.rename(columns={"stop_id": "first_stop_id"})
    last_stops = last_stops.rename(columns={"stop_id": "last_stop_id"})

    trips_merged = trips_merged.merge(
        first_stops[["trip_id", "first_stop_id"]], on="trip_id"
    ).merge(last_stops[["trip_id", "last_stop_id"]], on="trip_id")

    stops_ren_first = stops[["stop_id", "stop_name"]].rename(
        columns={"stop_id": "first_stop_id", "stop_name": "first_stop_name"}
    )
    stops_ren_last = stops[["stop_id", "stop_name"]].rename(
        columns={"stop_id": "last_stop_id", "stop_name": "last_stop_name"}
    )

    trips_merged = trips_merged.merge(stops_ren_first, on="first_stop_id")
    trips_merged = trips_merged.merge(stops_ren_last, on="last_stop_id")

    first_departures = stop_times[stop_times["stop_sequence"] == 1][["trip_id", "departure_time"]]
    return trips_merged.merge(first_departures, on="trip_id")


def determine_dominant_shapes(final_data: pd.DataFrame) -> pd.DataFrame:
    """Determines the dominant shape per route and direction.

    A dominant shape is the one associated with the most trips for a given
    (route_short_name, direction_id) pair. This function flags these shapes
    with a boolean column 'is_dominant' in the returned DataFrame.

    Args:
        final_data: Merged trip-shape-stop DataFrame.

    Returns:
        DataFrame with 'is_dominant' column added, True for dominant shapes.
    """
    shape_counts = (
        final_data.groupby(["route_short_name", "direction_id", "shape_id"])
        .size()
        .reset_index(name="trip_count")
    )

    idx_max = shape_counts.groupby(["route_short_name", "direction_id"])["trip_count"].idxmax()

    dominant_shapes = shape_counts.loc[idx_max]
    dominant_shapes["is_dominant"] = True

    return final_data.merge(
        dominant_shapes[["route_short_name", "direction_id", "shape_id", "is_dominant"]],
        on=["route_short_name", "direction_id", "shape_id"],
        how="left",
    )


def export_excel_summaries(summary: pd.DataFrame, final_data: pd.DataFrame) -> None:
    """Export Excel summary and per-route/direction departure files.

    Writes `Directions_Summary.xlsx` and `Route_<short_name>_Dir_<direction_id>_departures.xlsx`.
    """
    summary_path = os.path.join(OUTPUT_FOLDER, "Directions_Summary.xlsx")
    summary.to_excel(summary_path, index=False)

    grouped_fd = final_data.groupby(["route_short_name", "direction_id"])
    for (route, direction), group in grouped_fd:
        output_name = f"Route_{route}_Dir_{direction}_departures.xlsx"
        output_file = os.path.join(OUTPUT_FOLDER, output_name)
        group.sort_values("departure_time").to_excel(output_file, index=False)


def export_jpegs(summary: pd.DataFrame, gdf_shapes: gpd.GeoDataFrame) -> None:
    """Exports one JPEG map of the dominant shape per (route, direction)."""
    remaining_shape_ids = summary["shape_id"].unique().tolist()
    gdf_shapes_dominant = gdf_shapes[gdf_shapes["shape_id"].isin(remaining_shape_ids)]

    shape_info_lookup = summary[["shape_id", "route_short_name", "direction_id"]].drop_duplicates()

    for shape_id in remaining_shape_ids:
        row = shape_info_lookup[shape_info_lookup["shape_id"] == shape_id].iloc[0]
        route = row["route_short_name"]
        direction = row["direction_id"]
        route_shape_gdf = gdf_shapes_dominant[gdf_shapes_dominant["shape_id"] == shape_id]

        output_name = f"Route_{route}_Dir_{direction}_DominantShape.jpeg"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        plot_route_shape(route_shape_gdf, route, direction, output_path)


def flag_suspicious_data(summary: pd.DataFrame) -> None:
    """Flag inconsistent shape_direction vs. direction_id combinations.

    Exports `Suspicious_RouteDirections.xlsx` when inconsistencies are detected.
    """
    summary_simplified = summary[
        ["route_short_name", "direction_id", "shape_direction"]
    ].drop_duplicates()

    flags = []
    route_groups = summary_simplified.groupby("route_short_name")
    for route_name, grp in route_groups:
        unique_dirs = grp["direction_id"].unique()
        unique_shape_dirs = grp["shape_direction"].unique()
        if len(unique_dirs) > 1 and len(unique_shape_dirs) == 1:
            flags.append(
                {
                    "route_short_name": route_name,
                    "direction_id": list(unique_dirs),
                    "problem": (
                        f"Multiple direction_ids {list(unique_dirs)} but only one "
                        f"shape_direction '{unique_shape_dirs[0]}'"
                    ),
                }
            )

    def is_cardinal_direction(direct: str) -> bool:
        return direct in ("NB", "SB", "EB", "WB")

    rd_group = summary_simplified.groupby(["route_short_name", "direction_id"])
    for (rname, did), subgrp in rd_group:
        cardinal_dirs = [d for d in subgrp["shape_direction"] if is_cardinal_direction(d)]
        if len(set(cardinal_dirs)) > 1:
            flags.append(
                {
                    "route_short_name": rname,
                    "direction_id": did,
                    "problem": f"Conflicting cardinal directions {list(set(cardinal_dirs))}",
                }
            )

    flagged_df = pd.DataFrame(flags)
    if not flagged_df.empty:
        flagged_df_path = os.path.join(OUTPUT_FOLDER, "Suspicious_RouteDirections.xlsx")
        flagged_df.to_excel(flagged_df_path, index=False)
        logging.info("[INFO] Suspicious combinations flagged. See %s.", flagged_df_path)
    else:
        logging.info("[INFO] No suspicious route/direction combos found.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Primary entry point for GTFS direction classification.

    1. Reads GTFS files (routes, trips, stop_times, shapes, stops).
    2. Filters the routes according to user-defined inclusion/exclusion lists.
    3. Merges shape data with trip data, creating LineString geometries and classifying them.
    4. Identifies first/last stops for each trip, merges stop names.
    5. Determines dominant shapes within each route/direction based on trip counts.
    6. Exports summarized results and per-route files to Excel, if enabled.
    7. Exports JPEG maps of dominant shapes, if enabled.
    8. Flags suspicious data where shape_direction and direction_id seem inconsistent.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 1: Read GTFS
    routes, trips, stop_times, shapes, stops = read_gtfs_data()

    # Step 2: Filter routes
    routes_filtered = filter_routes(routes)

    # Step 3: Build shape geometries and classify directions
    gdf_shapes = create_lines_from_shapes(shapes)
    trips_merged = merge_and_classify_shapes(trips, routes_filtered, gdf_shapes)

    # Step 4: Identify first and last stops
    final_data = identify_first_last_stops(trips_merged, stop_times, stops)

    # Step 5: Determine dominant shapes
    final_data = determine_dominant_shapes(final_data)
    if ANALYZE_ONLY_DOMINANT_SHAPE:
        final_data = final_data[final_data["is_dominant"].fillna(False)]

    # Rebuild summary from final_data
    summary = (
        final_data.groupby(
            [
                "route_short_name",
                "direction_id",
                "shape_id",
                "shape_direction",
                "first_stop_name",
                "last_stop_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="trip_count")
    )

    # Step 6: Export to Excel
    if EXPORT_XLSX:
        export_excel_summaries(summary, final_data)

    # Step 7: Export JPEGs
    if EXPORT_JPEG:
        export_jpegs(summary, gdf_shapes)

    # Step 8: Flag suspicious data
    flag_suspicious_data(summary)

    logging.info("Script execution completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
