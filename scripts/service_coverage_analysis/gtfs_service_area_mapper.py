"""
Script Name:
        gtfs_service_area_mapper.py

Purpose:
        Processes GTFS data to generate transit stop buffers, route shapes,
        and optional network-based isochrones for spatial analysis.
        This is useful for transit service area mapping and manual analysis.

Inputs:
        1. GTFS text files (stops.txt, routes.txt, trips.txt, etc.)
        2. Optional roadway network Shapefile for isochrone generation.
        3. Configuration parameters (e.g., buffer distance, filters, CRS).

Outputs:
        1. Dissolved stop buffers per route and direction shapefile.
        2. Stop locations shapefiles, split by direction.
        3. Route geometry shapefiles, split by direction.
        4. Optional: Network-based isochrone shapefiles per route and direction.

Dependencies:
        geopandas, pandas, networkx, numpy, scipy, shapely
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple
import logging
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER_PATH = r"Path\To\Your\GTFS\Folder"
OUTPUT_DIR = r"Path\To\Your\Output\Folder"

# Optional filters (leave empty lists [] if you don’t need them)
FILTER_IN_STOP_CODES: list[str] = []  # e.g. ["1234", "5678"]
FILTER_OUT_STOP_CODES: list[str] = []
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = ["101", "202"]
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = ["9999A", "9999B", "9999C"]

BUFFER_DISTANCE_FEET: float = 1320.0  # ¼ mile
WORK_CRS: str = "EPSG:3857"  # metric – good for buffering
EXPORT_CRS: str = "EPSG:4326"  # WGS-84 (GTFS default)

NETWORK_SHP_PATH: str | None = (
    r"Path\To\Your\Roadway_Centerlines.shp"  # ← leave None to disable
)
ISOCHRONE_MINUTES: int = 5  # travel-time cut-off, corresponds to 1,320 feet at 3.0 MPH
DEFAULT_SPEED_MPH: float = 3.0  # applied when speed field missing (≈walking)
NETWORK_SPEED_FIELD: str | None = (
    None  # Replace None with your speed column name (e.g. "SPEED_MPH")
)
NETWORK_ONEWAY_FIELD: str | None = (
    None  # Replace None with your oneway column (e.g. "ONEWAY") if desired
)

# ── POLYGON-SHAPING KNOBS ────────────────────────────────────────────────────
EDGE_BUFFER_M = 10  # half street-width before polygonize
TRIM_BUFFER_M = 50  # outward “Trim Polygons” distance
SIMPLIFY_TOL_M = 10  # “Polygon Simplification” tolerance
MIN_POLY_AREA_M = 1000  # drop tiny islands (< ~0.25 acre)
# Optional extra buffer to smooth isochrone hulls
ISO_SMOOTH_BUFFER_M: float | None = 30.0        # metres; 0/None = no smoothing

# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------

def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from the specified directory.
    This function uses the logging module for output.

    Parameters:
        gtfs_folder_path (str): Path to the directory containing GTFS files.
        files (list[str], optional): GTFS filenames to load. Default is all
            standard GTFS files:
            [
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
                "transfers.txt"
            ]
        dtype (str or dict, optional): Pandas dtype to use. Default is str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        OSError: If gtfs_folder_path doesn't exist or if any required file is missing.
        ValueError: If a file is empty or there's a parsing error.
        RuntimeError: For OS errors during file reading.
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
        raise OSError(
            f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}"
        )

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"File '{file_name}' in '{gtfs_folder_path}' is empty."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------

def feet_to_meters(feet: float) -> float:
    """Convert feet to metres (1 ft = 0.3048 m)."""
    return feet * 0.3048


def apply_filters(
    data: Dict[str, pd.DataFrame],
    in_stop: list[str],
    out_stop: list[str],
    in_route: list[str],
    out_route: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return *filtered* copies of stops, routes, trips and stop_times.
    """
    stops = data["stops"].copy()
    routes = data["routes"].copy()
    trips = data["trips"].copy()
    stop_times = data["stop_times"].copy()

    # ─ stop_code filters ──────────────────────────────────────────────────
    if in_stop:
        stops = stops[stops["stop_code"].isin(in_stop)]
    if out_stop:
        stops = stops[~stops["stop_code"].isin(out_stop)]

    # ─ route_short_name filters ───────────────────────────────────────────
    if in_route:
        routes = routes[routes["route_short_name"].isin(in_route)]
    if out_route:
        routes = routes[~routes["route_short_name"].isin(out_route)]

    # Keep only trips referencing remaining routes
    trips = trips[trips["route_id"].isin(routes["route_id"])]

    # Keep only stop_times referencing remaining trips *and* stops
    stop_times = stop_times[
        stop_times["trip_id"].isin(trips["trip_id"])
        & stop_times["stop_id"].isin(stops["stop_id"])
    ]

    # Finally, only stops appearing in stop_times
    stops = stops[stops["stop_id"].isin(stop_times["stop_id"])]

    return stops, routes, trips, stop_times


def _snap_to_nearest_nodes(
    graph: nx.Graph,
    coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    For every (x, y) tuple in `coords` return the exact graph node that is
    spatially closest.  If the point is already a node, it comes back unchanged.
    """
    nodes_array = np.array(graph.nodes())
    kd = cKDTree(nodes_array)

    snapped = []
    for xy in coords:
        if xy in graph:
            snapped.append(xy)
        else:
            _, idx = kd.query(xy)
            snapped.append(tuple(nodes_array[idx]))
    return snapped


def export_buffers(  # very small wrapper around the old exporter
    gdf_buffers: gpd.GeoDataFrame,
    output_dir: str,
    export_crs: str,
) -> None:
    if gdf_buffers.empty:
        print("No buffers to export.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for _, row in gdf_buffers.iterrows():
        fname = f"{row.route_short_name}_dir{row.direction_id}_buffer.shp"
        gpd.GeoDataFrame([row], crs=gdf_buffers.crs).to_crs(export_crs).to_file(
            os.path.join(output_dir, fname)
        )
        print(f"✓ Exported buffer ({row.route_short_name} dir {row.direction_id})")


def build_stop_route_direction_gdf(
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Join stops, routes, trips and stop_times so each *stop occurrence* knows
    its `route_short_name` and `direction_id`.
    """
    gdf_stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )[["stop_id", "stop_code", "stop_name", "geometry"]]

    merged = (
        stop_times.merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id")
        .merge(routes[["route_id", "route_short_name"]], on="route_id")
        .merge(gdf_stops, on="stop_id")
    )

    merged = merged.drop_duplicates(
        subset=["stop_id", "route_short_name", "direction_id"]
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def build_route_lines_gdf(
    shapes: pd.DataFrame, routes: pd.DataFrame, trips: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Build a dissolved line for every surviving (route_short_name, direction_id).

    * Properly sorts by numeric `shape_pt_sequence` (or `shape_dist_traveled`).
    * Casts coordinates to float to avoid “string-to-string” vertex artefacts.
    """
    needed = {"shape_id", "shape_pt_lat", "shape_pt_lon"}
    if shapes.empty or not needed.issubset(shapes.columns):
        return gpd.GeoDataFrame(
            columns=["route_short_name", "direction_id", "geometry"], crs="EPSG:4326"
        )

    # ↓ keep only shapes referenced by surviving trips
    trips_has_shape = trips.dropna(subset=["shape_id"])
    shapes = shapes[shapes["shape_id"].isin(trips_has_shape["shape_id"].unique())]

    # ↓ ensure numeric types for correct ordering & geometry creation
    shapes["shape_pt_lat"] = shapes["shape_pt_lat"].astype(float)
    shapes["shape_pt_lon"] = shapes["shape_pt_lon"].astype(float)

    if "shape_pt_sequence" in shapes.columns:
        shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(int)
        order_col = "shape_pt_sequence"
    elif "shape_dist_traveled" in shapes.columns:
        shapes["shape_dist_traveled"] = shapes["shape_dist_traveled"].astype(float)
        order_col = "shape_dist_traveled"
    else:
        raise ValueError(
            "`shapes.txt` lacks both `shape_pt_sequence` and `shape_dist_traveled` – "
            "cannot reconstruct polyline order."
        )

    # ↓ build a LineString for every unique shape_id
    records: list[dict] = []
    for s_id, grp in shapes.groupby("shape_id"):
        grp = grp.sort_values(order_col)
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(coords) >= 2:
            records.append({"shape_id": s_id, "geometry": LineString(coords)})

    gdf_lines = gpd.GeoDataFrame(records, crs="EPSG:4326")
    if gdf_lines.empty:
        return gdf_lines

    # ↓ attach route_short_name + direction_id, then dissolve by both
    look = trips_has_shape[["shape_id", "route_id", "direction_id"]].drop_duplicates()
    gdf_lines = gdf_lines.merge(look, on="shape_id", how="left")
    gdf_lines = gdf_lines.merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )

    return gdf_lines.dissolve(
        by=["route_short_name", "direction_id"], as_index=False
    ).drop(columns=["shape_id", "route_id"])


def export_stops_by_direction(
    gdf_stops: gpd.GeoDataFrame, output_dir: str, export_crs: str
) -> None:
    if gdf_stops.empty:
        print("No stops to export.")
        return

    for direction, sub in gdf_stops.groupby("direction_id", dropna=False):
        dir_str = f"dir{int(direction) if pd.notna(direction) else 0}"
        sub.to_crs(export_crs).to_file(os.path.join(output_dir, f"stops_{dir_str}.shp"))
        print(f"✓ Exported stops ({dir_str})")


def export_routes_by_direction(
    gdf_routes: gpd.GeoDataFrame, output_dir: str, export_crs: str
) -> None:
    if gdf_routes.empty:
        print("No routes to export.")
        return

    for direction, sub in gdf_routes.groupby("direction_id", dropna=False):
        dir_str = f"dir{int(direction) if pd.notna(direction) else 0}"
        sub.to_crs(export_crs).to_file(
            os.path.join(output_dir, f"routes_{dir_str}.shp")
        )
        print(f"✓ Exported routes ({dir_str})")


def _mph_to_mps(mph: float) -> float:
    """Miles-per-hour ➝ metres-per-second."""
    return mph * 0.44704


def _yield_segments(
    geom: LineString | "MultiLineString",
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Yield consecutive coordinate pairs from (Multi)LineStrings."""
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        return list(zip(coords[:-1], coords[1:]))
    elif geom.geom_type == "MultiLineString":
        segs = []
        for line in geom.geoms:
            segs.extend(_yield_segments(line))
        return segs
    return []


def filter_network_to_buffers(
    gdf_net: gpd.GeoDataFrame,
    gdf_buffers: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Keep only network features that intersect at least one buffer polygon.
    """
    if gdf_buffers.empty or gdf_net.empty:
        return gpd.GeoDataFrame(columns=gdf_net.columns, crs=gdf_net.crs)

    # spatial index makes this quick even for large layers
    buffer_union = gdf_buffers.unary_union
    return gdf_net[gdf_net.intersects(buffer_union)].copy()


def build_network_graph_from_gdf(  # same logic as before, but no file I/O
    gdf_net: gpd.GeoDataFrame,
    speed_field: str | None = None,
    oneway_field: str | None = None,
    default_speed_mph: float = 3.0,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    for _, row in gdf_net.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # ─ speed & oneway ─
        try:
            mph_val = float(row[speed_field]) if speed_field else default_speed_mph
            if mph_val <= 0:
                mph_val = default_speed_mph
        except Exception:
            mph_val = default_speed_mph
        speed_mps = _mph_to_mps(mph_val)

        oneway_flag = str(row[oneway_field]).upper() if oneway_field else "N"
        oneway = oneway_flag.startswith("Y")

        # ─ break into individual segments ─
        for u_xy, v_xy in _yield_segments(geom):
            seg_len_m = LineString([u_xy, v_xy]).length
            if seg_len_m == 0:
                continue
            travel_time_s = seg_len_m / speed_mps
            graph.add_edge(u_xy, v_xy, weight=travel_time_s)
            if not oneway:
                graph.add_edge(v_xy, u_xy, weight=travel_time_s)

    print(
        f"  ✔ graph has {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges"
    )
    return graph


def generate_isochrone(
    graph: nx.DiGraph,
    origin_pts_xy: list[tuple[float, float]],
    cutoff_minutes: int,
    work_crs: str = WORK_CRS,
    export_crs: str = EXPORT_CRS,
) -> gpd.GeoSeries:
    """
    Build a street-exact isochrone and post-process it at reasonable detail level.
    """
    # 0️⃣  quick exit
    if not origin_pts_xy:
        return gpd.GeoSeries([], crs=export_crs)

    # 1️⃣  multi-source Dijkstra
    cutoff_s = cutoff_minutes * 60
    lengths, paths = nx.multi_source_dijkstra(
        graph,
        _snap_to_nearest_nodes(graph, origin_pts_xy),
        cutoff=cutoff_s,
        weight="weight",
    )
    if not lengths:
        return gpd.GeoSeries([], crs=export_crs)

    # 2️⃣  collect every traversed edge
    reach_edges = {
        (u, v) for path in paths.values() for u, v in zip(path[:-1], path[1:])
    }
    edge_lines = [LineString([u, v]) for u, v in reach_edges]

    # 3️⃣  buffer edges → union → polygonize → prune specks
    buff = gpd.GeoSeries(edge_lines, crs=work_crs).buffer(EDGE_BUFFER_M)
    raw_union = unary_union(buff)
    polys = list(polygonize(raw_union))
    gseries = gpd.GeoSeries(polys, crs=work_crs)
    gseries = gseries[gseries.area >= MIN_POLY_AREA_M]

    if gseries.empty:
        return gpd.GeoSeries([], crs=export_crs)

    hull = unary_union(gseries)  # dissolve to one multipart polygon

    # ────────────────────────────────────────────────────────────────
    # NEW  ➜  ArcGIS-style “trim + simplify” post-processing
    # ----------------------------------------------------------------
    if TRIM_BUFFER_M and TRIM_BUFFER_M != 0:
        hull = hull.buffer(TRIM_BUFFER_M)  # outward trim

    if SIMPLIFY_TOL_M and SIMPLIFY_TOL_M > 0:
        hull = shapely.simplify(hull, SIMPLIFY_TOL_M, preserve_topology=True)
    # ────────────────────────────────────────────────────────────────

    # 4️⃣  wrap in a GeoSeries and return in export CRS
    return gpd.GeoSeries([hull], crs=work_crs).to_crs(export_crs)


def export_isochrones_by_direction(
    gdf_stops: gpd.GeoDataFrame,
    graph: nx.DiGraph,
    output_dir: str,
    cutoff_min: int,
    smooth_buffer_m: float | None = None,
) -> None:
    """
    For each (route_short_name, direction_id) build an isochrone from all
    stops in that group, union them, optionally smooth the hull with a small
    buffer, and export one Shapefile.

    Parameters
    ----------
    gdf_stops : GeoDataFrame
        Per-stop occurrences with route_short_name & direction_id attached.
    graph : nx.DiGraph
        Pre-built, time-weighted street network graph.
    output_dir : str
        Destination folder for Shapefiles (will be created if absent).
    cutoff_min : int
        Travel-time cut-off in **minutes**.
    smooth_buffer_m : float | None, optional
        Extra buffer (in metres, applied in WORK_CRS) to smooth jagged
        isochrone edges.  Set 0 or None to disable.
    """
    if gdf_stops.empty:
        print("No stops ➝ no isochrones produced.")
        return

    os.makedirs(output_dir, exist_ok=True)

    grouped = gdf_stops.groupby(["route_short_name", "direction_id"], dropna=False)
    for (route, direction), sub in grouped:
        # 1⃣  project stops → WORK_CRS and pull XY tuples
        origin_xy = [(pt.x, pt.y) for pt in sub.to_crs(WORK_CRS).geometry]

        # 2⃣  build raw isochrone (in EXPORT_CRS)
        iso = generate_isochrone(
            graph,
            origin_xy,
            cutoff_minutes=cutoff_min,
        )
        if iso.empty:
            print(
                f"  (skip) no reach within {cutoff_min} min for {route or 'None'} dir {direction}"
            )
            continue

        # 3⃣  optional smoothing (tiny buffer ➝ soften angles)
        if smooth_buffer_m and smooth_buffer_m > 0:
            iso = iso.to_crs(WORK_CRS).buffer(  # back to metric CRS
                smooth_buffer_m
            )  # buffer/union in metres
            iso = gpd.GeoSeries(iso, crs=WORK_CRS).to_crs(EXPORT_CRS)

        # 4⃣  write out a small GeoDataFrame with the polygon
        fname = (
            f"{route or 'None'}_dir{int(direction) if pd.notna(direction) else 0}"
            f"_iso{cutoff_min}min.shp"
        )
        gpd.GeoDataFrame(
            {
                "route": [route or "None"],
                "dir_id": [direction],
                "minutes": [cutoff_min],
            },
            geometry=iso,
            crs=iso.crs,
        ).to_file(os.path.join(output_dir, fname))

        print(
            f"✓ Exported isochrone ({route or 'None'} dir {direction}, ≤{cutoff_min} min)"
        )


def build_buffers_gdf(
    gdf_stops: gpd.GeoDataFrame,
    buffer_distance_ft: float,
    work_crs: str,
) -> gpd.GeoDataFrame:
    """
    Build one dissolved walking-distance buffer per
    (route_short_name, direction_id).

    Returns
    -------
    GeoDataFrame
        • CRS == `work_crs`
        • Columns: route_short_name, direction_id, geometry
    """
    if gdf_stops.empty:
        return gpd.GeoDataFrame(
            columns=["route_short_name", "direction_id", "geometry"],
            crs=work_crs,
        )

    meters = feet_to_meters(buffer_distance_ft)
    records: list[dict] = []

    # dissolve within each (route, direction) group
    for (route, direction), sub in gdf_stops.groupby(
        ["route_short_name", "direction_id"], dropna=False
    ):
        poly = sub.to_crs(work_crs).buffer(meters).unary_union
        records.append(
            {
                "route_short_name": route or "None",
                "direction_id": int(direction) if pd.notna(direction) else 0,
                "geometry": poly,
            }
        )

    return gpd.GeoDataFrame(records, crs=work_crs)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # ─────────────────────────────── LOGGING ──────────────────────────────
    # (If you placed this block at the top of the file instead, omit it here.)
    import logging

    logging.basicConfig(
        level=logging.INFO,                  # DEBUG for extra chatter
        format="%(levelname)s: %(message)s",
    )

    # ──────────────────────────────── GTFS ────────────────────────────────
    try:
        # new signature: pass the folder path explicitly
        data = load_gtfs_data(GTFS_FOLDER_PATH)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.critical("FATAL: %s", exc)
        sys.exit(1)

    stops, routes, trips, stop_times = apply_filters(
        data,
        in_stop=FILTER_IN_STOP_CODES,
        out_stop=FILTER_OUT_STOP_CODES,
        in_route=FILTER_IN_ROUTE_SHORT_NAMES,
        out_route=FILTER_OUT_ROUTE_SHORT_NAMES,
    )
    logging.info(
        "Post-filter counts — stops: %s, routes: %s, trips: %s, stop_times: %s",
        f"{len(stops):,}",
        f"{len(routes):,}",
        f"{len(trips):,}",
        f"{len(stop_times):,}",
    )

    # ── 2. per-stop occurrences (route & direction already attached) ─────
    gdf_buf = build_stop_route_direction_gdf(stops, routes, trips, stop_times)

    gdf_stops_by_dir = gdf_buf[
        [
            "stop_id",
            "stop_code",
            "stop_name",
            "route_short_name",
            "direction_id",
            "geometry",
        ]
    ].copy()

    # ── 3. route polylines ───────────────────────────────────────────────
    gdf_routes_lines = build_route_lines_gdf(data["shapes"], routes, trips)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 4. walk-access buffers ───────────────────────────────────────────
    gdf_buffers = build_buffers_gdf(gdf_buf, BUFFER_DISTANCE_FEET, WORK_CRS)
    export_buffers(gdf_buffers, OUTPUT_DIR, EXPORT_CRS)

    # ── 5. plain GTFS exports (unchanged) ────────────────────────────────
    export_stops_by_direction(gdf_stops_by_dir, OUTPUT_DIR, EXPORT_CRS)
    export_routes_by_direction(gdf_routes_lines, OUTPUT_DIR, EXPORT_CRS)

    # ── 6. optional street network ➝ isochrones ──────────────────────────
    if NETWORK_SHP_PATH and os.path.exists(NETWORK_SHP_PATH):
        try:
            gdf_net_all = gpd.read_file(NETWORK_SHP_PATH).to_crs(WORK_CRS)
            gdf_net_clip = filter_network_to_buffers(gdf_net_all, gdf_buffers)
            kept_pct = len(gdf_net_clip) / len(gdf_net_all) * 100
            logging.info(
                "Keeping %s of %s road features (%.1f %%) that intersect walking buffers",
                f"{len(gdf_net_clip):,}",
                f"{len(gdf_net_all):,}",
                kept_pct,
            )

            graph = build_network_graph_from_gdf(
                gdf_net_clip,
                speed_field=NETWORK_SPEED_FIELD,
                oneway_field=NETWORK_ONEWAY_FIELD,
                default_speed_mph=DEFAULT_SPEED_MPH,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to build network — %s", exc)
        else:
            export_isochrones_by_direction(
                gdf_stops_by_dir,
                graph,
                OUTPUT_DIR,
                cutoff_min=ISOCHRONE_MINUTES,
                smooth_buffer_m=ISO_SMOOTH_BUFFER_M,
            )
    else:
        logging.info("No network file supplied — skipping isochrones.")

    logging.info("All done! 🎉")


if __name__ == "__main__":
    main()
