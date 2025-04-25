"""
Simple GTFS-to-buffer export script + per-direction stops & real route shapes

• Loads GTFS text files into pandas / GeoPandas DataFrames
• Optionally filters on `stop_code` and/or `route_short_name`
• Buffers stops by a user-supplied distance (default ¼ mile ≈ 1 320 ft)
• Exports four groups of layers (all in WGS-84):

      1. A dissolved buffer polygon for every (route_short_name, direction_id)
      2. All stop *occurrences* split into one Shapefile per direction_id
      3. All route/shape lines split into one Shapefile per direction_id
      4. (hook left for any additional layers you may add)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER_PATH = r"Path\To\Your\GTFS\Data\Folder"
OUTPUT_DIR = r"Path\To\Your\Output\Folder"

# Optional filters (leave empty lists [] if you don’t need them)
FILTER_IN_STOP_CODES: list[str] = []  # e.g. ["1234", "5678"]
FILTER_OUT_STOP_CODES: list[str] = []
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = ["101", "202"]
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = []

BUFFER_DISTANCE_FEET: float = 1_320.0  # ¼ mile
WORK_CRS: str = "EPSG:3857"  # metric – good for buffering
EXPORT_CRS: str = "EPSG:4326"  # WGS-84 (GTFS default)

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_gtfs_data(
    files: list[str] | None = None, dtype: str | dict = str
) -> Dict[str, pd.DataFrame]:
    """
    Read standard GTFS text files into a dict keyed by filename (sans .txt).
    """
    if not os.path.exists(GTFS_FOLDER_PATH):
        raise FileNotFoundError(f"The directory '{GTFS_FOLDER_PATH}' does not exist.")

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

    missing = [fn for fn in files if not os.path.exists(os.path.join(GTFS_FOLDER_PATH, fn))]
    if missing:
        raise FileNotFoundError(f"Missing GTFS files in '{GTFS_FOLDER_PATH}': {', '.join(missing)}")

    data: Dict[str, pd.DataFrame] = {}
    for fn in files:
        key = fn[:-4]  # strip ".txt"
        path = os.path.join(GTFS_FOLDER_PATH, fn)
        try:
            df = pd.read_csv(path, dtype=dtype)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{fn}' is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"Parser error in '{fn}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error loading '{fn}': {exc}") from exc

        data[key] = df
        print(f"Loaded {fn} ({len(df):,} records).")

    return data


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
        stop_times["trip_id"].isin(trips["trip_id"]) & stop_times["stop_id"].isin(stops["stop_id"])
    ]

    # Finally, only stops appearing in stop_times
    stops = stops[stops["stop_id"].isin(stop_times["stop_id"])]

    return stops, routes, trips, stop_times


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

    merged = merged.drop_duplicates(subset=["stop_id", "route_short_name", "direction_id"])
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
    gdf_lines = gdf_lines.merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")

    return gdf_lines.dissolve(by=["route_short_name", "direction_id"], as_index=False).drop(
        columns=["shape_id", "route_id"]
    )


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
        sub.to_crs(export_crs).to_file(os.path.join(output_dir, f"routes_{dir_str}.shp"))
        print(f"✓ Exported routes ({dir_str})")


def buffer_and_export(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    buffer_distance_ft: float,
    work_crs: str,
    export_crs: str,
) -> None:
    if gdf.empty:
        print("Nothing to export – GeoDataFrame is empty.")
        return

    meters = feet_to_meters(buffer_distance_ft)
    os.makedirs(output_dir, exist_ok=True)

    for (route, direction), sub in gdf.groupby(["route_short_name", "direction_id"], dropna=False):
        route = route or "None"
        direction = int(direction) if pd.notna(direction) else 0
        fname = f"{route}_dir{direction}_buffer.shp"

        sub_proj = sub.to_crs(work_crs)
        dissolved = sub_proj.buffer(meters).unary_union
        gpd.GeoDataFrame(
            {"route": [route], "dir_id": [direction], "geometry": [dissolved]},
            crs=work_crs,
        ).to_crs(export_crs).to_file(os.path.join(output_dir, fname))

        print(f"✓ Exported buffer ({route} dir {direction})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    try:
        data = load_gtfs_data()
    except Exception as exc:  # noqa: BLE001 – broad, prints then exits
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)

    stops, routes, trips, stop_times = apply_filters(
        data,
        in_stop=FILTER_IN_STOP_CODES,
        out_stop=FILTER_OUT_STOP_CODES,
        in_route=FILTER_IN_ROUTE_SHORT_NAMES,
        out_route=FILTER_OUT_ROUTE_SHORT_NAMES,
    )
    print(
        f"Post-filter counts – stops: {len(stops):,}, "
        f"routes: {len(routes):,}, trips: {len(trips):,}, "
        f"stop_times: {len(stop_times):,}"
    )

    gdf_buf = build_stop_route_direction_gdf(stops, routes, trips, stop_times)

    gdf_stops_by_dir = gdf_buf[
        ["stop_id", "stop_code", "stop_name", "route_short_name", "direction_id", "geometry"]
    ].copy()

    gdf_routes_lines = build_route_lines_gdf(data["shapes"], routes, trips)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    buffer_and_export(gdf_buf, OUTPUT_DIR, BUFFER_DISTANCE_FEET, WORK_CRS, EXPORT_CRS)
    export_stops_by_direction(gdf_stops_by_dir, OUTPUT_DIR, EXPORT_CRS)
    export_routes_by_direction(gdf_routes_lines, OUTPUT_DIR, EXPORT_CRS)

    print("All done! 🎉")


if __name__ == "__main__":
    main()
