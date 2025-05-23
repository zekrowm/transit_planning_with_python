"""
Script Name:
    gtfs_stop_spacing_calculator.py

Purpose:
    Processes General Transit Feed Specification (GTFS) data to generate essential
    geospatial shapefiles for transit analysis. It reads standard GTFS text files
    (stops, routes, trips, stop_times, shapes) and converts them into projected
    point (stops) and line (routes, segments) features.

    A key output is the 'segments' layer, which breaks down each route's path
    (per direction) into individual LineString segments connecting consecutive stops.
    These segments include a calculated length attribute (`length_ft`) in US survey feet,
    facilitating analyses like stop spacing. The script requires specifying a
    projected Coordinate Reference System (CRS) appropriate for distance measurement.

Outputs:
    1.  `stops.shp`: Projected points for served stops.
    2.  `routes.shp`: Projected lines representing route patterns (by shape_id or route/direction).
    3.  `segments.shp`: The main output; projected stop-to-stop lines with calculated
    `length_ft` for all routes/directions.
    4.  `<route>_<dir>.shp`: Individual shapefiles containing segments, split out
    for each unique route_id and direction_id combination.

Inputs:
    1. GTFS data (folder or zip) including:
    `stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `shapes.txt`.

Dependencies:
    pandas, geopandas, shapely, numpy
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint
from shapely.ops import split as split_line

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_PATH: str = r"C:\Path\To\Your\GTFS\Folder"
OUTPUT_FOLDER: str = r"C:\Path\To\Your\Output\Folder"

FILTER_OUT_LIST: list[str] = ["9999A", "9999B", "9999C"]
INCLUDE_ROUTE_IDS: list[str] = []

ROUTE_UNION: bool = False
# Replace with your preferred EPSG
PROJECTED_CRS: str = "EPSG:2263"  # DC / MD StatePlane (US survey ft)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _ensure_output_folder(folder: str) -> Path:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_gtfs_tables(gtfs_path: Path) -> Dict[str, pd.DataFrame]:
    """Read core GTFS tables into DataFrames (folder or .zip)."""
    filenames = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    if gtfs_path.is_dir():
        return {k: pd.read_csv(gtfs_path / v) for k, v in filenames.items()}

    if gtfs_path.is_file() and gtfs_path.suffix.lower() == ".zip":
        print("Detected GTFS zip – extracting to temp dir…")
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        return {k: pd.read_csv(root / v) for k, v in filenames.items()}

    raise ValueError("GTFS_PATH must be a folder or a .zip file")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    required: Dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon"},
        "routes": {"route_id", "route_short_name"},
        "trips": {"trip_id", "route_id", "shape_id", "direction_id"},
        "stop_times": {"trip_id", "stop_id"},
        "shapes": {
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lat",
            "shape_pt_lon",
        },
    }

    missing_messages: list[str] = []
    for tbl, needed in required.items():
        present = set(dfs[tbl].columns)
        missing = needed - present
        if missing:
            missing_messages.append(
                f"• {tbl}.txt is missing: {', '.join(sorted(missing))}"
            )

    if missing_messages:
        error_msg = (
            "GTFS validation failed – required columns not found:\n"
            + "\n".join(missing_messages)
            + "\n\nPlease fix the GTFS feed or adjust the script."
        )
        raise ValueError(error_msg)


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str],
    exclude_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = routes.loc[~routes["route_id"].isin(exclude_ids)].copy()
    if include_ids:
        routes = routes.loc[routes["route_id"].isin(include_ids)].copy()
    trips = trips.loc[trips["route_id"].isin(routes["route_id"])].copy()
    return routes, trips


def _build_stops_gdf(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Make a GeoDataFrame of served stops with route_id & direction_id lists."""
    served = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
    stops = stops[stops["stop_id"].isin(served["stop_id"])].copy()

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(crs)

    trip_attrs = trips[["trip_id", "route_id", "direction_id"]].merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    merged = served[["trip_id", "stop_id"]].merge(trip_attrs, on="trip_id", how="left")
    agg = (
        merged.groupby("stop_id")[["route_id", "direction_id", "route_short_name"]]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    gdf = gdf.merge(agg, on="stop_id", how="left")

    print(f"Stops GDF – kept {len(gdf):,} served stops")
    return gdf


def _build_routes_gdf(
    shapes: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
    union_shapes: bool,
) -> gpd.GeoDataFrame:
    """Make a GeoDataFrame of route shapes with direction_id."""
    shape_cols = ["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]

    lines = (
        shapes[shape_cols]
        .sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id")
        .apply(lambda g: LineString(zip(g.shape_pt_lon, g.shape_pt_lat)))
        .to_frame("geometry")
        .reset_index()
    )

    gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(crs)

    # keep direction_id
    gdf = gdf.merge(
        trips.drop_duplicates("shape_id")[["shape_id", "route_id", "direction_id"]],
        on="shape_id",
        how="left",
    ).merge(routes, on="route_id", how="left")

    if union_shapes:
        gdf = gdf.dissolve(
            by=["route_id", "direction_id"],
            as_index=False,
            aggfunc={"route_short_name": "first", "route_long_name": "first"},
        ).explode(ignore_index=True)

    print(f"Routes GDF – built {len(gdf):,} shapes")
    return gdf


def _split_into_segments(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Split each route/direction shape at its own stops only."""
    seg_records: list[dict] = []
    sindex = stops_gdf.sindex

    for _, r in routes_gdf.iterrows():
        line = r.geometry
        rid = r.route_id
        drn = r.direction_id

        # bbox filter, then restrict to stops on same route & direction
        cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
        cand = cand[
            cand.route_id.apply(lambda ids: rid in ids)
            & cand.direction_id.apply(lambda ids: drn in ids)
        ]
        if cand.empty:
            continue

        # snap & unique distances
        dists = np.array([line.project(pt) for pt in cand.geometry])
        uniq = np.unique(dists)
        snap_pts = [line.interpolate(d) for d in uniq]

        pieces = split_line(line, MultiPoint(snap_pts))
        if isinstance(pieces, LineString):
            pieces = [pieces]
        else:
            pieces = [g for g in pieces.geoms if isinstance(g, LineString)]

        for seg in pieces:
            if seg.length:
                seg_records.append(
                    {
                        "route_id": rid,
                        "direction_id": drn,
                        "route_short": r.get("route_short_name"),
                        "geometry": seg,
                    }
                )

    seg_gdf = gpd.GeoDataFrame(seg_records, crs=crs)
    # length_ft in US survey feet if already projected, else convert from meters
    seg_gdf["length_ft"] = seg_gdf.length * (1.0 if "2263" in crs else 3.28084)

    print(f"Segments GDF – generated {len(seg_gdf):,} pieces")
    return seg_gdf


def _export(gdf: gpd.GeoDataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.shp"
    gdf.to_file(path)
    print(f"Wrote {path.name}")


def _export_segments_by_route_dir(seg_gdf: gpd.GeoDataFrame, out_dir: Path) -> None:
    """Write one .shp per (route_id, direction_id)."""
    for (rid, drn), grp in seg_gdf.groupby(["route_id", "direction_id"]):
        suffix = "dir0" if drn == 0 else "dir1" if drn == 1 else f"dir{drn}"
        fname = f"{rid}_{suffix}.shp"
        grp.to_file(out_dir / fname)
        print(f"Wrote {fname}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("STEP 0  Reading GTFS tables…")
    gtfs_path = Path(GTFS_PATH)
    dfs = _read_gtfs_tables(gtfs_path)

    try:
        _validate_columns(dfs)
    except ValueError as err:
        print("\nERROR – invalid GTFS feed:\n" + str(err))
        sys.exit(1)

    routes_df, trips_df = _filter_routes(
        dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, FILTER_OUT_LIST
    )
    out_dir = _ensure_output_folder(OUTPUT_FOLDER)

    print("STEP 1  Building stops shapefile…")
    stops_gdf = _build_stops_gdf(
        dfs["stops"], dfs["stop_times"], trips_df, routes_df, PROJECTED_CRS
    )
    _export(stops_gdf, out_dir, "stops")

    print("STEP 2  Building routes shapefile…")
    routes_gdf = _build_routes_gdf(
        dfs["shapes"], trips_df, routes_df, PROJECTED_CRS, ROUTE_UNION
    )
    _export(routes_gdf, out_dir, "routes")

    print("STEP 3  Splitting routes into stop-to-stop segments…")
    segs_gdf = _split_into_segments(routes_gdf, stops_gdf, PROJECTED_CRS)
    _export(segs_gdf, out_dir, "segments")  # master file
    _export_segments_by_route_dir(segs_gdf, out_dir)  # one per route/dir

    print("\nAll done! Outputs in:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nUNEXPECTED ERROR:", exc)
        sys.exit(1)
