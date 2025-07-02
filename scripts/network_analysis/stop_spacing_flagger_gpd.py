"""GTFS to GIS pipeline for stop-spacing analysis, via route-segments.

This module converts a General Transit Feed Specification (GTFS) package
(directory or .zip) into projected ESRI Shapefiles suitable for spatial
analytics and generates an optional text log that flags consecutive served
stops that violate a minimum spacing threshold.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split as split_line

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_PATH: str = r"Path\To\Your\GTFS_Data_Folder"  # folder or .zip
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

FILTER_OUT_LIST: list[str] = ["9999A", "9999B", "9999C"]
INCLUDE_ROUTE_IDS: list[str] = ["101", "202"]

ROUTE_UNION: bool = False
PROJECTED_CRS: str = "EPSG:2263"  # feet-based CRS

MIN_SPACING_FT: float = 400.0  # 1/8 mile default
SPACING_LOG_FILE: str = "short_spacing_segments.txt"

# =============================================================================
# FUNCTIONS
# =============================================================================


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if necessary) and return the output folder as a ``Path``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_gtfs_tables(gtfs_path: Path) -> Dict[str, pd.DataFrame]:
    """Load the five core GTFS tables into DataFrames.

    Parameters
    ----------
    gtfs_path
        Path to either a directory containing ``*.txt`` files or a ``.zip`` GTFS.

    Returns:
    -------
    dict
        Keys ``stops, routes, trips, stop_times, shapes`` → dataframes.
    """
    filenames: Dict[str, str] = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    if gtfs_path.is_dir():
        return {k: pd.read_csv(gtfs_path / v) for k, v in filenames.items()}

    if gtfs_path.is_file() and gtfs_path.suffix.lower() == ".zip":
        print("Detected GTFS zip – extracting to temporary directory …")
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        return {k: pd.read_csv(root / v) for k, v in filenames.items()}

    raise ValueError("GTFS_PATH must be a folder or a .zip file.")


def _validate_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any required GTFS column is missing."""
    required: Dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon", "stop_name"},
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

    missing_msgs: list[str] = []
    for tbl, needed in required.items():
        present = set(dfs[tbl].columns)
        missing = needed - present
        if missing:
            missing_msgs.append(f"{tbl}.txt → missing {', '.join(sorted(missing))}")

    if missing_msgs:
        joined = "\n".join(" • " + msg for msg in missing_msgs)
        raise ValueError(f"GTFS validation failed – required columns not found:\n{joined}")


def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    include_ids: Sequence[str],
    exclude_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply include/exclude lists and return filtered ``routes`` and ``trips``."""
    routes_ok = routes.loc[~routes["route_id"].isin(exclude_ids)].copy()
    if include_ids:
        routes_ok = routes_ok.loc[routes_ok["route_id"].isin(include_ids)].copy()
    trips_ok = trips.loc[trips["route_id"].isin(routes_ok["route_id"])].copy()
    return routes_ok, trips_ok


def _build_stops_gdf(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of **served** stops with list fields for routes/directions."""
    served = stop_times.loc[stop_times["trip_id"].isin(trips["trip_id"])]
    stops = stops.loc[stops["stop_id"].isin(served["stop_id"])].copy()

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

    print(f"Stops GDF – kept {len(gdf):,} served stops.")
    return gdf


def _build_routes_gdf(
    shapes: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
    union_shapes: bool,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of polylines keyed by ``(route_id, direction_id)``."""
    shape_cols: list[str] = [
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lat",
        "shape_pt_lon",
    ]

    lines = (
        shapes[shape_cols]
        .sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id")
        .apply(lambda g: LineString(zip(g.shape_pt_lon, g.shape_pt_lat)))
        .to_frame("geometry")
        .reset_index()
    )

    gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(crs)

    gdf = gdf.merge(
        trips.drop_duplicates("shape_id")[["shape_id", "route_id", "direction_id"]],
        on="shape_id",
        how="left",
    ).merge(routes, on="route_id", how="left")

    # ---- NEW ---------------------------------------------------------------
    before = len(gdf)
    gdf = gdf[gdf["direction_id"].notna()].copy()
    dropped = before - len(gdf)
    if dropped:
        print(
            f"Routes GDF – {dropped:,} of {before:,} shapes were missing "
            "`direction_id` and were skipped."
        )
    # ------------------------------------------------------------------------

    if union_shapes:
        gdf = gdf.dissolve(
            by=["route_id", "direction_id"],
            as_index=False,
            aggfunc={"route_short_name": "first", "route_long_name": "first"},
        ).explode(ignore_index=True)

    print(f"Routes GDF – built {len(gdf):,} shapes.")
    return gdf


def _split_into_segments(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """Split each route polyline at its own stops and return segment GDF."""
    seg_records: list[dict[str, object]] = []
    sindex = stops_gdf.sindex

    for _, r in routes_gdf.iterrows():
        # -------------------------------------------------------------------
        if pd.isna(r.direction_id):  # extra safety – should not occur
            continue
        # -------------------------------------------------------------------

        line: LineString = r.geometry
        rid: str = str(r.route_id)
        drn: int = int(r.direction_id)

        cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
        cand = cand[
            cand.route_id.apply(lambda ids: rid in ids)
            & cand.direction_id.apply(lambda ids: drn in ids)
        ]
        if cand.empty:
            continue

        dists = np.array([line.project(pt) for pt in cand.geometry if isinstance(pt, Point)])
        uniq_dists = np.unique(dists)
        snap_pts: list[Point] = [line.interpolate(d) for d in uniq_dists]

        pieces = split_line(line, MultiPoint(snap_pts))
        geoms: Iterable[LineString]
        if isinstance(pieces, LineString):
            geoms = [pieces]
        else:
            geoms = (g for g in pieces.geoms if isinstance(g, LineString))

        for seg in geoms:
            if seg.length > 0:
                seg_records.append(
                    {
                        "route_id": rid,
                        "direction_id": drn,
                        "route_short": r.get("route_short_name"),
                        "geometry": seg,
                    }
                )

    seg_gdf = gpd.GeoDataFrame(seg_records, crs=crs)
    seg_gdf["length_ft"] = seg_gdf.length * (1.0 if "2263" in crs else 3.28084)
    print(f"Segments GDF – generated {len(seg_gdf):,} pieces.")
    return seg_gdf


def _export(gdf: gpd.GeoDataFrame, out_dir: Path, name: str) -> None:
    """Write *gdf* to ESRI Shapefile ``<out_dir>/<name>.shp``."""
    path = out_dir / f"{name}.shp"
    gdf.to_file(path)
    print(f"Wrote {path.name}")


def _export_segments_by_route_dir(seg_gdf: gpd.GeoDataFrame, out_dir: Path) -> None:
    """Write one shapefile per ``(route_id, direction_id)``."""
    for (rid, drn), grp in seg_gdf.groupby(["route_id", "direction_id"]):
        suffix = f"dir{drn}"
        fname = f"{rid}_{suffix}.shp"
        grp_gdf: gpd.GeoDataFrame = grp  # type: ignore[assignment]
        grp_gdf.to_file(out_dir / fname)
        print(f"Wrote {fname}")


def _flag_short_spacing(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    threshold_ft: float,
    log_path: Path,
) -> None:
    """Write a log of consecutive stops spaced closer than *threshold_ft* along their route polyline."""
    crs_str = str(stops_gdf.crs) if stops_gdf.crs is not None else ""
    factor_ft: float = 1.0 if "2263" in crs_str else 3.28084
    sindex = stops_gdf.sindex

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "route_id\tdirection_id\tbegin_stop_id\tbegin_stop_name\t"
            "end_stop_id\tend_stop_name\tspacing_ft\n"
        )

        for _, row in routes_gdf.iterrows():
            rid = str(row.route_id)
            drn = int(row.direction_id)
            line: LineString = row.geometry

            cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
            cand = cand[
                cand.route_id.apply(lambda xs: rid in xs)
                & cand.direction_id.apply(lambda xs: drn in xs)
            ].copy()

            if len(cand) < 2:
                continue

            cand["dist_along"] = cand.geometry.apply(line.project)
            cand = cand.drop_duplicates("dist_along").sort_values("dist_along")

            for i in range(len(cand) - 1):
                s0, s1 = cand.iloc[i], cand.iloc[i + 1]
                spacing_ft = (s1.dist_along - s0.dist_along) * factor_ft
                if spacing_ft < threshold_ft:
                    fh.write(
                        f"{rid}\t{drn}\t"
                        f"{s0.stop_id}\t{s0.stop_name}\t"
                        f"{s1.stop_id}\t{s1.stop_name}\t"
                        f"{spacing_ft:.1f}\n"
                    )

    print(f"Wrote short-spacing log → {log_path.name}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:  # noqa: D401
    """Run the entire GTFS shapefile & spacing-log pipeline."""
    print("STEP 0  Reading GTFS tables …")
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

    print("STEP 1  Building stops shapefile …")
    stops_gdf = _build_stops_gdf(
        dfs["stops"], dfs["stop_times"], trips_df, routes_df, PROJECTED_CRS
    )
    _export(stops_gdf, out_dir, "stops")

    print("STEP 2  Building routes shapefile …")
    routes_gdf = _build_routes_gdf(dfs["shapes"], trips_df, routes_df, PROJECTED_CRS, ROUTE_UNION)
    _export(routes_gdf, out_dir, "routes")

    print("STEP 3  Splitting routes into stop-to-stop segments …")
    segs_gdf = _split_into_segments(routes_gdf, stops_gdf, PROJECTED_CRS)
    _export(segs_gdf, out_dir, "segments")  # master file
    _export_segments_by_route_dir(segs_gdf, out_dir)  # per-route files

    print("STEP 4  Flagging closely-spaced stops …")
    _flag_short_spacing(
        routes_gdf,
        stops_gdf,
        MIN_SPACING_FT,
        out_dir / SPACING_LOG_FILE,
    )

    print("\nAll done! Outputs in:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print("\nUNEXPECTED ERROR:", exc)
        sys.exit(1)
