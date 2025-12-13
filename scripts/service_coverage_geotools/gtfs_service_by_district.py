"""Generates a matrix showing which public transit routes operate near which districts.

This script processes General Transit Feed Specification (GTFS) data and performs a
spatial analysis using ArcPy to determine the relationship between transit stops
and specified districts.

It follows these steps:
1. Loads required GTFS files (routes, trips, stops, stop_times).
2. Filters GTFS stops to include only active boarding locations.
3. Converts the filtered stops to a point feature class in WGS84 (EPSG:4326).
4. Projects both the stops and the districts feature class (shapefile) to a
   local projected coordinate system (TARGET_EPSG).
5. Performs a spatial join (WITHIN_A_DISTANCE, defined by SEARCH_DISTANCE_FEET)
   to link GTFS stops to districts.
6. Maps routes to districts if any stop on the route is spatially joined to that district.
7. Outputs the final route-district matrix as an Excel file.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from typing import Any, Mapping

import pandas as pd
import arcpy


# =============================================================================
# CONFIGURATION
# =============================================================================

DISTRICTS_FC = r"Path\To\Your\Districts.shp"
GTFS_DIR = r"Path\To\Your\GTFS_data"
GTFS_FILES = ["routes.txt", "stops.txt", "trips.txt", "stop_times.txt"]

TARGET_EPSG = 2248
SEARCH_DISTANCE_FEET = 1320.0
DISTRICT_FIELD = "DISTRICT"

# FAST: local SSD for intermediates
WORK_DIR = r"C:\temp\gtfs_district_matrix_work"
WORK_GDB_NAME = "work.gdb"

OUTPUT_EXCEL = r"Path\To\Your\Excel_File.xlsx"
LOG_DIR = r"Path\To\Your\Logs"

# =============================================================================
# LOGGING
# =============================================================================

def configure_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gtfs_route_district_matrix.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.info("Logging to: %s", log_path)


def log_arcpy_messages(level: int = logging.INFO) -> None:
    msg = arcpy.GetMessages()
    if msg:
        logging.log(level, "ArcPy messages:\n%s", msg)


# =============================================================================
# UTILITIES
# =============================================================================

def safe_name(prefix: str, workspace: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    return arcpy.ValidateTableName(f"{prefix}_{suffix}", workspace)


def ensure_work_gdb(work_dir: str, gdb_name: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    gdb = os.path.join(work_dir, gdb_name)
    if not arcpy.Exists(gdb):
        logging.info("Creating work GDB: %s", gdb)
        arcpy.management.CreateFileGDB(work_dir, gdb_name)
        log_arcpy_messages()
    return gdb


# =============================================================================
# GTFS LOADING + FILTERING
# =============================================================================

def load_gtfs_data(
    gtfs_folder: str,
    files: list[str],
    dtype: str | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for f in files:
        path = os.path.join(gtfs_folder, f)
        if not os.path.exists(path):
            raise OSError(f"Missing GTFS file: {f}")
        df = pd.read_csv(path, dtype=dtype, low_memory=False)
        if df.empty:
            raise ValueError(f"GTFS file is empty: {f}")
        data[f.replace(".txt", "")] = df
        logging.info("Loaded %s (%s rows)", f, len(df))
    return data


def filter_stops(gtfs_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Keep only stops that actually appear in stop_times and are boarding locations."""
    stops = gtfs_data["stops"].copy()
    stop_times = gtfs_data["stop_times"]

    used_ids = set(stop_times["stop_id"].astype(str).unique())
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops = stops[stops["stop_id"].isin(used_ids)]

    if "location_type" in stops.columns:
        stops = stops[stops["location_type"].fillna("0").astype(str) == "0"]

    stops = stops[["stop_id", "stop_lon", "stop_lat"]].dropna()
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops["stop_lat"] = stops["stop_lat"].astype(float)

    logging.info("Filtered to %s active boarding stops", len(stops))
    return stops


# =============================================================================
# SPATIAL WORK
# =============================================================================

def csv_to_points(
    csv_path: str,
    out_gdb: str,
) -> str:
    out_fc = os.path.join(out_gdb, safe_name("stops_wgs84", out_gdb))
    logging.info("XYTableToPoint from CSV")
    arcpy.management.XYTableToPoint(
        in_table=csv_path,
        out_feature_class=out_fc,
        x_field="stop_lon",
        y_field="stop_lat",
        coordinate_system=arcpy.SpatialReference(4326),
    )
    log_arcpy_messages()
    return out_fc


def project_fc(in_fc: str, out_gdb: str, epsg: int, prefix: str) -> str:
    out_fc = os.path.join(out_gdb, safe_name(prefix, out_gdb))
    logging.info("Projecting %s → EPSG:%s", os.path.basename(in_fc), epsg)
    arcpy.management.Project(in_fc, out_fc, arcpy.SpatialReference(epsg))
    log_arcpy_messages()
    return out_fc


def spatial_join_stops_to_districts(
    stops_fc: str,
    districts_fc: str,
    district_field: str,
    search_dist_ft: float,
    out_gdb: str,
) -> str:
    out_fc = os.path.join(out_gdb, safe_name("stops_districts", out_gdb))
    logging.info("SpatialJoin WITHIN_A_DISTANCE (%s ft)", search_dist_ft)

    arcpy.analysis.SpatialJoin(
        target_features=stops_fc,
        join_features=districts_fc,
        out_feature_class=out_fc,
        join_operation="JOIN_ONE_TO_MANY",
        join_type="KEEP_COMMON",
        match_option="WITHIN_A_DISTANCE",
        search_radius=f"{search_dist_ft} Feet",
    )
    log_arcpy_messages()
    return out_fc


def extract_stop_districts(fc: str, district_field: str) -> dict[str, set[str]]:
    stop_to_districts: dict[str, set[str]] = {}
    with arcpy.da.SearchCursor(fc, ["stop_id", district_field]) as cur:
        for stop_id, dist in cur:
            if stop_id and dist:
                stop_to_districts.setdefault(str(stop_id), set()).add(str(dist))
    return stop_to_districts


# =============================================================================
# MATRIX BUILD
# =============================================================================

def build_route_district_matrix(
    gtfs_data: dict[str, pd.DataFrame],
    stop_to_districts: dict[str, set[str]],
) -> pd.DataFrame:
    routes = gtfs_data["routes"]
    trips = gtfs_data["trips"]
    stop_times = gtfs_data["stop_times"]

    route_name = dict(
        zip(routes["route_id"].astype(str), routes["route_short_name"].astype(str))
    )
    trip_route = dict(
        zip(trips["trip_id"].astype(str), trips["route_id"].astype(str))
    )

    stop_routes: dict[str, set[str]] = {}
    for t_id, s_id in stop_times[["trip_id", "stop_id"]].itertuples(index=False):
        rid = trip_route.get(str(t_id))
        if rid:
            stop_routes.setdefault(str(s_id), set()).add(rid)

    route_districts: dict[str, set[str]] = {}
    for stop_id, districts in stop_to_districts.items():
        for rid in stop_routes.get(stop_id, []):
            route_districts.setdefault(rid, set()).update(districts)

    all_routes = sorted(route_districts, key=lambda r: route_name.get(r, "zzz"))
    all_districts = sorted({d for ds in route_districts.values() for d in ds})

    rows = []
    for rid in all_routes:
        row = {"route_short_name": route_name.get(rid, rid)}
        for d in all_districts:
            row[d] = "y" if d in route_districts[rid] else "n"
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    configure_logging(LOG_DIR)
    arcpy.env.overwriteOutput = True

    work_gdb = ensure_work_gdb(WORK_DIR, WORK_GDB_NAME)
    arcpy.env.workspace = work_gdb
    logging.info("Workspace: %s", work_gdb)

    gtfs = load_gtfs_data(GTFS_DIR, GTFS_FILES)
    stops_df = filter_stops(gtfs)

    csv_path = os.path.join(WORK_DIR, "filtered_stops.csv")
    stops_df.to_csv(csv_path, index=False)

    stops_wgs84 = csv_to_points(csv_path, work_gdb)
    stops_proj = project_fc(stops_wgs84, work_gdb, TARGET_EPSG, "stops_proj")
    districts_proj = project_fc(DISTRICTS_FC, work_gdb, TARGET_EPSG, "districts_proj")

    sj_fc = spatial_join_stops_to_districts(
        stops_proj,
        districts_proj,
        DISTRICT_FIELD,
        SEARCH_DISTANCE_FEET,
        work_gdb,
    )

    stop_to_districts = extract_stop_districts(sj_fc, DISTRICT_FIELD)
    df = build_route_district_matrix(gtfs, stop_to_districts)

    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, index=False)
    logging.info("Done. Excel written to: %s", OUTPUT_EXCEL)


if __name__ == "__main__":
    main()
