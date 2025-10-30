"""GTFS Stop Conflict Checker.

Uses ArcPy and pandas to identify GTFS stops that directly INTERSECT
(spatially overlap) with roadway, driveway, or building footprint layers.

This script performs a simple overlap check and does not use any buffers
or proximity analysis. It includes an optional pandas-based deduplication
step (by key fields or XY tolerance) before the ArcPy analysis.

Outputs a report (CSV, GDB Feature Class, or Shapefile) containing only
the stops that were flagged for one or more conflicts.

All configuration is set in the 'Configuration' section below.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import arcpy


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Output destinations and names ---
OUTPUT_DIR: str = r"C:\projects\my_stop_analysis\output"
OUTPUT_GDB: str = r".\stop_conflicts.gdb"      # created if missing
OUTPUT_FC_NAME: str = "stops_conflicts"
OUTPUT_BASENAME: str = "stop_conflicts"
OVERWRITE_OUTPUT: bool = True

# --- Input: specify exactly ONE of the following for stops ---
STOPS_TXT_PATH: str = r""  # e.g., r"C:\data\gtfs\stops.txt"
GTFS_DIR: str = r"C:\data\gtfs_feed_2025_10_30"  # must hold stops.txt

# --- Optional context layers (any may be empty). Can be shapefile or FC (any geometry type).
ROADWAYS_PATH: str = r"C:\data\gis_layers\context_data.gdb\transportation\road_centerlines"
DRIVEWAYS_PATH: str = r"C:\data\gis_layers\parcels.gdb\driveways"
BUILDINGS_PATH: str = r"C:\data\gis_layers\buildings\building_footprints.shp"

# --- CRS for analysis (use a local projected CRS, units = meters) ---
ANALYSIS_SR: arcpy.SpatialReference = arcpy.SpatialReference(32618)  # e.g., WGS 84 / UTM zone 18N (m)
WGS84_SR: arcpy.SpatialReference = arcpy.SpatialReference(4326)       # for lon/lat export

# --- Conflict toggles ---
FLAG_ROADWAYS: bool = True
FLAG_DRIVEWAYS: bool = True
FLAG_BUILDINGS: bool = True

# --- Deduping (pandas) ---
DEDUPE_STOPS: bool = True
DEDUPE_KEYS: list[str] = ["stop_id", "stop_code", "stop_name"]  # only existing columns are used
DEDUPE_XY_TOL_M: float = 0.5  # 0 to disable XY tolerance; uses rounded lon/lat grid ~meters

# --- Tabular / vector outputs ---
EXPORT_CSV: bool = True
EXPORT_XLSX: bool = False
EXPORT_FC: bool = True
EXPORT_SHP: bool = False

# Scratch
TMP_WS = arcpy.env.scratchGDB

# =============================================================================
# FUNCTIONS
# =============================================================================

def _validate_config() -> None:
    """Validate configuration choices (paths, outputs, exclusivity)."""
    has_stops_txt = bool(STOPS_TXT_PATH.strip())
    has_gtfs_dir = bool(GTFS_DIR.strip())
    if has_stops_txt == has_gtfs_dir:
        raise ValueError("Specify exactly one of STOPS_TXT_PATH or GTFS_DIR.")

    if not any([EXPORT_CSV, EXPORT_XLSX, EXPORT_FC, EXPORT_SHP]):
        raise ValueError("Enable at least one export format.")


def _resolve_stops_path() -> str:
    """Return a filesystem path to stops.txt based on configuration."""
    if STOPS_TXT_PATH.strip():
        p = Path(STOPS_TXT_PATH)
        if not p.exists():
            raise FileNotFoundError(f"stops.txt not found at: {p}")
        return str(p)
    p = Path(GTFS_DIR) / "stops.txt"
    if not p.exists():
        raise FileNotFoundError(f"stops.txt not found in GTFS_DIR: {GTFS_DIR}")
    return str(p)


def _deg_tolerance_for_meters(lat_deg: float, tol_m: float) -> tuple[float, float]:
    """Convert a meter tolerance to approximate degree tolerances at a given latitude.

    Args:
        lat_deg: Latitude in degrees (use dataset mean latitude).
        tol_m: Tolerance in meters.

    Returns:
        (tol_lon_deg, tol_lat_deg)
    """
    if tol_m <= 0:
        return (0.0, 0.0)
    # ~ meters per degree latitude and longitude
    meters_per_deg_lat = 111_132.92  # good enough for a small tolerance grid
    meters_per_deg_lon = 111_412.84 * math.cos(math.radians(lat_deg))
    tol_lat_deg = tol_m / meters_per_deg_lat
    tol_lon_deg = tol_m / meters_per_deg_lon if meters_per_deg_lon > 0 else tol_m / 111_320.0
    return (tol_lon_deg, tol_lat_deg)


def _pandas_dedupe_stops(src_stops: str,
                         keys: Sequence[str],
                         xy_tol_m: float,
                         out_dir: Path) -> str:
    """Read stops.txt with pandas, dedupe, and write a temp CSV used for XYTableToPoint.

    Args:
        src_stops: Path to GTFS stops.txt.
        keys: Candidate keys to dedupe by (only existing are used).
        xy_tol_m: Optional XY tolerance in meters; 0 to disable.
        out_dir: Directory for the deduped temp CSV.

    Returns:
        Path to deduped CSV on disk.
    """
    usecols = None  # read all; GTFS files are small
    df = pd.read_csv(src_stops, dtype=str, usecols=usecols)

    # Ensure numeric lat/lon; keep originals if present
    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        raise ValueError("stops.txt must include 'stop_lat' and 'stop_lon' columns.")
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    # Build dedupe subset
    existing_keys: list[str] = [k for k in keys if k in df.columns]

    if xy_tol_m > 0:
        # Use a small rounding grid based on mean latitude
        mean_lat = float(df["stop_lat"].mean())
        tol_lon_deg, tol_lat_deg = _deg_tolerance_for_meters(mean_lat, xy_tol_m)

        # Quantize onto an approximate meter grid
        # Avoid division-by-zero when tol_*_deg ~ 0
        if tol_lon_deg > 0:
            df["_lon_q"] = (df["stop_lon"] / tol_lon_deg).round().astype("Int64")
        else:
            df["_lon_q"] = (df["stop_lon"]).round(7)  # fallback, very fine grid
        if tol_lat_deg > 0:
            df["_lat_q"] = (df["stop_lat"] / tol_lat_deg).round().astype("Int64")
        else:
            df["_lat_q"] = (df["stop_lat"]).round(7)

        subset: list[str] = existing_keys + ["_lon_q", "_lat_q"]
    else:
        subset = existing_keys if existing_keys else ["stop_lon", "stop_lat"]

    # Deduplicate, keep first occurrence
    df = df.drop_duplicates(subset=subset, keep="first").copy()

    # Write to temp CSV for XYTableToPoint
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / "stops_dedup_tmp.csv"
    df.to_csv(tmp_csv, index=False)
    return str(tmp_csv)


def _xy_to_points(stops_csv: str, out_sr: arcpy.SpatialReference) -> str:
    """Create a projected point FC from a CSV with stop_lon / stop_lat."""
    xy_fc = arcpy.management.XYTableToPoint(
        in_table=stops_csv,
        out_feature_class=arcpy.CreateUniqueName("stops_xy", "in_memory"),
        x_field="stop_lon",
        y_field="stop_lat",
        coordinate_system=WGS84_SR,
    )[0]
    projected_fc = arcpy.management.Project(
        in_dataset=xy_fc,
        out_dataset=arcpy.CreateUniqueName("stops_prj", TMP_WS),
        out_coor_system=out_sr,
    )[0]
    return projected_fc


def _ensure_projected(in_fc: str, out_sr: arcpy.SpatialReference) -> str:
    """Project a feature class to the analysis CRS if needed, returns path to projected FC."""
    if not in_fc:
        return ""
    desc = arcpy.Describe(in_fc)
    in_sr = desc.spatialReference
    if in_sr and int(in_sr.factoryCode or 0) == int(out_sr.factoryCode or -1):
        return in_fc
    return arcpy.management.Project(
        in_dataset=in_fc,
        out_dataset=arcpy.CreateUniqueName("prj_", TMP_WS),
        out_coor_system=out_sr,
    )[0]


def _add_flag_field(fc: str, name: str) -> None:
    """Add a SHORT field initialized to 0 if missing."""
    if name not in [f.name for f in arcpy.ListFields(fc)]:
        arcpy.management.AddField(fc, name, "SHORT")
    with arcpy.da.UpdateCursor(fc, [name]) as cur:
        for _ in cur:
            cur.updateRow([0])


def _flag_intersections(target_fc: str,
                        context_fc: Optional[str],
                        flag_field: str) -> None:
    """Set flag_field = 1 for target points that INTERSECT context_fc (any geometry type)."""
    _add_flag_field(target_fc, flag_field)
    if not context_fc:
        return
    lyr_pts = arcpy.management.MakeFeatureLayer(target_fc, arcpy.CreateUniqueName("pts_", "in_memory"))[0]
    arcpy.management.SelectLayerByLocation(
        in_layer=lyr_pts,
        overlap_type="INTERSECT",
        select_features=context_fc,
        selection_type="NEW_SELECTION",
    )
    with arcpy.da.UpdateCursor(lyr_pts, ["OID@", flag_field]) as cur:
        for oid, _ in cur:
            cur.updateRow([oid, 1])


def _concat_conflict_labels(fc: str, flags: Sequence[str], out_field: str = "conflict_types") -> None:
    """Write comma-separated list of true flags to out_field."""
    if out_field not in [f.name for f in arcpy.ListFields(fc)]:
        arcpy.management.AddField(fc, out_field, "TEXT", field_length=255)
    with arcpy.da.UpdateCursor(fc, list(flags) + [out_field]) as cur:
        for *vals, _existing in cur:
            labels = [name for name, v in zip(flags, vals) if int(v or 0) == 1]
            cur.updateRow(vals + [",".join(labels)])


def _add_has_conflict(fc: str, flags: Sequence[str], name: str = "has_conflict") -> None:
    """Add has_conflict = 1 if any flag is 1."""
    if name not in [f.name for f in arcpy.ListFields(fc)]:
        arcpy.management.AddField(fc, name, "SHORT")
    with arcpy.da.UpdateCursor(fc, list(flags) + [name]) as cur:
        for *vals, _ in cur:
            cur.updateRow(vals + [1 if any(int(v or 0) == 1 for v in vals) else 0])


def _add_lon_lat(fc_analysis: str, out_lon: str = "lon", out_lat: str = "lat") -> None:
    """Add WGS84 lon/lat to the analysis FC (for CSV convenience)."""
    wgs_fc = arcpy.management.Project(
        in_dataset=fc_analysis,
        out_dataset=arcpy.CreateUniqueName("wgs_", TMP_WS),
        out_coor_system=WGS84_SR,
    )[0]
    arcpy.management.AddXY(wgs_fc)  # adds POINT_X / POINT_Y
    if out_lon not in [f.name for f in arcpy.ListFields(fc_analysis)]:
        arcpy.management.AddField(fc_analysis, out_lon, "DOUBLE")
    if out_lat not in [f.name for f in arcpy.ListFields(fc_analysis)]:
        arcpy.management.AddField(fc_analysis, out_lat, "DOUBLE")
    oid_main = arcpy.Describe(fc_analysis).OIDFieldName
    oid_wgs = arcpy.Describe(wgs_fc).OIDFieldName
    lut: dict[int, tuple[float | None, float | None]] = {}
    with arcpy.da.SearchCursor(wgs_fc, [oid_wgs, "POINT_X", "POINT_Y"]) as cur:
        for oid, x, y in cur:
            lut[int(oid)] = (float(x), float(y))
    with arcpy.da.UpdateCursor(fc_analysis, [oid_main, out_lon, out_lat]) as cur:
        for oid, _x, _y in cur:
            x, y = lut.get(int(oid), (None, None))
            cur.updateRow([oid, x, y])


def _export_conflicts(fc_conflicts: str,
                      csv_path: Optional[str],
                      xlsx_path: Optional[str],
                      shp_path: Optional[str],
                      gdb_path: Optional[str],
                      fc_name: Optional[str]) -> None:
    """Export conflicts to requested formats."""
    # Vector
    if gdb_path and fc_name:
        if not arcpy.Exists(gdb_path):
            folder, gdb = os.path.split(gdb_path)
            folder = folder or "."
            arcpy.management.CreateFileGDB(folder, gdb.replace(".gdb", ""))
        arcpy.management.CopyFeatures(fc_conflicts, os.path.join(gdb_path, fc_name))
    if shp_path:
        arcpy.management.CopyFeatures(fc_conflicts, shp_path)

    # Tabular
    field_names = [f.name for f in arcpy.ListFields(fc_conflicts) if f.type not in ("Geometry", "Raster")]
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(field_names)
            with arcpy.da.SearchCursor(fc_conflicts, field_names) as cur:
                for row in cur:
                    writer.writerow(row)
    if xlsx_path:
        arcpy.conversion.TableToExcel(
            Input_Table=fc_conflicts,
            Output_Excel_File=xlsx_path,
            Use_field_alias_as_column_header="NAME",
            Use_domain_and_subtype_description="CODE",
        )

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run minimal stop conflict checker (overlap-only; pandas dedupe retained)."""
    arcpy.env.overwriteOutput = OVERWRITE_OUTPUT
    _validate_config()

    # Resolve stops.txt and dedupe with pandas
    src_stops = _resolve_stops_path()
    out_dir = Path(OUTPUT_DIR)
    if DEDUPE_STOPS:
        dedup_csv = _pandas_dedupe_stops(
            src_stops=src_stops,
            keys=DEDUPE_KEYS,
            xy_tol_m=DEDUPE_XY_TOL_M,
            out_dir=out_dir,
        )
    else:
        dedup_csv = src_stops  # no dedupe

    # Points in analysis CRS (meters)
    stops_fc = _xy_to_points(dedup_csv, ANALYSIS_SR)

    # Prepare optional layers (project only if provided)
    road_fc = _ensure_projected(ROADWAYS_PATH, ANALYSIS_SR) if ROADWAYS_PATH.strip() else None
    drv_fc = _ensure_projected(DRIVEWAYS_PATH, ANALYSIS_SR) if DRIVEWAYS_PATH.strip() else None
    bld_fc = _ensure_projected(BUILDINGS_PATH, ANALYSIS_SR) if BUILDINGS_PATH.strip() else None

    # Working copy for flags
    work_fc = arcpy.management.CopyFeatures(stops_fc, arcpy.CreateUniqueName("stops_work", "in_memory"))[0]

    # Flag overlaps (INTERSECT) — any geometry type is fine
    if FLAG_ROADWAYS:
        _flag_intersections(work_fc, road_fc, "in_roadway")
    else:
        _add_flag_field(work_fc, "in_roadway")

    if FLAG_DRIVEWAYS:
        _flag_intersections(work_fc, drv_fc, "in_driveway")
    else:
        _add_flag_field(work_fc, "in_driveway")

    if FLAG_BUILDINGS:
        _flag_intersections(work_fc, bld_fc, "in_building")
    else:
        _add_flag_field(work_fc, "in_building")

    # Summaries
    flags: list[str] = ["in_roadway", "in_driveway", "in_building"]
    _concat_conflict_labels(work_fc, flags, out_field="conflict_types")
    _add_has_conflict(work_fc, flags, name="has_conflict")

    # Lon/Lat for convenience in CSV
    _add_lon_lat(work_fc, out_lon="lon", out_lat="lat")

    # Filter to conflicts
    conf_lyr = arcpy.management.MakeFeatureLayer(work_fc, arcpy.CreateUniqueName("conf_lyr", "in_memory"))[0]
    arcpy.management.SelectLayerByAttribute(conf_lyr, "NEW_SELECTION", "has_conflict = 1")
    conflicts_fc = arcpy.management.CopyFeatures(conf_lyr, arcpy.CreateUniqueName("conflicts_fc", "in_memory"))[0]

    # Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = str(out_dir / f"{OUTPUT_BASENAME}.shp") if EXPORT_SHP else None
    csv_path = str(out_dir / f"{OUTPUT_BASENAME}.csv") if EXPORT_CSV else None
    xlsx_path = str(out_dir / f"{OUTPUT_BASENAME}.xlsx") if EXPORT_XLSX else None
    gdb_path = OUTPUT_GDB if EXPORT_FC else None
    fc_name = OUTPUT_FC_NAME if EXPORT_FC else None

    _export_conflicts(
        fc_conflicts=conflicts_fc,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        shp_path=shp_path,
        gdb_path=gdb_path,
        fc_name=fc_name,
    )

    # Console summary
    total = int(arcpy.management.GetCount(work_fc)[0])
    n_conf = int(arcpy.management.GetCount(conflicts_fc)[0])
    pct = (n_conf / total * 100.0) if total else 0.0
    print(f"Stops checked (post-pandas-dedupe): {total}")
    print(f"Stops with conflicts: {n_conf} ({pct:.1f}%)")
    print(f"Vector export: {os.path.abspath(gdb_path) if gdb_path else '(GDB disabled)'} "
          f"{'and ' + shp_path if shp_path else ''}")
    print(f"Tabular export: {csv_path if csv_path else '(CSV disabled)'} "
          f"{'and ' + xlsx_path if xlsx_path else ''}")


if __name__ == "__main__":
    main()
