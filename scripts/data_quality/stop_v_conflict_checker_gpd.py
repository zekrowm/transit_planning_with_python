"""GTFS Stop Conflict Checker (GeoPandas port).

Uses GeoPandas and pandas to identify GTFS stops that directly INTERSECT
(spatially overlap) with roadway, driveway, or building footprint layers.

This script performs a simple overlap check and does not use any buffers
or proximity analysis. It includes an optional pandas-based deduplication
step (by key fields or XY tolerance) before the spatial analysis.

Outputs a report (CSV, GeoPackage, Shapefile, and/or XLSX) containing only
the stops that were flagged for one or more conflicts.

All configuration is set in the 'Configuration' section below.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Output destinations and names ---
OUTPUT_DIR: str = r"projects\my_stop_analysis\output"
OUTPUT_GPKG: str = "./stop_conflicts.gpkg"  # created if missing
OUTPUT_LAYER_NAME: str = "stops_conflicts"
OUTPUT_BASENAME: str = "stop_conflicts"
OVERWRITE_OUTPUT: bool = True

# --- Input: specify exactly ONE of the following for stops ---
STOPS_TXT_PATH: str = r""  # e.g., r"C:\data\gtfs\stops.txt"
GTFS_DIR: str = r"data\gtfs_feed_2025_10_30"  # must hold stops.txt

# --- Optional context layers (any may be empty). Any vector format readable by GeoPandas/Fiona.
ROADWAYS_PATH: str = r"data\gis_layers\roads\road_centerlines.shp"
DRIVEWAYS_PATH: str = r"data\gis_layers\parcels\driveways.shp"
BUILDINGS_PATH: str = r"data\gis_layers\buildings\building_footprints.shp"

# --- CRS for analysis (use a local projected CRS, units = meters) ---
ANALYSIS_CRS: str = "EPSG:32618"  # e.g., WGS 84 / UTM zone 18N (m)
WGS84_CRS: str = "EPSG:4326"  # for lon/lat export

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
EXPORT_GPKG: bool = True
EXPORT_SHP: bool = False

LOG_LEVEL: int = logging.INFO  # DEBUG / INFO / WARNING / ERROR

# =============================================================================
# FUNCTIONS
# =============================================================================


def _validate_config() -> None:
    """Validate configuration choices (paths, outputs, exclusivity)."""
    has_stops_txt = bool(STOPS_TXT_PATH.strip())
    has_gtfs_dir = bool(GTFS_DIR.strip())
    if has_stops_txt == has_gtfs_dir:
        raise ValueError("Specify exactly one of STOPS_TXT_PATH or GTFS_DIR.")

    if not any([EXPORT_CSV, EXPORT_XLSX, EXPORT_GPKG, EXPORT_SHP]):
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
    meters_per_deg_lat = 111_132.92
    meters_per_deg_lon = 111_412.84 * math.cos(math.radians(lat_deg))
    tol_lat_deg = tol_m / meters_per_deg_lat
    tol_lon_deg = tol_m / meters_per_deg_lon if meters_per_deg_lon > 0 else tol_m / 111_320.0
    return (tol_lon_deg, tol_lat_deg)


def _pandas_dedupe_stops(
    src_stops: str, keys: Sequence[str], xy_tol_m: float
) -> pd.DataFrame:
    """Read stops.txt with pandas and return a deduplicated DataFrame.

    Args:
        src_stops: Path to GTFS stops.txt.
        keys: Candidate keys to dedupe by (only existing are used).
        xy_tol_m: Optional XY tolerance in meters; 0 to disable.

    Returns:
        Deduplicated stops DataFrame with numeric stop_lat / stop_lon columns.
    """
    df = pd.read_csv(src_stops, dtype=str)

    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        raise ValueError("stops.txt must include 'stop_lat' and 'stop_lon' columns.")
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    existing_keys: list[str] = [k for k in keys if k in df.columns]

    if xy_tol_m > 0:
        mean_lat = float(df["stop_lat"].mean())
        tol_lon_deg, tol_lat_deg = _deg_tolerance_for_meters(mean_lat, xy_tol_m)

        if tol_lon_deg > 0:
            df["_lon_q"] = (df["stop_lon"] / tol_lon_deg).round().astype("Int64")
        else:
            df["_lon_q"] = (df["stop_lon"]).round(7)
        if tol_lat_deg > 0:
            df["_lat_q"] = (df["stop_lat"] / tol_lat_deg).round().astype("Int64")
        else:
            df["_lat_q"] = (df["stop_lat"]).round(7)

        subset: list[str] = existing_keys + ["_lon_q", "_lat_q"]
    else:
        subset = existing_keys if existing_keys else ["stop_lon", "stop_lat"]

    df = df.drop_duplicates(subset=subset, keep="first").copy()
    df = df.drop(columns=[c for c in ("_lon_q", "_lat_q") if c in df.columns])
    return df


def _stops_to_gdf(stops_df: pd.DataFrame, analysis_crs: str) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from stops with stop_lon / stop_lat and project it."""
    gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=WGS84_CRS,
    )
    return gdf.to_crs(analysis_crs)


def _load_context(path: str, analysis_crs: str) -> Optional[gpd.GeoDataFrame]:
    """Load a context layer if a path is provided, projected to the analysis CRS."""
    if not path or not path.strip():
        return None
    if not Path(path).exists():
        logging.warning("Context layer not found, skipping: %s", path)
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        logging.warning(
            "Context layer has no CRS; assuming it matches the analysis CRS: %s", path
        )
        gdf = gdf.set_crs(analysis_crs)
    elif str(gdf.crs) != str(analysis_crs):
        gdf = gdf.to_crs(analysis_crs)
    return gdf


def _flag_intersections(
    stops_gdf: gpd.GeoDataFrame,
    context_gdf: Optional[gpd.GeoDataFrame],
    flag_field: str,
) -> gpd.GeoDataFrame:
    """Set ``flag_field`` = 1 for stops that INTERSECT any feature in ``context_gdf``."""
    stops_gdf[flag_field] = 0
    if context_gdf is None or context_gdf.empty:
        return stops_gdf

    joined = gpd.sjoin(
        stops_gdf[[stops_gdf.geometry.name]],
        context_gdf[[context_gdf.geometry.name]],
        how="inner",
        predicate="intersects",
    )
    hit_idx = joined.index.unique()
    stops_gdf.loc[hit_idx, flag_field] = 1
    return stops_gdf


def _add_conflict_summary(
    stops_gdf: gpd.GeoDataFrame, flags: Sequence[str]
) -> gpd.GeoDataFrame:
    """Add ``conflict_types`` (comma-joined) and ``has_conflict`` columns."""
    flag_df = stops_gdf[list(flags)].fillna(0).astype(int)
    stops_gdf["conflict_types"] = flag_df.apply(
        lambda row: ",".join(name for name, v in row.items() if v == 1),
        axis=1,
    )
    stops_gdf["has_conflict"] = (flag_df.sum(axis=1) > 0).astype(int)
    return stops_gdf


def _add_lon_lat(stops_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add WGS84 lon/lat columns for CSV convenience."""
    wgs = stops_gdf.geometry.to_crs(WGS84_CRS)
    stops_gdf["lon"] = wgs.x.values
    stops_gdf["lat"] = wgs.y.values
    return stops_gdf


def _export_conflicts(
    conflicts_gdf: gpd.GeoDataFrame,
    csv_path: Optional[str],
    xlsx_path: Optional[str],
    shp_path: Optional[str],
    gpkg_path: Optional[str],
    layer_name: Optional[str],
    overwrite: bool,
) -> None:
    """Export conflicts to requested formats."""
    geom_col = conflicts_gdf.geometry.name
    attr_df = pd.DataFrame(conflicts_gdf.drop(columns=[geom_col]))

    if gpkg_path and layer_name:
        gpkg_p = Path(gpkg_path)
        gpkg_p.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and gpkg_p.exists():
            gpkg_p.unlink()
        conflicts_gdf.to_file(gpkg_p, layer=layer_name, driver="GPKG")

    if shp_path:
        shp_p = Path(shp_path)
        shp_p.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                sib = shp_p.with_suffix(ext)
                if sib.exists():
                    sib.unlink()
        conflicts_gdf.to_file(shp_p, driver="ESRI Shapefile")

    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        attr_df.to_csv(csv_path, index=False)

    if xlsx_path:
        Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
        attr_df.to_excel(xlsx_path, index=False)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run minimal stop conflict checker (overlap-only; pandas dedupe retained)."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if (
        OUTPUT_DIR == r"projects\my_stop_analysis\output"
        or GTFS_DIR == r"data\gtfs_feed_2025_10_30"
    ):
        logging.warning(
            "OUTPUT_DIR and/or GTFS_DIR are still set to placeholder values. "
            "Please update them in the CONFIGURATION section before running."
        )
        return
    _validate_config()

    src_stops = _resolve_stops_path()
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if DEDUPE_STOPS:
        stops_df = _pandas_dedupe_stops(
            src_stops=src_stops, keys=DEDUPE_KEYS, xy_tol_m=DEDUPE_XY_TOL_M
        )
    else:
        stops_df = pd.read_csv(src_stops, dtype=str)
        stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
        stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
        stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    stops_gdf = _stops_to_gdf(stops_df, ANALYSIS_CRS)

    road_gdf = _load_context(ROADWAYS_PATH, ANALYSIS_CRS) if ROADWAYS_PATH.strip() else None
    drv_gdf = _load_context(DRIVEWAYS_PATH, ANALYSIS_CRS) if DRIVEWAYS_PATH.strip() else None
    bld_gdf = _load_context(BUILDINGS_PATH, ANALYSIS_CRS) if BUILDINGS_PATH.strip() else None

    work_gdf = stops_gdf.copy()

    work_gdf = _flag_intersections(
        work_gdf, road_gdf if FLAG_ROADWAYS else None, "in_roadway"
    )
    work_gdf = _flag_intersections(
        work_gdf, drv_gdf if FLAG_DRIVEWAYS else None, "in_driveway"
    )
    work_gdf = _flag_intersections(
        work_gdf, bld_gdf if FLAG_BUILDINGS else None, "in_building"
    )

    flags: list[str] = ["in_roadway", "in_driveway", "in_building"]
    work_gdf = _add_conflict_summary(work_gdf, flags)
    work_gdf = _add_lon_lat(work_gdf)

    conflicts_gdf = work_gdf[work_gdf["has_conflict"] == 1].copy()

    shp_path = str(out_dir / f"{OUTPUT_BASENAME}.shp") if EXPORT_SHP else None
    csv_path = str(out_dir / f"{OUTPUT_BASENAME}.csv") if EXPORT_CSV else None
    xlsx_path = str(out_dir / f"{OUTPUT_BASENAME}.xlsx") if EXPORT_XLSX else None
    gpkg_path = OUTPUT_GPKG if EXPORT_GPKG else None
    layer_name = OUTPUT_LAYER_NAME if EXPORT_GPKG else None

    _export_conflicts(
        conflicts_gdf=conflicts_gdf,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        shp_path=shp_path,
        gpkg_path=gpkg_path,
        layer_name=layer_name,
        overwrite=OVERWRITE_OUTPUT,
    )

    total = len(work_gdf)
    n_conf = len(conflicts_gdf)
    pct = (n_conf / total * 100.0) if total else 0.0
    logging.info("Stops checked (post-pandas-dedupe): %d", total)
    logging.info("Stops with conflicts: %d (%.1f%%)", n_conf, pct)
    logging.info(
        "Vector export: %s %s",
        os.path.abspath(gpkg_path) if gpkg_path else "(GPKG disabled)",
        "and " + shp_path if shp_path else "",
    )
    logging.info(
        "Tabular export: %s %s",
        csv_path if csv_path else "(CSV disabled)",
        "and " + xlsx_path if xlsx_path else "",
    )
    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
