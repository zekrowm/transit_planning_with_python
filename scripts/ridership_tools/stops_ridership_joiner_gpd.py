"""Join ridership data to bus stop point features (GeoPandas port; no dataclass).

This script merges stop-level ridership data from an Excel file with stop locations
(from a shapefile/GeoPackage/GeoJSON/etc. or GTFS stops.txt), and optionally performs
a spatial join to polygons (e.g., Census Blocks) for geographic aggregation.

Outputs:
- Stops with ridership attributes (one file, or split by route)
- CSV summaries (per-stop and optional per-polygon aggregation)
- Optional polygon layer with aggregated ridership
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# INPUTS --------------------------------------------------------------------
BUS_STOPS_INPUT = Path(r"Your\File\Path\To\stops.txt")  # GTFS stops.txt OR vector file
EXCEL_FILE = Path(r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX")

ROUTE_FILTER_LIST: list[str] = []
SPLIT_BY_ROUTE = False

OUTPUT_FOLDER = Path(r"Your\Folder\Path\To\Output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Optional polygons (set to None to disable)
POLYGON_LAYER: Optional[Path] = Path(r"Your\File\Path\To\census_blocks.shp")

# OUTPUT FORMAT: "gpkg" strongly recommended; "shp" supported
OUT_FORMAT = "gpkg"  # "gpkg" | "shp"

# FIELDS & JOIN KEYS --------------------------------------------------------
GTFS_KEY_FIELD = "stop_code"
SHAPE_KEY_FIELD = "StopId"

GTFS_SECONDARY_ID_FIELD = "stop_id"
SHAPE_SECONDARY_ID_FIELD = "StopNum"

POLYGON_JOIN_FIELD = "GEOID"
POLYGON_FIELDS_TO_KEEP = ["NAME", "GEOID", "GEOIDFQ"]

GTFS_LON_FIELD = "stop_lon"
GTFS_LAT_FIELD = "stop_lat"

# Excel fields expected
EXCEL_STOP_ID_FIELD = "STOP_ID"
EXCEL_ROUTE_FIELD = "ROUTE_NAME"
EXCEL_BOARD_FIELD = "XBOARDINGS"
EXCEL_ALIGHT_FIELD = "XALIGHTINGS"

# Output ridership fields (short for shapefile compatibility)
OUT_BOARD = "XBOARD"
OUT_ALIGHT = "XALIGHT"
OUT_TOTAL = "XTOTAL"


# =============================================================================
# LOGGING
# =============================================================================


def configure_logging() -> logging.Logger:
    """Configure module logging."""
    logger = logging.getLogger("ridership_join")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Avoid duplicate handlers in notebooks
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(handler)

    return logger


LOGGER = configure_logging()


# =============================================================================
# HELPERS
# =============================================================================


def is_gtfs_txt(path: Path) -> bool:
    """Return True if input should be treated as GTFS stops.txt."""
    return path.suffix.lower() == ".txt"


def _safe_to_str(series: pd.Series) -> pd.Series:
    """Convert values to string, preserving NaNs."""
    return series.astype("string").astype(object)


def _require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def _to_common_crs(
    points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject points/polygons to a common CRS (prefers polygons CRS)."""
    if polygons.crs is None and points.crs is None:
        raise ValueError("Both points and polygons are missing CRS; cannot spatial-join safely.")
    if polygons.crs is None:
        raise ValueError("Polygon layer has no CRS; define it before running.")
    if points.crs is None:
        raise ValueError("Stop layer has no CRS; define it before running.")

    if points.crs != polygons.crs:
        points = points.to_crs(polygons.crs)

    return points, polygons


def output_path(base: str, route: Optional[str] = None) -> Path:
    """Build an output file path for the chosen output format."""
    suffix = ".gpkg" if OUT_FORMAT.lower() == "gpkg" else ".shp"
    name = f"{base}_{route}{suffix}" if route else f"{base}{suffix}"
    return OUTPUT_FOLDER / name


def write_vector(gdf: gpd.GeoDataFrame, path: Path, layer: Optional[str] = None) -> None:
    """Write a GeoDataFrame to disk as GPKG or SHP."""
    if path.suffix.lower() == ".gpkg":
        gdf.to_file(path, layer=layer or "data", driver="GPKG")
    elif path.suffix.lower() == ".shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


# =============================================================================
# CORE STEPS
# =============================================================================


def load_bus_stops(logger: logging.Logger) -> tuple[gpd.GeoDataFrame, str]:
    """Load bus stop points as a GeoDataFrame and return (gdf, key_field)."""
    if is_gtfs_txt(BUS_STOPS_INPUT):
        df = pd.read_csv(BUS_STOPS_INPUT)
        _require_columns(
            df,
            [GTFS_KEY_FIELD, GTFS_SECONDARY_ID_FIELD, "stop_name", GTFS_LON_FIELD, GTFS_LAT_FIELD],
            context=f"GTFS stops file {BUS_STOPS_INPUT}",
        )

        gdf = gpd.GeoDataFrame(
            data=df,
            geometry=gpd.points_from_xy(df[GTFS_LON_FIELD], df[GTFS_LAT_FIELD]),
            crs="EPSG:4326",
        )
        logger.info("Loaded GTFS stops.txt with %d records.", len(gdf))
        return gdf, GTFS_KEY_FIELD

    gdf = gpd.read_file(BUS_STOPS_INPUT)
    _require_columns(
        gdf,
        [SHAPE_KEY_FIELD, SHAPE_SECONDARY_ID_FIELD],
        context=f"stop layer {BUS_STOPS_INPUT}",
    )
    logger.info("Loaded stop layer with %d features: %s", len(gdf), BUS_STOPS_INPUT)
    return gdf, SHAPE_KEY_FIELD


def spatial_join_to_polygons(
    stops: gpd.GeoDataFrame, logger: logging.Logger
) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """Optionally spatial-join stops to polygons; returns (stops_joined, polygons_or_none)."""
    if not POLYGON_LAYER:
        logger.info("POLYGON_LAYER is None; skipping spatial join.")
        return stops, None

    polygons = gpd.read_file(POLYGON_LAYER)
    _require_columns(polygons, [POLYGON_JOIN_FIELD], context=f"polygon layer {POLYGON_LAYER}")

    keep = list(dict.fromkeys(POLYGON_FIELDS_TO_KEEP + [POLYGON_JOIN_FIELD]))
    keep = [c for c in keep if c in polygons.columns]
    polygons = polygons[keep + ["geometry"]].copy()

    stops, polygons = _to_common_crs(stops, polygons)

    # "within" = point must lie inside polygon. Use "intersects" if you want boundary hits.
    joined = gpd.sjoin(stops, polygons, how="left", predicate="within")
    joined = joined.drop(
        columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore"
    )

    logger.info("Spatial join complete. Stops rows: %d.", len(joined))
    return joined, polygons


def read_and_filter_excel(logger: logging.Logger) -> pd.DataFrame:
    """Read ridership data from Excel and optionally filter by routes; adds TOTAL."""
    df = pd.read_excel(EXCEL_FILE)

    _require_columns(
        df,
        [EXCEL_STOP_ID_FIELD, EXCEL_ROUTE_FIELD, EXCEL_BOARD_FIELD, EXCEL_ALIGHT_FIELD],
        context=f"Excel ridership file {EXCEL_FILE}",
    )

    if ROUTE_FILTER_LIST:
        before = len(df)
        df = df[df[EXCEL_ROUTE_FIELD].isin(ROUTE_FILTER_LIST)].copy()
        logger.info("Route filter applied. Records: %d -> %d", before, len(df))
    else:
        logger.info("No route filter applied.")

    df["TOTAL"] = df[EXCEL_BOARD_FIELD] + df[EXCEL_ALIGHT_FIELD]
    df[EXCEL_STOP_ID_FIELD] = _safe_to_str(df[EXCEL_STOP_ID_FIELD])

    return df


def aggregate_excel_per_stop(df_excel: pd.DataFrame) -> pd.DataFrame:
    """Collapse Excel ridership rows to one row per STOP_ID."""
    return df_excel.groupby(EXCEL_STOP_ID_FIELD, as_index=False).agg(
        {
            EXCEL_BOARD_FIELD: "sum",
            EXCEL_ALIGHT_FIELD: "sum",
            "TOTAL": "sum",
        }
    )


def merge_ridership(
    stops: gpd.GeoDataFrame,
    df_excel: pd.DataFrame,
    stops_key_field: str,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Inner-join ridership to stops on STOP_ID vs the chosen stop key field."""
    if stops_key_field not in stops.columns:
        raise ValueError(f"Stop key field '{stops_key_field}' not found in stops layer.")

    stops_copy = stops.copy()
    stops_copy[stops_key_field] = _safe_to_str(stops_copy[stops_key_field])

    out = stops_copy.merge(
        df_excel,
        left_on=stops_key_field,
        right_on=EXCEL_STOP_ID_FIELD,
        how="inner",
        validate="one_to_one" if df_excel[EXCEL_STOP_ID_FIELD].is_unique else "many_to_one",
    )

    logger.info("Matched stops after join: %d", len(out))
    return gpd.GeoDataFrame(data=out, geometry="geometry", crs=stops.crs)


def add_output_ridership_fields(stops_joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create standardized output fields (XBOARD/XALIGHT/XTOTAL)."""
    out = stops_joined.copy()
    out[OUT_BOARD] = out[EXCEL_BOARD_FIELD].astype(float)
    out[OUT_ALIGHT] = out[EXCEL_ALIGHT_FIELD].astype(float)
    out[OUT_TOTAL] = out["TOTAL"].astype(float)
    return out


def aggregate_by_polygon(
    matched_stops: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Aggregate stop ridership by POLYGON_JOIN_FIELD and join to polygons."""
    if POLYGON_JOIN_FIELD not in matched_stops.columns:
        raise ValueError(
            f"Matched stops missing polygon join field '{POLYGON_JOIN_FIELD}'. "
            "Confirm the spatial join ran and that field was kept."
        )

    df_agg = matched_stops.groupby(POLYGON_JOIN_FIELD, as_index=False).agg(
        {EXCEL_BOARD_FIELD: "sum", EXCEL_ALIGHT_FIELD: "sum", "TOTAL": "sum"}
    )
    df_agg = df_agg.rename(
        columns={
            EXCEL_BOARD_FIELD: "XBOARD_SUM",
            EXCEL_ALIGHT_FIELD: "XALITE_SUM",
            "TOTAL": "TOTAL_SUM",
        }
    )

    polygons_out = polygons.merge(df_agg, on=POLYGON_JOIN_FIELD, how="left")
    for c in ["XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"]:
        polygons_out[c] = polygons_out[c].fillna(0.0)

    logger.info("Polygon aggregation complete. Polygons: %d", len(polygons_out))
    return gpd.GeoDataFrame(data=polygons_out, geometry="geometry", crs=polygons.crs)


# =============================================================================
# PIPELINES
# =============================================================================


def run_single(logger: logging.Logger) -> None:
    """Run the non-split pipeline (one output for all matched stops)."""
    stops, stops_key_field = load_bus_stops(logger)
    stops_joined, polygons = spatial_join_to_polygons(stops, logger)

    df_excel = read_and_filter_excel(logger)
    df_excel_stop = aggregate_excel_per_stop(df_excel)

    agg_per_stop_csv = OUTPUT_FOLDER / "agg_ridership_per_stop.csv"
    df_excel_stop.to_csv(agg_per_stop_csv, index=False)
    logger.info("Wrote %s", agg_per_stop_csv)

    matched = merge_ridership(stops_joined, df_excel_stop, stops_key_field, logger)
    matched = add_output_ridership_fields(matched)

    stops_out = output_path("bus_stops_matched")
    layer = "bus_stops_matched" if stops_out.suffix.lower() == ".gpkg" else None
    write_vector(matched, stops_out, layer=layer)
    logger.info("Wrote %s", stops_out)

    matched_csv = OUTPUT_FOLDER / "bus_stops_with_polygon.csv"
    matched.drop(columns="geometry").to_csv(matched_csv, index=False)
    logger.info("Wrote %s", matched_csv)

    if polygons is not None:
        poly_out = aggregate_by_polygon(matched, polygons, logger)

        poly_out_path = output_path("polygon_with_ridership")
        layer = "polygon_with_ridership" if poly_out_path.suffix.lower() == ".gpkg" else None
        write_vector(poly_out, poly_out_path, layer=layer)
        logger.info("Wrote %s", poly_out_path)

        poly_csv = OUTPUT_FOLDER / "agg_ridership_by_polygon.csv"
        poly_out.drop(columns="geometry").to_csv(poly_csv, index=False)
        logger.info("Wrote %s", poly_csv)


def run_split_by_route(logger: logging.Logger) -> None:
    """Run the split-by-route pipeline (one output per route)."""
    stops, stops_key_field = load_bus_stops(logger)
    stops_joined, polygons = spatial_join_to_polygons(stops, logger)

    df_excel = read_and_filter_excel(logger)
    unique_routes = sorted(pd.unique(df_excel[EXCEL_ROUTE_FIELD].dropna()))
    logger.info("Found %d routes.", len(unique_routes))

    for route in unique_routes:
        df_route = df_excel[df_excel[EXCEL_ROUTE_FIELD] == route].copy()
        if df_route.empty:
            continue

        df_route_stop = aggregate_excel_per_stop(df_route)

        matched = merge_ridership(stops_joined, df_route_stop, stops_key_field, logger)
        if matched.empty:
            logger.warning("No matched stops for route %s; skipping.", route)
            continue

        matched = add_output_ridership_fields(matched)

        stops_out = output_path("bus_stops_matched", route=str(route))
        layer = f"bus_stops_matched_{route}" if stops_out.suffix.lower() == ".gpkg" else None
        write_vector(matched, stops_out, layer=layer)
        logger.info("Wrote %s", stops_out)

        matched_csv = OUTPUT_FOLDER / f"bus_stops_with_polygon_{route}.csv"
        matched.drop(columns="geometry").to_csv(matched_csv, index=False)
        logger.info("Wrote %s", matched_csv)

    # Optional: aggregate polygons across ALL filtered Excel records (not per-route)
    if polygons is not None:
        df_all_stop = aggregate_excel_per_stop(df_excel)
        matched_all = merge_ridership(stops_joined, df_all_stop, stops_key_field, logger)
        matched_all = add_output_ridership_fields(matched_all)

        poly_out = aggregate_by_polygon(matched_all, polygons, logger)
        poly_out_path = output_path("polygon_with_ridership")
        layer = "polygon_with_ridership" if poly_out_path.suffix.lower() == ".gpkg" else None
        write_vector(poly_out, poly_out_path, layer=layer)
        logger.info("Wrote %s", poly_out_path)

        poly_csv = OUTPUT_FOLDER / "agg_ridership_by_polygon.csv"
        poly_out.drop(columns="geometry").to_csv(poly_csv, index=False)
        logger.info("Wrote %s", poly_csv)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point."""
    if not BUS_STOPS_INPUT.exists():
        raise FileNotFoundError(f"BUS_STOPS_INPUT not found: {BUS_STOPS_INPUT}")
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"EXCEL_FILE not found: {EXCEL_FILE}")
    if POLYGON_LAYER is not None and not POLYGON_LAYER.exists():
        raise FileNotFoundError(f"POLYGON_LAYER not found: {POLYGON_LAYER}")

    LOGGER.info("Output folder: %s", OUTPUT_FOLDER)
    LOGGER.info("Split by route: %s", SPLIT_BY_ROUTE)
    LOGGER.info("Output format: %s", OUT_FORMAT)

    if SPLIT_BY_ROUTE:
        run_split_by_route(LOGGER)
    else:
        run_single(LOGGER)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
