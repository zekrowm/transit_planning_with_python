"""GTFS stops + route patterns to WGS84 shapefiles (ArcPy version).

This script converts a GTFS feed into two outputs:

* gtfs_stops.shp  – Stops as WGS84 point features
* gtfs_lines.shp  – One line geometry per route, with pattern selection and
                    optional direction merging

By default, lines are exported as one feature per (route_id, direction_id),
representing a selected "pattern" (shape_id). If MERGE_DIRECTIONS is True,
all directions are merged into a single feature per route_id.

Pattern selection options:

* "longest"     – choose the shape_id with the greatest geodesic length
* "most_stops"  – choose the shape_id that serves the most distinct stops
* "most_common" – choose the shape_id used by the most trips
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import arcpy
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# GTFS source – folder with *.txt OR a .zip file.
GTFS_PATH: str = r"Path\To\YourGTFS_Folder"

# Output folder for shapefiles (NOT a geodatabase).
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

ExportKind = Literal["stops", "lines", "both"]
EXPORT_KIND: ExportKind = "both"

PatternMode = Literal["longest", "most_stops", "most_common"]
PATTERN_MODE: PatternMode = "most_common"

# Optional route filters for lines export (by route_id).
# - ROUTE_FILTER_IN: if not None, only routes in this list are kept.
# - ROUTE_FILTER_OUT: if not None, routes in this list are removed.
ROUTE_FILTER_IN: Optional[List[str]] = None
ROUTE_FILTER_OUT: Optional[List[str]] = None

# If True, merge/dissolve all directions into a single feature per route_id.
# If False, keep one feature per (route_id, direction_id).
MERGE_DIRECTIONS: bool = False

# WGS 84 (GTFS lat/lon CRS).
WGS84_WKID: int = 4326

# Shapefile field lengths (text).
STOP_ID_LEN: int = 64
STOP_NAME_LEN: int = 128
ROUTE_ID_LEN: int = 64
ROUTE_SHORT_LEN: int = 64
SHAPE_ID_LEN: int = 64
PATTERN_MODE_LEN: int = 16

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"),
    )
    LOGGER.addHandler(_handler)


# ============================================================================
# HELPERS – I/O AND VALIDATION
# ============================================================================


def _ensure_output_folder(folder: str | Path) -> Path:
    """Create (if needed) and return the output folder path."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    if not out.is_dir():
        raise ValueError(f"OUTPUT_FOLDER is not a directory: {out}")
    return out


def _read_gtfs_tables(gtfs_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load GTFS tables into DataFrames."""
    gtfs = Path(gtfs_path)
    filenames: Dict[str, str] = {
        "stops": "stops.txt",
        "shapes": "shapes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "routes": "routes.txt",
    }

    def _read_from_dir(root: Path) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for key, name in filenames.items():
            p = root / name
            if p.exists():
                LOGGER.info("Reading %s", p)
                out[key] = pd.read_csv(p)
            else:
                LOGGER.warning("GTFS: %s not found at %s", name, p)
        return out

    if gtfs.is_dir():
        LOGGER.info("Detected GTFS directory at %s", gtfs)
        tables = _read_from_dir(gtfs)
    elif gtfs.is_file() and gtfs.suffix.lower() == ".zip":
        LOGGER.info("Detected GTFS zip at %s – extracting …", gtfs)
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs, "r") as zf:
            zf.extractall(tmp.name)
        tables = _read_from_dir(Path(tmp.name))
    else:
        raise ValueError("GTFS_PATH must be a folder or a .zip file")

    if "stops" not in tables:
        raise FileNotFoundError("stops.txt is required but was not found in the GTFS package")

    return tables


def _validate_tables(
    dfs: Dict[str, pd.DataFrame],
    export_kind: ExportKind,
    pattern_mode: PatternMode,
) -> None:
    """Validate that required GTFS columns are present."""
    missing_msgs: List[str] = []

    if export_kind in ("stops", "both"):
        if "stops" not in dfs:
            missing_msgs.append("missing stops.txt entirely")
        else:
            need = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
            have = set(dfs["stops"].columns)
            miss = need - have
            if miss:
                missing_msgs.append(f"stops.txt → missing {', '.join(sorted(miss))}")

    if export_kind in ("lines", "both"):
        if "shapes" not in dfs:
            missing_msgs.append("missing shapes.txt (required for route lines)")
        else:
            need = {
                "shape_id",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_pt_sequence",
            }
            have = set(dfs["shapes"].columns)
            miss = need - have
            if miss:
                missing_msgs.append(f"shapes.txt → missing {', '.join(sorted(miss))}")

        if "trips" not in dfs:
            missing_msgs.append("missing trips.txt (required for pattern selection)")
        else:
            need = {"trip_id", "route_id", "shape_id", "direction_id"}
            have = set(dfs["trips"].columns)
            miss = need - have
            if miss:
                missing_msgs.append(f"trips.txt → missing {', '.join(sorted(miss))}")

        if pattern_mode in ("most_stops", "most_common"):
            if "stop_times" not in dfs:
                missing_msgs.append(
                    "missing stop_times.txt (required for pattern mode "
                    f"'{pattern_mode}')",
                )
            else:
                need = {"trip_id", "stop_id"}
                have = set(dfs["stop_times"].columns)
                miss = need - have
                if miss:
                    missing_msgs.append(
                        f"stop_times.txt → missing {', '.join(sorted(miss))}",
                    )

    if missing_msgs:
        joined = "\n".join(" • " + msg for msg in missing_msgs)
        raise ValueError(f"GTFS validation failed:\n{joined}")


# ============================================================================
# ARCPY UTILS
# ============================================================================


def _wgs84_sr() -> arcpy.SpatialReference:
    """Return WGS 84 spatial reference."""
    sr = arcpy.SpatialReference(WGS84_WKID)
    if sr.name == "Unknown":
        raise ValueError(f"Spatial reference WKID {WGS84_WKID} is not recognized.")
    return sr


def _safe_add_field(
    table: str,
    name: str,
    field_type: str,
    field_precision: Optional[int] = None,
    field_scale: Optional[int] = None,
    field_length: Optional[int] = None,
    field_alias: Optional[str] = None,
    field_is_nullable: Optional[str] = None,
    field_is_required: Optional[str] = None,
    field_domain: Optional[str] = None,
) -> bool:
    """Add a field if it does not already exist.

    Returns:
        True if the field exists after this call (already present or added).
        False if AddField failed.
    """
    existing = {f.name.lower() for f in arcpy.ListFields(table) or []}
    if name.lower() in existing:
        LOGGER.info("Field %s already exists on %s – skipping AddField.", name, table)
        return True

    try:
        arcpy.management.AddField(
            in_table=table,
            field_name=name,
            field_type=field_type,
            field_precision=field_precision,
            field_scale=field_scale,
            field_length=field_length,
            field_alias=field_alias,
            field_is_nullable=field_is_nullable,
            field_is_required=field_is_required,
            field_domain=field_domain,
        )
        return True
    except arcpy.ExecuteError:
        LOGGER.warning(
            "AddField failed for %s on %s (type=%s). Messages:\n%s",
            name,
            table,
            field_type,
            arcpy.GetMessages(2),
        )
        return False


# ============================================================================
# GEOMETRY BUILDERS (WGS84)
# ============================================================================


def _build_stop_dataframe(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the stops DataFrame."""
    df = stops_df.copy()

    required = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"stops.txt missing required column '{col}'")

    df["stop_id"] = df["stop_id"].astype(str)
    df["stop_name"] = df["stop_name"].astype(str)

    for col in ("stop_lat", "stop_lon"):
        before = len(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        if dropped > 0:
            LOGGER.warning(
                "Dropped %d stops due to invalid values in %s.",
                dropped,
                col,
            )

    if df.empty:
        LOGGER.warning("No valid stops remain after cleaning.")

    return df[["stop_id", "stop_name", "stop_lat", "stop_lon"]]


def _build_shape_geometries(shapes_df: pd.DataFrame) -> Dict[str, arcpy.Polyline]:
    """Build WGS84 Polyline geometries keyed by shape_id."""
    df = shapes_df.copy()

    required = {
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"shapes.txt missing required columns: {', '.join(sorted(missing))}")

    df["shape_id"] = df["shape_id"].astype(str)

    for col in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
        before = len(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        if dropped > 0:
            LOGGER.warning(
                "Dropped %d shape points due to invalid values in %s.",
                dropped,
                col,
            )

    if df.empty:
        LOGGER.warning("No valid shape points remain after cleaning.")
        return {}

    df["shape_pt_sequence"] = df["shape_pt_sequence"].astype(int)
    df_sorted = df.sort_values(["shape_id", "shape_pt_sequence"])

    sr = _wgs84_sr()
    out: Dict[str, arcpy.Polyline] = {}

    for shape_id, group in df_sorted.groupby("shape_id"):
        array = arcpy.Array()
        for _, row in group.iterrows():
            lon = float(row["shape_pt_lon"])
            lat = float(row["shape_pt_lat"])
            array.add(arcpy.Point(lon, lat))

        if array.count < 2:
            LOGGER.debug("Shape %s has fewer than 2 points; skipping.", shape_id)
            continue

        out[shape_id] = arcpy.Polyline(array, sr)

    LOGGER.info("Built %d shape geometries.", len(out))
    return out


# ============================================================================
# PATTERN SELECTION
# ============================================================================


def _build_route_patterns(
    trips_df: pd.DataFrame,
    stop_times_df: Optional[pd.DataFrame],
    routes_df: Optional[pd.DataFrame],
    shape_geoms: Dict[str, arcpy.Polyline],
    pattern_mode: PatternMode,
    route_filter_in: Optional[Set[str]] = None,
    route_filter_out: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    """Return chosen pattern geometries per (route_id, direction_id).

    Route filtering is applied by route_id before pattern selection:
    - If route_filter_in is not None, only those route_ids are retained.
    - If route_filter_out is not None, those route_ids are removed.
    """
    if trips_df.empty:
        LOGGER.warning("trips.txt is empty – no route patterns can be built.")
        return []

    trips = trips_df.copy()

    needed = {"trip_id", "route_id", "shape_id", "direction_id"}
    missing = needed - set(trips.columns)
    if missing:
        raise ValueError(f"trips.txt missing required columns: {', '.join(sorted(missing))}")

    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trips["shape_id"] = trips["shape_id"].astype(str)

    trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce")
    before = len(trips)
    trips = trips.dropna(subset=["shape_id", "direction_id"])
    trips["direction_id"] = trips["direction_id"].astype(int)
    dropped = before - len(trips)
    if dropped > 0:
        LOGGER.warning(
            "Dropped %d trips due to missing shape_id or direction_id.",
            dropped,
        )

    # Apply route filters.
    if route_filter_in is not None:
        before = len(trips)
        trips = trips[trips["route_id"].isin(route_filter_in)]
        removed = before - len(trips)
        LOGGER.info(
            "Route filter IN applied – kept %d trips (dropped %d).",
            len(trips),
            removed,
        )

    if route_filter_out is not None:
        before = len(trips)
        trips = trips[~trips["route_id"].isin(route_filter_out)]
        removed = before - len(trips)
        LOGGER.info(
            "Route filter OUT applied – kept %d trips (dropped %d).",
            len(trips),
            removed,
        )

    if trips.empty:
        LOGGER.warning(
            "No valid trips remain after cleaning and filtering – cannot build patterns.",
        )
        return []

    route_short_lookup: Dict[str, str] = {}
    if routes_df is not None and not routes_df.empty:
        if "route_id" in routes_df.columns and "route_short_name" in routes_df.columns:
            tmp = routes_df[["route_id", "route_short_name"]].copy()
            tmp["route_id"] = tmp["route_id"].astype(str)
            route_short_lookup = (
                tmp.set_index("route_id")["route_short_name"].astype(str).to_dict()
            )

    if pattern_mode == "most_common":
        if stop_times_df is None or stop_times_df.empty:
            raise ValueError(
                "stop_times.txt is required for 'most_common' pattern mode.",
            )

        counts = (
            trips.groupby(["route_id", "direction_id", "shape_id"])["trip_id"]
            .count()
            .reset_index(name="trip_count")
        )

        if counts.empty:
            LOGGER.warning("No (route, direction, shape) combinations found.")
            return []

        chosen = (
            counts.sort_values(
                ["route_id", "direction_id", "trip_count", "shape_id"],
                ascending=[True, True, False, True],
            )
            .groupby(["route_id", "direction_id"], as_index=False)
            .head(1)
        )

    elif pattern_mode == "most_stops":
        if stop_times_df is None or stop_times_df.empty:
            raise ValueError(
                "stop_times.txt is required for 'most_stops' pattern mode.",
            )

        st = stop_times_df.copy()
        need = {"trip_id", "stop_id"}
        miss = need - set(st.columns)
        if miss:
            raise ValueError(
                f"stop_times.txt missing required columns: {', '.join(sorted(miss))}",
            )

        st["trip_id"] = st["trip_id"].astype(str)
        st["stop_id"] = st["stop_id"].astype(str)

        merged = trips[["trip_id", "route_id", "direction_id", "shape_id"]].merge(
            st[["trip_id", "stop_id"]],
            on="trip_id",
            how="inner",
        )

        if merged.empty:
            LOGGER.warning("No joined trip/stop records; cannot compute 'most_stops'.")
            return []

        stop_counts = (
            merged.groupby(["route_id", "direction_id", "shape_id"])["stop_id"]
            .nunique()
            .reset_index(name="n_stops")
        )

        chosen = (
            stop_counts.sort_values(
                ["route_id", "direction_id", "n_stops", "shape_id"],
                ascending=[True, True, False, True],
            )
            .groupby(["route_id", "direction_id"], as_index=False)
            .head(1)
        )

    elif pattern_mode == "longest":
        if not shape_geoms:
            LOGGER.warning("No shape geometries; cannot compute 'longest' patterns.")
            return []

        combos = (
            trips[["route_id", "direction_id", "shape_id"]]
            .drop_duplicates()
            .copy()
        )

        def _length_km(shape_id: str) -> float:
            geom = shape_geoms.get(shape_id)
            if geom is None:
                return float("nan")
            return geom.getLength("GEODESIC", "KILOMETERS")

        combos["length_km"] = combos["shape_id"].apply(_length_km)
        combos = combos.dropna(subset=["length_km"])

        if combos.empty:
            LOGGER.warning("No valid shape lengths for 'longest' patterns.")
            return []

        chosen = (
            combos.sort_values(
                ["route_id", "direction_id", "length_km", "shape_id"],
                ascending=[True, True, False, True],
            )
            .groupby(["route_id", "direction_id"], as_index=False)
            .head(1)
        )

    else:
        raise ValueError(f"Unsupported pattern_mode: {pattern_mode}")

    if chosen.empty:
        LOGGER.warning("Pattern selection produced no rows.")
        return []

    records: List[Dict[str, object]] = []
    for row in chosen.itertuples(index=False):
        route_id = str(row.route_id)
        direction_id = int(row.direction_id)
        shape_id = str(row.shape_id)
        geom = shape_geoms.get(shape_id)

        if geom is None:
            LOGGER.debug(
                "Missing geometry for shape_id=%s (route_id=%s, dir=%s); skipping.",
                shape_id,
                route_id,
                direction_id,
            )
            continue

        records.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "shape_id": shape_id,
                "geometry": geom,
                "route_short": route_short_lookup.get(route_id),
            },
        )

    LOGGER.info(
        "Built %d route pattern records (pattern_mode=%s).",
        len(records),
        pattern_mode,
    )
    return records


# ============================================================================
# LINES MERGE / DISSOLVE UTIL
# ============================================================================


def _merge_route_directions(
    routes: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Merge geometries across directions into one feature per route_id.

    The merged records:
    - Keep route_id and route_short.
    - Set direction_id to -1 to indicate "all directions".
    - Use a union of all direction geometries for that route.
    """
    merged: Dict[str, Dict[str, object]] = {}

    for rec in routes:
        geom = rec.get("geometry")
        if not isinstance(geom, arcpy.Polyline) or geom.length == 0:
            continue

        route_id = str(rec.get("route_id"))

        if route_id not in merged:
            base = rec.copy()
            base["direction_id"] = -1
            merged[route_id] = base
        else:
            existing_geom = merged[route_id].get("geometry")
            if isinstance(existing_geom, arcpy.Polyline):
                try:
                    merged[route_id]["geometry"] = existing_geom.union(geom)
                except arcpy.ExecuteError:
                    LOGGER.warning(
                        "Union failed for route %s; keeping existing geometry only.",
                        route_id,
                    )

    LOGGER.info(
        "Merged directions: %d input records → %d route-level records.",
        len(routes),
        len(merged),
    )
    return list(merged.values())


# ============================================================================
# SHAPEFILE EXPORTS
# ============================================================================


def _export_stops_shapefile(stops_df: pd.DataFrame, out_folder: Path) -> None:
    """Export GTFS stops to gtfs_stops.shp (WGS84 points)."""
    if stops_df.empty:
        LOGGER.warning("No stops to export – skipping gtfs_stops.shp.")
        return

    arcpy.env.overwriteOutput = True
    sr = _wgs84_sr()

    out_folder_str = str(out_folder)
    out_name = "gtfs_stops"

    existing_fc = os.path.join(out_folder_str, out_name + ".shp")
    if arcpy.Exists(existing_fc):
        LOGGER.info("Deleting existing %s", existing_fc)
        arcpy.management.Delete(existing_fc)

    try:
        result = arcpy.management.CreateFeatureclass(
            out_path=out_folder_str,
            out_name=out_name,
            geometry_type="POINT",
            template="",
            has_m="DISABLED",
            has_z="DISABLED",
            spatial_reference=sr,
        )
        fc_path = result[0]
    except arcpy.ExecuteError:
        LOGGER.error("ArcPy error in CreateFeatureclass (stops): %s", arcpy.GetMessages(2))
        raise

    _safe_add_field(fc_path, "stop_id", "TEXT", field_length=STOP_ID_LEN)
    _safe_add_field(fc_path, "stop_nm", "TEXT", field_length=STOP_NAME_LEN)
    _safe_add_field(fc_path, "stop_lat", "DOUBLE")
    _safe_add_field(fc_path, "stop_lon", "DOUBLE")

    fields = ["stop_id", "stop_nm", "stop_lat", "stop_lon", "SHAPE@"]

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, fields) as cursor:
        for row in stops_df.itertuples(index=False):
            stop_id = str(row.stop_id)
            stop_name = str(row.stop_name)
            lat = float(row.stop_lat)
            lon = float(row.stop_lon)

            pt = arcpy.Point(lon, lat)
            geom = arcpy.PointGeometry(pt, sr)
            cursor.insertRow([stop_id, stop_name, lat, lon, geom])
            rows_written += 1

    LOGGER.info("Wrote %s (%d features).", fc_path, rows_written)


def _export_lines_shapefile(
    routes: List[Dict[str, object]],
    out_folder: Path,
    pattern_mode: PatternMode,
    merge_directions: bool,
) -> None:
    """Export selected route patterns to gtfs_lines.shp (WGS84 polylines)."""
    if not routes:
        LOGGER.warning("No route patterns to export – skipping gtfs_lines.shp.")
        return

    if merge_directions:
        LOGGER.info("Merging all directions to one feature per route_id.")
        routes = _merge_route_directions(routes)
        if not routes:
            LOGGER.warning(
                "No routes remain after merging directions – skipping gtfs_lines.shp.",
            )
            return

    arcpy.env.overwriteOutput = True
    sr = _wgs84_sr()

    out_folder_str = str(out_folder)
    out_name = "gtfs_lines"

    existing_fc = os.path.join(out_folder_str, out_name + ".shp")
    if arcpy.Exists(existing_fc):
        LOGGER.info("Deleting existing %s", existing_fc)
        arcpy.management.Delete(existing_fc)

    try:
        result = arcpy.management.CreateFeatureclass(
            out_path=out_folder_str,
            out_name=out_name,
            geometry_type="POLYLINE",
            template="",
            has_m="DISABLED",
            has_z="DISABLED",
            spatial_reference=sr,
        )
        fc_path = result[0]
    except arcpy.ExecuteError:
        LOGGER.error("ArcPy error in CreateFeatureclass (lines): %s", arcpy.GetMessages(2))
        raise

    # Try to add fields; track which ones actually exist.
    added_fields: List[str] = []

    if _safe_add_field(fc_path, "route_id", "TEXT", field_length=ROUTE_ID_LEN):
        added_fields.append("route_id")

    # Use dir_id instead of dir to avoid 000852 conflicts.
    if _safe_add_field(fc_path, "dir_id", "SHORT"):
        added_fields.append("dir_id")

    if _safe_add_field(fc_path, "shape_id", "TEXT", field_length=SHAPE_ID_LEN):
        added_fields.append("shape_id")
    else:
        LOGGER.error(
            "Field shape_id could not be added to %s. "
            "Continuing without shape_id attribute.",
            fc_path,
        )

    if _safe_add_field(fc_path, "rshort", "TEXT", field_length=ROUTE_SHORT_LEN):
        added_fields.append("rshort")

    if _safe_add_field(fc_path, "pmode", "TEXT", field_length=PATTERN_MODE_LEN):
        added_fields.append("pmode")

    required = {"route_id", "dir_id", "pmode"}
    missing_required = required - set(added_fields)
    if missing_required:
        raise RuntimeError(
            "Missing required fields on gtfs_lines: "
            + ", ".join(sorted(missing_required)),
        )

    include_shape_id = "shape_id" in added_fields

    fields: List[str] = ["route_id", "dir_id"]
    if include_shape_id:
        fields.append("shape_id")
    fields.extend(["rshort", "pmode", "SHAPE@"])

    rows_written = 0
    with arcpy.da.InsertCursor(fc_path, fields) as cursor:
        for rec in routes:
            geom = rec.get("geometry")
            if not isinstance(geom, arcpy.Polyline) or geom.length == 0:
                continue

            route_id = str(rec.get("route_id"))
            direction_id = int(rec.get("direction_id"))

            route_short = rec.get("route_short")
            route_short_str = str(route_short) if route_short is not None else ""

            values: List[object] = [route_id, direction_id]

            if include_shape_id:
                shape_id_val = rec.get("shape_id")
                shape_id = str(shape_id_val) if shape_id_val is not None else ""
                values.append(shape_id)

            values.extend([route_short_str, pattern_mode, geom])

            cursor.insertRow(values)
            rows_written += 1

    LOGGER.info("Wrote %s (%d features).", fc_path, rows_written)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:  # noqa: D401
    """Run the GTFS-to-shapefile pipeline with pattern selection."""
    arcpy.env.overwriteOutput = True

    out_dir = _ensure_output_folder(OUTPUT_FOLDER)
    LOGGER.info("Output folder: %s", out_dir)

    LOGGER.info("STEP 0  Reading GTFS tables …")
    dfs = _read_gtfs_tables(GTFS_PATH)

    try:
        _validate_tables(dfs, EXPORT_KIND, PATTERN_MODE)
    except ValueError as err:
        LOGGER.error("ERROR – invalid GTFS feed:\n%s", err)
        sys.exit(1)

    if EXPORT_KIND in ("stops", "both"):
        LOGGER.info("STEP 1  Processing stops …")
        stops_df = _build_stop_dataframe(dfs["stops"])
        _export_stops_shapefile(stops_df, out_dir)

    if EXPORT_KIND in ("lines", "both"):
        LOGGER.info("STEP 2  Processing shapes and patterns (mode=%s) …", PATTERN_MODE)

        shapes_df = dfs.get("shapes")
        trips_df = dfs.get("trips")
        stop_times_df = dfs.get("stop_times")
        routes_df = dfs.get("routes")

        if shapes_df is None or trips_df is None:
            LOGGER.error(
                "shapes.txt and trips.txt are required for lines export; "
                "skipping gtfs_lines.shp.",
            )
        else:
            shape_geoms = _build_shape_geometries(shapes_df)

            route_filter_in_set: Optional[Set[str]] = (
                set(ROUTE_FILTER_IN) if ROUTE_FILTER_IN else None
            )
            route_filter_out_set: Optional[Set[str]] = (
                set(ROUTE_FILTER_OUT) if ROUTE_FILTER_OUT else None
            )

            routes = _build_route_patterns(
                trips_df,
                stop_times_df,
                routes_df,
                shape_geoms,
                PATTERN_MODE,
                route_filter_in=route_filter_in_set,
                route_filter_out=route_filter_out_set,
            )

            _export_lines_shapefile(
                routes,
                out_dir,
                PATTERN_MODE,
                MERGE_DIRECTIONS,
            )

    LOGGER.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except arcpy.ExecuteError:
        LOGGER.error("ArcPy ExecuteError:\n%s", arcpy.GetMessages())
        raise
    except Exception:
        LOGGER.exception("UNEXPECTED ERROR")
        sys.exit(1)
