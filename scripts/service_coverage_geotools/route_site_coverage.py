"""Analyze GTFS route coverage of strategic land uses using ArcPy geometries.

This module builds per-route buffers from GTFS shapes (or stop-based geometry for
express routes / missing shapes), counts intersecting facilities from configured
shapefile layers, and writes per-route and systemwide CSV summaries.

Optionally, it computes inter-route transfer opportunities from GTFS stop_times,
including same-stop and nearby-stop transfers within configured distance and
time windows.

Typical usage:
    Run as a script after updating the CONFIGURATION section.

Outputs:
    - File geodatabase feature classes (route buffers; optional parcel-buffered facilities)
    - Per-route facility summary CSVs and an all-routes summary CSV
    - Optional per-route transfer CSVs and an all-routes transfer summary CSV
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
import math
import arcpy
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top-level directories
GTFS_DIR = Path(r"Path\To\Your\GTFS_Data")  # folder containing GTFS .txt files
# SHP_INPUT_DIR = Path(r"data/shapefiles")  # folder with .shp layers to test
SHP_INPUT_DIR = Path(r"Path\To\Your\Shapefile_Data_Directory")  # folder with .shp layers to test
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")  # where CSVs and GDB output are written

# File geodatabase for outputs
GDB_NAME = "transit_coverage.gdb"

# List of `(filename, id_column)` describing each layer to test
# (filenames are relative to SHP_INPUT_DIR; search is recursive and case-insensitive)
LAYER_SPECS: List[Tuple[str, str]] = [
    ("school_facilities.shp", "SCHOOL_NAM"),
    ("metro_stations.shp", "NAME"),
    ("transit_stations.shp", "Name"),
    ("activity_centers.shp", "Activity_C"),
    ("colleges_and_universities.shp", "DESCRIPTIO"),
    ("hospitals.shp", "DESCRIPTIO"),
    ("park_and_rides.shp", "FACILITY_N"),
    ("gov_and_community_centers.shp", "DESCRIPTIO"),
]

# Optional filter: only analyze these route_id values using shape-based buffers.
# Leave empty (`[]`) to process every route in routes.txt
ROUTE_FILTER: List[str] = [
    "101",
    "202",
    "303",
]

# Routes that should use stop-based buffers instead of shape-based buffers.
EXPRESS_ROUTES: List[str] = [
    "909",
]

# Analysis options
USE_SHAPE_BUFFER = True  # True → buffer route geometry; False → buffer stops
BUFFER_DIST_FT = 1320.0  # ¼ mile in feet

# Optional parcels polygon layer: if set, facility coverage will be based on
# buffered parcels intersecting the facility features, rather than buffering
# the facility features themselves.
PARCELS_SHP = Path(r"Path\To\Your\parcels.shp")
#PARCELS_SHP: Optional[Path] = None

# Transfer analysis options
EXPORT_ROUTE_TRANSFERS: bool = True  # Set False to skip transfer CSVs
TRANSFER_DISTANCE_FT: float = 150.0   # Max stop-to-stop distance for a transfer (feet)
TRANSFER_TIME_MIN: float = 40.0      # Max time difference between trips at transfer (minutes)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("gtfs_buffer_analysis_arcpy")


# =============================================================================
# HELPERS
# =============================================================================


def _add_message(msg: str, level: str = "INFO") -> None:
    """Write message to both Python logging and ArcPy, if available."""
    if level.upper() == "ERROR":
        log.error(msg)
        try:
            arcpy.AddError(msg)
        except Exception:
            pass
    elif level.upper() == "WARNING":
        log.warning(msg)
        try:
            arcpy.AddWarning(msg)
        except Exception:
            pass
    else:
        log.info(msg)
        try:
            arcpy.AddMessage(msg)
        except Exception:
            pass


def _log_available_shapefiles(shp_dir: Path, max_examples: int = 50) -> None:
    """Log the shapefiles discovered under a directory (for debugging).

    Args:
        shp_dir: Root directory to search for shapefiles.
        max_examples: Maximum number of shapefiles to list explicitly.
    """
    shps = sorted(shp_dir.rglob("*.shp"))
    if not shps:
        _add_message(
            f"No shapefiles found anywhere under {shp_dir}",
            "WARNING",
        )
        return

    rel_paths = [str(p.relative_to(shp_dir)) for p in shps]
    preview = ", ".join(rel_paths[:max_examples])
    extra_count = len(rel_paths) - max_examples
    suffix = f" (and {extra_count} more...)" if extra_count > 0 else ""

    _add_message(
        f"Shapefiles discovered under {shp_dir}: {preview}{suffix}",
        "INFO",
    )


# =============================================================================
# GTFS LOADING AND PREP
# =============================================================================


def _load_gtfs_tables(gtfs_dir: Path) -> Mapping[str, pd.DataFrame]:
    """Load GTFS text files into pandas DataFrames.

    Args:
        gtfs_dir: Directory containing GTFS .txt files.

    Returns:
        Mapping keyed by table name (without .txt) to DataFrame.

    Raises:
        FileNotFoundError: If any required GTFS file is missing.
    """
    required = ["routes", "trips", "stop_times", "stops", "shapes"]
    tables: Dict[str, pd.DataFrame] = {}

    for fn in required:
        path = gtfs_dir / f"{fn}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        tables[fn] = df
        _add_message(f"Loaded {path} ({len(df)} rows)", "INFO")

    return tables


def _get_route_shape_mapping(trips: pd.DataFrame) -> Mapping[str, List[str]]:
    """Create mapping from route_id -> list of shape_ids.

    Args:
        trips: GTFS trips table.

    Returns:
        Mapping from route_id to list of distinct shape_ids.
    """
    if {"route_id", "trip_id", "shape_id"}.difference(trips.columns):
        raise ValueError("trips.txt missing required columns: route_id, trip_id, shape_id")

    route_shapes = (
        trips.dropna(subset=["shape_id"])
        .drop_duplicates(subset=["route_id", "shape_id"])
        .groupby("route_id")["shape_id"]
        .apply(list)
    )
    return route_shapes.to_dict()


def _compute_buffer_distance_units(
    target_sr: arcpy.SpatialReference,
    buffer_dist_ft: float,
) -> float:
    """Convert buffer distance in feet to units of the target spatial reference.

    Args:
        target_sr: Spatial reference used for geometry operations.
        buffer_dist_ft: Buffer distance in feet.

    Returns:
        Buffer distance expressed in the units of target_sr.
    """
    if not target_sr or target_sr.name == "Unknown":
        _add_message(
            "Target spatial reference is unknown. Assuming meters for buffer distance conversion.",
            "WARNING",
        )

    meters_per_unit = target_sr.metersPerUnit if target_sr and target_sr.metersPerUnit else 1.0
    if meters_per_unit == 0:
        meters_per_unit = 1.0

    buffer_meters = buffer_dist_ft * 0.3048
    buffer_units = buffer_meters / meters_per_unit

    _add_message(
        f"Buffer distance: {buffer_dist_ft:.2f} ft -> {buffer_units:.2f} "
        f"({target_sr.linearUnitName} units)",
        "INFO",
    )
    return buffer_units


def _parse_gtfs_time_to_seconds(time_str: object) -> Optional[int]:
    """Parse a GTFS time string (HH:MM[:SS]) into seconds from midnight.

    GTFS allows hours >= 24. Returns None if parsing fails.

    Args:
        time_str: Raw time value from stop_times.txt.

    Returns:
        Number of seconds from midnight, or None if invalid/blank.
    """
    if time_str is None:
        return None
    if isinstance(time_str, float) and math.isnan(time_str):
        return None

    s = str(time_str).strip()
    if not s:
        return None

    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None

    return hours * 3600 + minutes * 60 + seconds


def _haversine_distance_ft(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    """Compute great-circle distance between two WGS84 points in feet.

    Args:
        lat1_deg: Latitude of first point in degrees.
        lon1_deg: Longitude of first point in degrees.
        lat2_deg: Latitude of second point in degrees.
        lon2_deg: Longitude of second point in degrees.

    Returns:
        Distance in feet.
    """
    radius_m = 6_371_000.0
    radius_ft = radius_m * 3.28084

    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    a = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_ft * c


def _build_nearby_stop_pairs(
    stops: pd.DataFrame,
    max_dist_ft: float,
) -> pd.DataFrame:
    """Identify pairs of stops that are within max_dist_ft of each other.

    This catches transfers between distinct but nearby stops (e.g., bays or
    opposing stops across the street).

    Args:
        stops: GTFS stops table with stop_id, stop_lat, stop_lon.
        max_dist_ft: Maximum center-to-center distance in feet.

    Returns:
        DataFrame with columns:
        - stop_id_a
        - stop_id_b
        - distance_ft

        Each row is an unordered pair; stop_id_a != stop_id_b.
    """
    if stops.empty or max_dist_ft <= 0.0:
        return pd.DataFrame(columns=["stop_id_a", "stop_id_b", "distance_ft"])

    work = stops[["stop_id", "stop_lat", "stop_lon"]].copy()
    work = work.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)

    if work.empty:
        return pd.DataFrame(columns=["stop_id_a", "stop_id_b", "distance_ft"])

    # Simple spatial hashing grid: ~180 ft cells in latitude.
    cell_size_deg = 0.0005
    bin_map: Dict[int, Tuple[int, int]] = {}
    for idx in range(len(work)):
        lat = float(work.at[idx, "stop_lat"])
        lon = float(work.at[idx, "stop_lon"])
        bin_map[idx] = (
            int(lat / cell_size_deg),
            int(lon / cell_size_deg),
        )

    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, key in bin_map.items():
        grid.setdefault(key, []).append(idx)

    pairs: List[Dict[str, object]] = []
    seen: set[Tuple[int, int]] = set()

    for (bx, by), idxs in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb_key = (bx + dx, by + dy)
                if nb_key not in grid:
                    continue

                for i in idxs:
                    for j in grid[nb_key]:
                        if i >= j:
                            continue

                        key = (i, j)
                        if key in seen:
                            continue
                        seen.add(key)

                        lat1 = float(work.at[i, "stop_lat"])
                        lon1 = float(work.at[i, "stop_lon"])
                        lat2 = float(work.at[j, "stop_lat"])
                        lon2 = float(work.at[j, "stop_lon"])
                        dist_ft = _haversine_distance_ft(lat1, lon1, lat2, lon2)

                        if dist_ft <= max_dist_ft:
                            pairs.append(
                                {
                                    "stop_id_a": work.at[i, "stop_id"],
                                    "stop_id_b": work.at[j, "stop_id"],
                                    "distance_ft": dist_ft,
                                },
                            )

    if not pairs:
        return pd.DataFrame(columns=["stop_id_a", "stop_id_b", "distance_ft"])

    return pd.DataFrame(pairs)


# =============================================================================
# LAYER DISCOVERY
# =============================================================================


def _find_shapefile_case_insensitive(root: Path, filename: str) -> Optional[Path]:
    """Search recursively under root for a .shp matching filename (case-insensitive).

    Args:
        root: Root directory to search.
        filename: Target filename (with extension).

    Returns:
        First matching path in lexicographic order, or None if not found.
    """
    target_lower = filename.lower()
    matches = sorted(p for p in root.rglob("*.shp") if p.name.lower() == target_lower)
    if not matches:
        return None
    if len(matches) > 1:
        _add_message(
            f"Multiple copies of {filename} found; using {matches[0]}",
            "WARNING",
        )
    return matches[0]


def _load_layers(
    layer_specs: Iterable[Tuple[str, str]],
    shp_dir: Path,
) -> Tuple[Dict[str, Dict[str, str]], Optional[arcpy.SpatialReference]]:
    """Discover each designated shapefile and validate ID fields.

    Args:
        layer_specs: Tuples of (filename, id_column).
        shp_dir: Root directory to search.

    Returns:
        Tuple of:
        - Mapping of filename -> {"path": <full path>, "id_field": <column name>}
        - Target spatial reference inferred from the first valid layer, or None.
    """
    layers: Dict[str, Dict[str, str]] = {}
    target_sr: Optional[arcpy.SpatialReference] = None

    for filename, id_col in layer_specs:
        path = _find_shapefile_case_insensitive(shp_dir, filename)
        if path is None:
            _add_message(
                f"Layer {filename} NOT FOUND anywhere under {shp_dir}",
                "WARNING",
            )
            _log_available_shapefiles(shp_dir)
            continue

        desc = arcpy.Describe(str(path))
        field_names = {f.name for f in desc.fields}

        if id_col not in field_names:
            existing_fields = ", ".join(sorted(field_names))
            _add_message(
                f"Column {id_col} missing in {path} – layer skipped. "
                f"Existing fields: {existing_fields}",
                "WARNING",
            )
            continue

        layer_sr = desc.spatialReference
        if not target_sr:
            target_sr = layer_sr
            _add_message(
                f"Using spatial reference '{target_sr.name}' "
                f"from layer {path.name} as analysis CRS",
                "INFO",
            )
        else:
            # Warn if SRs disagree in name. We still handle reprojection later via cursors.
            if layer_sr.name != target_sr.name:
                _add_message(
                    f"Layer {path.name} has spatial reference '{layer_sr.name}' "
                    f"which differs from target '{target_sr.name}'. "
                    "Geometries will be reprojected on the fly.",
                    "WARNING",
                )

        layers[filename] = {"path": str(path), "id_field": id_col}
        _add_message(
            f"Loaded layer definition for {path.relative_to(shp_dir)}",
            "INFO",
        )

    return layers, target_sr


def _maybe_replace_facility_layers_with_parcel_buffers(
    layers: Mapping[str, Dict[str, str]],
    layer_specs: Iterable[Tuple[str, str]],
    parcels_path: Optional[Path],
    gdb_path: Path,
    target_sr: arcpy.SpatialReference,
    buffer_dist_ft: float,
) -> Dict[str, Dict[str, str]]:
    """Optionally replace facility layers with parcel-buffered versions.

    For each facility layer in ``layers``, this function will:
      * Find all parcels that intersect each facility feature.
      * Union those parcels into a single geometry.
      * Buffer that union by ``buffer_dist_ft`` (converted to target CRS units).
      * Write the result to a new polygon feature class in ``gdb_path`` that
        preserves the layer's ID field.

    If ``parcels_path`` is None or does not exist, the original ``layers`` are
    returned unchanged.

    Args:
        layers: Mapping of filename -> {"path": <fc path>, "id_field": <field>}.
        layer_specs: Layer spec list defining output order.
        parcels_path: Path to parcels polygon feature class/shapefile, or None.
        gdb_path: Path to the output file geodatabase.
        target_sr: Spatial reference for analysis and output geometries.
        buffer_dist_ft: Buffer distance in feet applied to parcel unions.

    Returns:
        Possibly modified layer mapping whose paths now point to parcel-buffer
        feature classes instead of the original facility layers.
    """
    if parcels_path is None:
        _add_message(
            "No parcels shapefile configured; using facility geometries directly.",
            "INFO",
        )
        return dict(layers)

    if not parcels_path.exists():
        _add_message(
            f"Parcels shapefile {parcels_path} not found – "
            "using facility geometries directly.",
            "WARNING",
        )
        return dict(layers)

    parcels_desc = arcpy.Describe(str(parcels_path))
    if parcels_desc.shapeType.upper() != "POLYGON":
        _add_message(
            f"Parcels layer {parcels_path} is not a polygon dataset "
            f"(shapeType={parcels_desc.shapeType}) – ignoring parcels.",
            "WARNING",
        )
        return dict(layers)

    parcels_layer_name = "parcels_buffer_tmp_lyr"
    arcpy.management.MakeFeatureLayer(str(parcels_path), parcels_layer_name)

    buffer_dist_units = _compute_buffer_distance_units(target_sr, buffer_dist_ft)
    updated_layers: Dict[str, Dict[str, str]] = dict(layers)

    _add_message(
        f"Building parcel-based buffers for facilities using parcels {parcels_path}",
        "INFO",
    )

    for filename, _ in layer_specs:
        if filename not in updated_layers:
            # Layer was skipped earlier (e.g., missing id field); nothing to do.
            continue

        layer_info = updated_layers[filename]
        facility_path = layer_info["path"]
        id_field = layer_info["id_field"]

        if not arcpy.Exists(facility_path):
            _add_message(
                f"Facility layer {facility_path} for {filename} does not exist – "
                "skipping parcel buffering for this layer.",
                "WARNING",
            )
            continue

        out_name = f"{Path(filename).stem}_parcelbuf"
        out_fc = str(gdb_path / out_name)

        if arcpy.Exists(out_fc):
            _add_message(f"Deleting existing parcel buffer FC {out_fc}", "INFO")
            arcpy.management.Delete(out_fc)

        arcpy.management.CreateFeatureclass(
            str(gdb_path),
            out_name,
            "POLYGON",
            spatial_reference=target_sr,
        )
        arcpy.management.AddField(out_fc, id_field, "TEXT", field_length=255)

        inserted_count = 0

        with arcpy.da.SearchCursor(
            facility_path,
            [id_field, "SHAPE@"],
            spatial_reference=target_sr,
        ) as fac_cur, arcpy.da.InsertCursor(
            out_fc,
            [id_field, "SHAPE@"],
        ) as ins_cur:
            for id_val, fac_geom in fac_cur:
                if fac_geom is None:
                    continue

                arcpy.management.SelectLayerByLocation(
                    parcels_layer_name,
                    "INTERSECT",
                    fac_geom,
                    selection_type="NEW_SELECTION",
                )

                parcel_geoms: List[arcpy.Geometry] = []
                with arcpy.da.SearchCursor(
                    parcels_layer_name,
                    ["SHAPE@"],
                    spatial_reference=target_sr,
                ) as parc_cur:
                    for (pgeom,) in parc_cur:
                        if pgeom:
                            parcel_geoms.append(pgeom)

                if not parcel_geoms:
                    # Facility does not intersect any parcels – it will not receive
                    # a parcel buffer geometry.
                    continue

                union_geom = parcel_geoms[0]
                for pgeom in parcel_geoms[1:]:
                    union_geom = union_geom.union(pgeom)

                try:
                    buf_geom = union_geom.buffer(buffer_dist_units)
                except Exception as exc:
                    _add_message(
                        f"Parcel buffering failed for {filename} feature {id_val}: {exc}",
                        "WARNING",
                    )
                    continue

                ins_cur.insertRow((str(id_val), buf_geom))
                inserted_count += 1

        if inserted_count == 0:
            _add_message(
                f"No parcel-buffer geometries created for layer {filename}. "
                "Falling back to original facility geometries.",
                "WARNING",
            )
            continue

        updated_layers[filename] = {
            "path": out_fc,
            "id_field": id_field,
        }
        _add_message(
            f"Parcel-buffer facility layer created for {filename}: {out_fc} "
            f"({inserted_count} features)",
            "INFO",
        )

    return updated_layers


# =============================================================================
# ROUTE BUFFER CREATION (ARCPY GEOMETRY)
# =============================================================================


def _build_route_buffers_fc(
    tables: Mapping[str, pd.DataFrame],
    target_sr: arcpy.SpatialReference,
    buffer_dist_ft: float,
    use_shape_buffer: bool,
    out_fc: str,
) -> None:
    """Create a feature class with one buffered geometry per route_id.

    For each route_id, this function chooses how to construct the base
    geometry before buffering:

      * By default (use_shape_buffer=True), routes use their shapes.txt
        polylines where available.
      * Any route_id listed in EXPRESS_ROUTES is forced to use a
        stop-based MultiPoint geometry instead, regardless of the global
        use_shape_buffer flag.

    Args:
        tables: GTFS tables as pandas DataFrames.
        target_sr: Spatial reference for output geometries.
        buffer_dist_ft: Buffer distance in feet.
        use_shape_buffer: Default behaviour; if True routes use shapes
            unless overridden by EXPRESS_ROUTES.
        out_fc: Output feature class path (will be overwritten if exists).
    """
    shapes_df = tables["shapes"]
    trips_df = tables["trips"]
    stop_times_df = tables["stop_times"]
    stops_df = tables["stops"]

    required_shapes_cols = {
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
    }
    missing = required_shapes_cols.difference(shapes_df.columns)
    if missing:
        raise ValueError(
            f"shapes.txt missing required columns: {sorted(missing)}",
        )

    # Mapping from route_id -> list of shape_ids (may be empty for some routes).
    route_shapes = _get_route_shape_mapping(trips_df)

    # All route_ids present in trips.txt.
    all_route_ids = sorted(
        r for r in trips_df["route_id"].dropna().unique().tolist()
    )

    buffer_dist_units = _compute_buffer_distance_units(target_sr, buffer_dist_ft)
    wgs84_sr = arcpy.SpatialReference(4326)

    if arcpy.Exists(out_fc):
        _add_message(f"Deleting existing feature class {out_fc}", "INFO")
        arcpy.management.Delete(out_fc)

    out_path, out_name = str(Path(out_fc).parent), Path(out_fc).name
    arcpy.management.CreateFeatureclass(
        out_path,
        out_name,
        "POLYGON",
        spatial_reference=target_sr,
    )
    arcpy.management.AddField(out_fc, "route_id", "TEXT", field_length=64)

    express_set = set(EXPRESS_ROUTES)
    insert_fields = ["route_id", "SHAPE@"]

    with arcpy.da.InsertCursor(out_fc, insert_fields) as icur:
        for route_id in all_route_ids:
            if ROUTE_FILTER and route_id not in ROUTE_FILTER:
                continue

            shape_ids = route_shapes.get(route_id, [])

            # Decide geometry source for this route.
            #  - Express routes (in EXPRESS_ROUTES) use stops-based geometry.
            #  - Non-express routes use shapes if allowed and shapes exist.
            #  - If shapes are not available, fall back to stops.
            use_shapes_for_route = (
                use_shape_buffer
                and route_id not in express_set
                and bool(shape_ids)
            )

            mode = "shapes" if use_shapes_for_route else "stops"

            try:
                if use_shapes_for_route:
                    route_geom = _build_route_geometry_from_shapes(
                        route_id=route_id,
                        shape_ids=shape_ids,
                        shapes_df=shapes_df,
                        target_sr=target_sr,
                        wgs84_sr=wgs84_sr,
                    )
                else:
                    route_geom = _build_route_geometry_from_stops(
                        route_id=route_id,
                        trips_df=trips_df,
                        stop_times_df=stop_times_df,
                        stops_df=stops_df,
                        target_sr=target_sr,
                        wgs84_sr=wgs84_sr,
                    )
            except Exception as exc:
                _add_message(
                    f"Failed to build {mode}-based geometry for route {route_id}: {exc}",
                    "WARNING",
                )
                continue

            if route_geom is None:
                _add_message(
                    f"No {mode}-based geometry for route {route_id} – skipped",
                    "WARNING",
                )
                continue

            try:
                buf_geom = route_geom.buffer(buffer_dist_units)
            except Exception as exc:
                _add_message(
                    f"Buffering failed for route {route_id}: {exc}",
                    "WARNING",
                )
                continue

            icur.insertRow((route_id, buf_geom))
            _add_message(
                f"Buffered route {route_id} using {mode}-based geometry",
                "INFO",
            )

    _add_message(f"Route buffers written to {out_fc}", "INFO")


def _build_route_geometry_from_shapes(
    route_id: str,
    shape_ids: List[str],
    shapes_df: pd.DataFrame,
    target_sr: arcpy.SpatialReference,
    wgs84_sr: arcpy.SpatialReference,
) -> Optional[arcpy.Geometry]:
    """Construct a merged Polyline geometry from shapes for a route."""
    polylines: List[arcpy.Polyline] = []

    for sid in shape_ids:
        seg = shapes_df[shapes_df["shape_id"] == sid].copy()
        if seg.empty:
            continue
        seg.sort_values("shape_pt_sequence", inplace=True)

        pts = [
            arcpy.Point(lon, lat)
            for lon, lat in zip(seg["shape_pt_lon"], seg["shape_pt_lat"])
        ]
        if len(pts) < 2:
            continue

        array = arcpy.Array(pts)
        pl = arcpy.Polyline(array, wgs84_sr)
        pl_proj = pl.projectAs(target_sr)
        polylines.append(pl_proj)

    if not polylines:
        _add_message(
            f"No valid shape geometries for route {route_id}",
            "WARNING",
        )
        return None

    geom_union: arcpy.Geometry = polylines[0]
    for pl in polylines[1:]:
        geom_union = geom_union.union(pl)

    return geom_union


def _build_route_geometry_from_stops(
    route_id: str,
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    target_sr: arcpy.SpatialReference,
    wgs84_sr: arcpy.SpatialReference,
) -> Optional[arcpy.Geometry]:
    """Construct a MultiPoint geometry from all stops used by a route."""
    trip_ids = trips_df.loc[trips_df["route_id"] == route_id, "trip_id"].unique()
    if len(trip_ids) == 0:
        _add_message(f"No trips found for route {route_id}", "WARNING")
        return None

    rt_stop_ids = (
        stop_times_df[stop_times_df["trip_id"].isin(trip_ids)]["stop_id"]
        .dropna()
        .unique()
    )
    if len(rt_stop_ids) == 0:
        _add_message(f"No stops found for route {route_id}", "WARNING")
        return None

    stops_subset = stops_df[stops_df["stop_id"].isin(rt_stop_ids)]
    if stops_subset.empty:
        _add_message(
            f"No matching stops in stops.txt for route {route_id}",
            "WARNING",
        )
        return None

    points: List[arcpy.Point] = []
    for _, row in stops_subset.iterrows():
        try:
            lon = float(row["stop_lon"])
            lat = float(row["stop_lat"])
        except Exception:
            continue
        points.append(arcpy.Point(lon, lat))

    if not points:
        _add_message(
            f"No valid stop coordinates for route {route_id}",
            "WARNING",
        )
        return None

    multi = arcpy.Multipoint(arcpy.Array(points), wgs84_sr)
    multi_proj = multi.projectAs(target_sr)
    return multi_proj


# =============================================================================
# FEATURE COUNTING
# =============================================================================


def _count_features(
    route_buffers_fc: str,
    layers: Mapping[str, Dict[str, str]],
    layer_specs: Iterable[Tuple[str, str]],
    target_sr: arcpy.SpatialReference,
    output_dir: Path,
) -> pd.DataFrame:
    """For each route, count intersecting features and write per-route CSV.

    Args:
        route_buffers_fc: Feature class containing buffers with field 'route_id'.
        layers: Mapping of filename -> {"path": <fc path>, "id_field": <field name>}.
        layer_specs: Layer spec list defining output order.
        target_sr: Spatial reference to which geometries will be projected.
        output_dir: Directory where CSVs will be written.

    Returns:
        Summary DataFrame indexed by route_id with feature counts.
    """
    summary_records: List[Dict[str, object]] = []

    with arcpy.da.SearchCursor(
        route_buffers_fc,
        ["route_id", "SHAPE@"],
        spatial_reference=target_sr,
    ) as rcur:
        for route_id, buf_geom in rcur:
            if buf_geom is None:
                continue

            per_route_counts: Dict[str, object] = {"route_id": route_id}
            feature_name_lists: Dict[str, List[str]] = {}

            for filename, _id_col in layer_specs:
                if filename not in layers:
                    continue

                layer_info = layers[filename]
                layer_path = layer_info["path"]
                id_field = layer_info["id_field"]

                names: List[str] = []
                count = 0

                with arcpy.da.SearchCursor(
                    layer_path,
                    [id_field, "SHAPE@"],
                    spatial_reference=target_sr,
                ) as lcur:
                    for id_val, geom in lcur:
                        if geom is None or buf_geom is None:
                            continue
                        try:
                            if not geom.disjoint(buf_geom):
                                count += 1
                                names.append(str(id_val))
                        except Exception as exc:
                            _add_message(
                                f"Geometry error in {filename} for route {route_id}: {exc}",
                                "WARNING",
                            )

                per_route_counts[filename] = count
                feature_name_lists[filename] = names

            csv_rows = [
                {
                    "layer": fname,
                    "count": int(per_route_counts.get(fname, 0)),
                    "names": ", ".join(feature_name_lists.get(fname, [])),
                }
                for fname, _ in layer_specs
                if fname in layers
            ]
            if csv_rows:
                per_route_path = output_dir / f"{route_id}_feature_summary.csv"
                pd.DataFrame(csv_rows).to_csv(per_route_path, index=False)
                _add_message(
                    f"Per-route CSV written: {per_route_path}",
                    "INFO",
                )

            summary_records.append(per_route_counts)

    if not summary_records:
        _add_message("No feature counts produced.", "WARNING")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_records).set_index("route_id")
    summary_df = summary_df.fillna(0).astype(int)
    return summary_df


def _compute_route_transfer_tables(
    tables: Mapping[str, pd.DataFrame],
    distance_ft: float,
    time_window_min: float,
    subject_routes: Optional[Iterable[str]],
    output_dir: Path,
) -> None:
    """Compute inter-route transfer options and write CSV files.

    A "transfer opportunity" is any pair of trips (route A, route B) where:

      * The two trips serve the same stop, or two stops within distance_ft
        feet of each other; and
      * The absolute time difference between the events at those stops is
        <= time_window_min minutes; and
      * route_id_B != route_id_A.

    For each subject route, a CSV is written with one row per other route
    that it can transfer to, with counts and basic distance/time stats.

    In addition, an all-routes summary CSV named
    ``all_routes_transfer_summary.csv`` is written, containing one row
    per (from_route_id, transfer_route_id) pair across all subject routes.
    """
    stop_times = tables["stop_times"].copy()
    trips = tables["trips"][["trip_id", "route_id"]].copy()
    stops = tables["stops"][["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    routes = tables["routes"][
        ["route_id", "route_short_name", "route_long_name"]
    ].copy()

    # Parse GTFS times into seconds-from-midnight.
    stop_times["arrival_sec"] = stop_times["arrival_time"].apply(
        _parse_gtfs_time_to_seconds,
    )
    stop_times["departure_sec"] = stop_times["departure_time"].apply(
        _parse_gtfs_time_to_seconds,
    )
    stop_times["event_sec"] = stop_times["arrival_sec"].fillna(
        stop_times["departure_sec"],
    )
    stop_times = stop_times.dropna(subset=["event_sec"])
    stop_times["event_sec"] = stop_times["event_sec"].astype(int)

    # Attach route_id and stop metadata.
    events = stop_times.merge(trips, on="trip_id", how="inner").merge(
        stops,
        on="stop_id",
        how="left",
    )

    if events.empty:
        _add_message(
            "No stop-time events after merging with trips and stops – "
            "skipping transfer analysis.",
            "WARNING",
        )
        return

    all_routes = sorted(events["route_id"].dropna().unique().tolist())
    if subject_routes:
        subjects = [r for r in subject_routes if r in all_routes]
    else:
        subjects = all_routes

    if not subjects:
        _add_message(
            "No subject routes found in stop_times/trips – skipping transfers.",
            "WARNING",
        )
        return

    _add_message(
        f"Computing candidate nearby stop pairs within {distance_ft:.1f} ft.",
        "INFO",
    )
    near_pairs = _build_nearby_stop_pairs(stops, distance_ft)
    if near_pairs.empty:
        near_pairs_long = pd.DataFrame(
            columns=["from_stop_id", "to_stop_id", "distance_ft"],
        )
        _add_message(
            "No distinct stop pairs found within transfer distance – "
            "only same-stop transfers will be considered.",
            "INFO",
        )
    else:
        # Make pairs directional so we can join in either direction.
        near_pairs_long = pd.concat(
            [
                near_pairs.rename(
                    columns={"stop_id_a": "from_stop_id", "stop_id_b": "to_stop_id"},
                ),
                near_pairs.rename(
                    columns={"stop_id_b": "from_stop_id", "stop_id_a": "to_stop_id"},
                ),
            ],
            ignore_index=True,
        )

    time_window_sec = int(time_window_min * 60.0)

    # Collect per-route aggregates so we can write a global summary after the loop.
    summary_frames: List[pd.DataFrame] = []

    for route_id in subjects:
        ev_from = events[events["route_id"] == route_id].copy()
        if ev_from.empty:
            _add_message(
                f"No stop_time events for subject route {route_id} – skipped.",
                "WARNING",
            )
            continue

        # ------------------------------------------------------------------
        # Same-stop transfers (distance = 0).
        # ------------------------------------------------------------------
        same = ev_from.merge(
            events,
            on="stop_id",
            suffixes=("_from", "_to"),
        )
        same = same[same["route_id_to"] != route_id]

        if not same.empty:
            same["time_diff_sec"] = (same["event_sec_to"] - same["event_sec_from"]).abs()
            same = same[same["time_diff_sec"] <= time_window_sec]
            if not same.empty:
                same["distance_ft"] = 0.0

        # ------------------------------------------------------------------
        # Nearby-stop transfers (distinct stops within distance_ft).
        # ------------------------------------------------------------------
        if near_pairs_long.empty:
            near = pd.DataFrame()
        else:
            tmp = ev_from.merge(
                near_pairs_long,
                left_on="stop_id",
                right_on="from_stop_id",
                how="left",
            )
            tmp = tmp.dropna(subset=["to_stop_id"])
            tmp = tmp.merge(
                events,
                left_on="to_stop_id",
                right_on="stop_id",
                how="inner",
                suffixes=("_from", "_to"),
            )
            near = tmp[tmp["route_id_to"] != route_id]

            if not near.empty:
                near["time_diff_sec"] = (
                    near["event_sec_to"] - near["event_sec_from"]
                ).abs()
                near = near[near["time_diff_sec"] <= time_window_sec]

        frames: List[pd.DataFrame] = []
        if not same.empty:
            frames.append(same[["route_id_to", "time_diff_sec", "distance_ft"]])
        if not near.empty:
            frames.append(near[["route_id_to", "time_diff_sec", "distance_ft"]])

        if not frames:
            _add_message(
                f"No transfer opportunities found for route {route_id}.",
                "INFO",
            )
            continue

        all_tf = pd.concat(frames, ignore_index=True)

        agg = (
            all_tf.groupby("route_id_to")
            .agg(
                num_transfer_events=("route_id_to", "size"),
                min_distance_ft=("distance_ft", "min"),
                max_distance_ft=("distance_ft", "max"),
                min_time_diff_min=("time_diff_sec", lambda s: s.min() / 60.0),
                max_time_diff_min=("time_diff_sec", lambda s: s.max() / 60.0),
            )
            .reset_index()
            .rename(columns={"route_id_to": "transfer_route_id"})
        )

        # Attach route names for readability (for the *transfer* route).
        agg = agg.merge(
            routes,
            left_on="transfer_route_id",
            right_on="route_id",
            how="left",
        ).drop(columns=["route_id"])

        # Optional rounding for readability.
        agg["min_distance_ft"] = agg["min_distance_ft"].round(1)
        agg["max_distance_ft"] = agg["max_distance_ft"].round(1)
        agg["min_time_diff_min"] = agg["min_time_diff_min"].round(1)
        agg["max_time_diff_min"] = agg["max_time_diff_min"].round(1)

        # Persist per-route CSV exactly as before.
        out_path = output_dir / f"{route_id}_transfer_routes.csv"
        agg.to_csv(out_path, index=False)
        _add_message(
            f"Transfer routes CSV written for {route_id}: {out_path}",
            "INFO",
        )

        # Keep a copy with from_route_id for the all-routes summary.
        agg_for_summary = agg.copy()
        agg_for_summary.insert(0, "from_route_id", route_id)
        summary_frames.append(agg_for_summary)

    # ----------------------------------------------------------------------
    # Systemwide summary: one row per (from_route_id, transfer_route_id).
    # ----------------------------------------------------------------------
    if not summary_frames:
        _add_message(
            "No transfer opportunities found for any subject route – "
            "no all-routes transfer summary written.",
            "INFO",
        )
        return

    all_transfers = pd.concat(summary_frames, ignore_index=True)

    # Clarify that these name columns are for the transfer (to) route.
    all_transfers = all_transfers.rename(
        columns={
            "route_short_name": "to_route_short_name",
            "route_long_name": "to_route_long_name",
        },
    )

    # Attach from-route names.
    routes_from = routes.rename(
        columns={
            "route_id": "from_route_id",
            "route_short_name": "from_route_short_name",
            "route_long_name": "from_route_long_name",
        },
    )
    all_transfers = all_transfers.merge(routes_from, on="from_route_id", how="left")

    # Sort for readability.
    all_transfers = all_transfers.sort_values(
        ["from_route_id", "num_transfer_events", "transfer_route_id"],
        ascending=[True, False, True],
    )

    summary_path = output_dir / "all_routes_transfer_summary.csv"
    all_transfers.to_csv(summary_path, index=False)
    _add_message(
        f"All-routes transfer summary written to {summary_path}",
        "INFO",
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the GTFS feature-coverage analysis using ArcPy geometries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdb_path = OUTPUT_DIR / GDB_NAME

    if not gdb_path.exists():
        _add_message(f"Creating file geodatabase at {gdb_path}", "INFO")
        arcpy.management.CreateFileGDB(str(OUTPUT_DIR), GDB_NAME)
    else:
        _add_message(f"Using existing file geodatabase {gdb_path}", "INFO")

    route_buffers_fc = str(gdb_path / "route_buffers")

    _add_message(f"Loading GTFS from {GTFS_DIR}", "INFO")
    tables = _load_gtfs_tables(GTFS_DIR)

    _add_message("Loading designated shapefile definitions", "INFO")
    layers, target_sr = _load_layers(LAYER_SPECS, SHP_INPUT_DIR)

    if not layers:
        _add_message("No valid layers loaded – nothing to analyze", "ERROR")
        return

    if not target_sr or target_sr.name == "Unknown":
        _add_message(
            "Could not determine a valid analysis spatial reference from layers. "
            "Please ensure at least one layer has a defined projected CRS.",
            "ERROR",
        )
        return

    layers = _maybe_replace_facility_layers_with_parcel_buffers(
        layers=layers,
        layer_specs=LAYER_SPECS,
        parcels_path=PARCELS_SHP,
        gdb_path=gdb_path,
        target_sr=target_sr,
        buffer_dist_ft=BUFFER_DIST_FT,
    )

    _add_message(
        f"Building route buffers in CRS '{target_sr.name}' "
        f"(use_shape_buffer={USE_SHAPE_BUFFER})",
        "INFO",
    )
    _build_route_buffers_fc(
        tables=tables,
        target_sr=target_sr,
        buffer_dist_ft=BUFFER_DIST_FT,
        use_shape_buffer=USE_SHAPE_BUFFER,
        out_fc=route_buffers_fc,
    )

    if not arcpy.Exists(route_buffers_fc):
        _add_message("Route buffers feature class not created – nothing to do", "ERROR")
        return

    _add_message("Counting features per route", "INFO")
    summary_df = _count_features(
        route_buffers_fc=route_buffers_fc,
        layers=layers,
        layer_specs=LAYER_SPECS,
        target_sr=target_sr,
        output_dir=OUTPUT_DIR,
    )

    if summary_df.empty:
        _add_message("No summary produced – check logs for errors/warnings.", "ERROR")
        return

    summary_path = OUTPUT_DIR / "all_routes_feature_summary.csv"
    summary_df.to_csv(summary_path)
    _add_message(f"Summary written to {summary_path}", "INFO")

    # -------------------------------------------------------------------------
    # Optional: per-route transfer network CSVs based on GTFS stop_times.
    # -------------------------------------------------------------------------
    if EXPORT_ROUTE_TRANSFERS:
        _add_message(
            "Computing per-route transfer opportunities (same/nearby stops).",
            "INFO",
        )
        try:
            _compute_route_transfer_tables(
                tables=tables,
                distance_ft=TRANSFER_DISTANCE_FT,
                time_window_min=TRANSFER_TIME_MIN,
                subject_routes=ROUTE_FILTER or None,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            _add_message(
                f"Route transfer analysis failed: {exc}",
                "WARNING",
            )

    _add_message("Done.", "INFO")


if __name__ == "__main__":
    main()
