"""Service-area demographics from transit stops.

Builds an ArcPy pipeline that buffers stops, dissolves buffers, clips ACS
demographic polygons, and computes area-weighted counts (households, population,
jobs). Supports network-wide runs from a stops shapefile or GTFS, and optional
per-route summaries from GTFS.

Configuration:
  - Set absolute input/output paths in CONFIGURATION.
  - Choose STOPS_INPUT_MODE: {"shapefile", "gtfs"}.
  - Optionally filter GTFS via GTFS_ROUTE_SHORT_NAMES.

Requires:
  - ArcGIS Pro with arcpy; GTFS stop coordinates assumed WGS84.
"""


from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arcpy
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION (ABSOLUTE PATHS ONLY)
# =============================================================================
# Overwrite behavior
OVERWRITE_OUTPUTS: bool = True

# --- Stops input mode ---
# Choose one: "shapefile" or "gtfs"
STOPS_INPUT_MODE: str = "shapefile"

# For "shapefile" mode
STOPS_FEATURE_CLASS: str = (
    r"Path\To\Your\bus_route.shp"
)

# For "gtfs" mode
GTFS_FOLDER: str = r"Folder\Path\To\Your\GTFS"

# Optional: filter GTFS to the following route_short_name values.
# Example: ["101", "202"]. Leave as [] or None to include all routes (network run).
GTFS_ROUTE_SHORT_NAMES: Optional[Sequence[str]] = [ ]

# Inputs (absolute paths)
DEMOGRAPHICS_SHP: str = (
    r"Path\To\Your\census_demographics.shp"
)

# -----------------------------------------------------------------------------
# DEMOGRAPHICS SOURCE FIELD MAP
# -----------------------------------------------------------------------------
# Map the clipped ACS layer columns to the semantic inputs the pipeline expects.
# Leave a value as None if the source field does not exist in your dataset.
DEMOG_SRC_FIELDS: dict[str, Optional[str]] = {
    # Households
    "loinc_hh_src": "Lowincome_",  # low-income households
    "total_hh_src": "TotHH",       # total households

    # Population
    "minor_pop_src": "Minority",   # minority population
    "total_pop_src": "Tot_Pop",    # total population

    # Jobs (ACS layer has none; keep None to emit zeros)
    "loinc_jobs_src": None,        # low-wage jobs
    "all_jobs_src": None,          # total jobs
}

# Outputs (absolute paths) — used by the network run path
BUFFERED_STOPS_SHP: str =       r"Path\To\Your\buffered_stops.shp"
DISSOLVED_BUFFERS_SHP: str =    r"Path\To\Your\dissolved_buffers.shp"
CLIPPED_DEMOGRAPHICS_SHP: str = r"Path\To\Your\clipped_demographics.shp"
FINAL_EXPORT_DIR: str =         r"Path\To\Your\output_demogs.shp"

# Processing parameters
BUFFER_DISTANCE_MILES: float = 0.25
DISSOLVE_FIELD_NAME: str = "dissolve"

# Run label used for final export naming and console prints
RUN_TAG: str = "tsp_service"  # e.g., "weekday_2024q3"

# -----------------------------------------------------------------------------
# OUTPUT MODE
# -----------------------------------------------------------------------------
# One of: "network" | "by_route" | "both"
OUTPUT_MODE: str = "both"

# Per-route runs are supported only when STOPS_INPUT_MODE == "gtfs".
# When True, also write a CSV of per-route results to FINAL_EXPORT_DIR.
BY_ROUTE_WRITE_CSV: bool = True

# Optional: export per-route clipped polygons as shapefiles.
# File names will be {RUN_TAG}_route_{route_short_name}_service_buffer_data.shp
BY_ROUTE_EXPORT_FEATURES: bool = False

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------
# METRIC FIELDS (single source of truth)
# -----------------------------------------------------------------------------
CLIPPED_FIELDS: List[str] = [
    "loinc_hh",
    "total_hh",
    "minor_pop",
    "total_pop",
    "loinc_jobs",
    "all_jobs",
]

# =============================================================================
# REUSABLE FUNCTIONS
# =============================================================================

def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
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
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# UTILITIES
# =============================================================================
def _as_distance_miles(miles: float) -> str:
    """Return an ArcPy-friendly distance string in miles."""
    return f"{miles} Miles"


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def field_exists(feature_class: str, field_name: str) -> bool:
    """Return True if a field exists in the feature class."""
    return any(f.name.lower() == field_name.lower() for f in arcpy.ListFields(feature_class))


def safe_add_field(
    feature_class: str,
    field_name: str,
    field_type: str,
    **kwargs: Any,
) -> None:
    """Add a field if it does not already exist."""
    if not field_exists(feature_class, field_name):
        arcpy.management.AddField(feature_class, field_name, field_type, **kwargs)


def calc_field(feature_class: str, field_name: str, expression: str) -> None:
    """Calculate a field using PYTHON3 expression type."""
    arcpy.management.CalculateField(
        feature_class,
        field_name,
        expression,
        expression_type="PYTHON3",
    )


def _sanitize_name(token: str) -> str:
    """Return a safe token for in_memory names and filenames."""
    return "".join(ch for ch in str(token) if ch.isalnum() or ch in ("_", "-"))[:40]


def _route_scoped_temp(name: str, scope: str) -> str:
    """Build an in_memory path unique per scope (e.g., route short name)."""
    safe = _sanitize_name(scope)
    return rf"in_memory\{name}_{safe}"


# =============================================================================
# GTFS → FEATURE LAYER
# =============================================================================
def _filter_gtfs_stops_by_route_short_name(
    gtfs: dict[str, pd.DataFrame],
    route_short_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Return stops DF filtered to those used by routes with given short names.

    Args:
        gtfs: Mapping of GTFS table name to DataFrame (from load_gtfs_data()).
        route_short_names: Optional list/sequence of route_short_name values to keep.

    Returns:
        DataFrame of filtered stops (subset of gtfs["stops"]).
    """
    stops = gtfs["stops"]
    if not route_short_names:
        return stops.copy()

    # Normalize filter to strings for robust matching.
    target = {str(x) for x in route_short_names}

    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]

    # route_ids where route_short_name is in target
    routes_sel = routes[routes["route_short_name"].astype(str).isin(target)]
    if routes_sel.empty:
        raise ValueError(
            "No routes matched the provided route_short_name filter: "
            f"{sorted(target)}"
        )

    route_ids = set(routes_sel["route_id"].astype(str))
    trips_sel = trips[trips["route_id"].astype(str).isin(route_ids)]
    if trips_sel.empty:
        raise ValueError(
            "Routes matched, but no trips found for the selected routes. "
            "Check GTFS integrity."
        )

    trip_ids = set(trips_sel["trip_id"].astype(str))
    st_sel = stop_times[stop_times["trip_id"].astype(str).isin(trip_ids)]
    if st_sel.empty:
        raise ValueError(
            "Trips matched, but no stop_times rows found. Check GTFS integrity."
        )

    stop_ids = set(st_sel["stop_id"].astype(str))
    out = stops[stops["stop_id"].astype(str).isin(stop_ids)].copy()
    if out.empty:
        raise ValueError(
            "Filtering produced zero stops. Verify stop_times and stops tables."
        )
    return out


def _points_layer_from_stops_df(stops_df: pd.DataFrame, layer_name: str = "gtfs_stops") -> str:
    """Create an in-memory point feature layer from a GTFS stops DataFrame.

    Expects columns: stop_id, stop_name, stop_lat, stop_lon.
    Uses a strict NumPy structured array (no 'object' dtypes) so that
    arcpy.da.NumPyArrayToTable can consume it safely.
    """
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required.difference(stops_df.columns)
    if missing:
        raise ValueError(f"GTFS stops missing required columns: {sorted(missing)}")

    df = stops_df.loc[:, ["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()

    # Coerce coordinates to float; drop rows with invalid coords.
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    # Normalize text columns to strings and clamp to safe lengths.
    max_id_len = 64
    max_name_len = 255
    df["stop_id"] = df["stop_id"].astype(str).str.slice(0, max_id_len)
    df["stop_name"] = df["stop_name"].astype(str).str.slice(0, max_name_len)

    # Structured array (no 'object' dtype)
    dtype = np.dtype(
        [
            ("stop_id", f"<U{max_id_len}"),
            ("stop_name", f"<U{max_name_len}"),
            ("stop_lat", "<f8"),
            ("stop_lon", "<f8"),
        ]
    )
    rec = np.empty(len(df), dtype=dtype)
    rec["stop_id"] = df["stop_id"].to_numpy()
    rec["stop_name"] = df["stop_name"].to_numpy()
    rec["stop_lat"] = df["stop_lat"].to_numpy(dtype=np.float64)
    rec["stop_lon"] = df["stop_lon"].to_numpy(dtype=np.float64)

    # Write to in_memory, then convert XY to points.
    tbl_path = r"in_memory\gtfs_stops_tbl"
    pt_path = r"in_memory\gtfs_stops_points"

    if arcpy.Exists(tbl_path):
        arcpy.management.Delete(tbl_path)
    if arcpy.Exists(pt_path):
        arcpy.management.Delete(pt_path)

    arcpy.da.NumPyArrayToTable(rec, tbl_path)

    wgs84 = arcpy.SpatialReference(4326)
    arcpy.management.XYTableToPoint(
        tbl_path,
        pt_path,
        x_field="stop_lon",
        y_field="stop_lat",
        coordinate_system=wgs84,
    )

    lyr_result = arcpy.management.MakeFeatureLayer(pt_path, layer_name)
    return str(lyr_result)


def make_stops_layer(
    mode: str,
    shapefile_fc: Optional[str] = None,
    gtfs_folder: Optional[str] = None,
    route_short_names: Optional[Sequence[str]] = None,
    layer_name: str = "stops",
) -> str:
    """Create a feature layer of stops for downstream processing.

    Args:
        mode: One of {"shapefile", "gtfs"}.
        shapefile_fc: Absolute path to stops feature class when mode == "shapefile".
        gtfs_folder: Absolute path to GTFS folder when mode == "gtfs".
        route_short_names: Optional filter for GTFS by route_short_name.
        layer_name: Output layer name.

    Returns:
        An ArcPy feature layer ready to use.

    Raises:
        FileNotFoundError: When the specified input does not exist.
        ValueError: When configuration is inconsistent or GTFS filtering fails.
    """
    m = mode.strip().lower()
    if m not in {"shapefile", "gtfs"}:
        raise ValueError("STOPS_INPUT_MODE must be 'shapefile' or 'gtfs'.")

    if m == "shapefile":
        if not shapefile_fc or not arcpy.Exists(shapefile_fc):
            raise FileNotFoundError(f"Stops feature class not found: {shapefile_fc}")
        return str(arcpy.management.MakeFeatureLayer(shapefile_fc, layer_name))

    # GTFS mode
    if not gtfs_folder or not os.path.exists(gtfs_folder):
        raise FileNotFoundError(f"GTFS folder not found: {gtfs_folder}")

    # Load required GTFS tables using provided helper
    gtfs = load_gtfs_data(
        gtfs_folder,
        files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt"),
        dtype=str,
    )
    stops_df = _filter_gtfs_stops_by_route_short_name(gtfs, route_short_names)
    return _points_layer_from_stops_df(stops_df, layer_name=layer_name)


# =============================================================================
# PIPELINE STEPS
# =============================================================================
def buffer_stops(stops_layer: str, out_path: str, miles: float) -> str:
    """Buffer stops by the given distance in miles."""
    distance = _as_distance_miles(miles)
    result = arcpy.analysis.Buffer(stops_layer, out_path, distance)
    return str(result)


def dissolve_buffers(
    buffer_fc: str,
    out_path: str,
    dissolve_field: Optional[str] = None,
) -> str:
    """Dissolve buffered polygons.

    If ``dissolve_field`` is falsy, dissolve ALL features (no field).
    This avoids shapefile schema edits and lock issues.
    If a field is requested and adding it fails, we fall back to dissolve-all.

    Args:
        buffer_fc: Path to the buffer feature class.
        out_path: Output feature class path (e.g., .shp).
        dissolve_field: Optional name of a field to dissolve on. If None/"" → all.

    Returns:
        Path to the dissolved feature class.
    """
    # If caller doesn't insist on a field, do an "all-in-one" dissolve.
    if not dissolve_field:
        result = arcpy.management.Dissolve(buffer_fc, out_path, dissolve_field="")
        return str(result)

    # Try the field-based dissolve; if schema edit fails, fall back gracefully.
    try:
        if not field_exists(buffer_fc, dissolve_field):
            arcpy.management.AddField(buffer_fc, dissolve_field, "SHORT")
        calc_field(buffer_fc, dissolve_field, "1")
        result = arcpy.management.Dissolve(buffer_fc, out_path, dissolve_field)
        return str(result)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "Field-based dissolve failed on '%s' (%s). Falling back to dissolve-all.",
            dissolve_field,
            exc,
        )
        result = arcpy.management.Dissolve(buffer_fc, out_path, dissolve_field="")
        return str(result)


def add_original_area_acres(demographics_fc: str, field_name: str = "area_ac_og") -> None:
    """Add original area (acres) to demographic polygons."""
    safe_add_field(demographics_fc, field_name, "DOUBLE")
    # area in square meters / 4046.86 => acres
    calc_field(demographics_fc, field_name, "!shape.area@SQUAREMETERS! / 4046.86")


def clip_demographics_to_buffers(
    demographics_fc: str,
    buffers_fc: str,
    out_path: str,
) -> str:
    """Clip demographic polygons to the dissolved service buffer."""
    result = arcpy.analysis.Clip(demographics_fc, buffers_fc, out_path)
    return str(result)


def add_clipped_area_and_percentage(
    clipped_fc: str,
    area_field: str = "area_ac_cl",
    pct_field: str = "area_perc",
    original_area_field: str = "area_ac_og",
) -> None:
    """Add clipped area (acres) and area percentage vs. original."""
    safe_add_field(clipped_fc, area_field, "DOUBLE")
    calc_field(clipped_fc, area_field, "!shape.area@SQUAREMETERS! / 4046.86")

    safe_add_field(clipped_fc, pct_field, "DOUBLE")
    calc_field(clipped_fc, pct_field, f"!{area_field}! / !{original_area_field}!")


def print_fields(feature_class: str) -> None:
    """List and print field names and types for a feature class."""
    print(f"Fields in {feature_class}:")
    for f in arcpy.ListFields(feature_class):
        print(f"  - {f.name} ({f.type})")


def add_synthetic_fields(
    clipped_fc: str,
    area_pct_field: str = "area_perc",
    src_map: Mapping[str, Optional[str]] = DEMOG_SRC_FIELDS,
) -> None:
    """Create synthetic counts based on area fraction and declared source fields.

    The function expects the clipped demographics to contain:
      - A fractional area field (default: 'area_perc').
      - Source columns named in ``src_map`` when present.

    Any metric whose source field is absent (None) will be emitted as zeros
    so downstream schemas remain stable.

    Args:
        clipped_fc: Path to the clipped ACS feature class.
        area_pct_field: Name of the fractional area field in the clipped features.
        src_map: Mapping from semantic inputs to source field names (or None).

    Raises:
        ValueError: If the required area fraction field is missing.
    """
    # Remove legacy collision if present
    if field_exists(clipped_fc, "minrty_pop"):
        arcpy.management.DeleteField(clipped_fc, ["minrty_pop"])

    # Validate area fraction
    if not field_exists(clipped_fc, area_pct_field):
        raise ValueError(
            f"Missing '{area_pct_field}' on {clipped_fc}. "
            "Run add_clipped_area_and_percentage() first."
        )

    # Build list of source fields that are actually present
    # Each entry is (semantic_key, source_field_or_None)
    sources = [
        ("loinc_hh_src", src_map.get("loinc_hh_src")),
        ("total_hh_src", src_map.get("total_hh_src")),
        ("minor_pop_src", src_map.get("minor_pop_src")),
        ("total_pop_src", src_map.get("total_pop_src")),
        ("loinc_jobs_src", src_map.get("loinc_jobs_src")),  # optional in ACS
        ("all_jobs_src", src_map.get("all_jobs_src")),      # optional in ACS
    ]

    # Outputs to guarantee (stable schema)
    out_specs = [
        ("loinc_hh",  "DOUBLE"),
        ("total_hh",  "DOUBLE"),
        ("minor_pop", "DOUBLE"),
        ("total_pop", "DOUBLE"),
        ("loinc_jobs","DOUBLE"),
        ("all_jobs",  "DOUBLE"),
    ]
    for fname, ftype in out_specs:
        safe_add_field(clipped_fc, fname, ftype)

    # Helper for numeric conversion
    def _f(x: Any) -> float:
        if x in (None, "", " "):
            return 0.0
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    # Cursor fields: area fraction + present sources + outputs
    src_field_names: List[Optional[str]] = [s[1] for s in sources]
    present_src_fields: List[str] = [f for f in src_field_names if f and field_exists(clipped_fc, f)]
    missing_required = [
        name for name, f in [
            ("loinc_hh_src", src_map.get("loinc_hh_src")),
            ("total_hh_src", src_map.get("total_hh_src")),
            ("minor_pop_src", src_map.get("minor_pop_src")),
            ("total_pop_src", src_map.get("total_pop_src")),
        ]
        if f and not field_exists(clipped_fc, f)
    ]
    if missing_required:
        raise ValueError(
            "Synthetic field calculation aborted. Missing expected ACS input columns on "
            f"{clipped_fc}: {', '.join(missing_required)}"
        )

    cursor_fields = [area_pct_field] + [f for f in src_field_names if f] + [p[0] for p in out_specs]
    # Indices to pull safely
    n_sources = len(present_src_fields)

    with arcpy.da.UpdateCursor(clipped_fc, cursor_fields) as cur:
        for row in cur:
            a = _f(row[0])

            # Build a dict from present source names → values
            src_vals: dict[str, float] = {}
            for i, f in enumerate(present_src_fields, start=1):
                src_vals[f] = _f(row[i])

            # Resolve each output's source and compute
            def v(src_key: str) -> float:
                src_name = src_map.get(src_key)
                if not src_name:  # no source declared → zero metric
                    return 0.0
                return src_vals.get(src_name, 0.0)

            # Write outputs at the tail positions
            base = 1 + n_sources  # first output index in 'row'
            row[base + 0] = a * v("loinc_hh_src")
            row[base + 1] = a * v("total_hh_src")
            row[base + 2] = a * v("minor_pop_src")
            row[base + 3] = a * v("total_pop_src")
            row[base + 4] = a * v("loinc_jobs_src")  # zeros for ACS
            row[base + 5] = a * v("all_jobs_src")    # zeros for ACS
            cur.updateRow(row)


def summarize_fields(feature_class: str, fields: Iterable[str]) -> Dict[str, int]:
    """Sum a set of numeric fields and return rounded integers."""
    totals = {f: 0.0 for f in fields}
    with arcpy.da.SearchCursor(feature_class, list(fields)) as cur:
        for row in cur:
            for i, f in enumerate(fields):
                val = row[i]
                if val is not None:
                    totals[f] += float(val)
    return {k: int(round(v)) for k, v in totals.items()}


def export_final_copy(
    in_feature_class: str,
    out_dir: str,
    run_tag: str,
) -> str:
    """Copy features to a final shapefile named with the run tag.

    Args:
      in_feature_class: Absolute path to the feature class to export.
      out_dir: Absolute path to the destination directory.
      run_tag: Tag to embed in the output filename.

    Returns:
      Absolute path to the exported shapefile.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{run_tag}_service_buffer_data.shp")
    arcpy.management.CopyFeatures(in_feature_class, out_path)
    return out_path


def summarize_census_layer(demographics_fc: str) -> Dict[str, int]:
    """Summarize raw totals over the full demographics layer for reference.

    Uses DEMOG_SRC_FIELDS to locate source columns. Missing metrics are returned as zero.
    """
    lyr = str(arcpy.management.MakeFeatureLayer(demographics_fc, "census_data"))

    # Map outputs → source field names
    src = DEMOG_SRC_FIELDS
    mapping = {
        "tt_hh_b": src.get("total_hh_src"),
        "est__60": src.get("loinc_hh_src"),
        "tt_pp_b": src.get("total_pop_src"),
        "est_mnr": src.get("minor_pop_src"),
        "tot_mpl": src.get("all_jobs_src"),   # may be None
        "low_wag": src.get("loinc_jobs_src"), # may be None
    }

    # Build the list of actual fields to read
    read_fields = [f for f in mapping.values() if f and field_exists(lyr, f)]
    if not read_fields:
        return {k: 0 for k in mapping.keys()}

    # Sum the present fields
    sums_present = summarize_fields(lyr, read_fields)

    # Re-express in the expected output keys, with zeros for missing
    out: Dict[str, int] = {}
    for out_key, src_field in mapping.items():
        if src_field and src_field in sums_present:
            out[out_key] = int(sums_present[src_field])
        else:
            out[out_key] = 0
    return out


# =============================================================================
# CONSOLIDATED EXECUTION HELPERS
# =============================================================================
def _process_service_area_from_stops_layer(
    stops_layer: str,
    run_tag: str,
    *,
    export_final: bool,
    final_export_dir: str,
    # Optional explicit disk paths; if provided, they will be used (network path).
    buffered_path: Optional[str] = None,
    dissolved_path: Optional[str] = None,
    clipped_path: Optional[str] = None,
) -> Tuple[Dict[str, int], Optional[str]]:
    """Run the buffer→dissolve→clip→synthetics→summaries pipeline for any stops layer.

    When disk paths are provided for buffered/dissolved/clipped, they are used.
    Otherwise, unique in_memory paths are allocated (per-route use case).

    Args:
        stops_layer: Feature layer of stop points to buffer.
        run_tag: Label for prints/filenames.
        export_final: If True, writes a final shapefile to final_export_dir.
        final_export_dir: Absolute directory for exports when export_final is True.
        buffered_path: Optional explicit output for buffered polygons.
        dissolved_path: Optional explicit output for dissolved polygons.
        clipped_path: Optional explicit output for clipped demographics.

    Returns:
        (totals_dict, exported_fc_path or None)
    """
    # Allocate unique ephemeral outputs when not explicitly provided
    if not buffered_path:
        buffered_path = _route_scoped_temp("buffered_stops", run_tag)
    if not dissolved_path:
        dissolved_path = _route_scoped_temp("dissolved_buffers", run_tag)
    if not clipped_path:
        clipped_path = _route_scoped_temp("clipped_demog", run_tag)

    # 2) Buffer
    logging.info("Buffering stops (%s)…", run_tag)
    buffered = buffer_stops(stops_layer, buffered_path, BUFFER_DISTANCE_MILES)

    # 3) Dissolve all features for this scope
    logging.info("Dissolving buffers (%s)…", run_tag)
    dissolved = dissolve_buffers(buffered, dissolved_path, dissolve_field=None)

    # 5) Clip demographics to dissolved
    logging.info("Clipping demographics (%s)…", run_tag)
    clipped = clip_demographics_to_buffers(DEMOGRAPHICS_SHP, dissolved, clipped_path)

    # 6–8) Derived area fields and synthetic metrics
    add_clipped_area_and_percentage(clipped)
    add_synthetic_fields(clipped)

    # 9) Summaries
    totals = summarize_fields(clipped, CLIPPED_FIELDS)

    # Optional export to disk
    exported_path: Optional[str] = None
    if export_final:
        exported_path = export_final_copy(clipped, final_export_dir, run_tag)

    return totals, exported_path


def _run_network_total(stops_layer: str) -> None:
    """Whole-network summary using the already-prepared stops layer.

    Uses in_memory for all intermediate edits to avoid shapefile field limits.
    Only the final, slimmed copy is exported as a shapefile in FINAL_EXPORT_DIR.
    """
    print(f"Buffering stops to {BUFFER_DISTANCE_MILES} mi → in_memory (intermediates)")

    # Do NOT pass explicit disk paths; this keeps buffer/dissolve/clip + field edits in_memory.
    svc_totals, exported = _process_service_area_from_stops_layer(
        stops_layer=stops_layer,
        run_tag=RUN_TAG,
        export_final=True,
        final_export_dir=FINAL_EXPORT_DIR,
        # buffered_path=None,
        # dissolved_path=None,
        # clipped_path=None,
    )

    print(f"[{RUN_TAG}] Service buffer totals (area-weighted, rounded):")
    for k in CLIPPED_FIELDS:
        print(f"  {k}: {svc_totals[k]:,}")

    print("Summarizing reference totals over the full demographics layer…")
    full_totals = summarize_census_layer(DEMOGRAPHICS_SHP)
    for k in ["tt_hh_b", "est__60", "tt_pp_b", "est_mnr", "tot_mpl", "low_wag"]:
        print(f"  {k}: {full_totals[k]:,}")

    if exported:
        print(f"Final shapefile exported: {exported}")


def _run_by_route(gtfs_folder: str, route_short_names: Sequence[str]) -> pd.DataFrame:
    """Per-route summaries (GTFS only). Returns a DataFrame of results.

    For each route_short_name, rebuilds its own stops layer from GTFS,
    runs the service-area pipeline (in_memory by default), and collects totals.
    """
    results: List[Dict[str, int | str]] = []
    for route_sn in route_short_names:
        tag = f"{RUN_TAG}_route_{_sanitize_name(route_sn)}"
        print(f"— Processing route {route_sn} —")

        # Build a route-scoped stops layer directly from GTFS
        stops_layer = make_stops_layer(
            mode="gtfs",
            gtfs_folder=gtfs_folder,
            route_short_names=[route_sn],
            layer_name=f"stops_{_sanitize_name(route_sn)}",
        )

        totals, exported = _process_service_area_from_stops_layer(
            stops_layer=stops_layer,
            run_tag=tag,
            export_final=BY_ROUTE_EXPORT_FEATURES,
            final_export_dir=FINAL_EXPORT_DIR,
        )

        row: Dict[str, int | str] = {"route_short_name": str(route_sn)}
        row.update({k: int(totals[k]) for k in CLIPPED_FIELDS})
        results.append(row)

        if BY_ROUTE_EXPORT_FEATURES and exported:
            print(f"  Exported per-route shapefile: {exported}")

    df = pd.DataFrame(results).sort_values("route_short_name").reset_index(drop=True)
    if BY_ROUTE_WRITE_CSV and not df.empty:
        ensure_dir(FINAL_EXPORT_DIR)
        csv_path = os.path.join(FINAL_EXPORT_DIR, f"{RUN_TAG}_by_route_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Per-route summary written: {csv_path}")

    # Also print a quick console view
    if not df.empty:
        print("\nPer-route totals (area-weighted, rounded):")
        cols = ["route_short_name"] + CLIPPED_FIELDS
        print(df[cols].to_string(index=False))

    return df


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Run the end-to-end pipeline according to OUTPUT_MODE."""
    arcpy.env.overwriteOutput = OVERWRITE_OUTPUTS

    # Basic input checks (mode-specific)
    m = STOPS_INPUT_MODE.lower()
    if m == "shapefile":
        if not arcpy.Exists(STOPS_FEATURE_CLASS):
            raise FileNotFoundError(f"Stops feature class not found: {STOPS_FEATURE_CLASS}")
    elif m == "gtfs":
        if not os.path.exists(GTFS_FOLDER):
            raise FileNotFoundError(f"GTFS folder not found: {GTFS_FOLDER}")
    else:
        raise ValueError("STOPS_INPUT_MODE must be 'shapefile' or 'gtfs'.")

    if not arcpy.Exists(DEMOGRAPHICS_SHP):
        raise FileNotFoundError(f"Input not found: {DEMOGRAPHICS_SHP}")

    # One-time prep on demographics
    print("Adding original area (acres) to demographics…")
    add_original_area_acres(DEMOGRAPHICS_SHP)

    # Build the base stops layer (network scope) once
    print("Preparing stops layer…")
    stops_layer = make_stops_layer(
        mode=STOPS_INPUT_MODE,
        shapefile_fc=STOPS_FEATURE_CLASS,
        gtfs_folder=GTFS_FOLDER,
        route_short_names=GTFS_ROUTE_SHORT_NAMES,
        layer_name="stops_for_buffering",
    )

    mode = OUTPUT_MODE.lower().strip()
    if mode not in {"network", "by_route", "both"}:
        raise ValueError("OUTPUT_MODE must be one of {'network','by_route','both'}.")

    run_by_route = mode in {"by_route", "both"}
    run_network = mode in {"network", "both"}

    if run_network:
        _run_network_total(stops_layer)

    if run_by_route:
        if STOPS_INPUT_MODE.lower() != "gtfs":
            raise RuntimeError(
                "Per-route output requires STOPS_INPUT_MODE='gtfs' so that "
                "stops can be reconstructed by route_short_name."
            )
        route_list = list(GTFS_ROUTE_SHORT_NAMES or [])
        if not route_list:
            # Guardrail: avoid accidental 'all routes' on very large feeds.
            raise ValueError(
                "GTFS_ROUTE_SHORT_NAMES is empty; populate it to enable per-route processing."
            )
        _ = _run_by_route(GTFS_FOLDER, route_list)

    print("✔ Processing completed successfully.")


if __name__ == "__main__":
    main()
