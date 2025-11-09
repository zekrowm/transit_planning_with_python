"""Compute service-area demographics from transit stops.

Builds stop buffers (from GTFS or a point FC), dissolves to service areas,
clips a demographics layer, and produces area-weighted counts for equity and
employment metrics. Supports whole-network and per-route runs and optional CSV
and feature exports.

Pipeline:
  1) Create stops layer (GTFS filter by route_short_name or shapefile).
  2) Buffer and dissolve (optional processing CRS for stability).
  3) Clip demographics (SR alignment, PairwiseClip with fallbacks).
  4) Add areas (`area_ac_cl`, `area_perc`) and synthetic counts
     (prefer `PCT_* × *_TOT`; fallback to `*_CNT` with safety checks).
  5) Summarize totals and export.

Inputs:
  - GTFS: stops.txt, routes.txt, trips.txt, stop_times.txt
  - Demographics FC with HH_LOWINC/PCT_LOWINC/HH_TOT, MINOR_CNT/PCT_MINOR/POP_TOT,
    EMP_LO/EMP_TOT; LEP/YOUTH/ELDER optional.

Outputs:
  - Clipped polygons with area fields and synthetic metrics:
    loinc_hh, total_hh, minor_pop, total_pop, loinc_jobs, all_jobs (+ extras).
  - Final FC/shapefile; optional per-route CSV.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import arcpy
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
# Overwrite behavior
OVERWRITE_OUTPUTS: bool = True

# --- Stops input mode ---
# Choose one: "shapefile" or "gtfs"
STOPS_INPUT_MODE: str = "gtfs"

# For "shapefile" mode
STOPS_FEATURE_CLASS: str = r""

# For "gtfs" mode
GTFS_FOLDER: str = r"Path\To\Your\GTFS_Folder"

# Optional: filter GTFS to the following route_short_name values.
# Example: ["101", "202"]. Leave as [] or None to include all routes (network run).
GTFS_ROUTE_SHORT_NAMES: Optional[Sequence[str]] = ["101", "202"]

# Input demographics (from the census-join pipeline).
# This may be a FileGDB feature class or a shapefile; both work.
DEMOGRAPHICS_FC: str = (
    # r"File\Path\To\Your\output_final\scratch_join.gdb\joined_blocks"
    r"File\Path\To\Your\output_final\blocks_with_attrs.shp"
)

# Optional disk outputs for intermediates (used only for the "network" run).
# If left as empty strings, intermediates are kept in_memory.
BUFFERED_STOPS_OUT: str = r""
DISSOLVED_BUFFERS_OUT: str = r""
CLIPPED_DEMOGRAPHICS_OUT: str = r""

# Final export target:
#   If this ends with ".gdb", results are written as a feature class into that GDB.
#   Otherwise it's treated as a directory and a shapefile is written there.
FINAL_EXPORT_TARGET: str = r"File\Path\To\Your\output_final\output.gdb"

# Processing parameters
BUFFER_DISTANCE_MILES: float = 0.25

# Optional: project just the *processing* into a local planar CRS (WKID),
# while area computations remain geodesic. Leave as None to skip reprojection.
PROCESS_CRS_WKID: Optional[int] = None

# Run label used for final export naming and console prints
RUN_TAG: str = "tsp_service"  # e.g., "weekday_2024q3"

# -----------------------------------------------------------------------------
# OUTPUT MODE
# -----------------------------------------------------------------------------
# One of: "network" | "by_route" | "both"
OUTPUT_MODE: str = "by_route"

# Per-route runs are supported only when STOPS_INPUT_MODE == "gtfs".
# When True, also write a CSV of per-route results to FINAL_EXPORT_TARGET (folder or peer to gdb).
BY_ROUTE_WRITE_CSV: bool = True

# Optional: export per-route clipped polygons to FINAL_EXPORT_TARGET (gdb or folder).
BY_ROUTE_EXPORT_FEATURES: bool = False

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# =============================================================================
# PREFERRED DEMOGRAPHIC FIELD NAMES (from the census-join pipeline)
# =============================================================================


class _Pref(NamedTuple):
    """Preference tuple for sourcing a metric: count, percent, total."""

    count: str | None
    pct: str | None
    total: str | None


# What we want to produce in the clipped layer → how to source it.
# Prefer direct counts; otherwise derive via percent * total.
_PREFS: dict[str, _Pref] = {
    # Households
    "loinc_hh": _Pref(count="HH_LOWINC", pct="PCT_LOWINC", total="HH_TOT"),
    "total_hh": _Pref(count="HH_TOT", pct=None, total=None),
    # Population
    "minor_pop": _Pref(count="MINOR_CNT", pct="PCT_MINOR", total="POP_TOT"),
    "total_pop": _Pref(count="POP_TOT", pct=None, total=None),
    # Jobs (LODES)
    "loinc_jobs": _Pref(count="EMP_LO", pct=None, total=None),
    "all_jobs": _Pref(count="EMP_TOT", pct=None, total=None),
    # Optional extras (auto-included if present/derivable)
    "lep_cnt": _Pref(count="LEP_CNT", pct="PCT_LEP", total="POP_TOT"),
    "youth_cnt": _Pref(count="YOUTH_CNT", pct="PCT_YOUTH", total="POP_TOT"),
    "elder_cnt": _Pref(count="ELDER_CNT", pct="PCT_ELDER", total="POP_TOT"),
}


@dataclass(frozen=True)
class DemogSchema:
    """Resolved strategies for computing metrics from a demographics layer."""

    outputs: tuple[str, ...]
    # strategy:
    #   ("count", "FIELD") or ("derived", ("PCT_FIELD","TOTAL_FIELD"))
    strategies: dict[str, tuple[str, str | tuple[str, str]]]
    resolved_inputs: Dict[str, str]  # mapping of canonical→resolved name (for diagnostics)


# =============================================================================
# UTILITIES
# =============================================================================
def _as_distance_miles(miles: float) -> str:
    """Return an ArcPy-friendly distance string in miles."""
    return f"{miles} Miles"


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not path:
        return
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


def _resolve_field(dataset: str, desired: str) -> Optional[str]:
    """Resolve a logical field name to the actual field on *dataset*.

    Strategies (in order):
      1) Exact match (case-insensitive).
      2) Endswith match (case-insensitive), e.g., 'attrs_csv_POP_TOT' for 'POP_TOT'.
      3) Endswith match ignoring prefixes before the last underscore in the dataset.
    """
    desired_l = desired.lower()
    fields = arcpy.ListFields(dataset)
    # 1) Exact (case-insensitive)
    for f in fields:
        if f.name.lower() == desired_l:
            return f.name
    # 2) Endswith w/ case-insensitive
    for f in fields:
        if f.name.lower().endswith(desired_l):
            return f.name
    # 3) Try to strip to tail in dataset names and compare
    for f in fields:
        tail = f.name.split("_")[-1].lower()
        if tail == desired_l:
            return f.name
    return None


def _resolve_many(dataset: str, names: Iterable[str]) -> Dict[str, Optional[str]]:
    """Resolve many desired names → actual names on dataset."""
    out: Dict[str, Optional[str]] = {}
    for n in names:
        out[n] = _resolve_field(dataset, n)
    return out


# =============================================================================
# GTFS LOADING
# =============================================================================
def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory (all columns as str by default)."""
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
        df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
        data[key] = df
        logging.info("Loaded %s (%d records).", file_name, len(df))

    return data


def _filter_gtfs_stops_by_route_short_name(
    gtfs: dict[str, pd.DataFrame],
    route_short_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Return stops DataFrame filtered to those used by routes with given short names."""
    stops = gtfs["stops"]
    if not route_short_names:
        return stops.copy()

    target = {str(x) for x in route_short_names}
    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]

    routes_sel = routes[routes["route_short_name"].astype(str).isin(target)]
    if routes_sel.empty:
        raise ValueError(
            f"No routes matched the provided route_short_name filter: {sorted(target)}"
        )

    route_ids = set(routes_sel["route_id"].astype(str))
    trips_sel = trips[trips["route_id"].astype(str).isin(route_ids)]
    if trips_sel.empty:
        raise ValueError("Routes matched, but no trips found for the selected routes.")

    trip_ids = set(trips_sel["trip_id"].astype(str))
    st_sel = stop_times[stop_times["trip_id"].astype(str).isin(trip_ids)]
    if st_sel.empty:
        raise ValueError("Trips matched, but no stop_times rows found.")

    stop_ids = set(st_sel["stop_id"].astype(str))
    out = stops[stops["stop_id"].astype(str).isin(stop_ids)].copy()
    if out.empty:
        raise ValueError("Filtering produced zero stops. Verify stop_times and stops tables.")
    return out


def _points_layer_from_stops_df(stops_df: pd.DataFrame, layer_name: str = "gtfs_stops") -> str:
    """Create an in-memory point feature layer from a GTFS stops DataFrame.

    Expects columns: stop_id, stop_name, stop_lat, stop_lon.
    Uses a strict NumPy structured array (no 'object' dtypes).
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
    """Create a feature layer of stops for downstream processing."""
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

    gtfs = load_gtfs_data(
        gtfs_folder,
        files=("stops.txt", "routes.txt", "trips.txt", "stop_times.txt"),
        dtype=str,
    )
    stops_df = _filter_gtfs_stops_by_route_short_name(gtfs, route_short_names)
    return _points_layer_from_stops_df(stops_df, layer_name=layer_name)


# =============================================================================
# DEMOGRAPHIC SCHEMA DETECTION & AREA-WEIGHTED METRICS
# =============================================================================
def _has(dataset: str, field: Optional[str]) -> bool:
    return bool(field) and field_exists(dataset, field or "")


def detect_demog_schema(demographics_fc: str) -> DemogSchema:
    """Inspect the demographics layer and decide how to compute each metric.

    Preference (safer order):
      1) If percent + total exist (e.g., PCT_MINOR + POP_TOT), derive.
      2) Otherwise, use a direct count field if present (e.g., MINOR_CNT).
      3) Otherwise, skip that metric.

    Rationale: joined pipelines sometimes duplicate or mislabel *_CNT fields
    across geographies; deriving from PCT_* × POP_TOT is monotonic and avoids
    over-counting relative to the total.
    """
    strategies: dict[str, tuple[str, str | tuple[str, str]]] = {}
    outputs: list[str] = []
    resolved_inputs: Dict[str, str] = {}

    # Pre-resolve all canonical names that might be referenced
    all_needed: set[str] = set()
    for pref in _PREFS.values():
        for cand in (pref.count, pref.pct, pref.total):
            if cand:
                all_needed.add(cand)
    resolved = _resolve_many(demographics_fc, sorted(all_needed))

    for out_name, pref in _PREFS.items():
        # 1) Prefer derived if possible
        if pref.pct and pref.total:
            rpct = resolved.get(pref.pct)
            rtot = resolved.get(pref.total)
            if rpct and rtot and _has(demographics_fc, rpct) and _has(demographics_fc, rtot):
                strategies[out_name] = ("derived", (rpct, rtot))  # type: ignore[arg-type]
                outputs.append(out_name)
                resolved_inputs[pref.pct] = rpct
                resolved_inputs[pref.total] = rtot
                continue

        # 2) Fallback to direct count
        if pref.count:
            rcount = resolved.get(pref.count)
            if rcount and _has(demographics_fc, rcount):
                strategies[out_name] = ("count", rcount)  # type: ignore[arg-type]
                outputs.append(out_name)
                resolved_inputs[pref.count] = rcount
                continue

    return DemogSchema(
        outputs=tuple(outputs), strategies=strategies, resolved_inputs=resolved_inputs
    )


def add_original_area_acres(demographics_fc: str, field_name: str = "area_ac_og") -> None:
    """Add original area (acres) to demographic polygons (idempotent)."""
    safe_add_field(demographics_fc, field_name, "DOUBLE")
    calc_field(demographics_fc, field_name, "!shape.area@SQUAREMETERS! / 4046.86")


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


def _normalize_pct(x: float) -> float:
    """Normalize a percentage that may be expressed 0–1 or 0–100."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1.0:
        # Heuristic: treat as 0–100
        return v / 100.0
    return v


def _geom_area_m2(fc: str) -> Optional[float]:
    """Compute total geodesic area (m^2) of a feature class, safely."""
    total = 0.0
    try:
        with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
            for (geom,) in cur:
                if geom:
                    total += geom.getArea("GEODESIC", "SQUAREMETERS")
        return total
    except Exception as exc:  # noqa: BLE001
        logging.warning("Diagnostics: could not compute geodesic area (%s)", exc)
        return None


def add_synthetic_fields(
    clipped_fc: str,
    area_pct_field: str = "area_perc",
    demographics_fc_for_schema: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Create area-weighted counts on the clipped layer using auto-detected schema.

    If a direct count field exists, we scale it by area_perc.
    If only a percent exists (with its appropriate total), we scale (pct * total) by area_perc.

    Safety:
      - When a metric resolves to a direct count but the corresponding pct+total
        pair is also present on the clipped data, use the derived value if the
        raw count would exceed the implied total for that row.
    """
    if not field_exists(clipped_fc, area_pct_field):
        raise ValueError(
            f"Missing '{area_pct_field}' on {clipped_fc}. "
            "Run add_clipped_area_and_percentage() first."
        )

    probe_fc = demographics_fc_for_schema or clipped_fc
    schema = detect_demog_schema(probe_fc)
    outputs = schema.outputs
    strategies = schema.strategies

    if not outputs:
        logging.info("Synthetic outputs: <none>")
        return tuple(), {}

    # Ensure output fields exist
    for out_name in outputs:
        safe_add_field(clipped_fc, out_name, "DOUBLE")

    # Determine needed inputs (resolved names from schema)
    needed_inputs: set[str] = set()
    for mode, spec in strategies.values():
        if mode == "count":
            needed_inputs.add(str(spec))  # spec is the count field
        else:
            pct_field, tot_field = spec  # type: ignore[misc]
            needed_inputs.add(str(pct_field))
            needed_inputs.add(str(tot_field))

    # Also try to fetch pct/total for count-based metrics (for sanity fallback)
    derived_helpers: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for out_name, (mode, spec) in strategies.items():
        if mode == "count":
            # Look up matching pct/total names from _PREFS (canonical), then resolve on clipped FC
            pref = _PREFS.get(out_name)
            if pref and pref.pct and pref.total:
                derived_helpers[out_name] = (pref.pct, pref.total)
                needed_inputs.add(pref.pct)
                needed_inputs.add(pref.total)

    # Re-resolve against the *clipped* layer (it inherits prefixes from source)
    desired_again = sorted(needed_inputs)
    remap = _resolve_many(clipped_fc, desired_again)

    # Build final cursor field list
    # area fraction + present inputs + outputs
    def _f(x: Any) -> float:
        if x in (None, "", " "):
            return 0.0
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    present_inputs = [v for v in remap.values() if v is not None]
    cursor_fields = [area_pct_field] + present_inputs + list(outputs)
    idx = {name: i for i, name in enumerate(cursor_fields)}

    # Build helper maps for quick lookups
    input_on_clip: Dict[str, str] = {k: v for k, v in remap.items() if v is not None}

    with arcpy.da.UpdateCursor(clipped_fc, cursor_fields) as cur:
        for row in cur:
            a = _f(row[idx[area_pct_field]])

            # Read all input values present on this row
            vals: dict[str, float] = {}
            for fname in present_inputs:
                vals[fname] = _f(row[idx[fname]])

            for out_name in outputs:
                mode, spec = strategies[out_name]
                if mode == "count":
                    src = str(spec)
                    src_clip = input_on_clip.get(src)
                    raw = a * (vals.get(src_clip or "", 0.0))

                    # Optional derived fallback if pct+total are available
                    dspec = derived_helpers.get(out_name)
                    if dspec:
                        pct_req, tot_req = dspec
                        pct_clip = input_on_clip.get(pct_req or "")
                        tot_clip = input_on_clip.get(tot_req or "")
                        if pct_clip and tot_clip:
                            pct_val = vals.get(pct_clip, 0.0)
                            pct_val = pct_val / 100.0 if pct_val > 1.0 else pct_val
                            tot_val = vals.get(tot_clip, 0.0)
                            derived = a * (pct_val * tot_val)
                            # If raw would exceed derived-total expectation, use the safer derived value
                            if derived > 0.0 and raw > derived:
                                row[idx[out_name]] = derived
                            else:
                                row[idx[out_name]] = raw
                        else:
                            row[idx[out_name]] = raw
                    else:
                        row[idx[out_name]] = raw

                else:
                    pct_field, tot_field = spec  # type: ignore[misc]
                    pct_clip = input_on_clip.get(str(pct_field))
                    tot_clip = input_on_clip.get(str(tot_field))
                    pct_val = vals.get(pct_clip or "", 0.0)
                    pct_val = pct_val / 100.0 if pct_val > 1.0 else pct_val
                    tot_val = vals.get(tot_clip or "", 0.0)
                    row[idx[out_name]] = a * (pct_val * tot_val)

            cur.updateRow(row)

    pretty_map: Dict[str, str] = {}
    for out_name, (mode, spec) in strategies.items():
        if mode == "count":
            pretty_map[out_name] = f"{out_name} := area_perc * {spec}"
        else:
            pct_field, tot_field = spec  # type: ignore[misc]
            pretty_map[out_name] = f"{out_name} := area_perc * ({pct_field} * {tot_field})"

    logging.info("Synthetic outputs: %s", ", ".join(outputs))
    for k, v in pretty_map.items():
        logging.info("  %s -> %s", k, v)

    # Quick diagnostics on area-perc distribution
    try:
        vals = []
        with arcpy.da.SearchCursor(clipped_fc, [area_pct_field]) as cur2:
            for (v,) in cur2:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        if vals:
            logging.info(
                "Diagnostics: area_perc per-source: max=%.3f, mean=%.3f (N=%d)",
                max(vals),
                sum(vals) / max(len(vals), 1),
                len(vals),
            )
    except Exception:
        pass

    return outputs, pretty_map


def resolved_clipped_fields(demographics_fc: str) -> List[str]:
    """Return the list of outputs we can produce given the source schema.

    Keeps the original six at the front if present; appends extras in a stable order.
    """
    sch = detect_demog_schema(demographics_fc)
    base = ["loinc_hh", "total_hh", "minor_pop", "total_pop", "loinc_jobs", "all_jobs"]
    extras = [x for x in sch.outputs if x not in base]
    return [x for x in base if x in sch.outputs] + sorted(extras)


# =============================================================================
# GEOPROCESSING STEPS
# =============================================================================
def _maybe_project_for_processing(in_fc: str, name_hint: str) -> str:
    """Optionally project to PROCESS_CRS_WKID for buffering/clipping stability."""
    if PROCESS_CRS_WKID is None:
        return in_fc
    spref = arcpy.SpatialReference(PROCESS_CRS_WKID)
    out_fc = _route_scoped_temp(f"proc_{name_hint}", "proj")
    arcpy.management.Project(in_fc, out_fc, spref)
    return out_fc


def buffer_stops(stops_layer: str, out_path: str, miles: float) -> str:
    """Buffer stops by the given distance in miles (optionally after projection)."""
    src = _maybe_project_for_processing(stops_layer, "stops")
    distance = _as_distance_miles(miles)
    result = arcpy.analysis.Buffer(src, out_path, distance)
    return str(result)


def dissolve_buffers(
    buffer_fc: str,
    out_path: str,
) -> str:
    """Dissolve buffered polygons (all features into one or a few multipart features)."""
    src = _maybe_project_for_processing(buffer_fc, "buffer")
    result = arcpy.management.Dissolve(src, out_path, dissolve_field="")
    return str(result)


def _sr_name(obj_path: str) -> str:
    try:
        return arcpy.Describe(obj_path).spatialReference.name
    except Exception:
        return "<unknown>"


def _count(obj_path: str) -> int:
    try:
        return int(arcpy.management.GetCount(obj_path).getOutput(0))
    except Exception:
        return -1


def _log_env() -> None:
    env = arcpy.env
    logging.info(
        "Env: extent=%s mask=%s snapRaster=%s",
        getattr(env, "extent", None),
        getattr(env, "mask", None),
        getattr(env, "snapRaster", None),
    )


def _project_to_match_sr(in_fc: str, like_fc: str, name_hint: str) -> str:
    """Project *in_fc* to the spatial reference of *like_fc* if they differ.

    Writes to a persistent shapefile in the inspection folder returned by
    _intermediate_buffer_folder(). This avoids scratch.gdb visibility issues
    and keeps the intermediate available for manual inspection.

    Args:
        in_fc: Input feature class or layer to (maybe) project.
        like_fc: Dataset whose spatial reference we want to match.
        name_hint: Short token used in naming the output.

    Returns:
        Path to a shapefile in the same spatial reference as *like_fc*.
        Returns *in_fc* unchanged when spatial references already match.
    """
    try:
        in_sr = arcpy.Describe(in_fc).spatialReference
        like_sr = arcpy.Describe(like_fc).spatialReference
    except Exception as exc:  # noqa: BLE001
        logging.warning("SR inspection failed; proceeding without projection (%s)", exc)
        return in_fc

    def _sr_eq(a: arcpy.SpatialReference, b: arcpy.SpatialReference) -> bool:
        try:
            if a.factoryCode and b.factoryCode:
                return int(a.factoryCode) == int(b.factoryCode)
        except Exception:
            pass
        try:
            return (a.name or "").strip().lower() == (b.name or "").strip().lower()
        except Exception:
            return False

    if _sr_eq(in_sr, like_sr):
        return in_fc

    # Persistent shapefile path for inspection
    folder = _intermediate_buffer_folder()
    base = f"proj_{_sanitize_name(name_hint)}.shp"
    out_fc = arcpy.CreateUniqueName(base, folder)

    # Best-effort: choose a geographic transformation if recommended
    transform = None
    try:
        cands = arcpy.ListTransformations(in_sr, like_sr) or []
        if cands:
            transform = cands[0]
    except Exception:
        pass

    logging.info(
        "Projecting '%s' SR [%s] → match '%s' SR [%s] (transform=%s) → '%s' …",
        in_fc,
        getattr(in_sr, "name", "<unknown>"),
        like_fc,
        getattr(like_sr, "name", "<unknown>"),
        transform or "<none>",
        out_fc,
    )
    arcpy.management.Project(in_fc, out_fc, like_sr, transform)
    return out_fc


def _intermediate_buffer_folder() -> str:
    """Return a stable folder for projected buffer intermediates (shapefiles).

    Uses the user-specified inspection directory so outputs persist for review.

    Returns:
        Absolute path to the inspection folder, created if missing.
    """
    folder = r"G:\projects\dot\zkrohmal\census_merge_test_2025_10_31\output_buffer_intermediate"
    ensure_dir(folder)
    return folder


def _scratch_gdb() -> str:
    """Return a writable scratch file geodatabase path, creating it if needed.

    Prefers arcpy.env.scratchGDB. Falls back to <scratchFolder>/sr_temp.gdb,
    or OS temp if scratchFolder is unavailable.
    """
    gdb = getattr(arcpy.env, "scratchGDB", None)
    if gdb and arcpy.Exists(gdb):
        return gdb

    folder = getattr(arcpy.env, "scratchFolder", None) or os.environ.get("TEMP") or os.getcwd()
    ensure_dir(folder)
    gdb = os.path.join(folder, "sr_temp.gdb")
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(folder, "sr_temp.gdb")
    return gdb


def clip_demographics_to_buffers(
    demographics_fc: str,
    buffers_fc: str,
    out_path: str,
) -> str:
    """Clip demographics by buffers with SR-alignment, diagnostics, and fallbacks.

    Steps:
      1) Ensure the buffer geometry is in the demographics SR (project if needed).
      2) Add a spatial index to demographics (idempotent).
      3) Prefer PairwiseClip; fall back to Clip; then Intersect on failure.

    Args:
        demographics_fc: Polygon feature class with demographic attributes.
        buffers_fc: Dissolved service-area polygons (any SR).
        out_path: Destination path for clipped output.

    Returns:
        Path to the clipped feature class.
    """
    _log_env()

    # Align SRs for robust and predictable overlay
    buffers_for_clip = _project_to_match_sr(
        in_fc=buffers_fc,
        like_fc=demographics_fc,
        name_hint="buffers_for_clip",
    )

    # Pre-flight diagnostics
    logging.info(
        "Clip preflight: DEMO SR=%s, count=%s",
        _sr_name(demographics_fc),
        _count(demographics_fc),
    )
    logging.info(
        "Clip preflight: BUFF SR=%s, count=%s",
        _sr_name(buffers_for_clip),
        _count(buffers_for_clip),
    )

    # Make sure large FCs have a spatial index (idempotent)
    try:
        arcpy.management.AddSpatialIndex(demographics_fc)
    except Exception:
        pass

    # Ensure output container exists and is writable
    out_dir = os.path.dirname(out_path)
    if out_dir and out_dir.lower().endswith(".gdb") and not arcpy.Exists(out_dir):
        dir_, name = os.path.split(out_dir)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        arcpy.management.CreateFileGDB(dir_, name)

    # Primary attempt: PairwiseClip (when available) or Clip
    result_path = out_path
    try:
        if hasattr(arcpy.analysis, "PairwiseClip"):
            logging.info("Running PairwiseClip …")
            arcpy.analysis.PairwiseClip(demographics_fc, buffers_for_clip, result_path)
        else:
            logging.info("Running Clip …")
            arcpy.analysis.Clip(demographics_fc, buffers_for_clip, result_path)
    except Exception as exc1:
        logging.warning("Primary clip failed: %s", exc1)
        # Fallback: Intersect polygons with buffer
        try:
            logging.info("Fallback: Intersect (polygons ∩ buffer) …")
            tmp = result_path + "_ix_tmp"
            arcpy.analysis.Intersect([demographics_fc, buffers_for_clip], tmp, "ALL", "", "INPUT")
            if arcpy.Exists(result_path):
                arcpy.management.Delete(result_path)
            arcpy.management.CopyFeatures(tmp, result_path)
            arcpy.management.Delete(tmp)
        except Exception:
            logging.error("Fallback Intersect also failed.\nGP Messages:\n%s", arcpy.GetMessages(2))
            raise

    n = _count(result_path)
    logging.info(
        "Clip result: %s features at %s (SR=%s)",
        n,
        result_path,
        _sr_name(result_path),
    )
    if n == 0:
        logging.warning(
            "Clip returned ZERO features. Likely causes:\n"
            "  • Extent/Mask env excludes overlaps (current env logged above).\n"
            "  • No actual overlap (check buffer distance and SR).\n"
            "  • Demographics is multipart with tiny slivers and buffer in a different CRS.\n"
            "  • Selection on inputs (layer with empty selection)."
        )
        logging.warning("GP Messages:\n%s", arcpy.GetMessages(2))

    return str(result_path)


def summarize_fields(feature_class: str, fields: Iterable[str]) -> Dict[str, int]:
    """Sum a set of numeric fields and return rounded integers."""
    fields_list = list(fields)
    if not fields_list:
        return {}

    totals = {f: 0.0 for f in fields_list}
    with arcpy.da.SearchCursor(feature_class, fields_list) as cur:
        for row in cur:
            for i, f in enumerate(fields_list):
                val = row[i]
                if val is not None:
                    totals[f] += float(val)
    return {k: int(round(v)) for k, v in totals.items()}


def export_final_copy(
    in_feature_class: str,
    out_target: str,
    run_tag: str,
) -> str:
    """Copy features to final destination.

    If out_target ends with '.gdb', create a feature class '{run_tag}_service_buffer_data'.
    Otherwise treat out_target as a directory and write a shapefile with that name.
    """
    if out_target.lower().endswith(".gdb"):
        gdb = out_target
        if not arcpy.Exists(gdb):
            dir_, name = os.path.split(gdb)
            ensure_dir(dir_)
            arcpy.management.CreateFileGDB(dir_, name)
        out_name = f"{run_tag}_service_buffer_data"
        out_fc = os.path.join(gdb, out_name)
        if arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)
        arcpy.management.CopyFeatures(in_feature_class, out_fc)
        return out_fc

    # shapefile directory
    ensure_dir(out_target)
    out_fc = os.path.join(out_target, f"{run_tag}_service_buffer_data.shp")
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.CopyFeatures(in_feature_class, out_fc)
    return out_fc


# =============================================================================
# CONSOLIDATED EXECUTION HELPERS
# =============================================================================


def calc_original_area_for_intersecting(
    demographics_fc: str,
    buffers_fc: str,
    *,
    insurance_distance_ft: float = 100.0,
    field_name: str = "area_ac_og",
) -> None:
    """Populate original area (acres) only for demo polygons near the buffer.

    Creates/ensures the `area_ac_og` field on the source demographics feature
    class, then calculates it *only* for features that intersect the buffer
    expanded by `insurance_distance_ft`. Non-intersecting features are left
    untouched, avoiding a full-table CalculateField over ~40k rows.

    Args:
        demographics_fc: Input demographics polygon feature class to update.
        buffers_fc: Dissolved service-area polygons.
        insurance_distance_ft: Extra distance (feet) to expand the buffer
            before selecting intersecting demographics.
        field_name: Name of the target field holding original area in acres.
    """
    # 1) Ensure target field exists on the *source* FC (idempotent schema change).
    safe_add_field(demographics_fc, field_name, "DOUBLE")

    # 2) Create an expanded selection buffer (temporary).
    sel_buf = _route_scoped_temp("sel_buffer_expanded", "intersect_area")
    try:
        arcpy.analysis.Buffer(
            in_features=buffers_fc,
            out_feature_class=sel_buf,
            buffer_distance_or_field=f"{float(insurance_distance_ft)} Feet",
            line_side="FULL",
            line_end_type="ROUND",
            dissolve_option="ALL",
        )

        # 3) Make a selectable layer from the demographics and select by location.
        lyr = arcpy.management.MakeFeatureLayer(demographics_fc, "demog_for_area_sel")
        arcpy.management.SelectLayerByLocation(
            in_layer=lyr,
            overlap_type="INTERSECT",
            select_features=sel_buf,
            selection_type="NEW_SELECTION",
        )

        # 4) If nothing selected, bail early.
        sel_count = int(arcpy.management.GetCount(lyr).getOutput(0))
        logging.info(
            "Area-prep selection: %d demographics features within %.1f ft of buffer.",
            sel_count,
            insurance_distance_ft,
        )
        if sel_count == 0:
            # Clear selection to be tidy and return.
            arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
            return

        # 5) Calculate area only for the selected rows.
        arcpy.management.CalculateField(
            in_table=lyr,
            field=field_name,
            expression="!shape.area@SQUAREMETERS! / 4046.86",
            expression_type="PYTHON3",
        )

        # 6) Clear selection.
        arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")

    finally:
        # Best-effort cleanup of the temp selection buffer.
        try:
            if arcpy.Exists(sel_buf):
                arcpy.management.Delete(sel_buf)
        except Exception:
            pass


def _process_service_area_from_stops_layer(
    stops_layer: str,
    run_tag: str,
    *,
    export_final: bool,
    final_export_target: str,
    buffered_path: Optional[str] = None,
    dissolved_path: Optional[str] = None,
    clipped_path: Optional[str] = None,
) -> Tuple[Dict[str, int], Optional[str]]:
    """Run the buffer→dissolve→clip→synthetics→summaries pipeline for any stops layer.

    When disk paths are provided for buffered/dissolved/clipped, they are used.
    Otherwise, unique in_memory paths are allocated (per-route use case).
    """
    # Allocate paths if not given
    buffered_path = buffered_path or _route_scoped_temp("buffered_stops", run_tag)
    dissolved_path = dissolved_path or _route_scoped_temp("dissolved_buffers", run_tag)
    clipped_path = clipped_path or _route_scoped_temp("clipped_demog", run_tag)

    # 1) Buffer
    logging.info("Buffering stops (%s)…", run_tag)
    buffered = buffer_stops(stops_layer, buffered_path, BUFFER_DISTANCE_MILES)

    # 2) Dissolve
    logging.info("Dissolving buffers (%s)…", run_tag)
    dissolved = dissolve_buffers(buffered, dissolved_path)

    # 2b) Diagnostics: compute area of dissolved
    area_m2 = _geom_area_m2(dissolved)
    if area_m2 is not None:
        logging.info("Diagnostics: dissolved geodesic area = %.2f sq.m", area_m2)

    # 3) Clip demographics
    logging.info("Clipping demographics (%s)…", run_tag)
    clipped = clip_demographics_to_buffers(DEMOGRAPHICS_FC, dissolved, clipped_path)

    # 4) Precompute original area ONLY for demographics that matter (near the buffer)
    calc_original_area_for_intersecting(
        demographics_fc=DEMOGRAPHICS_FC,
        buffers_fc=dissolved,
        insurance_distance_ft=100.0,
        field_name="area_ac_og",
    )

    # 5) Areas and percentages on the clipped output
    add_clipped_area_and_percentage(clipped)

    # 6) Synthetic metrics (auto-detected schema from original FC; resolved again on clip)
    outputs, _strategy_map = add_synthetic_fields(
        clipped,
        area_pct_field="area_perc",
        demographics_fc_for_schema=DEMOGRAPHICS_FC,
    )

    # 7) Summaries
    fields_for_summary = resolved_clipped_fields(DEMOGRAPHICS_FC) if outputs else []
    totals = summarize_fields(clipped, fields_for_summary)

    # Optional export to disk/GDB
    exported_path: Optional[str] = None
    if export_final:
        exported_path = export_final_copy(clipped, final_export_target, run_tag)

    return totals, exported_path


def _run_network_total(stops_layer: str) -> None:
    """Whole-network summary using the already-prepared stops layer.

    Intermediates use explicit disk outputs if provided in config; otherwise in_memory.
    """
    print(
        f"Buffering stops to {BUFFER_DISTANCE_MILES} mi → "
        f"{'disk' if BUFFERED_STOPS_OUT else 'in_memory'} (intermediates)"
    )

    svc_totals, exported = _process_service_area_from_stops_layer(
        stops_layer=stops_layer,
        run_tag=RUN_TAG,
        export_final=True,
        final_export_target=FINAL_EXPORT_TARGET,
        buffered_path=BUFFERED_STOPS_OUT or None,
        dissolved_path=DISSOLVED_BUFFERS_OUT or None,
        clipped_path=CLIPPED_DEMOGRAPHICS_OUT or None,
    )

    print(f"[{RUN_TAG}] Service buffer totals (area-weighted, rounded):")
    for k in resolved_clipped_fields(DEMOGRAPHICS_FC):
        print(f"  {k}: {svc_totals.get(k, 0):,}")

    if exported:
        print(f"Final export: {exported}")


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
            final_export_target=FINAL_EXPORT_TARGET,
        )

        row: Dict[str, int | str] = {"route_short_name": str(route_sn)}
        for k in resolved_clipped_fields(DEMOGRAPHICS_FC):
            row[k] = int(totals.get(k, 0))
        results.append(row)

        if BY_ROUTE_EXPORT_FEATURES and exported:
            print(f"  Exported per-route features: {exported}")

    df = pd.DataFrame(results).sort_values("route_short_name").reset_index(drop=True)

    # CSV output adjacent to the target (folder or alongside GDB)
    if BY_ROUTE_WRITE_CSV and not df.empty:
        out_dir = (
            FINAL_EXPORT_TARGET
            if not FINAL_EXPORT_TARGET.lower().endswith(".gdb")
            else os.path.dirname(FINAL_EXPORT_TARGET)
        )
        ensure_dir(out_dir)
        csv_path = os.path.join(out_dir, f"{RUN_TAG}_by_route_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Per-route summary written: {csv_path}")

    # Console view
    if not df.empty:
        print("\nPer-route totals (area-weighted, rounded):")
        cols = ["route_short_name"] + resolved_clipped_fields(DEMOGRAPHICS_FC)
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

    if not arcpy.Exists(DEMOGRAPHICS_FC):
        raise FileNotFoundError(f"Input not found: {DEMOGRAPHICS_FC}")

    # Prepare stops layer (network scope) once
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
