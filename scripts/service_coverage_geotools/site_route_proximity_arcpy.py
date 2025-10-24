"""GTFS proximity to sites (points, polygons, or points+parcels) with ArcPy.

This script finds nearby GTFS routes for user-specified locations. It supports:
  - A single input feature class of points (e.g., access/entrance points), or
  - A single input feature class of polygons (e.g., parcels), or
  - Two inputs: points + parcels. In this mode the workflow is point-driven:
    for each park-and-ride point, intersect with parcels to (optionally) use
    the matching parcel geometry as context. The CSV always has one row per
    input point, and parcel mismatches are flagged without dropping the point.

For each analysis unit (site), the script:
  1) Builds a search geometry (buffer of parcel or point).
  2) Selects GTFS stops within the search area.
  3) Computes planar distance to an anchor geometry (parcel polygon or the point).
  4) Joins to GTFS trips/routes; picks the nearest stop for each
     (route_short_name, direction_id) pair.
  5) Writes a CSV row with location metadata, route/direction pairs, and stop IDs.

Outputs
-------
- CSV with one row per site.
- Optional PNGs per site in PLOT_DIR.
- QA CSV (shared/identical stop issues).

Requires
--------
ArcGIS Pro (arcpy) and pandas (bundled with Pro).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import arcpy
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- GTFS paths
# Folder containing GTFS text files (stops.txt, stop_times.txt, trips.txt, routes.txt)
GTFS_FOLDER = r"C:\data\gtfs\your_gtfs_folder"

# ---- Mode and inputs
MODE = "points_plus_parcels"  # "single_fc" | "points_plus_parcels"

# When MODE == "single_fc":
# A single feature class of points or polygons representing sites
LOCATIONS_FC = r"C:\data\sites\locations.shp"

# When MODE == "points_plus_parcels":
# Entrance/access points (points) and a parcels layer (polygons)
ENTRANCE_POINTS_FC = r"C:\data\sites\entrance_points.shp"
PARCELS_FC = r"C:\data\sites\parcels.shp"

# ---- Site identifiers / attributes
# Preferred display names; fallbacks handled in code.
LOCATION_NAME_FIELD = "SITE_NAME"          # used by single_fc (if present)
PARCEL_ID_FIELD = "PARCEL_KEY"             # preferred key on parcels (robustly resolved)
PARCEL_NAME_FIELD = "FACILITY_N"           # parcel-side display name fallback
ENTRANCE_NAME_FIELD = "FACILITY_N"         # preferred display name on the entrance points

# Extra attributes to copy to output if present
LOCATION_EXTRA_FIELDS = ["OWNER", "PARKING", "HANDICAPPE", "BICYCLE", "LIGHTING"]

# ---- Geometry handling & distance (Northern VA state plane feet)
PROJECTED_CRS_WKID = 2283                  # NAD_1983_StatePlane_Virginia_North_FIPS_4501_Feet
BUFFER_DISTANCE = 0.25
BUFFER_UNIT = "miles"                      # "miles" | "feet"

# For single_fc with polygons:
SINGLE_POLY_REPRESENTATION = "buffer"      # "buffer" | "centroid" | "inside_point"

# ---- Route filters (short names)
ROUTE_FILTER_IN: list[str] = []            # keep only these (empty => no whitelist)
ROUTE_FILTER_OUT: list[str] = []           # drop these

# ---- Output
OUTPUT_FOLDER = r"C:\data\outputs\gtfs_proximity"
OUTPUT_FILE_NAME = "gtfs_proximity_results.csv"

# ---- Optional plotting
MAKE_PLOTS = True
PLOT_DIR = os.path.join(OUTPUT_FOLDER, "plots")
PLOT_FIG_DPI = 220

# ---- Large-parcel handling
# If a matched parcel exceeds either threshold, treat it as "too large"
# and buffer a representative point instead of the parcel polygon.
MAX_PARCEL_AREA_ACRES: float | None = 15.0     # None disables the area test
MAX_PARCEL_DIAMETER_FT: float | None = 2500.0  # None disables the diameter test
LARGE_PARCEL_POINT_METHOD: str = "INSIDE"      # "INSIDE" | "CENTROID"

# ---- QA reporting
LOG_SHARED_STOPS: bool = True
QA_REPORT_CSV: str = os.path.join(OUTPUT_FOLDER, "qa_shared_stops.csv")


# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def _feet(value: float, unit: str) -> float:
    """Convert `value` (miles or feet) to feet."""
    return float(value) * 5280.0 if unit.lower() == "miles" else float(value)


def _acres(area_sqft: float) -> float:
    """Convert square feet to acres."""
    return float(area_sqft) / 43560.0


def _parcel_diameter_ft(geom: arcpy.Geometry) -> float:
    """Approximate parcel 'diameter' as the larger of width/height in feet."""
    ext = geom.extent
    width = abs(ext.XMax - ext.XMin)
    height = abs(ext.YMax - ext.YMin)
    return float(max(width, height))


def _parcel_is_too_large(geom: arcpy.Geometry) -> bool:
    """Return True if the parcel exceeds MAX_PARCEL_* thresholds (if set)."""
    try:
        area_ac = _acres(geom.area)
    except Exception:
        area_ac = float("nan")
    try:
        diam_ft = _parcel_diameter_ft(geom)
    except Exception:
        diam_ft = float("nan")

    too_big_area = (
        MAX_PARCEL_AREA_ACRES is not None
        and area_ac == area_ac
        and area_ac > MAX_PARCEL_AREA_ACRES
    )
    too_big_diam = (
        MAX_PARCEL_DIAMETER_FT is not None
        and diam_ft == diam_ft
        and diam_ft > MAX_PARCEL_DIAMETER_FT
    )
    return bool(too_big_area or too_big_diam)


def _ensure_dir(path: str | os.PathLike) -> None:
    """Create the output directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _unique_fullpath(base: str, workspace: str) -> str:
    """Return a unique full dataset path under `workspace`.

    Use for single-parameter outputs (Project, FeatureToPoint, CopyFeatures, etc.).
    """
    return arcpy.CreateUniqueName(base, workspace)


def _unique_basename(base: str, workspace: str) -> str:
    """Return a unique *basename* (no path) under `workspace`.

    Use for tools with separate out_path + out_name (CreateFeatureclass).
    """
    return os.path.basename(arcpy.CreateUniqueName(base, workspace))


def _polygon_to_point_fc(poly_geom: arcpy.Geometry, sr_proj: arcpy.SpatialReference) -> str:
    """Create a one-row point FC from a polygon geometry using INSIDE or CENTROID."""
    # Build a single-feature polygon FC (CreateFeatureclass uses out_path + out_name)
    poly_fc = arcpy.management.CreateFeatureclass(
        "in_memory",
        _unique_basename("tmp_poly", "in_memory"),
        "POLYGON",
        spatial_reference=sr_proj,
    )[0]
    with arcpy.da.InsertCursor(poly_fc, ["SHAPE@"]) as icur:
        icur.insertRow([poly_geom])

    method = "INSIDE" if LARGE_PARCEL_POINT_METHOD.upper() == "INSIDE" else "CENTROID"

    # FeatureToPoint requires a full output path
    pt_fc = arcpy.management.FeatureToPoint(
        poly_fc,
        _unique_fullpath("tmp_pt", "in_memory"),
        method,
    )[0]

    arcpy.management.Delete(poly_fc)
    return pt_fc


def _gtfs_read_required(folder: str) -> dict[str, pd.DataFrame]:
    """Read required GTFS tables into pandas DataFrames.

    Args:
        folder: Directory containing GTFS text files.

    Returns:
        Mapping of table name to DataFrame.

    Raises:
        FileNotFoundError: If any required GTFS file is missing.
    """
    required = ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt")
    for fn in required:
        fp = os.path.join(folder, fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing GTFS file → {fp}")

    dfs = {
        "stops": pd.read_csv(os.path.join(folder, "stops.txt"), dtype=str),
        "stop_times": pd.read_csv(os.path.join(folder, "stop_times.txt"), dtype=str),
        "trips": pd.read_csv(os.path.join(folder, "trips.txt"), dtype=str),
        "routes": pd.read_csv(os.path.join(folder, "routes.txt"), dtype=str),
    }
    return dfs


def _apply_route_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply whitelist/blacklist filters by route_short_name."""
    if "route_short_name" not in df.columns:
        return df
    out = df
    if ROUTE_FILTER_IN:
        out = out[out.route_short_name.isin(ROUTE_FILTER_IN)]
    if ROUTE_FILTER_OUT:
        out = out[~out.route_short_name.isin(ROUTE_FILTER_OUT)]
    return out


def _make_stops_fc_from_gtfs(stops_df: pd.DataFrame, sr_proj: arcpy.SpatialReference) -> str:
    """Create a projected point FC from GTFS stops, using a disk scratch GDB for Project output."""
    if not {"stop_id", "stop_lat", "stop_lon"}.issubset(stops_df.columns):
        raise ValueError("stops.txt must contain stop_id, stop_lat, stop_lon")

    mem = "in_memory"
    sr_wgs84 = arcpy.SpatialReference(4326)

    # Build WGS84 points in-memory (CreateFeatureclass: out_name must be basename)
    pts_wgs = arcpy.management.CreateFeatureclass(
        mem,
        _unique_basename("gtfs_stops_wgs84", mem),
        "POINT",
        spatial_reference=sr_wgs84,
    )[0]
    arcpy.management.AddField(pts_wgs, "stop_id", "TEXT", field_length=64)

    with arcpy.da.InsertCursor(pts_wgs, ["SHAPE@XY", "stop_id"]) as icur:
        for _, r in stops_df.iterrows():
            try:
                x = float(r["stop_lon"])
                y = float(r["stop_lat"])
            except Exception:
                continue
            icur.insertRow(((x, y), str(r["stop_id"])))

    # Project to target CRS into a disk workspace (single-parameter output ⇒ full path)
    scratch = _scratch_gdb()
    out_path = _unique_fullpath("gtfs_stops_proj", scratch)
    pts_proj = arcpy.management.Project(pts_wgs, out_path, sr_proj)[0]
    return pts_proj


def _project_if_needed(fc_path: str, sr_proj: arcpy.SpatialReference) -> str:
    """Project a feature class to `sr_proj` if needed; write output to scratch GDB."""
    desc = arcpy.Describe(fc_path)
    src_sr = desc.spatialReference
    if src_sr is None or src_sr.factoryCode != sr_proj.factoryCode:
        scratch = _scratch_gdb()
        out_fc = arcpy.management.Project(
            fc_path,
            _unique_fullpath(f"{Path(fc_path).stem}_proj", scratch),
            sr_proj,
        )[0]
        return out_fc
    return fc_path


def _ensure_name_field(fc_path: str, name_field: str) -> str:
    """Ensure a 'name' field exists on the feature class.

    If `name_field` exists, copy it to 'name'. If not, populate from OID.
    """
    fields = [f.name for f in arcpy.ListFields(fc_path)]
    if "name" not in fields:
        arcpy.management.AddField(fc_path, "name", "TEXT", field_length=128)
    oid = arcpy.Describe(fc_path).OIDFieldName

    if name_field in fields and name_field != "name":
        with arcpy.da.UpdateCursor(fc_path, ["name", name_field]) as ucur:
            for nm, src in ucur:
                if not nm:
                    ucur.updateRow([str(src) if src is not None else "", src])

    with arcpy.da.UpdateCursor(fc_path, ["name", oid]) as ucur:
        for nm, oidv in ucur:
            if not nm:
                ucur.updateRow([f"feat_{oidv}", oidv])
    return fc_path


def _feature_to_point(fc: str, method: str) -> str:
    """Convert polygon features to points.

    Args:
        fc: Input polygon feature class.
        method: "CENTROID" or "INSIDE".

    Returns:
        Path to in-memory point feature class.
    """
    assert method in ("CENTROID", "INSIDE")
    return arcpy.management.FeatureToPoint(
        fc, _unique_fullpath(f"{Path(fc).name}_{method.lower()}", "in_memory"), method
    )[0]


def _buffer_fc(fc: str, dist_ft: float) -> str:
    """Planar buffer by feet; returns the buffered feature class.

    Args:
        fc: Input feature class to buffer.
        dist_ft: Buffer distance in feet. Must be strictly greater than zero.

    Returns:
        Path to the buffered feature class in the in_memory workspace.

    Raises:
        ValueError: If `dist_ft` is not strictly greater than zero.
    """
    if dist_ft is None or dist_ft <= 0:
        raise ValueError(f"Buffer distance must be > 0 feet; got {dist_ft!r}")
    return arcpy.analysis.Buffer(
        fc,
        _unique_fullpath(f"{Path(fc).name}_buf", "in_memory"),
        f"{dist_ft} Feet",
        dissolve_option="NONE",
    )[0]


def _select_stops_within(buffer_fc: str, stops_fc: str) -> str:
    """Return a fresh feature layer of stops selected WITHIN buffer_fc."""
    lyr_name = _unique_basename(f"stops_{Path(buffer_fc).name}_lyr", "in_memory")
    lyr = arcpy.management.MakeFeatureLayer(stops_fc, lyr_name)[0]
    arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
    arcpy.management.SelectLayerByLocation(lyr, "WITHIN", buffer_fc, selection_type="NEW_SELECTION")
    return lyr


def _near(selected_stops_lyr: str, anchor_fc: str) -> str:
    """Copy selected stops and run Near to `anchor_fc`; returns a feature class."""
    out = arcpy.management.CopyFeatures(
        selected_stops_lyr, _unique_fullpath(f"stops_sel_{Path(anchor_fc).name}", "in_memory")
    )[0]
    arcpy.analysis.Near(out, anchor_fc, method="PLANAR")
    return out


def _read_near_to_df(stops_fc_with_near: str) -> pd.DataFrame:
    """Read stop_id and NEAR_DIST from a feature class to pandas."""
    data = []
    with arcpy.da.SearchCursor(stops_fc_with_near, ["stop_id", "NEAR_DIST"]) as cur:
        for sid, dist in cur:
            data.append((str(sid), float(dist) if dist is not None else float("inf")))
    return pd.DataFrame(data, columns=["stop_id", "NEAR_DIST"])


def _base_row_from_fields(field_names: Iterable[str], values: Iterable) -> dict:
    """Build a dict for CSV row from a sequence of field names and values."""
    base = {}
    names = list(field_names)
    vals = list(values)
    for i, f in enumerate(names):
        key = "Location" if f == "name" else f
        base[key] = "" if vals[i] is None else str(vals[i])
    # ensure extras exist even if missing
    for c in LOCATION_EXTRA_FIELDS:
        if c not in base:
            base[c] = ""
    if "Location" not in base or not base["Location"]:
        base["Location"] = "Unnamed"
    return base


def _most_common(values: list[str]) -> str:
    """Return the most common non-empty string from values (ties: first encountered)."""
    counts: dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _quick_plot_site(
    parcel_geom: arcpy.Geometry | None,
    entrance_points: list,
    stop_points: list,
    site_name: str,
    out_dir: str,
    dpi: int,
) -> None:
    """Write a simple PNG: parcel outline, entrance points, and selected stops."""
    import matplotlib.pyplot as plt

    outlines = []
    try:
        gi = parcel_geom.__geo_interface__  # ArcPy geometry may provide this
        if gi["type"] == "Polygon":
            outlines.append(gi["coordinates"])
        elif gi["type"] == "MultiPolygon":
            for poly in gi["coordinates"]:
                outlines.append(poly)
    except Exception:
        outlines = []

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    # Parcel outline(s)
    for poly in outlines:
        xs = [pt[0] for pt in poly[0]]
        ys = [pt[1] for pt in poly[0]]
        ax.plot(xs, ys, linewidth=1)

    # Entrance points
    if entrance_points:
        xs = [p.firstPoint.X for p in entrance_points if p]
        ys = [p.firstPoint.Y for p in entrance_points if p]
        ax.scatter(xs, ys, s=12, marker="^", label="entrances")

    # Stops
    if stop_points:
        xs = [p.firstPoint.X for p in stop_points if p]
        ys = [p.firstPoint.Y for p in stop_points if p]
        ax.scatter(xs, ys, s=10, marker="o", label="stops")

    ax.set_title(site_name or "site")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in (site_name or "site"))
    out_png = Path(out_dir) / f"{safe}.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _scratch_gdb() -> str:
    """Return a writable scratch file geodatabase path; create one if needed."""
    sgdb = arcpy.env.scratchGDB
    if sgdb and arcpy.Exists(sgdb):
        return sgdb
    out_dir = OUTPUT_FOLDER if os.path.isdir(OUTPUT_FOLDER) else os.path.dirname(OUTPUT_FOLDER)
    gdb_path = os.path.join(out_dir, "_scratch.gdb")
    if not arcpy.Exists(gdb_path):
        arcpy.management.CreateFileGDB(out_dir, "_scratch.gdb")
    return gdb_path


def _in_ipython() -> bool:
    """Return True if running inside IPython/Jupyter."""
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


# -----------------------------------------------------------------------------
# SINGLE-FC WORKFLOW
# -----------------------------------------------------------------------------

def _single_fc_sites(
    sr_proj: arcpy.SpatialReference,
    dist_ft: float,
    stops_fc: str,
    st_trips_routes: pd.DataFrame,
) -> list[dict]:
    """Process a single locations FC (points or polygons)."""
    loc_fc = _project_if_needed(LOCATIONS_FC, sr_proj)
    loc_fc = _ensure_name_field(loc_fc, LOCATION_NAME_FIELD)

    shape_type = arcpy.Describe(loc_fc).shapeType.lower()
    fields_needed = {"name", *LOCATION_EXTRA_FIELDS}
    present = [f.name for f in arcpy.ListFields(loc_fc) if f.name in fields_needed]
    oid = arcpy.Describe(loc_fc).OIDFieldName

    # Choose anchor/search depending on geometry
    if shape_type in ("point", "multipoint"):
        anchor_fc = loc_fc
        search_fc = loc_fc  # buffer points
    elif shape_type in ("polygon", "multipatch"):
        if SINGLE_POLY_REPRESENTATION == "buffer":
            anchor_fc = loc_fc
            search_fc = loc_fc
        elif SINGLE_POLY_REPRESENTATION == "centroid":
            anchor_fc = _feature_to_point(loc_fc, "CENTROID")
            search_fc = anchor_fc
        elif SINGLE_POLY_REPRESENTATION == "inside_point":
            anchor_fc = _feature_to_point(loc_fc, "INSIDE")
            search_fc = anchor_fc
        else:
            raise ValueError("SINGLE_POLY_REPRESENTATION must be buffer|centroid|inside_point")
    else:
        raise ValueError("LOCATIONS_FC must be points or polygons")

    results: list[dict] = []

    # Iterate locations; per-feature buffer/select/near
    with arcpy.da.SearchCursor(search_fc, [oid, "SHAPE@"] + present) as scur:
        # Map OID to anchor geometry (could be same FC)
        anchor_geom = {r[0]: r[1] for r in arcpy.da.SearchCursor(anchor_fc, [oid, "SHAPE@"])}
        for oidv, search_geom, *vals in scur:
            # Create single-row FCs for search and anchor
            single_search = arcpy.management.CreateFeatureclass(
                "in_memory",
                _unique_basename(f"search_{oidv}", "in_memory"),
                "POLYGON" if search_geom.type.lower() != "point" else "POINT",
                spatial_reference=sr_proj,
            )[0]
            with arcpy.da.InsertCursor(single_search, ["SHAPE@"]) as icur:
                icur.insertRow([search_geom])

            single_anchor = arcpy.management.CreateFeatureclass(
                "in_memory",
                _unique_basename(f"anchor_{oidv}", "in_memory"),
                "POLYGON" if anchor_geom[oidv].type.lower() != "point" else "POINT",
                spatial_reference=sr_proj,
            )[0]
            with arcpy.da.InsertCursor(single_anchor, ["SHAPE@"]) as icur:
                icur.insertRow([anchor_geom[oidv]])

            buf_fc = _buffer_fc(single_search, dist_ft)
            stops_sel_lyr = _select_stops_within(buf_fc, stops_fc)
            count_within = int(arcpy.management.GetCount(stops_sel_lyr)[0])

            base = _base_row_from_fields(present, vals)
            base["ParcelID"] = ""          # N/A in single_fc
            base["ParcelMatch"] = ""       # N/A in single_fc
            base["AnchorType"] = "parcel" if search_geom.type.lower() != "point" else "point"
            base["ParcelLarge"] = ""
            base["ParcelAreaAc"] = ""
            base["ParcelDiamFt"] = ""
            base["StopsWithinBufferCount"] = count_within

            if count_within == 0:
                results.append({**base, "Routes": "No routes", "Stops": "No stops",
                                "StopNamesSample": "", "NearestDistMinFt": "",
                                "NearestDistMedFt": "", "NearestDistMaxFt": ""})
            else:
                near_fc = _near(stops_sel_lyr, single_anchor)
                dist_df = _read_near_to_df(near_fc)
                merged = dist_df.merge(
                    st_trips_routes[["stop_id", "route_short_name", "direction_id"]],
                    on="stop_id",
                    how="inner",
                ).drop_duplicates()
                if merged.empty:
                    results.append({**base, "Routes": "No routes", "Stops": "No stops",
                                    "StopNamesSample": "", "NearestDistMinFt": "",
                                    "NearestDistMedFt": "", "NearestDistMaxFt": ""})
                else:
                    idx = merged.groupby(["route_short_name", "direction_id"])["NEAR_DIST"].idxmin()
                    nearest = merged.loc[idx]
                    pair_set = {(str(r), str(d)) for r, d in zip(nearest.route_short_name,
                                                                 nearest.direction_id)}
                    routes_str = ", ".join(sorted(f"{rt} (dir {di})" for rt, di in pair_set))
                    stops_str = ", ".join(sorted(nearest.stop_id.astype(str).unique()))
                    stopnames = (
                        nearest.merge(gtfs["stops"][["stop_id", "stop_name"]],
                                      on="stop_id", how="left")["stop_name"]
                        .dropna().astype(str).unique()
                    )
                    stopnames_sample = "; ".join(sorted(stopnames)[:10])

                    base["NearestDistMinFt"] = round(float(nearest["NEAR_DIST"].min()), 1)
                    base["NearestDistMedFt"] = round(float(nearest["NEAR_DIST"].median()), 1)
                    base["NearestDistMaxFt"] = round(float(nearest["NEAR_DIST"].max()), 1)
                    base["StopNamesSample"] = stopnames_sample

                    results.append({**base, "Routes": routes_str, "Stops": stops_str})

            # cleanup
            for p in (single_search, single_anchor, buf_fc):
                arcpy.management.Delete(p)

    return results

# -----------------------------------------------------------------------------
# POINT-DRIVEN POINTS+PARCELS WORKFLOW
# -----------------------------------------------------------------------------


def _resolve_parcel_id_field(prc_fc_path: str, desired: str) -> str:
    """Resolve a usable parcel ID field name robustly."""
    fields = [f.name for f in arcpy.ListFields(prc_fc_path)]
    lower_map = {f.lower(): f for f in fields}
    if desired and desired.lower() in lower_map:
        return lower_map[desired.lower()]
    for cand in ("PARCEL_ID", "PARCELID", "PIN", "GPIN", "ACCOUNT", "MAPID", "UPI", "GIS_ID"):
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if "PARCEL_ID" not in fields:
        arcpy.management.AddField(prc_fc_path, "PARCEL_ID", "TEXT", field_length=64)
        oidf = arcpy.Describe(prc_fc_path).OIDFieldName
        arcpy.management.CalculateField(
            prc_fc_path, "PARCEL_ID", f'"parc_" + str(!{oidf}!)', "PYTHON3"
        )
    return "PARCEL_ID"


def _points_driven_sites(
    sr_proj: arcpy.SpatialReference,
    dist_ft: float,
    stops_fc: str,
    st_trips_routes: pd.DataFrame,
) -> list[dict]:
    """Process park-and-ride points one-by-one, with optional parcel context."""
    # Projected inputs
    pts_fc = _project_if_needed(ENTRANCE_POINTS_FC, sr_proj)
    prc_fc = _project_if_needed(PARCELS_FC, sr_proj)

    # Names on points (preferred)
    _ = _ensure_name_field(pts_fc, ENTRANCE_NAME_FIELD or LOCATION_NAME_FIELD)

    # Resolve parcel ID and build a parcel lookup
    parcel_id_field = _resolve_parcel_id_field(prc_fc, PARCEL_ID_FIELD)
    prc_fields_all = [f.name for f in arcpy.ListFields(prc_fc)]
    keep_fields_prc = ["name"] + [f for f in LOCATION_EXTRA_FIELDS if f in prc_fields_all]

    parcel_lookup: dict[str, tuple] = {}
    with arcpy.da.SearchCursor(prc_fc, [parcel_id_field, "SHAPE@", *keep_fields_prc]) as cur:
        for pid, geom, *vals in cur:
            if pid is not None:
                parcel_lookup[str(pid)] = (geom, vals)

    # SpatialJoin points→parcels; KEEP_ALL and JOIN_ONE_TO_MANY to inspect cardinality
    sj = arcpy.analysis.SpatialJoin(
        target_features=pts_fc,
        join_features=prc_fc,
        out_feature_class=_unique_fullpath("pnr_points_with_parcels", "in_memory"),
        join_operation="JOIN_ONE_TO_MANY",
        join_type="KEEP_ALL",
        match_option="INTERSECT",
    )[0]

    # Identify parcel key on join output (handle suffixing)
    sj_fields = [f.name for f in arcpy.ListFields(sj)]
    parcel_id_on_sj = None
    if parcel_id_field in sj_fields:
        parcel_id_on_sj = parcel_id_field
    else:
        pref = parcel_id_field + "_"
        for fn in sj_fields:
            if fn.startswith(pref):
                parcel_id_on_sj = fn
                break

    pts_oid = arcpy.Describe(pts_fc).OIDFieldName

    # Build mapping: point OID -> list of parcel IDs (0, 1, or many)
    parcels_by_point: dict[int, list[str]] = {}
    with arcpy.da.SearchCursor(
        sj, [pts_oid, parcel_id_on_sj] if parcel_id_on_sj else [pts_oid]
    ) as cur:
        for row in cur:
            poid = int(row[0])
            if len(row) == 1:
                parcels_by_point.setdefault(poid, [])
                continue
            pid = row[1]
            if pid is not None:
                parcels_by_point.setdefault(poid, []).append(str(pid))
            else:
                parcels_by_point.setdefault(poid, [])

    # Prepare iteration over the original points
    pt_fields_all = [f.name for f in arcpy.ListFields(pts_fc)]
    keep_fields_pts = ["name"] + [f for f in LOCATION_EXTRA_FIELDS if f in pt_fields_all]
    read_fields_pts = [pts_oid, "SHAPE@", *keep_fields_pts]

    results: list[dict] = []

    with arcpy.da.SearchCursor(pts_fc, read_fields_pts) as cur:
        for poid, pgeom, *pvals in cur:
            poid = int(poid)
            matched_pids = parcels_by_point.get(poid, [])
            status = "OK"
            parcel_id_str = ""

            # Anchor & search FCs per point
            if len(matched_pids) == 1:
                parcel_id_str = matched_pids[0]
                parcel_geom, _prc_vals = parcel_lookup.get(parcel_id_str, (None, None))
                if parcel_geom is None:
                    status = "ParcelMissingInLookup"
                    anchor_fc = arcpy.management.CreateFeatureclass(
                        "in_memory", _unique_basename(f"anc_pt_{poid}", "in_memory"),
                        "POINT", spatial_reference=sr_proj
                    )[0]
                    with arcpy.da.InsertCursor(anchor_fc, ["SHAPE@"]) as icur:
                        icur.insertRow([pgeom])
                    search_fc = _buffer_fc(anchor_fc, dist_ft)
                    anchor_type = "point"
                    parcel_large = False
                else:
                    # Large-Parcel check
                    if _parcel_is_too_large(parcel_geom):
                        # Use a representative point for both anchor and buffer center
                        pt_fc = _polygon_to_point_fc(parcel_geom, sr_proj)
                        anchor_fc = pt_fc
                        search_fc = _buffer_fc(pt_fc, dist_ft)
                        anchor_type = "point"   # overridden due to parcel size
                        parcel_large = True
                    else:
                        # Anchor is parcel polygon; search area = buffered parcel
                        anchor_fc = arcpy.management.CreateFeatureclass(
                            "in_memory", _unique_basename(f"anc_parcel_{poid}", "in_memory"),
                            "POLYGON", spatial_reference=sr_proj
                        )[0]
                        with arcpy.da.InsertCursor(anchor_fc, ["SHAPE@"]) as icur:
                            icur.insertRow([parcel_geom])
                        tmp = arcpy.management.CreateFeatureclass(
                            "in_memory", _unique_basename(f"srch_parcel_{poid}", "in_memory"),
                            "POLYGON", spatial_reference=sr_proj
                        )[0]
                        with arcpy.da.InsertCursor(tmp, ["SHAPE@"]) as icur:
                            icur.insertRow([parcel_geom])
                        search_fc = _buffer_fc(tmp, dist_ft)
                        anchor_type = "parcel"
                        parcel_large = False
            else:
                status = "NoParcel" if len(matched_pids) == 0 else "MultiParcel"
                parcel_id_str = "" if len(matched_pids) == 0 else ";".join(matched_pids)
                anchor_fc = arcpy.management.CreateFeatureclass(
                    "in_memory", _unique_basename(f"anc_pt_{poid}", "in_memory"),
                    "POINT", spatial_reference=sr_proj
                )[0]
                with arcpy.da.InsertCursor(anchor_fc, ["SHAPE@"]) as icur:
                    icur.insertRow([pgeom])
                search_fc = _buffer_fc(anchor_fc, dist_ft)
                anchor_type = "point"
                parcel_large = False

            # Select nearby stops and compute Near
            stops_sel_lyr = _select_stops_within(search_fc, stops_fc)
            count_within = int(arcpy.management.GetCount(stops_sel_lyr)[0])

            base = _base_row_from_fields(keep_fields_pts, pvals)
            base["ParcelID"] = parcel_id_str
            base["ParcelMatch"] = status  # OK | NoParcel | MultiParcel | ParcelMissingInLookup
            base["AnchorType"] = anchor_type  # "parcel" or "point"
            base["ParcelLarge"] = str(parcel_large)
            base["StopsWithinBufferCount"] = count_within

            # Parcel diagnostics (if available)
            if len(matched_pids) == 1 and parcel_id_str in parcel_lookup:
                pg = parcel_lookup[parcel_id_str][0]
                try:
                    base["ParcelAreaAc"] = round(_acres(pg.area), 3)
                    base["ParcelDiamFt"] = round(_parcel_diameter_ft(pg), 1)
                except Exception:
                    base["ParcelAreaAc"] = ""
                    base["ParcelDiamFt"] = ""
            else:
                base["ParcelAreaAc"] = ""
                base["ParcelDiamFt"] = ""

            if count_within == 0:
                results.append({**base, "Routes": "No routes", "Stops": "No stops",
                                "StopNamesSample": "", "NearestDistMinFt": "",
                                "NearestDistMedFt": "", "NearestDistMaxFt": ""})
                near_fc = None
            else:
                near_fc = _near(stops_sel_lyr, anchor_fc)
                dist_df = _read_near_to_df(near_fc)
                merged = dist_df.merge(
                    st_trips_routes[["stop_id", "route_short_name", "direction_id"]],
                    on="stop_id",
                    how="inner",
                ).drop_duplicates()

                if merged.empty:
                    results.append({**base, "Routes": "No routes", "Stops": "No stops",
                                    "StopNamesSample": "", "NearestDistMinFt": "",
                                    "NearestDistMedFt": "", "NearestDistMaxFt": ""})
                else:
                    idx = merged.groupby(["route_short_name", "direction_id"])["NEAR_DIST"].idxmin()
                    nearest = merged.loc[idx]
                    pair_set = {(str(r), str(d)) for r, d in zip(nearest.route_short_name,
                                                                 nearest.direction_id)}
                    routes_str = ", ".join(sorted(f"{rt} (dir {di})" for rt, di in pair_set))
                    stops_str = ", ".join(sorted(nearest.stop_id.astype(str).unique()))

                    # Stop names preview and distance stats
                    stopnames = (
                        nearest.merge(gtfs["stops"][["stop_id", "stop_name"]],
                                      on="stop_id", how="left")["stop_name"]
                        .dropna().astype(str).unique()
                    )
                    stopnames_sample = "; ".join(sorted(stopnames)[:10])

                    base["NearestDistMinFt"] = round(float(nearest["NEAR_DIST"].min()), 1)
                    base["NearestDistMedFt"] = round(float(nearest["NEAR_DIST"].median()), 1)
                    base["NearestDistMaxFt"] = round(float(nearest["NEAR_DIST"].max()), 1)
                    base["StopNamesSample"] = stopnames_sample

                    results.append({**base, "Routes": routes_str, "Stops": stops_str})

            # Optional plot
            if MAKE_PLOTS:
                try:
                    parcel_geom_for_plot = None
                    if len(matched_pids) == 1 and parcel_id_str in parcel_lookup:
                        parcel_geom_for_plot = parcel_lookup[parcel_id_str][0]

                    stop_geoms = []
                    src_fc_for_points = near_fc if near_fc else stops_sel_lyr
                    with arcpy.da.SearchCursor(src_fc_for_points, ["SHAPE@"]) as c2:
                        for g in c2:
                            stop_geoms.append(g[0])

                    _quick_plot_site(
                        parcel_geom=parcel_geom_for_plot,
                        entrance_points=[pgeom],
                        stop_points=stop_geoms,
                        site_name=base.get("Location", f"pnr_{poid}"),
                        out_dir=PLOT_DIR,
                        dpi=PLOT_FIG_DPI,
                    )
                except Exception:
                    pass

            # Cleanup temporary per-point FCs
            for pth in [anchor_fc, search_fc, near_fc] if near_fc else [anchor_fc, search_fc]:
                arcpy.management.Delete(pth)

    return results


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the GTFS proximity analysis."""
    arcpy.env.overwriteOutput = True
    try:
        _ensure_dir(OUTPUT_FOLDER)
        if MAKE_PLOTS:
            _ensure_dir(PLOT_DIR)

        sr_proj = arcpy.SpatialReference(PROJECTED_CRS_WKID)

        # Compute distance in feet and guard against invalid values
        dist_ft = _feet(BUFFER_DISTANCE, BUFFER_UNIT)
        if dist_ft <= 0:
            raise ValueError(
                f"Configured BUFFER_DISTANCE resolves to {dist_ft} feet (must be > 0)."
            )

        # Load GTFS and build stop→trip→route mapping
        global gtfs  # used in helper merges for stop_name previews
        gtfs = _gtfs_read_required(GTFS_FOLDER)

        # Sanity: enforce unique stop_id
        if gtfs["stops"]["stop_id"].duplicated().any():
            dup_ids = gtfs["stops"]["stop_id"][gtfs["stops"]["stop_id"].duplicated()].unique()
            raise ValueError(
                f"Non-unique stop_id values detected in stops.txt (e.g., {dup_ids[:10]!r}). "
                "Deduplicate or create a surrogate key before proceeding."
            )

        st_trips_routes = (
            gtfs["stop_times"]
            .merge(gtfs["trips"], on="trip_id", how="inner")
            .merge(gtfs["routes"], on="route_id", how="inner")
        )
        st_trips_routes = _apply_route_filters(st_trips_routes)
        if st_trips_routes.empty:
            print("Route filters removed every route – nothing to analyze.")
            return

        # Create projected stops FC
        stops_fc = _make_stops_fc_from_gtfs(gtfs["stops"], sr_proj)

        # Branch by mode
        if MODE == "single_fc":
            rows = _single_fc_sites(sr_proj, dist_ft, stops_fc, st_trips_routes)
        elif MODE == "points_plus_parcels":
            rows = _points_driven_sites(sr_proj, dist_ft, stops_fc, st_trips_routes)
        else:
            raise ValueError("MODE must be 'single_fc' or 'points_plus_parcels'.")

        if not rows:
            print("No results.")
            return

        # ---------- QA: shared stops across facilities ----------
        if LOG_SHARED_STOPS:
            df = pd.DataFrame(rows).copy()
            df["Stops"] = df["Stops"].astype(str).fillna("")
            df["Location"] = df["Location"].astype(str).fillna("")

            def _norm_stopset(s: str) -> str:
                toks = [t.strip() for t in s.split(",") if t.strip() and t.strip().lower() != "no stops"]
                return ",".join(sorted(set(toks)))

            df["StopSetNorm"] = df["Stops"].map(_norm_stopset)
            nonempty = df[df["StopSetNorm"] != ""]

            # (A) identical stop sets across locations
            identical_groups = (
                nonempty.groupby("StopSetNorm")["Location"]
                .agg(list)
                .reset_index()
            )
            identical_groups["n_locations"] = identical_groups["Location"].map(len)
            identical_groups = identical_groups[identical_groups["n_locations"] > 1]
            identical_groups["IssueType"] = "IdenticalStopSet"

            # (B) any single stop_id assigned to multiple facilities
            explode = []
            for _, r in nonempty.iterrows():
                for sid in r["StopSetNorm"].split(","):
                    if sid:
                        explode.append((sid, r["Location"]))
            df_expl = pd.DataFrame(explode, columns=["stop_id", "Location"])
            shared_stops = (
                df_expl.groupby("stop_id")["Location"]
                .agg(lambda xs: sorted(set(xs)))
                .reset_index()
            )
            shared_stops["n_locations"] = shared_stops["Location"].map(len)
            shared_stops = shared_stops[shared_stops["n_locations"] > 1]
            shared_stops["IssueType"] = "StopIdShared"

            qa_rows = []
            for _, r in identical_groups.iterrows():
                qa_rows.append({
                    "IssueType": r["IssueType"],
                    "Key": r["StopSetNorm"],
                    "Locations": "; ".join(r["Location"]),
                    "Count": r["n_locations"],
                })
            for _, r in shared_stops.iterrows():
                qa_rows.append({
                    "IssueType": r["IssueType"],
                    "Key": r["stop_id"],
                    "Locations": "; ".join(r["Location"]),
                    "Count": r["n_locations"],
                })

            if qa_rows:
                pd.DataFrame(qa_rows).to_csv(QA_REPORT_CSV, index=False, encoding="utf-8-sig")
                print(f"⚑ QA report written → {QA_REPORT_CSV}")
            else:
                print("✓ QA: no shared/identical stop-set issues detected.")

        # ---------- Write primary CSV ----------
        out_csv = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_NAME)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✔ Results written → {out_csv}")

    except Exception as exc:  # pylint: disable=broad-except
        # In notebooks, re-raise to avoid IPython's nested 'inspect' failure with sys.exit
        if _in_ipython():
            print(f"✖ {exc}")
            raise
        print(f"✖ {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
