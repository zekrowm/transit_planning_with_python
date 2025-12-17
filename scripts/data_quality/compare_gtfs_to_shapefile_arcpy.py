"""GTFS Route Data Validation and Comparison.

This script compares an agency's authoritative route centerline features (Shapefile/FC)
against the GTFS shapes and stops files for the same route system.

The primary function is to:
1.  **Match Routes:** Link agency route keys (via configured normalization) to GTFS routes.
2.  **Calculate Divergence:** Compute spatial mismatch (minimum buffer required to contain
    GTFS features, reported as the 'max' or 'p95' distance in feet) and name similarity
    (Levenshtein/Token Sort Ratio).
3.  **Generate Diagnostics:** Produce a summary CSV report and optional debug plots
    showing the spatial overlap for each route.
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Inputs ---
AGENCY_ROUTE_FC: Path = Path(r"File\Path\To\Your\Bus_System.shp")  # .shp or feature class
GTFS_DIR: Path = Path(r"Folder\Path\To\Your\GTFS")

# --- Output (CSV + plots) ---
OUTPUT_CSV: Path = Path(r"File\Path\To\Your\route_comparison_summary.csv")
PLOT_DEBUG: bool = True
PLOT_DIR: Path = Path(r"Folder\Path\To\Your\Output\Plots")
PLOT_DPI: int = 150
PLOT_FIGSIZE: tuple[int, int] = (8, 8)
PLOT_MAX_ROUTES: Optional[int] = None  # None = all

# --- Agency fields ---
AGENCY_ROUTE_KEY_FIELD: str = "ROUTE_NUMB"
AGENCY_ROUTE_NAME_FIELD: str = "ROUTE_NAME"

# --- Join policy (strict short-name equality after normalization) ---
SHORTNAME_STRIP_NONALNUM: bool = True
SHORTNAME_UPPERCASE: bool = True
STRIP_LEADING_ZEROS: bool = True
STRIP_TRAILING_LETTERS: bool = False

# --- GTFS-as-truth knobs ---
USE_REPRESENTATIVE_SHAPE: bool = True
STOPS_FROM_REP_SHAPE_ONLY: bool = True

# --- Buffer statistic ---
# "max" -> full directed max; "p95" -> 95th percentile (robust to outliers)
BUFFER_STAT: str = "p95"

# --- Sampling for directed line→line distance (CRS units) ---
# IMPORTANT: if your CRS is feet, 5.0 means 5 feet and is usually overkill.
LINE_SAMPLING_STEP: float = 25.0

# Optional: if True and CRS units are feet and LINE_SAMPLING_STEP < 10, bump to 25 for safety.
AUTO_BUMP_SAMPLING_IN_FEET: bool = False

# --- Reporting units (always feet in output) ---
TARGET_MAX_BUFFER_FEET: float = 100.0
TARGET_MAX_NAME_DEVIATION_PCT: float = 0.0

# --- Shapes construction safeguard ---
MIN_POINTS_PER_SHAPE: int = 2

# --- Diagnostics ---
VERBOSE: bool = True


def log(msg: str) -> None:
    """Log message immediately if VERBOSE is True."""
    if VERBOSE:
        print(msg, flush=True)


# =============================================================================
# OPTIONAL RAPIDFUZZ (fallback if missing)
# =============================================================================

try:
    from rapidfuzz import fuzz  # type: ignore
    from rapidfuzz.distance import Levenshtein as RF_Levenshtein  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


def _levenshtein_py(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (fallback)."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _token_sort_ratio_fallback(a: str, b: str) -> float:
    """Rough fallback for token_sort_ratio using difflib."""
    import difflib

    def norm_tokens(s: str) -> str:
        toks = (s or "").split()
        toks = sorted(toks, key=lambda x: x.lower())
        return " ".join(toks)

    aa = norm_tokens(a)
    bb = norm_tokens(b)
    return 100.0 * difflib.SequenceMatcher(None, aa, bb).ratio()


def _name_divergence(a: str, b: str) -> tuple[int, float, float]:
    """Return (levenshtein_distance, similarity_pct, deviation_pct)."""
    a = (a or "").strip()
    b = (b or "").strip()

    if _HAS_RAPIDFUZZ:
        lev = int(RF_Levenshtein.distance(a, b))
        sim = float(fuzz.token_sort_ratio(a, b))
    else:
        lev = _levenshtein_py(a, b)
        sim = _token_sort_ratio_fallback(a, b)

    dev = 100.0 - sim
    return lev, sim, dev


# =============================================================================
# FILE + GTFS READ
# =============================================================================

def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file/folder: {path}")


def _read_gtfs_table(gtfs_dir: Path, name: str) -> pd.DataFrame:
    fp = gtfs_dir / f"{name}.txt"
    _ensure_exists(fp)
    df = pd.read_csv(fp, dtype=str)
    log(f"[INFO] Loaded {name}.txt -> {len(df):,} rows")
    return df


# =============================================================================
# NORMALIZATION
# =============================================================================

def _norm_short_name(val: object) -> str:
    s = "" if val is None else str(val).strip()
    if SHORTNAME_STRIP_NONALNUM:
        s = re.sub(r"[^0-9A-Za-z]+", "", s)
    if STRIP_LEADING_ZEROS and re.fullmatch(r"0+[0-9]+", s):
        s = re.sub(r"^0+([0-9])", r"\1", s)
    if SHORTNAME_UPPERCASE:
        s = s.upper()
    if STRIP_TRAILING_LETTERS:
        s = re.sub(r"^([0-9]+)[A-Z]+$", r"\1", s)
    return s


def _safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "", s)


# =============================================================================
# CRS + UNITS (ArcPy)
# =============================================================================

@dataclass(frozen=True)
class UnitInfo:
    """Spatial reference and unit conversion details for distance reporting."""
    sr: arcpy.SpatialReference
    crs_units_label: str  # "feet" or "meters" (or best guess)
    to_feet_factor: float


def _sr_is_projected(sr: arcpy.SpatialReference) -> bool:
    try:
        return bool(sr.PCSName)
    except Exception:
        return False


def _to_feet_factor_from_sr(sr: arcpy.SpatialReference) -> tuple[str, float]:
    """
    Return (label, factor) where factor converts SR linear units to feet.
    If unknown, assume meters.
    """
    try:
        unit_name = (sr.linearUnitName or "").lower()
    except Exception:
        unit_name = ""

    meters_per_unit: Optional[float]
    try:
        meters_per_unit = float(sr.metersPerUnit)
        if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
            meters_per_unit = None
    except Exception:
        meters_per_unit = None

    if "foot" in unit_name or "feet" in unit_name:
        if meters_per_unit is None:
            meters_per_unit = 0.3048
        return "feet", meters_per_unit * 3.28084
    if "meter" in unit_name or "metre" in unit_name:
        if meters_per_unit is None:
            meters_per_unit = 1.0
        return "meters", meters_per_unit * 3.28084

    if meters_per_unit is not None:
        if abs(meters_per_unit - 0.3048) < 5e-4 or abs(meters_per_unit - 0.3048006096) < 5e-4:
            return "feet", meters_per_unit * 3.28084
        if abs(meters_per_unit - 1.0) < 1e-6:
            return "meters", meters_per_unit * 3.28084

    return "meters", 3.28084


def _auto_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0 + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _project_sr_if_geographic(
    input_sr: arcpy.SpatialReference,
    geom_for_centroid: arcpy.Geometry,
) -> arcpy.SpatialReference:
    """
    If input SR is geographic, choose an auto UTM based on centroid.
    If already projected, return input SR.
    """
    if _sr_is_projected(input_sr):
        return input_sr

    wgs84 = arcpy.SpatialReference(4326)
    try:
        c_ll = geom_for_centroid.projectAs(wgs84).centroid
        lon = float(c_ll.X)
        lat = float(c_ll.Y)
    except Exception:
        c = geom_for_centroid.centroid
        lon = float(c.X)
        lat = float(c.Y)

    epsg = _auto_utm_epsg(lon, lat)
    log(f"[INFO] Agency CRS geographic; auto-projected to UTM EPSG:{epsg}")
    return arcpy.SpatialReference(epsg)


# =============================================================================
# AGENCY LOAD + DISSOLVE (ArcPy Geometry)
# =============================================================================

@dataclass
class AgencyRoute:
    """Normalized agency route key, display name, and dissolved route geometry."""
    route_key_raw: str
    route_key_norm: str
    agency_route_name: str
    geom: arcpy.Polyline


def _load_and_dissolve_agency_routes(
    fc_path: str,
    key_field: str,
    name_field: str,
) -> tuple[list[AgencyRoute], UnitInfo]:
    if not arcpy.Exists(fc_path):
        raise FileNotFoundError(f"Agency route dataset not found: {fc_path}")

    desc = arcpy.Describe(fc_path)
    in_sr: arcpy.SpatialReference = desc.spatialReference
    if in_sr is None:
        raise ValueError("Agency dataset has no spatial reference. Define projection first.")

    union_by_key: dict[str, arcpy.Polyline] = {}
    name_by_key: dict[str, str] = {}

    fields = [key_field, name_field, "SHAPE@"]
    count = 0

    first_geom: Optional[arcpy.Geometry] = None
    with arcpy.da.SearchCursor(fc_path, fields) as cur:
        for key_val, name_val, geom in cur:
            if geom is None:
                continue
            if first_geom is None:
                first_geom = geom

            key_raw = "" if key_val is None else str(key_val)
            nm = "" if name_val is None else str(name_val)

            if key_raw not in name_by_key:
                name_by_key[key_raw] = nm

            if key_raw not in union_by_key:
                union_by_key[key_raw] = geom
            else:
                # union() is generally fine; if it fails, you likely have invalid geometry.
                union_by_key[key_raw] = union_by_key[key_raw].union(geom)

            count += 1

    if count == 0 or not union_by_key:
        raise ValueError("Agency dataset contains no usable line features.")
    if first_geom is None:
        raise ValueError("Agency dataset contains no geometries.")

    out_sr = _project_sr_if_geographic(in_sr, first_geom)
    if out_sr.factoryCode != in_sr.factoryCode:
        log(f"[INFO] Reprojecting agency geometries to: {out_sr.name} (EPSG:{out_sr.factoryCode})")

    units_label, to_feet = _to_feet_factor_from_sr(out_sr)
    log(f"[INFO] Projected CRS units: {units_label} | units→feet factor={to_feet:.6f}")

    routes: list[AgencyRoute] = []
    for key_raw, geom in union_by_key.items():
        geom2 = geom.projectAs(out_sr) if out_sr.factoryCode != in_sr.factoryCode else geom
        routes.append(
            AgencyRoute(
                route_key_raw=str(key_raw),
                route_key_norm=_norm_short_name(key_raw),
                agency_route_name=name_by_key.get(key_raw, ""),
                geom=geom2,
            )
        )

    routes.sort(key=lambda r: r.route_key_raw)
    log(f"[INFO] Agency routes after dissolve: {len(routes):,}")
    return routes, UnitInfo(sr=out_sr, crs_units_label=units_label, to_feet_factor=float(to_feet))


# =============================================================================
# GTFS GEOMETRY BUILD (ArcPy)
# =============================================================================

def _build_shapes_dict(
    shapes_df: pd.DataFrame,
    out_sr: arcpy.SpatialReference,
) -> dict[str, arcpy.Polyline]:
    req = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    miss = req - set(shapes_df.columns)
    if miss:
        raise ValueError(f"shapes.txt missing columns: {sorted(miss)}")

    sdf = shapes_df.copy()
    sdf["shape_pt_sequence"] = sdf["shape_pt_sequence"].astype(int)
    sdf["shape_pt_lat"] = sdf["shape_pt_lat"].astype(float)
    sdf["shape_pt_lon"] = sdf["shape_pt_lon"].astype(float)
    sdf = sdf.sort_values(["shape_id", "shape_pt_sequence"])

    wgs84 = arcpy.SpatialReference(4326)
    out: dict[str, arcpy.Polyline] = {}

    for sid, grp in sdf.groupby("shape_id", sort=False):
        pts = list(zip(grp["shape_pt_lon"].tolist(), grp["shape_pt_lat"].tolist()))
        if len(pts) < MIN_POINTS_PER_SHAPE:
            continue
        arr = arcpy.Array([arcpy.Point(x, y) for x, y in pts])
        out[str(sid)] = arcpy.Polyline(arr, wgs84).projectAs(out_sr)

    if not out:
        raise ValueError("No valid shapes constructed from shapes.txt")

    log(f"[INFO] Built shapes dict with {len(out):,} polylines")
    return out


def _build_stops_dict(
    stops_df: pd.DataFrame,
    out_sr: arcpy.SpatialReference,
) -> dict[str, arcpy.PointGeometry]:
    req = {"stop_id", "stop_lat", "stop_lon"}
    miss = req - set(stops_df.columns)
    if miss:
        raise ValueError(f"stops.txt missing columns: {sorted(miss)}")

    sdf = stops_df.copy()
    sdf["stop_lat"] = sdf["stop_lat"].astype(float)
    sdf["stop_lon"] = sdf["stop_lon"].astype(float)

    wgs84 = arcpy.SpatialReference(4326)
    out: dict[str, arcpy.PointGeometry] = {}
    for _, row in sdf.iterrows():
        sid = str(row["stop_id"])
        pt = arcpy.Point(float(row["stop_lon"]), float(row["stop_lat"]))
        out[sid] = arcpy.PointGeometry(pt, wgs84).projectAs(out_sr)

    log(f"[INFO] Built stops dict with {len(out):,} points")
    return out


# =============================================================================
# GTFS INDEXES (PERFORMANCE)
# =============================================================================

@dataclass(frozen=True)
class GtfsIndex:
    """Spatial reference and unit conversion details for distance reporting."""
    route_to_trips: dict[str, list[str]]
    route_to_shapes: dict[str, list[str]]
    route_shape_to_trips: dict[tuple[str, str], list[str]]
    trip_to_stops: dict[str, set[str]]


def build_gtfs_indexes(trips_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> GtfsIndex:
    """
    Precompute lookups to avoid repeated DataFrame filtering.

    - route_to_trips: route_id -> [trip_id, ...]
    - route_to_shapes: route_id -> [shape_id, ...] (unique, non-null)
    - route_shape_to_trips: (route_id, shape_id) -> [trip_id, ...]
    - trip_to_stops: trip_id -> {stop_id, ...}
    """
    # Ensure string dtype
    t = trips_df[["route_id", "trip_id", "shape_id"]].copy()
    st = stop_times_df[["trip_id", "stop_id"]].copy()

    t["route_id"] = t["route_id"].astype(str)
    t["trip_id"] = t["trip_id"].astype(str)
    # shape_id can be null
    t["shape_id"] = t["shape_id"].where(t["shape_id"].notna(), None)

    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_id"] = st["stop_id"].astype(str)

    route_to_trips = t.groupby("route_id")["trip_id"].apply(list).to_dict()

    route_to_shapes = (
        t.dropna(subset=["shape_id"])
        .groupby("route_id")["shape_id"]
        .apply(lambda s: s.dropna().astype(str).unique().tolist())
        .to_dict()
    )

    route_shape_to_trips = (
        t.dropna(subset=["shape_id"])
        .groupby(["route_id", "shape_id"])["trip_id"]
        .apply(list)
        .to_dict()
    )

    trip_to_stops = (
        st.groupby("trip_id")["stop_id"]
        .apply(lambda s: set(x for x in s.dropna().tolist()))
        .to_dict()
    )

    log(
        "[INFO] Built GTFS indexes: "
        f"{len(route_to_trips):,} routes->trips, "
        f"{len(route_to_shapes):,} routes->shapes, "
        f"{len(route_shape_to_trips):,} (route,shape)->trips, "
        f"{len(trip_to_stops):,} trips->stops"
    )

    return GtfsIndex(
        route_to_trips=route_to_trips,
        route_to_shapes=route_to_shapes,
        route_shape_to_trips=route_shape_to_trips,
        trip_to_stops=trip_to_stops,
    )


# =============================================================================
# DISTANCES (ArcPy Geometry)
# =============================================================================

def _iter_polyline_vertices(pl: arcpy.Geometry) -> Iterable[tuple[float, float]]:
    """Yield all vertices (x,y) from a polyline-like geometry (multipart-safe)."""
    for part in pl:
        for p in part:
            if p is None:
                continue
            yield float(p.X), float(p.Y)


def _sample_points_along_polyline(pl: arcpy.Polyline, step: float) -> list[arcpy.PointGeometry]:
    """
    Sample points along a polyline every `step` units (in SR units).
    Includes both endpoints.
    """
    L = float(pl.length)
    if not math.isfinite(L) or L <= 0.0:
        c = pl.firstPoint
        return [arcpy.PointGeometry(arcpy.Point(c.X, c.Y), pl.spatialReference)]

    n = max(1, int(math.ceil(L / step)))
    dists = np.linspace(0.0, L, num=n + 1)

    pts: list[arcpy.PointGeometry] = []
    for d in dists:
        pts.append(pl.positionAlongLine(float(d), use_percentage=False))
    return pts


def _directed_line_to_line_distances(
    src_line: Optional[arcpy.Polyline],
    ref_line: arcpy.Polyline,
    step: float,
) -> np.ndarray:
    if src_line is None:
        return np.array([], dtype=float)

    d_all: list[float] = []
    for p in _sample_points_along_polyline(src_line, step):
        try:
            d_all.append(float(p.distanceTo(ref_line)))
        except Exception:
            continue
    return np.array(d_all, dtype=float)


def _point_set_to_line_distances(
    points: Iterable[arcpy.PointGeometry],
    ref_line: arcpy.Polyline,
) -> np.ndarray:
    vals: list[float] = []
    for p in points:
        try:
            vals.append(float(p.distanceTo(ref_line)))
        except Exception:
            continue
    return np.array(vals, dtype=float)


def _statistic(vals: np.ndarray, mode: str) -> float:
    if vals.size == 0:
        return float("nan")
    if mode == "max":
        return float(np.nanmax(vals))
    if mode == "p95":
        return float(np.nanpercentile(vals, 95))
    raise ValueError("BUFFER_STAT must be 'max' or 'p95'")


# =============================================================================
# PLOTTING (matplotlib from ArcPy geometry)
# =============================================================================

def _plot_route_debug(
    route_key: str,
    agency_line: arcpy.Polyline,
    gtfs_line: Optional[arcpy.Polyline],
    gtfs_stops: list[arcpy.PointGeometry],
    buffer_units: Optional[float],
    out_dir: Path,
    crs_units_label: str,
    figsize: tuple[int, int],
    dpi: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"route_{_safe_name(str(route_key))}.png"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(
        [x for x, _ in _iter_polyline_vertices(agency_line)],
        [y for _, y in _iter_polyline_vertices(agency_line)],
        linewidth=2.0,
        label="Agency line",
    )

    if gtfs_line is not None:
        ax.plot(
            [x for x, _ in _iter_polyline_vertices(gtfs_line)],
            [y for _, y in _iter_polyline_vertices(gtfs_line)],
            linewidth=1.2,
            linestyle="--",
            label="GTFS line",
        )

    if gtfs_stops:
        xs = [p.firstPoint.X for p in gtfs_stops]
        ys = [p.firstPoint.Y for p in gtfs_stops]
        ax.scatter(xs, ys, s=12, marker="o", label="GTFS stops")

    if buffer_units is not None and math.isfinite(buffer_units):
        try:
            buf = agency_line.buffer(float(buffer_units))
            boundary = buf.boundary()
            xs = [x for x, _ in _iter_polyline_vertices(boundary)]
            ys = [y for _, y in _iter_polyline_vertices(boundary)]
            ax.fill(xs, ys, alpha=0.2, label=f"Buffer ({crs_units_label})")
        except Exception:
            pass

    def _extent(g: arcpy.Geometry) -> tuple[float, float, float, float]:
        e = g.extent
        return float(e.XMin), float(e.YMin), float(e.XMax), float(e.YMax)

    exts = [_extent(agency_line)]
    if gtfs_line is not None:
        exts.append(_extent(gtfs_line))
    for p in gtfs_stops[:50_000]:
        exts.append(_extent(p))

    xmin = min(e[0] for e in exts)
    ymin = min(e[1] for e in exts)
    xmax = max(e[2] for e in exts)
    ymax = max(e[3] for e in exts)
    xpad = (xmax - xmin) * 0.05 or 1.0
    ypad = (ymax - ymin) * 0.05 or 1.0

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Route {route_key} | Units: {crs_units_label} | Buffer stat: {BUFFER_STAT}")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


# =============================================================================
# CORE
# =============================================================================

def compute_per_route_metrics() -> pd.DataFrame:
    """Compute per-route spatial/name divergence metrics and (optionally) plot diagnostics."""
    _ensure_exists(GTFS_DIR)

    routes_df = _read_gtfs_table(GTFS_DIR, "routes")
    trips_df = _read_gtfs_table(GTFS_DIR, "trips")
    shapes_df = _read_gtfs_table(GTFS_DIR, "shapes")
    stops_df = _read_gtfs_table(GTFS_DIR, "stops")
    stop_times_df = _read_gtfs_table(GTFS_DIR, "stop_times")

    for name, df, cols in [
        ("routes", routes_df, {"route_id", "route_long_name", "route_short_name"}),
        ("trips", trips_df, {"route_id", "trip_id", "shape_id"}),
        ("stop_times", stop_times_df, {"trip_id", "stop_id", "stop_sequence"}),
        ("stops", stops_df, {"stop_id", "stop_lat", "stop_lon"}),
        ("shapes", shapes_df, {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}),
    ]:
        miss = cols - set(df.columns)
        if miss:
            raise ValueError(f"{name}.txt missing columns: {sorted(miss)}")

    # --- Agency (ArcPy) ---
    agency_routes, unit_info = _load_and_dissolve_agency_routes(
        fc_path=str(AGENCY_ROUTE_FC),
        key_field=AGENCY_ROUTE_KEY_FIELD,
        name_field=AGENCY_ROUTE_NAME_FIELD,
    )

    # Auto bump sampling step if desired
    sampling_step = float(LINE_SAMPLING_STEP)
    if AUTO_BUMP_SAMPLING_IN_FEET and unit_info.crs_units_label == "feet" and sampling_step < 10.0:
        sampling_step = 25.0
        log(f"[INFO] AUTO_BUMP_SAMPLING_IN_FEET enabled; sampling step set to {sampling_step} feet")

    # Build GTFS geometries in agency projected SR
    shapes_by_id = _build_shapes_dict(shapes_df, unit_info.sr)
    stops_by_id = _build_stops_dict(stops_df, unit_info.sr)

    # --- GTFS indices ---
    routes_df["_norm_short"] = routes_df["route_short_name"].map(_norm_short_name)
    short_to_routeids = routes_df.groupby("_norm_short")["route_id"].apply(list).to_dict()

    empties = routes_df[routes_df["_norm_short"].isna() | (routes_df["_norm_short"] == "")]
    if len(empties):
        log(f"[WARN] {len(empties)} GTFS routes have empty normalized short names")

    trips_per_route = trips_df.groupby("route_id")["trip_id"].nunique().to_dict()
    trips_per_shape = trips_df.groupby("shape_id")["trip_id"].nunique().to_dict()

    idx = build_gtfs_indexes(trips_df, stop_times_df)

    records: list[dict[str, Any]] = []
    plotted = 0

    for a in agency_routes:
        try:
            a_key_raw = a.route_key_raw
            a_key_norm = a.route_key_norm
            a_name = a.agency_route_name
            a_geom = a.geom

            mapping_status = "ok"
            mapping_notes = ""
            chosen_rid: Optional[str] = None

            candidate_rids = short_to_routeids.get(a_key_norm, [])
            if not candidate_rids:
                mapping_status = "unmatched_short_name"
            elif len(candidate_rids) == 1:
                chosen_rid = str(candidate_rids[0])
            else:
                cand = pd.DataFrame({"route_id": candidate_rids})
                cand["trip_count"] = cand["route_id"].map(lambda r: trips_per_route.get(r, 0))
                cand = cand.sort_values(["trip_count", "route_id"], ascending=[False, True])
                chosen_rid = str(cand.iloc[0]["route_id"])
                mapping_status = "short_name_ambiguous_resolved"
                mapping_notes = (
                    f"Candidates={';'.join(map(str, candidate_rids))}; "
                    f"selected={chosen_rid} by route trip_count."
                )

            if chosen_rid:
                rrow = routes_df.loc[routes_df["route_id"] == chosen_rid].iloc[0]
                long_nm = str(rrow.get("route_long_name") or "")
                short_nm = str(rrow.get("route_short_name") or "")
            else:
                long_nm = ""
                short_nm = ""

            gtfs_line: Optional[arcpy.Polyline] = None
            stops_sel: list[arcpy.PointGeometry] = []
            shape_pick = "none"
            stops_pick = "none"

            if chosen_rid:
                shape_ids = idx.route_to_shapes.get(chosen_rid, [])
                rep_shape_id: Optional[str] = None

                if USE_REPRESENTATIVE_SHAPE and shape_ids:
                    rep_shape_id = max(shape_ids, key=lambda sid: trips_per_shape.get(sid, 0))

                if rep_shape_id and rep_shape_id in shapes_by_id:
                    gtfs_line = shapes_by_id[rep_shape_id]
                    shape_pick = f"rep:{rep_shape_id}"
                else:
                    # fallback: merge all shapes for the route (can be slow)
                    geoms = [shapes_by_id[sid] for sid in shape_ids if sid in shapes_by_id]
                    if geoms:
                        merged = geoms[0]
                        for g in geoms[1:]:
                            merged = merged.union(g)
                        gtfs_line = merged
                        shape_pick = "merged_all"
                    else:
                        gtfs_line = None
                        shape_pick = "none"

                # stops via pre-indexed dictionaries
                if STOPS_FROM_REP_SHAPE_ONLY and rep_shape_id:
                    trip_ids = idx.route_shape_to_trips.get((chosen_rid, rep_shape_id), [])
                    stops_pick = "rep_shape_trips"
                else:
                    trip_ids = idx.route_to_trips.get(chosen_rid, [])
                    stops_pick = "all_route_trips"

                stop_ids: set[str] = set()
                for tid in trip_ids:
                    stop_ids |= idx.trip_to_stops.get(tid, set())

                stops_sel = [stops_by_id[sid] for sid in stop_ids if sid in stops_by_id]

                # distances
                stop_dists = _point_set_to_line_distances(stops_sel, a_geom)
                stat_stop = _statistic(stop_dists, BUFFER_STAT) if stop_dists.size else float("nan")

                if gtfs_line is not None:
                    line_dists = _directed_line_to_line_distances(gtfs_line, a_geom, sampling_step)
                    stat_line = _statistic(line_dists, BUFFER_STAT) if line_dists.size else float("nan")
                else:
                    stat_line = float("nan")

                buffer_units = float(np.nanmax([stat_line, stat_stop]))
                buffer_feet = (
                    buffer_units * unit_info.to_feet_factor if math.isfinite(buffer_units) else float("nan")
                )
            else:
                buffer_units = float("nan")
                buffer_feet = float("nan")

            if chosen_rid:
                lev, sim, dev = _name_divergence(a_name, long_nm)
            else:
                lev, sim, dev = (np.nan, np.nan, np.nan)

            plot_path = ""
            if PLOT_DEBUG and chosen_rid and gtfs_line is not None:
                if (PLOT_MAX_ROUTES is None) or (plotted < int(PLOT_MAX_ROUTES)):
                    try:
                        out_png = _plot_route_debug(
                            route_key=str(a_key_raw),
                            agency_line=a_geom,
                            gtfs_line=gtfs_line,
                            gtfs_stops=stops_sel,
                            buffer_units=(buffer_units if math.isfinite(buffer_units) else None),
                            out_dir=PLOT_DIR,
                            crs_units_label=unit_info.crs_units_label,
                            figsize=PLOT_FIGSIZE,
                            dpi=PLOT_DPI,
                        )
                        plot_path = str(out_png)
                        plotted += 1
                    except Exception as e:
                        log(f"[WARN] Plot failed for route {a_key_raw}: {e}")

            records.append(
                {
                    "route_key_agency": str(a_key_raw),
                    "route_key_normalized": a_key_norm,
                    "agency_route_name": a_name,
                    "mapping_status": mapping_status,
                    "mapping_notes": mapping_notes,
                    "matched_gtfs_route_id": chosen_rid or "",
                    "gtfs_route_short_name": short_nm,
                    "gtfs_route_long_name": long_nm,
                    "shape_selection": shape_pick,
                    "stops_selection": stops_pick,
                    "buffer_stat": BUFFER_STAT,
                    "min_buffer_feet": None if not math.isfinite(buffer_feet) else round(float(buffer_feet), 2),
                    "name_levenshtein_distance": None
                    if (isinstance(lev, float) and np.isnan(lev))
                    else int(lev),
                    "name_similarity_pct": None
                    if (isinstance(sim, float) and np.isnan(sim))
                    else round(float(sim), 1),
                    "name_deviation_pct": None
                    if (isinstance(dev, float) and np.isnan(dev))
                    else round(float(dev), 1),
                    "exceeds_buffer_threshold": False
                    if not math.isfinite(buffer_feet)
                    else (buffer_feet > TARGET_MAX_BUFFER_FEET),
                    "exceeds_name_threshold": False
                    if not math.isfinite(float(dev))
                    else (float(dev) > TARGET_MAX_NAME_DEVIATION_PCT),
                    "crs_units": unit_info.crs_units_label,
                    "line_sampling_step_units": sampling_step,
                    "plot_path": plot_path,
                }
            )

        except Exception as e:
            log(f"[ERROR] Route '{a.route_key_raw}' failed: {e}")
            records.append(
                {
                    "route_key_agency": str(a.route_key_raw),
                    "route_key_normalized": _norm_short_name(a.route_key_raw),
                    "agency_route_name": str(a.agency_route_name),
                    "mapping_status": "exception",
                    "mapping_notes": str(e),
                    "matched_gtfs_route_id": "",
                    "gtfs_route_short_name": "",
                    "gtfs_route_long_name": "",
                    "shape_selection": "none",
                    "stops_selection": "none",
                    "buffer_stat": BUFFER_STAT,
                    "min_buffer_feet": None,
                    "name_levenshtein_distance": None,
                    "name_similarity_pct": None,
                    "name_deviation_pct": None,
                    "exceeds_buffer_threshold": False,
                    "exceeds_name_threshold": False,
                    "crs_units": unit_info.crs_units_label,
                    "line_sampling_step_units": sampling_step,
                    "plot_path": "",
                }
            )

    out = pd.DataFrame.from_records(records).sort_values(
        ["mapping_status", "exceeds_buffer_threshold", "exceeds_name_threshold", "route_key_agency"],
        ascending=[True, False, False, True],
    )
    log(f"[INFO] Produced {len(out):,} summary rows")
    return out


def write_summary_csv(df: pd.DataFrame, path: Path) -> None:
    """Write the summary DataFrame to CSV at the given path (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run metrics and write outputs."""
    try:
        df = compute_per_route_metrics()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        return

    try:
        write_summary_csv(df, OUTPUT_CSV)
    except OSError as exc:
        print(f"[ERROR] Failed to write CSV: {exc}", file=sys.stderr)
        return

    print(f"[OK] Wrote per-route comparison summary to: {OUTPUT_CSV.resolve()}")
    if PLOT_DEBUG:
        print(f"[OK] Plots (if any) saved under: {PLOT_DIR.resolve()}")
    print(
        f"[INFO] Representative shape: {USE_REPRESENTATIVE_SHAPE} | "
        f"Stops from rep shape only: {STOPS_FROM_REP_SHAPE_ONLY} | "
        f"Buffer stat: {BUFFER_STAT}"
    )
    print(
        f"[INFO] Targets: min_buffer_feet ≤ {TARGET_MAX_BUFFER_FEET}, "
        f"name_deviation_pct = {TARGET_MAX_NAME_DEVIATION_PCT}"
    )


if __name__ == "__main__":
    main()
