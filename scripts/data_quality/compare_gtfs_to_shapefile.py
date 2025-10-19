"""Compare agency route centerlines (shapefile) to GTFS shapes and stops, with diagnostics.

- Join on ROUTE_NUMB ⇔ GTFS route_short_name (normalized, strict).
- Pick a representative GTFS shape per route (by trip count) and use stops from trips
  that use that shape (GTFS-as-truth).
- Compute minimal buffer around the agency line to contain the representative GTFS line
  (directed sampled distances) and that shape’s stops.
- Optionally use P95 instead of max to reduce detour outliers.
- Emit per-route diagnostic plots.

Requires: geopandas, pandas, shapely, rapidfuzz, numpy, matplotlib, pyproj
"""

from __future__ import annotations

import re
import sys
import warnings
from math import isfinite
from pathlib import Path
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Inputs ---
AGENCY_ROUTE_SHAPEFILE: Path = Path(r"File\Path\To\Your\Bus_System.shp")
GTFS_DIR: Path = Path(r"Folder\Path\To\Your\GTFS")

# --- Output (CSV + plots) ---
OUTPUT_CSV: Path = Path(r"File\Path\To\Your\route_comparison_summary.csv")
PLOT_DEBUG: bool = True
PLOT_DIR: Path = Path(r"Folder\Path\To\Your\Output\Plots")
PLOT_DPI: int = 150
PLOT_FIGSIZE: tuple[int, int] = (8, 8)
PLOT_MAX_ROUTES: Optional[int] = None  # None = all

# --- Agency fields ---
AGENCY_ROUTE_KEY_FIELD: str = "ROUTE_NUMB" # Adjust to match your shapefile
AGENCY_ROUTE_NAME_FIELD: str = "ROUTE_NAME" # Adjust to match your shapefile

# --- Join policy (strict short-name equality after normalization) ---
SHORTNAME_STRIP_NONALNUM: bool = True
SHORTNAME_UPPERCASE: bool = True
STRIP_LEADING_ZEROS: bool = True
STRIP_TRAILING_LETTERS: bool = False  # keep strict unless you *want* families collapsed

# --- GTFS-as-truth knobs ---
USE_REPRESENTATIVE_SHAPE: bool = True
STOPS_FROM_REP_SHAPE_ONLY: bool = True

# --- Buffer statistic ---
# "max"  -> full directed max; "p95" -> 95th percentile (robust to outliers)
BUFFER_STAT: str = "p95"

# --- Sampling for directed line→line distance (CRS units) ---
LINE_SAMPLING_STEP: float = 5.0  # if meters CRS, ~5 m; if feet CRS, ~15.0

# --- CRS / reporting units ---
# If None and agency CRS is geographic, auto-pick UTM (meters). Reporting is always *feet*.
TARGET_EPSG: Optional[int] = None

# --- Triage flags (not hard filters) ---
TARGET_MAX_BUFFER_FEET: float = 100.0
TARGET_MAX_NAME_DEVIATION_PCT: float = 0.0

# --- Shapes construction safeguard ---
MIN_POINTS_PER_SHAPE: int = 2

# --- Diagnostics ---
VERBOSE: bool = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)

# =============================================================================
# FUNCTIONS
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


def _axis_unit_info(crs_obj) -> tuple[Optional[str], Optional[float]]:
    """
    Return (unit_name, meters_per_unit) from pyproj CRS.axis_info if possible.
    """
    try:
        crs = CRS.from_user_input(crs_obj)
        if not crs.is_projected:
            return None, None
        if crs.axis_info:
            ax = crs.axis_info[0]
            return (ax.unit_name or "").lower(), float(ax.unit_conversion_factor or 1.0)
    except Exception:
        pass
    return None, None


def _infer_units_is_feet(crs_obj) -> Optional[bool]:
    name, m_per_u = _axis_unit_info(crs_obj)
    if name:
        if "foot" in name or "feet" in name or "us_survey_foot" in name:
            return True
        if "metre" in name or "meter" in name or "metres" in name or "meters" in name:
            return False
    if m_per_u is not None:
        if abs(m_per_u - 0.3048) < 5e-4 or abs(m_per_u - 0.3048006096) < 5e-4:
            return True
        if abs(m_per_u - 1.0) < 1e-6:
            return False
    return None


def _units_to_feet_factor(crs_obj) -> Optional[float]:
    """
    Return factor to convert CRS units → feet, or None if cannot determine.
    """
    _, m_per_u = _axis_unit_info(crs_obj)
    if m_per_u is None:
        return None
    return m_per_u * 3.28084  # meters per unit * feet per meter


def _auto_utm_epsg_from_series(geom: gpd.GeoSeries) -> int:
    c = geom.unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int((lon + 180.0) // 6.0 + 1)
    north = lat >= 0
    return (32600 + zone) if north else (32700 + zone)


def _project_agency(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, bool, float]:
    """
    Project agency to a distance-suitable CRS.

    Returns:
        (projected_gdf, is_feet, to_feet_factor) where to_feet_factor converts
        CRS units to feet.
    """
    if gdf.crs is None:
        raise ValueError("Agency shapefile has no CRS; define it before running.")

    if TARGET_EPSG:
        g2 = gdf.to_crs(epsg=TARGET_EPSG)
        log(f"[INFO] Reprojected agency to EPSG:{TARGET_EPSG}")
    else:
        crs = CRS.from_user_input(gdf.crs)
        if crs.is_projected:
            g2 = gdf
            log(f"[INFO] Agency CRS already projected: {crs.to_authority() or crs.to_string()}")
        else:
            g_ll = gdf.to_crs(epsg=4326)
            epsg = _auto_utm_epsg_from_series(g_ll.geometry)
            g2 = g_ll.to_crs(epsg=epsg)
            log(f"[INFO] Agency CRS geographic; auto-projected to UTM EPSG:{epsg}")

    is_feet = bool(_infer_units_is_feet(g2.crs))
    to_feet = _units_to_feet_factor(g2.crs)
    if to_feet is None:
        to_feet = 3.28084  # assume meters if unknown
        is_feet = False
    log(f"[INFO] Projected CRS units: {'feet' if is_feet else 'meters'} | units→feet factor={to_feet:.6f}")
    return g2, is_feet, float(to_feet)


def _build_shapes_gdf(shapes_df: pd.DataFrame, target_epsg: int | None) -> gpd.GeoDataFrame:
    req = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    miss = req - set(shapes_df.columns)
    if miss:
        raise ValueError(f"shapes.txt missing columns: {sorted(miss)}")

    sdf = shapes_df.copy()
    sdf["shape_pt_sequence"] = sdf["shape_pt_sequence"].astype(int)
    sdf["shape_pt_lat"] = sdf["shape_pt_lat"].astype(float)
    sdf["shape_pt_lon"] = sdf["shape_pt_lon"].astype(float)
    sdf = sdf.sort_values(["shape_id", "shape_pt_sequence"])

    parts: list[tuple[str, LineString]] = []
    for sid, grp in sdf.groupby("shape_id", sort=False):
        pts = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(pts) >= MIN_POINTS_PER_SHAPE:
            parts.append((sid, LineString(pts)))
    if not parts:
        raise ValueError("No valid shapes constructed from shapes.txt")

    g = gpd.GeoDataFrame(
        [{"shape_id": sid, "geometry": geom} for sid, geom in parts],
        crs="EPSG:4326",
    )
    g = g.to_crs(epsg=target_epsg) if target_epsg else g
    g = g.set_index("shape_id", drop=True)
    log(f"[INFO] Built shapes GDF with {len(g):,} geometries | target_epsg={target_epsg}")
    return g


def _stops_gdf(stops_df: pd.DataFrame, target_epsg: int | None) -> gpd.GeoDataFrame:
    req = {"stop_id", "stop_lat", "stop_lon"}
    miss = req - set(stops_df.columns)
    if miss:
        # BUGFIX: variable name was wrong previously
        raise ValueError(f"stops.txt missing columns: {sorted(miss)}")
    sdf = stops_df.copy()
    sdf["stop_lat"] = sdf["stop_lat"].astype(float)
    sdf["stop_lon"] = sdf["stop_lon"].astype(float)

    g = gpd.GeoDataFrame(
        sdf[["stop_id"]].assign(
            geometry=[Point(xy) for xy in zip(sdf["stop_lon"], sdf["stop_lat"])]
        ),
        crs="EPSG:4326",
    )
    g = g.to_crs(epsg=target_epsg) if target_epsg else g
    g = g.set_index("stop_id", drop=True)
    log(f"[INFO] Built stops GDF with {len(g):,} points | target_epsg={target_epsg}")
    return g


def _merge_lines(geoms: Iterable) -> LineString | MultiLineString:
    return linemerge(unary_union(list(geoms)))


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


def _name_divergence(a: str, b: str) -> Tuple[int, float, float]:
    a = (a or "").strip()
    b = (b or "").strip()
    lev = Levenshtein.distance(a, b)
    sim = float(fuzz.token_sort_ratio(a, b))
    dev = 100.0 - sim
    return lev, sim, dev


# ---------- directed distances (sampling) ----------

def _sample_points_along_line(line: LineString, step: float) -> np.ndarray:
    L = line.length
    if L <= 0.0:
        x, y = line.coords[0]
        return np.array([[x, y]], dtype=float)
    n = max(1, int(np.ceil(L / step)))
    d = np.linspace(0.0, L, num=n + 1)
    pts = [line.interpolate(float(t)) for t in d]
    return np.array([[p.x, p.y] for p in pts], dtype=float)


def _directed_line_to_line_distances(
    src_line: LineString | MultiLineString,
    ref_line: LineString | MultiLineString,
    step: float,
) -> np.ndarray:
    if src_line is None:
        return np.array([], dtype=float)
    parts = list(src_line.geoms) if isinstance(src_line, MultiLineString) else [src_line]
    d_all = []
    for part in parts:
        coords = _sample_points_along_line(part, step)
        pts = gpd.GeoSeries([Point(x, y) for x, y in coords], crs=None)
        if len(pts):
            d = pts.distance(ref_line).to_numpy(dtype=float)
            d_all.append(d)
    return np.concatenate(d_all) if d_all else np.array([], dtype=float)


def _point_set_to_line_distances(points: gpd.GeoDataFrame, ref_line) -> np.ndarray:
    if points is None or points.empty:
        return np.array([], dtype=float)
    return points.geometry.distance(ref_line).to_numpy(dtype=float)


def _statistic(vals: np.ndarray, mode: str) -> float:
    if vals.size == 0:
        return float("nan")
    if mode == "max":
        return float(np.nanmax(vals))
    if mode == "p95":
        return float(np.nanpercentile(vals, 95))
    raise ValueError("BUFFER_STAT must be 'max' or 'p95'")


# ---------- plotting ----------

def _safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "", s)


def _plot_route_debug(
    route_key: str,
    agency_line,
    gtfs_line,
    gtfs_stops_gdf,
    buffer_units: float | None,
    out_dir: Path,
    crs_units_label: str,
    figsize: tuple[int, int],
    dpi: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"route_{_safe_name(str(route_key))}.png"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    gpd.GeoSeries([agency_line]).plot(ax=ax, linewidth=2.0, label="Agency line")
    if gtfs_line is not None:
        gpd.GeoSeries([gtfs_line]).plot(ax=ax, linewidth=1.2, linestyle="--", label="GTFS line")
    if gtfs_stops_gdf is not None and not gtfs_stops_gdf.empty:
        gtfs_stops_gdf.plot(ax=ax, markersize=10, marker="o", label="GTFS stops")
    if buffer_units is not None and isfinite(buffer_units):
        gpd.GeoSeries([agency_line.buffer(buffer_units)]).plot(ax=ax, alpha=0.2, label=f"Buffer ({crs_units_label})")

    bounds = np.array(gpd.GeoSeries([agency_line]).total_bounds)
    if gtfs_line is not None:
        bounds = np.vstack([bounds, gpd.GeoSeries([gtfs_line]).total_bounds])
    if gtfs_stops_gdf is not None and not gtfs_stops_gdf.empty:
        bounds = np.vstack([bounds, gtfs_stops_gdf.total_bounds])
    xmin, ymin, xmax, ymax = bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()
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

# -----------------------------------------------------------------------------
# CORE
# -----------------------------------------------------------------------------

def compute_per_route_metrics() -> pd.DataFrame:
    # --- GTFS ---
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

    # --- Agency ---
    _ensure_exists(AGENCY_ROUTE_SHAPEFILE)
    agency_gdf = gpd.read_file(AGENCY_ROUTE_SHAPEFILE)
    if agency_gdf.empty:
        raise ValueError("Agency shapefile contains no features.")
    for fld in [AGENCY_ROUTE_KEY_FIELD, AGENCY_ROUTE_NAME_FIELD]:
        if fld not in agency_gdf.columns:
            raise ValueError(
                f"Shapefile missing field '{fld}'. "
                f"Available: {', '.join(agency_gdf.columns)}"
            )
    agency_gdf = agency_gdf.rename(
        columns={AGENCY_ROUTE_KEY_FIELD: "route_key", AGENCY_ROUTE_NAME_FIELD: "agency_route_name"}
    )

    # Project agency; determine reporting conversion to feet
    agency_proj, is_feet, to_feet = _project_agency(agency_gdf)
    crs_units_label = "feet" if is_feet else "meters"

    # Dissolve to one per route_key
    agency_routes = (
        agency_proj[["route_key", "agency_route_name", "geometry"]]
        .dissolve(by="route_key", as_index=False, aggfunc={"agency_route_name": "first"})
    )
    log(f"[INFO] Agency routes after dissolve: {len(agency_routes):,}")

    target_epsg = int(agency_routes.crs.to_epsg()) if agency_routes.crs else None
    shapes_gdf = _build_shapes_gdf(shapes_df, target_epsg)
    stops_gdf = _stops_gdf(stops_df, target_epsg)

    # --- GTFS indices ---
    routes_df["_norm_short"] = routes_df["route_short_name"].map(_norm_short_name)
    short_to_routeids = routes_df.groupby("_norm_short")["route_id"].apply(list).to_dict()
    empties = routes_df[routes_df["_norm_short"].isna() | (routes_df["_norm_short"] == "")]
    if len(empties):
        log(f"[WARN] {len(empties)} GTFS routes have empty normalized short names")

    trips_per_route = trips_df.groupby("route_id")["trip_id"].nunique().to_dict()
    trips_per_shape = trips_df.groupby("shape_id")["trip_id"].nunique().to_dict()

    records: list[dict] = []
    plotted = 0

    for _, arow in agency_routes.iterrows():
        try:
            a_key_raw = arow["route_key"]
            a_key_norm = _norm_short_name(a_key_raw)
            a_name = str(arow.get("agency_route_name") or "")
            a_geom = arow.geometry

            mapping_status = "ok"
            mapping_notes = ""
            chosen_rid: Optional[str] = None
            candidate_rids = short_to_routeids.get(a_key_norm, [])

            if not candidate_rids:
                mapping_status = "unmatched_short_name"
            elif len(candidate_rids) == 1:
                chosen_rid = candidate_rids[0]
            else:
                cand = pd.DataFrame({"route_id": candidate_rids})
                cand["trip_count"] = cand["route_id"].map(lambda r: trips_per_route.get(r, 0))
                cand = cand.sort_values(["trip_count", "route_id"], ascending=[False, True])
                chosen_rid = str(cand.iloc[0]["route_id"])
                mapping_status = "short_name_ambiguous_resolved"
                mapping_notes = f"Candidates={';'.join(candidate_rids)}; selected={chosen_rid} by route trip_count."

            long_nm = routes_df.loc[routes_df["route_id"] == chosen_rid, "route_long_name"].values[0] if chosen_rid else ""
            short_nm = routes_df.loc[routes_df["route_id"] == chosen_rid, "route_short_name"].values[0] if chosen_rid else ""

            if chosen_rid:
                shape_ids = (
                    trips_df.loc[trips_df["route_id"] == chosen_rid, "shape_id"]
                    .dropna().unique().tolist()
                )
                rep_shape_id = None
                if USE_REPRESENTATIVE_SHAPE and shape_ids:
                    rep_shape_id = max(shape_ids, key=lambda sid: trips_per_shape.get(sid, 0))

                if rep_shape_id and rep_shape_id in shapes_gdf.index:
                    gtfs_line = shapes_gdf.loc[rep_shape_id].geometry
                    shape_pick = f"rep:{rep_shape_id}"
                else:
                    sel = shapes_gdf.loc[shapes_gdf.index.intersection(shape_ids)]
                    gtfs_line = _merge_lines(sel.geometry) if not sel.empty else None
                    shape_pick = "merged_all" if shape_ids else "none"

                if STOPS_FROM_REP_SHAPE_ONLY and rep_shape_id:
                    rep_trip_ids = trips_df.loc[
                        (trips_df["route_id"] == chosen_rid) & (trips_df["shape_id"] == rep_shape_id),
                        "trip_id",
                    ]
                    stop_ids = stop_times_df.loc[stop_times_df["trip_id"].isin(rep_trip_ids), "stop_id"].dropna().unique().tolist()
                    stops_sel = stops_gdf.loc[stops_gdf.index.intersection(stop_ids)]
                    stops_pick = "rep_shape_trips"
                else:
                    trip_ids = trips_df.loc[trips_df["route_id"] == chosen_rid, "trip_id"]
                    stop_ids = stop_times_df.loc[stop_times_df["trip_id"].isin(trip_ids), "stop_id"].dropna().unique().tolist()
                    stops_sel = stops_gdf.loc[stops_gdf.index.intersection(stop_ids)]
                    stops_pick = "all_route_trips"

                line_dists = _directed_line_to_line_distances(gtfs_line, a_geom, LINE_SAMPLING_STEP) if gtfs_line else np.array([])
                stop_dists = _point_set_to_line_distances(stops_sel, a_geom)

                stat_line = _statistic(line_dists, BUFFER_STAT) if line_dists.size else float("nan")
                stat_stop = _statistic(stop_dists, BUFFER_STAT) if stop_dists.size else float("nan")
                buffer_units = np.nanmax([stat_line, stat_stop])
                buffer_feet = buffer_units * to_feet if isfinite(buffer_units) else float("nan")
            else:
                gtfs_line = None
                stops_sel = gpd.GeoDataFrame(geometry=[], crs=stops_gdf.crs)
                shape_pick, stops_pick = "none", "none"
                buffer_units = float("nan")
                buffer_feet = float("nan")

            if chosen_rid:
                lev, sim, dev = _name_divergence(a_name, long_nm)
            else:
                lev, sim, dev = (np.nan, np.nan, np.nan)

            plot_path = ""
            if PLOT_DEBUG and chosen_rid and (gtfs_line is not None):
                if (PLOT_MAX_ROUTES is None) or (plotted < int(PLOT_MAX_ROUTES)):
                    try:
                        out_png = _plot_route_debug(
                            route_key=str(a_key_raw),
                            agency_line=a_geom,
                            gtfs_line=gtfs_line,
                            gtfs_stops_gdf=stops_sel,
                            buffer_units=(buffer_units if isfinite(buffer_units) else None),
                            out_dir=PLOT_DIR,
                            crs_units_label=crs_units_label,
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
                    "min_buffer_feet": None if not isfinite(buffer_feet) else round(float(buffer_feet), 2),
                    "name_levenshtein_distance": None if isinstance(lev, float) and np.isnan(lev) else (int(lev) if isinstance(lev, (int, np.integer)) else lev),
                    "name_similarity_pct": None if isinstance(sim, float) and np.isnan(sim) else round(float(sim), 1),
                    "name_deviation_pct": None if isinstance(dev, float) and np.isnan(dev) else round(float(dev), 1),
                    "exceeds_buffer_threshold": False if not isfinite(buffer_feet) else (buffer_feet > TARGET_MAX_BUFFER_FEET),
                    "exceeds_name_threshold": False if not isfinite(dev) else (dev > TARGET_MAX_NAME_DEVIATION_PCT),
                    "crs_units": crs_units_label,
                    "line_sampling_step_units": LINE_SAMPLING_STEP,
                    "plot_path": plot_path,
                }
            )
        except Exception as e:
            log(f"[ERROR] Route '{arow.get('route_key', '?')}' failed: {e}")
            # Keep a row to make failures visible in the CSV
            records.append(
                {
                    "route_key_agency": str(arow.get("route_key", "")),
                    "route_key_normalized": _norm_short_name(arow.get("route_key", "")),
                    "agency_route_name": str(arow.get("agency_route_name", "")),
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
                    "crs_units": crs_units_label,
                    "line_sampling_step_units": LINE_SAMPLING_STEP,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
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
