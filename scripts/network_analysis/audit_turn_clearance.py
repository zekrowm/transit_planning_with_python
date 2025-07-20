#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flags GTFS bus stops with insufficient distance before a downstream left turn.

This script identifies stops placed too close to a left-turn movement along the
same route and direction, which may violate clearance standards or safety
guidelines.

Overview:
    The script analyzes GTFS stops and shapes, reconstructs the served
    stop geometry by route and direction, and detects turning vertices
    with configurable thresholds. Stops upstream of left turns are flagged
    if the spacing is less than a specified minimum.

    Several geometric filters are applied to avoid false positives due to
    minor shape artifacts or route alignment noise:
    - Curb-proximity filter: suppresses turns that occur too close to any stop.
    - Lateral offset filter: rejects jogs that stay within a narrow corridor.

Workflow:
    1. Read and validate core GTFS files.
    2. Optionally filter to allowed or denied routes.
    3. Create GeoDataFrames for stops and route polylines.
    4. Identify left-turn vertices using angle and segment-length criteria.
    5. For each turn, find the last upstream served stop and calculate spacing.
    6. Flag violations where spacing is positive and less than MIN_CLEARANCE_FT.
    7. Write a tab-delimited log of flagged events.
    8. Optionally export:
        - A shapefile of flagged stops.
        - PNG figures showing a zoomed-in context map with stop, turn, route, and roads.
"""

import re
import numpy as np
from shapely.geometry import box, LineString, Point
from shapely.ops import substring
import matplotlib.pyplot as plt
from __future__ import annotations
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import split as split_line
from shapely.ops import substring
from shapely.ops import substring, split as split_line
from shapely.geometry import LineString, Point, box

# =============================================================================
# CONFIGURATION
# =============================================================================
GTFS_PATH: str = r"Path\To\Your\GTFS_Folder"          # folder or .zip
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"
ROAD_CENTERLINE_SHP: str | None = r"Path\To\Your\Roadway_Centerlines.shp"  # ← set to None to skip

INCLUDE_ROUTE_IDS: list[str] = []              # empty = keep all
EXCLUDE_ROUTE_IDS: list[str] = ["9999A", "9999B", "9999C"]              # e.g. ["9999A"]

PROJECTED_CRS: str = "EPSG:2263"               # feet
SIMPLIFY_FT: float = 5.0                       # pre‑simplify shapes (0 = off)

# Left‑turn detection parameters
MIN_DEFLECTION_DEG: float = 55.0               # ≥ this angle ⇒ ‘turn’
MIN_LEG_FT: float = 30.0                       # each incident segment ≥
MIN_CLEARANCE_FT: float = 60.0                 # spacing threshold (stop→turn)

LOG_FILENAME: str = "left_turn_spacing.txt"
EXPORT_FLAGGED_SHP: bool = True                # also write a shapefile
FLAGGED_SHP_NAME: str = "left_turn_flags.shp"

# optional plots
EXPORT_FLAGGED_PNGS: bool = True          # ⇦ turn on/off
PNG_SUBDIR: str = "left_turn_figures"     # folder inside OUTPUT_FOLDER
PNG_DPI: int = 150                        # resolution
PNG_WINDOW_FT: float = 400.0              # half‑width of zoom window around stop

# -----------------------------------------------------------------------------
# optional roadway backdrop
PLOT_ROAD_CENTERLINES: bool = True
ROAD_SIMPLIFY_FT: float = 0.0            # >0  ⇒ geometry.simplify() before use

TURN_STOP_BUFFER_FT: float = 15.0        # ignore ‘turns’ within this many feet of a stop

# -----------------------------------------------------------------------------
# ‘wiggle’ suppression
LOOK_AHEAD_FT: float = 40.0            # how far before/after the vertex to sample
MIN_LATERAL_OFFSET_FT: float = 10.0    # must exceed this to be a real turn

# -----------------------------------------------------------------------------
# direction‑arrow settings
ARROW_SPACING_FT: float = 100.0     # distance between arrow heads along the line
ARROW_SIZE: float = 10.0           # matplotlib arrow head size (points)

# =============================================================================
# FUNCTIONS
# =============================================================================

def _ensure_output(folder: str | Path) -> Path:
    """Create *folder* (if needed) and return it as a ``Path``."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_gtfs(gtfs_path: Path) -> Dict[str, pd.DataFrame]:
    """Load core GTFS tables into DataFrames.

    Parameters
    ----------
    gtfs_path
        Directory containing *.txt files or a *.zip* GTFS bundle.

    Returns
    -------
    dict
        Keys ``stops, routes, trips, stop_times, shapes`` → DataFrames.
    """
    names = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
        "shapes": "shapes.txt",
    }

    if gtfs_path.is_dir():
        return {k: pd.read_csv(gtfs_path / v) for k, v in names.items()}

    if gtfs_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        return {k: pd.read_csv(root / v) for k, v in names.items()}

    raise ValueError("GTFS_PATH must be a directory or .zip file.")


def _validate_gtfs(dfs: Dict[str, pd.DataFrame]) -> None:
    """Raise if any required column is missing."""
    req: Dict[str, set[str]] = {
        "stops": {"stop_id", "stop_lat", "stop_lon", "stop_name"},
        "routes": {"route_id", "route_short_name"},
        "trips": {"trip_id", "route_id", "shape_id", "direction_id"},
        "stop_times": {"trip_id", "stop_id"},
        "shapes": {"shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"},
    }
    missing: list[str] = []
    for tbl, need in req.items():
        miss = need - set(dfs[tbl].columns)
        if miss:
            missing.append(f"{tbl}: {', '.join(sorted(miss))}")
    if missing:
        joined = "; ".join(missing)
        raise ValueError(f"GTFS missing required columns → {joined}")


def _load_road_centerlines(
    shp_path: str | Path,
    target_crs: str,
    simplify_ft: float = 0.0,
) -> gpd.GeoDataFrame:
    """Read, re‑project, and (optionally) simplify a centerline layer."""
    roads = gpd.read_file(shp_path)
    if roads.crs != target_crs:
        roads = roads.to_crs(target_crs)

    if simplify_ft > 0:
        factor = 1.0 if "2263" in target_crs else 3.28084
        roads["geometry"] = roads.geometry.simplify(simplify_ft / factor)

    print(f"Road layer: {len(roads):,} features from {Path(shp_path).name}")
    return roads


# -----------------------------------------------------------------------------
# BUILD SPATIAL LAYERS
# -----------------------------------------------------------------------------

def _filter_routes(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    allow: Sequence[str],
    deny: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return filtered copies of *routes* and *trips*."""
    routes_ok = routes.copy()
    if deny:
        routes_ok = routes_ok[~routes_ok["route_id"].isin(deny)]
    if allow:
        routes_ok = routes_ok[routes_ok["route_id"].isin(allow)]
    trips_ok = trips[trips["route_id"].isin(routes_ok["route_id"])]
    return routes_ok, trips_ok


def _build_stops_gdf(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    """GeoDataFrame of served stops with per‑stop route/direction lists."""
    served = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
    stops = stops[stops["stop_id"].isin(served["stop_id"])].copy()

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs="EPSG:4326",
    ).to_crs(crs)

    trip_attrs = trips[["trip_id", "route_id", "direction_id"]].merge(
        routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )
    merged = served[["trip_id", "stop_id"]].merge(trip_attrs, on="trip_id", how="left")
    agg = (
        merged.groupby("stop_id")[["route_id", "direction_id", "route_short_name"]]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    gdf = gdf.merge(agg, on="stop_id", how="left")

    print(f"Stops layer: {len(gdf):,} served stops.")
    return gdf


def _build_routes_gdf(
    shapes: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    crs: str,
    simplify_ft: float,
) -> gpd.GeoDataFrame:
    """Polyline GeoDataFrame keyed by (route_id, direction_id)."""
    shape_cols = ["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"]
    lines = (
        shapes[shape_cols]
        .sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id")
        .apply(lambda g: LineString(zip(g.shape_pt_lon, g.shape_pt_lat)))
        .to_frame("geometry")
        .reset_index()
    )

    gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326").to_crs(crs)
    gdf = gdf.merge(
        trips.drop_duplicates("shape_id")[["shape_id", "route_id", "direction_id"]],
        on="shape_id",
        how="left",
    ).merge(routes, on="route_id", how="left")

    # discard shapes with missing direction_id
    gdf = gdf[gdf["direction_id"].notna()].copy()

    if simplify_ft > 0:
        factor = 1.0 if "2263" in crs else 3.28084
        gdf["geometry"] = gdf.geometry.apply(lambda ln: ln.simplify(simplify_ft / factor))

    print(f"Routes layer: {len(gdf):,} polylines.")
    return gdf


# -----------------------------------------------------------------------------
# LEFT‑TURN DETECTION & QA
# -----------------------------------------------------------------------------

def _find_left_turns(
    line: LineString,
    *,
    min_deflect_deg: float,
    min_leg_ft: float,
) -> list[tuple[float, float]]:
    """Detect left turns on *line*.

    Parameters
    ----------
    line
        Polyline in a projected CRS (x = east, y = north, units = ft or m).
    min_deflect_deg
        Deflection angle threshold (θ ≥ this ⇒ ‘turn’).
    min_leg_ft
        Each incident segment must be at least this long.

    Returns
    -------
    list[tuple[float, float]]
        ``[(dist_along, angle_deg), …]`` for every qualifying left turn,
        where *dist_along* is the line‑projected distance of the vertex.
    """
    pts = np.asarray(line.coords)
    out: list[tuple[float, float]] = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < min_leg_ft or len2 < min_leg_ft:
            continue

        cos_theta = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        if theta < min_deflect_deg:
            continue

        # sign of z‑component of v1×v2 indicates left (positive) or right (negative)
        if np.cross(v1, v2) > 0.0:
            out.append((line.project(Point(pts[i])), theta))
    return out


def _flag_left_turn_spacing(
    routes_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    *,
    min_deflect_deg: float,
    min_leg_ft: float,
    clearance_ft: float,
    log_path: Path,
) -> gpd.GeoDataFrame:
    """Detect stops with inadequate distance to the next left turn.

    Rejects false ‘turns’ that are
    • within TURN_STOP_BUFFER_FT of any served stop, *or*
    • have a lateral offset < MIN_LATERAL_OFFSET_FT from the straight‑line
      chord connecting points LOOK_AHEAD_FT before and after the vertex.
    """
    crs_str = str(stops_gdf.crs)
    ft_factor = 1.0 if "2263" in crs_str else 3.28084
    sindex = stops_gdf.sindex

    flagged: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "route_id\tdirection_id\tstop_id\tstop_name\tspacing_ft\tturn_angle_deg\n"
        )

        for _, r in routes_gdf.iterrows():
            rid: str = str(r.route_id)
            drn: int = int(r.direction_id)
            line: LineString = r.geometry

            # served stops for this route × direction
            cand = stops_gdf.iloc[list(sindex.intersection(line.bounds))]
            served = cand[
                cand.route_id.apply(lambda xs: rid in xs)
                & cand.direction_id.apply(lambda xs: drn in xs)
            ].copy()
            if len(served) < 2:
                continue

            served["dist_along"] = served.geometry.apply(line.project)
            served.sort_values("dist_along", inplace=True)
            served.drop_duplicates("dist_along", inplace=True)

            # left‑turn vertices
            lefts = _find_left_turns(
                line,
                min_deflect_deg=min_deflect_deg,
                min_leg_ft=min_leg_ft / ft_factor,
            )
            if not lefts:
                continue

            stop_idx = 0
            for d_turn, ang in lefts:
                turn_pt = line.interpolate(d_turn)

                # 1) curb‑poke filter
                min_stop_dist_ft = (
                    min(turn_pt.distance(pt) for pt in served.geometry) * ft_factor
                )
                if min_stop_dist_ft <= TURN_STOP_BUFFER_FT:
                    continue

                # 2) lateral‑offset filter
                look = LOOK_AHEAD_FT / ft_factor
                d0 = max(d_turn - look, 0.0)
                d1 = min(d_turn + look, line.length)
                chord = LineString(
                    [line.interpolate(d0), line.interpolate(d1)]
                )
                offset_ft = turn_pt.distance(chord) * ft_factor
                if offset_ft < MIN_LATERAL_OFFSET_FT:
                    continue

                # find last upstream stop
                while (
                    stop_idx + 1 < len(served)
                    and served.iloc[stop_idx + 1].dist_along < d_turn
                ):
                    stop_idx += 1
                stop_row = served.iloc[stop_idx]

                spacing_ft = (d_turn - stop_row.dist_along) * ft_factor
                if spacing_ft <= 0 or spacing_ft >= clearance_ft:
                    continue

                fh.write(
                    f"{rid}\t{drn}\t{stop_row.stop_id}\t"
                    f"{stop_row.stop_name}\t{spacing_ft:.1f}\t{ang:.1f}\n"
                )
                flagged.append(
                    {
                        "route_id": rid,
                        "direction_id": drn,
                        "stop_id": stop_row.stop_id,
                        "stop_name": stop_row.stop_name,
                        "spacing_ft": round(spacing_ft, 1),
                        "angle_deg": round(ang, 1),
                        "geometry": stop_row.geometry,
                    }
                )

    print(
        f"Wrote left‑turn spacing log → {log_path.name} "
        f"({'no issues' if not flagged else f'{len(flagged):,} issues'})"
    )
    return gpd.GeoDataFrame(flagged, crs=stops_gdf.crs)


import re
from shapely.geometry import box
from shapely.ops import substring


def _export_flagged_pngs(
    flagged_gdf: gpd.GeoDataFrame,
    routes_gdf: gpd.GeoDataFrame,
    *,
    out_dir: Path,
    subdir: str,
    dpi: int = 150,
    window_ft: float = PNG_WINDOW_FT,
    roads_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Write one zoom‑in PNG per flagged stop violation (roads + dashed green).

    Filename pattern:
        <route_id>_dir<direction_id>_<stop_id>_<stop_name_slug>.png
    """
    if flagged_gdf.empty:
        print("PNG export skipped – no flagged stops.")
        return

    png_dir = (Path(out_dir) / subdir).resolve()
    png_dir.mkdir(parents=True, exist_ok=True)

    roads_sindex = roads_gdf.sindex if roads_gdf is not None else None

    crs_str = str(routes_gdf.crs)
    ft2crs = 1.0 if "2263" in crs_str else (1.0 / 3.28084)

    for _, row in flagged_gdf.iterrows():
        rid, drn = row.route_id, row.direction_id
        stop_id, stop_name = row.stop_id, row.stop_name
        stop_pt: Point = row.geometry

        # route polyline for this direction
        line = routes_gdf.loc[
            (routes_gdf.route_id == rid) & (routes_gdf.direction_id == drn), "geometry"
        ].iloc[0]
        if not isinstance(line, LineString):
            line = max(line, key=lambda g: g.length)

        # distance chainages
        d_stop = line.project(stop_pt)
        d_turn = d_stop + (row.spacing_ft * ft2crs)
        turn_pt = line.interpolate(d_turn)
        try:
            seg = substring(line, d_stop, d_turn, normalized=False)
        except Exception:
            seg = LineString([stop_pt, turn_pt])

        # plotting window
        mid_pt = seg.interpolate(0.5, normalized=True)
        half = window_ft * ft2crs
        minx, miny, maxx, maxy = (
            mid_pt.x - half,
            mid_pt.y - half,
            mid_pt.x + half,
            mid_pt.y + half,
        )
        win_box = box(minx, miny, maxx, maxy)

        # ─── figure ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4, 4))

        # roads backdrop (dark‑gray)
        if roads_gdf is not None:
            idx = list(roads_sindex.intersection(win_box.bounds))
            for geom in roads_gdf.geometry.iloc[idx]:
                clipped = geom.intersection(win_box)
                if clipped.is_empty:
                    continue
                if clipped.geom_type == "LineString":
                    ax.plot(*clipped.xy, lw=0.5, color="darkgray")
                else:  # MultiLineString
                    for part in clipped.geoms:
                        ax.plot(*part.xy, lw=0.5, color="darkgray")

        # full route polyline (dashed green)
        ax.plot(
            *line.xy,
            lw=1,
            color="green",
            linestyle="--",
            label="Route polyline",
        )

        # offending segment (red)
        ax.plot(*seg.xy, lw=3, color="red", label="Stop → left turn")

        # stop marker (blue)
        ax.scatter(
            stop_pt.x,
            stop_pt.y,
            s=50,
            color="blue",
            zorder=3,
            label="Flagged stop",
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{rid} dir{drn}  stop {stop_id}\n"
            f"{row.spacing_ft:.0f} ft to left turn (θ≈{row.angle_deg:.0f}°)",
            fontsize=9,
        )
        ax.legend(loc="lower left", fontsize=7, frameon=False)

        # ---- filename with stop name slug ---------------------------------
        stop_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(stop_name))[:40].strip("_")
        fname = f"{rid}_dir{drn}_{stop_id}_{stop_slug}.png"

        fig.savefig(png_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {len(flagged_gdf):,} zoom PNGs → {png_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:  # noqa: D401
    """Run the GTFS left‑turn clearance QA pipeline."""
    # -------------------------------------------------------------
    # STEP 0 — Read & validate GTFS
    # -------------------------------------------------------------
    print("Reading GTFS …")
    dfs = _read_gtfs(Path(GTFS_PATH))
    _validate_gtfs(dfs)

    # -------------------------------------------------------------
    # STEP 0·1 — Route / trip filtering
    # -------------------------------------------------------------
    routes_df, trips_df = _filter_routes(
        dfs["routes"], dfs["trips"], INCLUDE_ROUTE_IDS, EXCLUDE_ROUTE_IDS
    )

    out_dir = _ensure_output(OUTPUT_FOLDER)

    # -------------------------------------------------------------
    # STEP 1 — Build spatial layers
    # -------------------------------------------------------------
    print("Building spatial layers …")
    stops_gdf = _build_stops_gdf(
        dfs["stops"], dfs["stop_times"], trips_df, routes_df, PROJECTED_CRS
    )
    routes_gdf = _build_routes_gdf(
        dfs["shapes"],
        trips_df,
        routes_df,
        PROJECTED_CRS,
        simplify_ft=SIMPLIFY_FT,
    )

    # -------------------------------------------------------------
    # STEP 2 — Left‑turn spacing QA
    # -------------------------------------------------------------
    print("Running left‑turn spacing QA …")
    flagged_gdf = _flag_left_turn_spacing(
        routes_gdf,
        stops_gdf,
        min_deflect_deg=MIN_DEFLECTION_DEG,
        min_leg_ft=MIN_LEG_FT,
        clearance_ft=MIN_CLEARANCE_FT,
        log_path=out_dir / LOG_FILENAME,
    )

    # -------------------------------------------------------------
    # STEP 2·1 — Optional road backdrop
    # -------------------------------------------------------------
    roads_gdf = None
    if PLOT_ROAD_CENTERLINES and ROAD_CENTERLINE_SHP:
        roads_gdf = _load_road_centerlines(
            ROAD_CENTERLINE_SHP,
            target_crs=PROJECTED_CRS,
            simplify_ft=ROAD_SIMPLIFY_FT,
        )

    # -------------------------------------------------------------
    # STEP 3 — Optional shapefile export
    # -------------------------------------------------------------
    if EXPORT_FLAGGED_SHP and not flagged_gdf.empty:
        shp_path = out_dir / FLAGGED_SHP_NAME
        flagged_gdf.to_file(shp_path)
        print(f"Wrote {shp_path.name}")

    # -------------------------------------------------------------
    # STEP 4 — Optional per‑stop PNG export (zoomed, with roads)
    # -------------------------------------------------------------
    if EXPORT_FLAGGED_PNGS:
        _export_flagged_pngs(
            flagged_gdf,
            routes_gdf,
            out_dir=out_dir,
            subdir=PNG_SUBDIR,
            dpi=PNG_DPI,
            roads_gdf=roads_gdf,
        )

    # -------------------------------------------------------------
    # FIN
    # -------------------------------------------------------------
    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print("\nUNEXPECTED ERROR:", exc)
        sys.exit(1)
