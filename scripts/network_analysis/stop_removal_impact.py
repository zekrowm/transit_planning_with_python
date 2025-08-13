"""Analyze sidewalk-access impacts from GTFS stop removals using a spatial network.

This module quantifies pedestrian-access changes caused by bus stop removals by
combining GTFS stop data with a segmented sidewalk network. It measures the
change in 0.25-mile sidewalk-buffer coverage before and after stop deletion
and calculates network-based walking distances from each removed stop to its
nearest retained stop.

Key Features:
    * Builds a segmented ("physical") pedestrian graph from sidewalk or road
      centerlines, treating each segment as a distinct edge.
    * Snaps stops to their nearest sidewalk segment and splits the segment at
      the snap location to maintain network topology.
    * Calculates coverage area lost when stops are removed.
    * Computes both linear and network-based walking distances to the nearest
      retained stop within the search radius.
    * Exports CSV, shapefile, and QA map outputs showing removed stops and
      shortest walking paths.

This script is intended for transportation planners and GIS analysts assessing
the accessibility impacts of stop consolidation, relocation, or removal.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box
from shapely.ops import substring
from shapely.strtree import STRtree

# =============================================================================
# CONFIGURATION
# =============================================================================

SIDEWALK_SHP = Path(r"Path\To\Your\Sidewalks_Centerline.shp")  # Must be a functional network
GTFS_DIR = Path(r"Path\To\Your\GTFS_Folder")  # folder path must contain stops.txt
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

# Plotting-only backdrop (not used for analysis)
PLOT_SIDEWALKS_SHP: Optional[Path] = Path(r"Path\To\Your\Sidewalks_Centerline.shp")
SIDEWALK_BACKDROP_PAD_FT: float = 300.0  # how far to expand the map view when clipping

IDENTIFIER_PRIORITY: Tuple[str, str] = ("stop_code", "stop_id")

# Analysis parameters
TARGET_CRS = 6447  # VA State Plane (US ft)
BUFFER_MILES = 0.25
FT_PER_MILE = 5_280
BUFFER_FT = BUFFER_MILES * FT_PER_MILE
MAX_SNAP_FT: float = 1500.0  # skip stops farther than this from the network

# Endpoint merging: snap endpoints to a grid to connect near-coincident nodes
NODE_GRID_FT: float = 5.0  # merge endpoints within ~±2.5 ft

# “Across-the-street” sanity guard
ACROSS_STREET_MAX_FT: float = 120.0  # straight-line threshold to consider “across street”
ACROSS_STREET_RATIO: float = 8.0  # network/linear ratio considered absurd
ACROSS_STREET_ABS_FT: float = 2000.0  # or absolute detour threshold

# Stop selection for analysis
DELETED_STOP_IDS: List[str] = ["1001", "1002"]

# Outputs
EXPORT_MAPS: bool = True  # write small PNG map per deleted stop

# Logging
LOG_LEVEL = logging.INFO

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

NodeKey = Tuple[float, float]  # quantized (x, y)
EdgeID = int

# =============================================================================
# FUNCTIONS
# =============================================================================


def _load_backdrop_layer_for_plots(
    path: Optional[Path], target_crs: int | str
) -> Optional[gpd.GeoDataFrame]:
    """Load a linework backdrop for plotting only."""
    if path is None:
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Backdrop load failed (%s); continuing without it.", exc)
        return None

    if gdf.empty or gdf.geometry.isna().all():
        logging.warning("Backdrop layer is empty; skipping.")
        return None

    if gdf.crs is None:
        logging.warning("Backdrop layer has no CRS; skipping to avoid misplotting.")
        return None

    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])][["geometry"]]
    gdf.reset_index(drop=True, inplace=True)
    return gdf


def _plot_backdrop_within_bounds(
    ax: Any,
    backdrop: Optional[gpd.GeoDataFrame],
    bounds: Tuple[float, float, float, float],
    crs: int | str,
) -> None:
    """Plot a clipped backdrop within expanded bounds."""
    if backdrop is None:
        return
    xmin, ymin, xmax, ymax = bounds
    pad = SIDEWALK_BACKDROP_PAD_FT
    clip_poly = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    try:
        # Fast pre-filter by bbox intersection
        sub = backdrop[backdrop.geometry.intersects(clip_poly)].copy()
        if sub.empty:
            return
        try:
            sub = gpd.clip(sub, clip_poly)
        except Exception:
            # If clip not available, fall back to intersects only
            pass
        sub.plot(ax=ax, linewidth=0.6, alpha=0.6, zorder=0)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Backdrop clip/plot failed: %s", exc)


def resolve_deleted_stop_ids(
    stops: gpd.GeoDataFrame,
    identifiers: Sequence[str],
    prefer_stop_code: bool = True,
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve human-entered identifiers to canonical GTFS stop_ids."""
    sid_series = stops["stop_id"].astype(str)
    if "stop_code" in stops.columns:
        sc_series = stops["stop_code"].astype(str)
    else:
        sc_series = pd.Series([""] * len(stops), index=stops.index, dtype=str)

    id_by_stop_id: dict[str, list[str]] = (
        sid_series.to_frame(name="stop_id").groupby("stop_id")["stop_id"].apply(list).to_dict()
    )
    id_by_stop_code: dict[str, list[str]] = (
        pd.DataFrame({"stop_code": sc_series, "stop_id": sid_series})
        .groupby("stop_code")["stop_id"]
        .apply(list)
        .to_dict()
    )

    first, second = ("stop_code", "stop_id") if prefer_stop_code else ("stop_id", "stop_code")
    lookups = {"stop_id": id_by_stop_id, "stop_code": id_by_stop_code}

    resolved: list[str] = []
    match_map: dict[str, list[str]] = {}

    for raw in identifiers:
        key = str(raw)
        matched: list[str] = []
        for field in (first, second):
            lst = lookups[field].get(key, [])
            if lst:
                matched = lst
                break
        match_map[key] = matched
        resolved.extend(matched)

    n_in = len(identifiers)
    n_res = len(set(resolved))
    n_only_sid = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_id) and (k not in id_by_stop_code)
    )
    n_only_sc = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_code) and (k not in id_by_stop_id)
    )
    n_both = sum(
        1 for k, v in match_map.items() if v and (k in id_by_stop_code) and (k in id_by_stop_id)
    )
    n_none = sum(1 for v in match_map.values() if not v)

    logging.info(
        "Deleted list resolved: %d identifiers → %d unique stop_ids "
        "(only stop_code: %d, only stop_id: %d, both: %d, unmatched: %d)",
        n_in,
        n_res,
        n_only_sc,
        n_only_sid,
        n_both,
        n_none,
    )
    if n_none:
        lost = [k for k, v in match_map.items() if not v]
        logging.warning(
            "Unmatched identifiers (neither stop_id nor stop_code): %s",
            ", ".join(lost[:20]) + ("…" if len(lost) > 20 else ""),
        )

    return resolved, match_map


def quantize_node(x: float, y: float, step_ft: float = NODE_GRID_FT) -> NodeKey:
    """Snap (x, y) to a square grid of size 'step_ft' (feet)."""
    return (round(float(x) / step_ft) * step_ft, round(float(y) / step_ft) * step_ft)


def linestring_length(line: LineString) -> float:
    """Return length as float (helps mypy and avoids numpy types)."""
    return float(line.length)


def safe_nearest(seg_index: STRtree, pt: Point) -> int | LineString:
    """Return the nearest result from an STRtree (Shapely 2.x or pygeos style)."""
    try:
        return seg_index.nearest(pt)
    except TypeError:
        return seg_index.nearest(pt, 1)[0]


def linestring_substring(line: LineString, start_m: float, end_m: float) -> LineString:
    """Return a portion of a LineString between two measures (absolute units)."""
    L = linestring_length(line)
    if L == 0.0:
        return line
    s0 = max(0.0, min(L, start_m))
    s1 = max(0.0, min(L, end_m))
    if s1 < s0:
        s0, s1 = s1, s0
    return substring(line, s0, s1, normalized=False)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------


def load_centerlines(path: Path, target_crs: int | str) -> gpd.GeoDataFrame:
    """Load centerlines and reproject to target CRS."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Sidewalk/centerline file has no CRS.")
    return gdf.to_crs(target_crs)


def explode_segments(centerlines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode to pure LineStrings and assign a stable edge_id."""
    segs = centerlines.explode(index_parts=False, ignore_index=True)
    segs = segs[segs.geometry.notnull()].copy()
    segs = segs[segs.geom_type == "LineString"].copy()
    segs.reset_index(drop=True, inplace=True)
    segs["edge_id"] = segs.index.astype(int)
    return segs[["edge_id", "geometry"]]


def build_graph(
    segments: gpd.GeoDataFrame,
    node_grid_ft: float = NODE_GRID_FT,
) -> tuple[nx.MultiGraph, dict[EdgeID, Tuple[NodeKey, NodeKey]]]:
    """Build an undirected MultiGraph from exploded segments."""
    G = nx.MultiGraph()
    edge_endpoints: dict[EdgeID, Tuple[NodeKey, NodeKey]] = {}
    for edge_id, geom in zip(segments.edge_id.values, segments.geometry.values):
        x1, y1 = geom.coords[0]
        x2, y2 = geom.coords[-1]
        u = quantize_node(x1, y1, node_grid_ft)
        v = quantize_node(x2, y2, node_grid_ft)
        L = linestring_length(geom)

        for n, (xx, yy) in ((u, u), (v, v)):
            if n not in G:
                G.add_node(n, x=xx, y=yy)

        G.add_edge(u, v, edge_id=int(edge_id), geometry=geom, length=L)
        edge_endpoints[int(edge_id)] = (u, v)
    return G, edge_endpoints


def build_segment_index(
    segments: gpd.GeoDataFrame,
) -> tuple[STRtree, List[LineString], Dict[bytes, EdgeID]]:
    """Create an STRtree over the segments and a WKB→edge_id lookup."""
    seg_geoms: List[LineString] = list(segments.geometry.values)  # same order as edge_id
    wkb_to_eid: Dict[bytes, int] = {g.wkb: int(eid) for g, eid in zip(seg_geoms, segments.edge_id)}
    index = STRtree(seg_geoms)
    return index, seg_geoms, wkb_to_eid


def load_gtfs_stops(gtfs_dir: Path, target_crs: int | str) -> gpd.GeoDataFrame:
    """Load GTFS stops.txt as a projected GeoDataFrame."""
    stops_csv = gtfs_dir / "stops.txt"
    df = pd.read_csv(stops_csv, dtype=str)
    df = df.drop_duplicates("stop_id")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.stop_lon.astype(float), df.stop_lat.astype(float)),
        crs="EPSG:4326",
    ).to_crs(target_crs)
    return gdf


# -----------------------------------------------------------------------------
# SNAP LOGIC (VIRTUAL CONNECTORS)
# -----------------------------------------------------------------------------


def snap_stops_to_segments(
    stops: gpd.GeoDataFrame,
    seg_index: STRtree,
    wkb_to_eid: Dict[bytes, EdgeID],
    segments: gpd.GeoDataFrame,
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]],
    max_snap_ft: float = MAX_SNAP_FT,
) -> pd.DataFrame:
    """Snap each stop to its nearest segment and compute offsets to endpoints.

    Returns DataFrame:
        stop_id, stop_name, stop_code, x, y,
        edge_id, s_ft, L_ft, u_node, v_node, to_u_ft, to_v_ft
    """
    rows: List[dict] = []

    seg_geoms = segments.geometry.values
    seg_eids = segments.edge_id.values

    for sid, sname, scode, pt in zip(
        stops["stop_id"],
        stops.get("stop_name", pd.Series("", index=stops.index)),
        stops.get("stop_code", pd.Series("", index=stops.index)),
        stops.geometry,
    ):
        nearest_obj = safe_nearest(seg_index, pt)

        if isinstance(nearest_obj, (int, np.integer)):
            idx = int(nearest_obj)
            seg = seg_geoms[idx]
            eid = int(seg_eids[idx])
        else:
            seg = nearest_obj  # assumed LineString
            eid = wkb_to_eid.get(seg.wkb, None)
            if eid is None:
                try:
                    eid = int(segments.loc[segments.geometry.wkb == seg.wkb, "edge_id"].iloc[0])
                except Exception:
                    rows.append(
                        dict(
                            stop_id=sid,
                            stop_name=str(sname),
                            stop_code=str(scode),
                            x=float(pt.x),
                            y=float(pt.y),
                            edge_id=np.nan,
                            s_ft=np.nan,
                            L_ft=np.nan,
                            u_node=None,
                            v_node=None,
                            to_u_ft=np.nan,
                            to_v_ft=np.nan,
                        )
                    )
                    continue

        dist_ft = float(pt.distance(seg))
        if dist_ft > max_snap_ft:
            rows.append(
                dict(
                    stop_id=sid,
                    stop_name=str(sname),
                    stop_code=str(scode),
                    x=float(pt.x),
                    y=float(pt.y),
                    edge_id=np.nan,
                    s_ft=np.nan,
                    L_ft=np.nan,
                    u_node=None,
                    v_node=None,
                    to_u_ft=np.nan,
                    to_v_ft=np.nan,
                )
            )
            continue

        s = float(seg.project(pt))
        L = float(seg.length)
        u, v = edge_endpoints[int(eid)]

        rows.append(
            dict(
                stop_id=sid,
                stop_name=str(sname),
                stop_code=str(scode),
                x=float(pt.x),
                y=float(pt.y),
                edge_id=int(eid),
                s_ft=s,
                L_ft=L,
                u_node=u,
                v_node=v,
                to_u_ft=s,
                to_v_ft=L - s,
            )
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# SHORTEST PATHS WITH VIRTUAL CONNECTORS
# -----------------------------------------------------------------------------

# Small symmetric cache keyed by unordered node pair.
_node_dist_cache: dict[tuple[NodeKey, NodeKey], float] = {}


def _node_dist_ft(G: nx.MultiGraph, a: NodeKey, b: NodeKey) -> float:
    """Cacheable node-to-node shortest path length in feet."""
    if a == b:
        return 0.0
    key = (a, b) if a <= b else (b, a)
    d = _node_dist_cache.get(key)
    if d is not None:
        return d
    d = float(nx.shortest_path_length(G, a, b, weight="length"))
    _node_dist_cache[key] = d
    return d


def _min_edge_data(G: nx.MultiGraph, u: NodeKey, v: NodeKey) -> dict:
    """Return edge data dict for the shortest parallel edge between u and v."""
    data = G.get_edge_data(u, v)
    if not data:
        raise nx.NetworkXNoPath(f"No edge between {u} and {v}")
    k_min = min(data, key=lambda k: data[k]["length"])
    return data[k_min]


def _concat_lines(lines: Sequence[LineString]) -> LineString:
    """Concatenate a sequence of LineStrings into a single LineString."""
    coords: List[Tuple[float, float]] = []
    for ln in lines:
        if ln.is_empty:
            continue
        cs = list(ln.coords)
        if not coords:
            coords.extend(cs)
        else:
            if coords[-1] == cs[0]:
                coords.extend(cs[1:])
            else:
                coords.extend(cs)
    if len(coords) < 2:
        return LineString(coords)
    return LineString(coords)


def _ensure_oriented(lines: Sequence[LineString], nodes: Sequence[NodeKey]) -> List[LineString]:
    """Orient each edge geometry to follow the node sequence."""
    out: List[LineString] = []
    for (u, v), ln in zip(zip(nodes[:-1], nodes[1:]), lines):
        cs = list(ln.coords)
        if cs[0] == u and cs[-1] == v:
            out.append(ln)
        elif cs[0] == v and cs[-1] == u:
            out.append(LineString(list(reversed(cs))))
        else:
            du0 = math.hypot(cs[0][0] - u[0], cs[0][1] - u[1])
            du1 = math.hypot(cs[-1][0] - u[0], cs[-1][1] - u[1])
            out.append(ln if du0 <= du1 else LineString(list(reversed(cs))))
    return out


def _build_path_geometry(
    G: nx.MultiGraph,
    a_seg: LineString,
    a_s: float,
    a_to_u: bool,
    node_path: Sequence[NodeKey],
    b_seg: LineString,
    b_s: float,
    b_from_u: bool,
) -> LineString:
    """Construct full path geometry: A partial, node path, B partial."""
    L_a = linestring_length(a_seg)
    L_b = linestring_length(b_seg)

    a_part = (
        linestring_substring(a_seg, a_s, 0.0) if a_to_u else linestring_substring(a_seg, a_s, L_a)
    )  # noqa: E501

    fulls: List[LineString] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        ed = _min_edge_data(G, u, v)
        geom: LineString = ed["geometry"]
        gcoords = list(geom.coords)
        if gcoords[0] != u and gcoords[-1] == u:
            geom = LineString(list(reversed(gcoords)))
        fulls.append(geom)

    b_part = (
        linestring_substring(b_seg, 0.0, b_s) if b_from_u else linestring_substring(b_seg, L_b, b_s)
    )  # noqa: E501

    return _concat_lines([a_part, *_ensure_oriented(fulls, node_path), b_part])


def stop_to_stop_network(
    G: nx.MultiGraph,
    segments: gpd.GeoDataFrame,
    edge_endpoints: Dict[EdgeID, Tuple[NodeKey, NodeKey]],
    snap_map: pd.DataFrame,
    sid_a: str,
    sid_b: str,
) -> tuple[float, Optional[LineString]]:
    """Compute network distance and geometry between two snapped stops."""
    rec_a = snap_map.loc[snap_map.stop_id == sid_a]
    rec_b = snap_map.loc[snap_map.stop_id == sid_b]
    if rec_a.empty or rec_b.empty:
        return math.inf, None
    ra = rec_a.iloc[0]
    rb = rec_b.iloc[0]
    if pd.isna(ra.edge_id) or pd.isna(rb.edge_id):
        return math.inf, None

    eid_a = int(ra.edge_id)
    eid_b = int(rb.edge_id)
    u_a, v_a = edge_endpoints[eid_a]
    u_b, v_b = edge_endpoints[eid_b]
    seg_a: LineString = segments.loc[segments.edge_id == eid_a, "geometry"].values[0]
    seg_b: LineString = segments.loc[segments.edge_id == eid_b, "geometry"].values[0]
    s_a, s_b = float(ra.s_ft), float(rb.s_ft)
    to_u_a, to_v_a = float(ra.to_u_ft), float(ra.to_v_ft)
    to_u_b, to_v_b = float(rb.to_u_ft), float(rb.to_v_ft)

    candidates: List[Tuple[float, Tuple[NodeKey, NodeKey], Tuple[bool, bool]]] = []

    for a_end, a_cost, a_is_u in ((u_a, to_u_a, True), (v_a, to_v_a, False)):
        for b_end, b_cost, b_is_u in ((u_b, to_u_b, True), (v_b, to_v_b, False)):
            try:
                middle = _node_dist_ft(G, a_end, b_end)
            except nx.NetworkXNoPath:
                continue
            total = a_cost + middle + b_cost
            candidates.append((total, (a_end, b_end), (a_is_u, b_is_u)))

    if not candidates:
        return math.inf, None

    total_ft, (a_end, b_end), (a_is_u, b_is_u) = min(candidates, key=lambda t: t[0])

    try:
        node_path: List[NodeKey] = nx.shortest_path(G, a_end, b_end, weight="length")
    except nx.NetworkXNoPath:
        return math.inf, None

    path_geom = _build_path_geometry(
        G,
        a_seg=seg_a,
        a_s=s_a,
        a_to_u=a_is_u,
        node_path=node_path,
        b_seg=seg_b,
        b_s=s_b,
        b_from_u=b_is_u,
    )
    return total_ft, path_geom


# -----------------------------------------------------------------------------
# COVERAGE (FEET-BASED CRS)
# -----------------------------------------------------------------------------


def coverage_polygon(stops: gpd.GeoDataFrame, buffer_miles: float) -> gpd.GeoDataFrame:
    """Dissolved coverage polygon using feet buffers (CRS in US ft)."""
    radius_ft = buffer_miles * FT_PER_MILE
    dissolved = stops.buffer(radius_ft).union_all()
    return gpd.GeoDataFrame(geometry=[dissolved], crs=stops.crs)


# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------


def export_results(
    results: Dict[str, Dict[str, object]],
    crs: int | str,
    out_dir: Path,
) -> None:
    """Export CSV and shapefiles (paths + deleted stops)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results.values())
    df.to_csv(out_dir / "deleted_stops_distances.csv", index=False)

    ok = df[df.path_geom.notna()]
    if not ok.empty:
        gdf_lines = gpd.GeoDataFrame(
            ok[
                [
                    "stop_name",
                    "stop_id",
                    "stop_code",
                    "nearest_stop_id",
                    "linear_dist_miles",
                    "network_dist_miles",
                ]
            ].copy(),
            geometry=ok.path_geom,
            crs=crs,
        )
        gdf_lines.to_file(out_dir / "deleted_to_nearest_paths.shp")

    gdf_pts = gpd.GeoDataFrame(
        df[
            [
                "stop_name",
                "stop_id",
                "stop_code",
                "linear_dist_miles",
                "network_dist_miles",
            ]
        ].copy(),
        geometry=[Point(xy) for xy in zip(df.x, df.y)],
        crs=crs,
    )
    gdf_pts.to_file(out_dir / "deleted_stops.shp")


def export_stop_maps(
    stops: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    results: Dict[str, Dict[str, object]],
    crs: int | str,
    out_dir: Path,
) -> None:
    """Save a simple PNG map per removed stop.

    Changes vs. previous version:
      * Plots BOTH the visual backdrop layer (if provided) AND the actual
        network centerlines (segments) used to build the graph.
      * Labels points and title with stop NAMES in addition to IDs.
      * Displays network distance in FEET instead of miles in the title.
    """
    if not EXPORT_MAPS:
        return

    def _feet_from_miles_str(val: object) -> str:
        """Robustly format miles (float or '> x') as a feet string."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        if isinstance(val, str) and val.strip().startswith(">"):
            try:
                mi = float(val.strip().lstrip(">").strip())
                return f"> {int(round(mi * FT_PER_MILE)):,.0f} ft"
            except Exception:  # noqa: BLE001
                return str(val)
        try:
            mi = float(val)  # handles ints/floats/num-strings
            return f"{int(round(mi * FT_PER_MILE)):,.0f} ft"
        except Exception:  # noqa: BLE001
            return str(val)

    map_dir = out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    sidewalks_backdrop = _load_backdrop_layer_for_plots(PLOT_SIDEWALKS_SHP, crs)

    # Quick lookups
    geom_by_id: dict[str, Point] = dict(zip(stops.stop_id.astype(str), stops.geometry))
    name_by_id: dict[str, str] = dict(
        zip(
            stops.stop_id.astype(str),
            stops.get("stop_name", pd.Series("", index=stops.index)).astype(str),
        )
    )

    for sid, rec in results.items():
        path = rec["path_geom"]
        tgt_sid = rec["nearest_stop_id"]
        if path is None or tgt_sid is None:
            continue

        sid = str(sid)
        tgt_sid = str(tgt_sid)

        removed_pt = geom_by_id.get(sid)
        kept_pt = geom_by_id.get(tgt_sid)
        if removed_pt is None or kept_pt is None:
            continue

        removed_name = name_by_id.get(sid, "")
        kept_name = name_by_id.get(tgt_sid, "")

        g_path = gpd.GeoSeries([path], crs=crs)

        # Figure & axes
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

        # Backdrop (optional)
        _plot_backdrop_within_bounds(ax, sidewalks_backdrop, tuple(g_path.total_bounds), crs)

        # Plot the ACTUAL network centerlines used to build the graph (clipped to view)
        xmin, ymin, xmax, ymax = g_path.total_bounds
        pad = SIDEWALK_BACKDROP_PAD_FT
        clip_poly = box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
        try:
            seg_sub = segments[segments.geometry.intersects(clip_poly)].copy()
            if not seg_sub.empty:
                try:
                    seg_sub = gpd.clip(seg_sub, clip_poly)
                except Exception:  # noqa: BLE001
                    pass
                # Slightly heavier than backdrop so it's visible beneath the path
                seg_sub.plot(ax=ax, linewidth=0.8, alpha=0.8, zorder=1)
        except Exception as exc:  # noqa: BLE001
            logging.debug("Segment clip/plot failed: %s", exc)

        # Path and points
        g_path.plot(ax=ax, linewidth=2.0, zorder=2)
        gpd.GeoSeries([removed_pt], crs=crs).plot(ax=ax, color="red", markersize=35, zorder=3)
        gpd.GeoSeries([kept_pt], crs=crs).plot(ax=ax, color="green", markersize=35, zorder=3)

        # Labels (names + IDs) with a small offset
        ax.annotate(
            f"{removed_name} ({sid})",
            xy=(removed_pt.x, removed_pt.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            zorder=4,
        )
        ax.annotate(
            f"{kept_name} ({tgt_sid})",
            xy=(kept_pt.x, kept_pt.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            zorder=4,
        )

        # Title: network distance in FEET
        dist_ft_txt = _feet_from_miles_str(rec.get("network_dist_miles"))
        title = (
            f"Deleted {removed_name} ({sid}) \u2192 "
            f"{kept_name} ({tgt_sid}) [Network: {dist_ft_txt}]"
        )
        ax.set_title(title, fontsize=8)

        ax.set_aspect("equal")
        ax.set_axis_off()

        fig.tight_layout()
        fig.savefig(map_dir / f"{sid}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Analyze sidewalk-access impacts using a virtual-connector network."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")

    logging.info("Reading centerlines …")
    centerlines = load_centerlines(SIDEWALK_SHP, TARGET_CRS)

    logging.info("Exploding to segments …")
    segments = explode_segments(centerlines)

    logging.info("Building graph …")
    G, edge_endpoints = build_graph(segments, NODE_GRID_FT)

    logging.info("Building spatial index …")
    seg_index, _seg_geoms, wkb_to_eid = build_segment_index(segments)

    logging.info("Loading GTFS stops …")
    stops = load_gtfs_stops(GTFS_DIR, TARGET_CRS)
    logging.info("Total unique stops: %d", len(stops))

    logging.info("Snapping stops (virtual connectors) …")
    snap_map = snap_stops_to_segments(
        stops, seg_index, wkb_to_eid, segments, edge_endpoints, MAX_SNAP_FT
    )

    # Coverage (feet buffers; EPSG:6447 in US ft)
    logging.info("Calculating coverage polygons …")
    orig_cov = coverage_polygon(stops, BUFFER_MILES)

    # Resolve the human-entered identifiers to canonical stop_ids and mark a boolean.
    logging.info("Resolving deleted identifiers …")
    prefer_stop_code = IDENTIFIER_PRIORITY[0] == "stop_code"
    resolved_ids, _match_map = resolve_deleted_stop_ids(
        stops, DELETED_STOP_IDS, prefer_stop_code=prefer_stop_code
    )
    stops["is_deleted"] = stops["stop_id"].isin(resolved_ids)

    kept = stops[~stops["is_deleted"]].copy()
    kept_cov = coverage_polygon(kept, BUFFER_MILES)
    lost_cov = gpd.overlay(orig_cov, kept_cov, how="difference")
    lost_area_sqmi = float(lost_cov.area.sum()) / (FT_PER_MILE**2)
    logging.info("Area lost: %.4f sq mi", lost_area_sqmi)

    logging.info("Computing network distances …")
    kept_stops = kept

    if kept_stops.empty:
        logging.warning("No kept stops; skipping network distances.")
    kept_coords = (
        np.array([(p.x, p.y) for p in kept_stops.geometry])
        if not kept_stops.empty
        else np.empty((0, 2))
    )  # noqa: E501
    kd = cKDTree(kept_coords) if kept_coords.size else None

    results: Dict[str, Dict[str, object]] = {}

    unique_deleted = sorted(set(resolved_ids))
    for sid in unique_deleted:
        row = stops.loc[stops.stop_id == sid]
        if row.empty:
            logging.warning("Deleted stop_id %s not found in GTFS.", sid)
            continue
        row = row.iloc[0]
        x, y = float(row.geometry.x), float(row.geometry.y)

        snapped_row = snap_map.loc[snap_map.stop_id == sid]
        off_net = snapped_row.empty or pd.isna(snapped_row.iloc[0].edge_id)

        if kd is None:
            results[sid] = dict(
                stop_name=row.get("stop_name", ""),
                stop_id=sid,
                stop_code=row.get("stop_code", ""),
                x=x,
                y=y,
                nearest_stop_id=None,
                linear_dist_miles="> 0.25",
                network_dist_miles="> 0.25",
                path_geom=None,
                sanity_flag=None,
            )
            continue

        # Euclidean prefilter
        idxs = kd.query_ball_point([x, y], r=BUFFER_FT)

        if off_net or not idxs:
            results[sid] = dict(
                stop_name=row.get("stop_name", ""),
                stop_id=sid,
                stop_code=row.get("stop_code", ""),
                x=x,
                y=y,
                nearest_stop_id=None,
                linear_dist_miles="> 0.25",
                network_dist_miles="> 0.25",
                path_geom=None,
                sanity_flag="off_network" if off_net else "no_kept_within_buffer",
            )
            continue

        best = (math.inf, None, None, None, None)  # net_ft, lin_ft, tgt_sid, path_geom, flag
        for i in idxs:
            tgt = kept_stops.iloc[i]
            tgt_sid = str(tgt.stop_id)

            lin_ft = float(row.geometry.distance(tgt.geometry))
            net_ft, path_geom = stop_to_stop_network(
                G, segments, edge_endpoints, snap_map, sid, tgt_sid
            )

            # Across-the-street sanity override
            flag = None
            if lin_ft <= ACROSS_STREET_MAX_FT and (
                math.isinf(net_ft)
                or net_ft > max(ACROSS_STREET_ABS_FT, ACROSS_STREET_RATIO * lin_ft)  # noqa: E501
            ):
                path_geom = LineString([(x, y), (float(tgt.geometry.x), float(tgt.geometry.y))])
                net_ft = lin_ft
                flag = "across_street_override"

            if net_ft < best[0]:
                best = (net_ft, lin_ft, tgt_sid, path_geom, flag)

        net_ft, lin_ft, tgt_sid, path_geom, flag = best
        if math.isinf(net_ft):
            results[sid] = dict(
                stop_name=row.get("stop_name", ""),
                stop_id=sid,
                stop_code=row.get("stop_code", ""),
                x=x,
                y=y,
                nearest_stop_id=None,
                linear_dist_miles=round(float(lin_ft) / FT_PER_MILE, 4)
                if lin_ft is not None
                else None,  # noqa: E501
                network_dist_miles="> 0.25",
                path_geom=None,
                sanity_flag=flag,
            )
        else:
            results[sid] = dict(
                stop_name=row.get("stop_name", ""),
                stop_id=sid,
                stop_code=row.get("stop_code", ""),
                x=x,
                y=y,
                nearest_stop_id=tgt_sid,
                linear_dist_miles=round(float(lin_ft) / FT_PER_MILE, 4),
                network_dist_miles=round(float(net_ft) / FT_PER_MILE, 4),
                path_geom=path_geom,
                sanity_flag=flag,
            )

    ok = sum(1 for v in results.values() if v["path_geom"] is not None)
    logging.info("Paths built for %d/%d deleted stops", ok, len(unique_deleted))

    logging.info("Exporting CSV and shapefiles …")
    export_results(results, TARGET_CRS, OUTPUT_DIR)
    export_stop_maps(stops, segments, results, TARGET_CRS, OUTPUT_DIR)

    if not lost_cov.empty:
        lost_cov.to_file(OUTPUT_DIR / "lost_coverage.shp")
    else:
        logging.info("No lost coverage within %.2f mi buffers; nothing to export.", BUFFER_MILES)

    logging.info("Done ✔")


if __name__ == "__main__":
    main()
