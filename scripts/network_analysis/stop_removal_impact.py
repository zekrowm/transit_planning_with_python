"""Analyze sidewalk-access impacts from GTFS stop removals using a spatial network.

This script quantifies network-based pedestrian access changes caused by bus stop removals.
It compares sidewalk-buffer coverage before and after stop deletions and calculates network
distances from each removed stop to its nearest retained stop along a segmented sidewalk graph.

Key Features:
    * Builds a "physical" graph where every road segment is a distinct edge.
    * Snaps each stop to its nearest sidewalk segment and splits the segment at that point.
    * Computes sidewalk-buffer coverage (0.25 miles) before and after stop deletion.
    * Calculates both linear and network distances to nearest kept stop within 0.25 mi.
    * Outputs CSV and QA shapefiles including paths and removed stop locations.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# =============================================================================
# CONFIGURATION
# =============================================================================

SIDEWALK_SHP = Path(r"Path\To\Your\Sidewalks_Centerline.shp")
GTFS_DIR = Path(r"Path\To\Your\GTFS_Fdoler\stops.txt")  # must contain stops.txt
OUTPUT_DIR = Path(r"Path\To\Your\Output_Folder")

DELETED_STOP_IDS: List[str] = ["1001", "1002"]

BUFFER_MILES = 0.25
TARGET_CRS = 6447  # VA State Plane (US ft)

# --- Unit Conversions ---
FT_PER_MILE = 5280
SQFT_TO_SQMI = 1 / (FT_PER_MILE**2)
BUFFER_FT = BUFFER_MILES * FT_PER_MILE

# --- Logging ---
LOG_LEVEL = logging.INFO

# =============================================================================
# FUNCTIONS
# =============================================================================

def build_pedestrian_network(
    sidewalks_gdf: gpd.GeoDataFrame,
) -> tuple[nx.Graph, gpd.GeoDataFrame]:
    """
    Builds a physical pedestrian network graph from sidewalk/road centerlines.

    Each complex LineString is decomposed into simple, two-point segments,
    and each segment becomes an edge in the graph.

    Args:
        sidewalks_gdf: A GeoDataFrame of sidewalk or road centerlines.

    Returns:
        A tuple containing:
        - graph: A NetworkX Graph with nodes as coordinate tuples and
                 edge weights ('weight') as segment lengths.
        - segments_gdf: A GeoDataFrame of all the two-point segments
                        that make up the network.
    """
    graph = nx.Graph()
    all_segments = []

    for _, row in sidewalks_gdf.iterrows():
        geom = row.geometry
        if geom.is_empty or geom is None:
            continue

        lines = getattr(geom, "geoms", [geom])  # Handle MultiLineString

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Decompose the polyline into 2-point segments
            for i in range(len(coords) - 1):
                u_coord, v_coord = coords[i], coords[i + 1]

                if Point(u_coord).equals(Point(v_coord)):
                    continue

                segment_geom = LineString([u_coord, v_coord])
                graph.add_edge(u_coord, v_coord, weight=segment_geom.length)
                all_segments.append({"u": u_coord, "v": v_coord, "geometry": segment_geom})

    if not all_segments:
        raise ValueError("No valid line segments found to build the network.")

    segments_gdf = gpd.GeoDataFrame(all_segments, crs=sidewalks_gdf.crs)
    return graph, segments_gdf


def snap_stops_and_update_graph(
    stops_gdf: gpd.GeoDataFrame, graph: nx.Graph, segments_gdf: gpd.GeoDataFrame
) -> dict[str, tuple[float, float]]:
    """
    Snaps stops to the nearest network segment, splitting the segment and
    updating the graph with a new node at the snap point.

    Args:
        stops_gdf: A GeoDataFrame of GTFS stops.
        graph: The NetworkX graph to be modified.
        segments_gdf: A GeoDataFrame of the simple, two-point segments
                      that constitute the graph's edges.

    Returns:
        A dictionary mapping each stop_id to its corresponding node coordinate tuple.
    """
    tree = STRtree(segments_gdf.geometry)
    stop_to_node_map = {}
    
    # Create a copy of the geometry column to avoid SettingWithCopyWarning
    segments_geoms = segments_gdf.geometry.copy()

    for _, stop_row in stops_gdf.iterrows():
        stop_id = stop_row["stop_id"]
        stop_geom = stop_row.geometry

        # Find the index of the nearest segment geometry
        nearest_geom_idx = tree.nearest(stop_geom)
        
        # Get the full segment information using the index
        edge_geom = segments_geoms.iloc[nearest_geom_idx]
        segment_info = segments_gdf.iloc[nearest_geom_idx]
        u, v = segment_info["u"], segment_info["v"]
        
        snap_pt = edge_geom.interpolate(edge_geom.project(stop_geom))
        snap_node = (snap_pt.x, snap_pt.y)

        # Avoid modifying the graph if the snap point is on an existing endpoint
        tol = 1e-6
        if Point(u).distance(snap_pt) < tol:
            stop_to_node_map[stop_id] = u
            continue
        if Point(v).distance(snap_pt) < tol:
            stop_to_node_map[stop_id] = v
            continue

        # Split the edge by removing the old one and adding two new ones
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)

        graph.add_node(snap_node)
        graph.add_edge(u, snap_node, weight=Point(u).distance(snap_pt))
        graph.add_edge(snap_node, v, weight=snap_pt.distance(Point(v)))
        
        stop_to_node_map[stop_id] = snap_node

    return stop_to_node_map


def load_gtfs_stops(gtfs_dir: Path, target_crs) -> gpd.GeoDataFrame:
    """Return GTFS stops as GeoDataFrame in *target_crs* (dedup by id)."""
    df = pd.read_csv(gtfs_dir / "stops.txt", dtype=str).drop_duplicates("stop_id")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.stop_lon.astype(float), df.stop_lat.astype(float)),
        crs="EPSG:4326",
    ).to_crs(target_crs)


def network_distances(
    deleted_ids: List[str],
    stops: gpd.GeoDataFrame,
    stop_nodes: Dict[str, Tuple[float, float]],
    graph: nx.Graph,
) -> Dict[str, Dict[str, object]]:
    """
    Calculates distances from each deleted stop to its nearest kept stop.

    The total walking distance is the sum of three parts:
    1. The "access" connector from the deleted stop to the network.
    2. The shortest path along the network.
    3. The "egress" connector from the network to the nearest kept stop.
    """
    results: Dict[str, Dict[str, object]] = {}

    kept_stops = stops[~stops.stop_id.isin(deleted_ids)].copy()
    kept_coords = np.array([(g.x, g.y) for g in kept_stops.geometry])
    kd = cKDTree(kept_coords)

    # Pre-calculate all connector distances for efficiency
    connector_distances = {
        sid: geom.distance(Point(stop_nodes[sid]))
        for sid, geom in zip(stops.stop_id, stops.geometry)
        if sid in stop_nodes
    }

    for sid in deleted_ids:
        row = stops.loc[stops.stop_id == sid].iloc[0]
        src_geom = row.geometry
        src_node = stop_nodes.get(sid)

        if src_node is None or not graph.has_node(src_node):
            logging.warning(f"Stop {sid} could not be mapped to a valid graph node. Skipping.")
            continue

        src_connector_dist = connector_distances.get(sid, 0)
        idxs = kd.query_ball_point([src_geom.x, src_geom.y], r=BUFFER_FT)
        
        best_total_ft = float('inf')
        best_lin_ft = float('inf')
        best_stop_id = None
        best_path_nodes = []

        if idxs:
            for i in idxs:
                tgt_row = kept_stops.iloc[i]
                tgt_sid = tgt_row.stop_id
                tgt_node = stop_nodes.get(tgt_sid)

                if tgt_node is None or not graph.has_node(tgt_node):
                    continue
                
                try:
                    # 1. Calculate on-network path distance
                    network_ft = nx.shortest_path_length(
                        graph, src_node, tgt_node, weight="weight"
                    )

                    # 2. Get target stop's connector distance
                    tgt_connector_dist = connector_distances.get(tgt_sid, 0)
                    
                    # 3. Sum the three parts for the total walking distance
                    total_ft = src_connector_dist + network_ft + tgt_connector_dist
                    
                    if total_ft < best_total_ft:
                        best_total_ft = total_ft
                        best_lin_ft = src_geom.distance(tgt_row.geometry)
                        best_stop_id = tgt_sid
                        best_path_nodes = nx.shortest_path(
                            graph, src_node, tgt_node, weight="weight"
                        )

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        results[sid] = {
            "stop_name": row.stop_name,
            "stop_id": sid,
            "stop_code": row.stop_code,
            "x": src_geom.x,
            "y": src_geom.y,
            "nearest_stop_id": best_stop_id,
            "linear_dist_miles": round(best_lin_ft / FT_PER_MILE, 4) if best_lin_ft != float('inf') else "> 0.25",
            # NOTE: This field now represents the TOTAL walking distance
            "network_dist_miles": round(best_total_ft / FT_PER_MILE, 4) if best_total_ft != float('inf') else "> 0.25",
            # The path geometry remains the on-network portion for visualization
            "path_geom": LineString(best_path_nodes) if best_path_nodes else None,
        }
    return results


def export_outputs(results: Dict, crs, out_dir: Path) -> None:
    """Dump CSV + QA shapefiles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results.values())
    df.to_csv(out_dir / "deleted_stops_distances.csv", index=False)
    
    # Paths Shapefile
    ok = df[df.path_geom.notna()]
    if not ok.empty:
        gpd.GeoDataFrame(
            ok.drop(columns=["x", "y", "path_geom"]), geometry=ok.path_geom, crs=crs
        ).to_file(out_dir / "deleted_to_nearest_paths.shp")
    
    # Points Shapefile
    gpd.GeoDataFrame(
        df.drop(columns=["x", "y", "path_geom"]),
        geometry=[Point(xy) for xy in zip(df.x, df.y)], crs=crs,
    ).to_file(out_dir / "deleted_stops.shp")

def export_stop_maps(
    stops: gpd.GeoDataFrame, results: Dict, crs: int, out_dir: Path
) -> None:
    """
    Save a simple .png map for each removed stop, including connector lines.

    The map shows:
        - Removed stop (red point)
        - Nearest kept stop (green point)
        - On-network path (solid blue line)
        - Access/egress connector lines (dashed gray lines)
    """
    map_dir = out_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    geom_by_id = dict(zip(stops.stop_id, stops.geometry))

    for sid, rec in results.items():
        path = rec["path_geom"]
        tgt_sid = rec["nearest_stop_id"]

        # Skip if no valid path was found
        if path is None or path.is_empty or tgt_sid is None:
            continue

        # Get original stop locations
        removed_pt = geom_by_id[sid]
        kept_pt = geom_by_id[tgt_sid]

        # Get the start and end points of the on-network path
        snapped_start_pt = Point(path.coords[0])
        snapped_end_pt = Point(path.coords[-1])

        # Create the connector line geometries
        connector_start = LineString([removed_pt, snapped_start_pt])
        connector_end = LineString([kept_pt, snapped_end_pt])

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

        # Plot on-network path (solid blue)
        gpd.GeoSeries([path], crs=crs).plot(
            ax=ax, linewidth=2, color="steelblue", zorder=2
        )
        # Plot connector lines (dashed gray)
        gpd.GeoSeries([connector_start, connector_end], crs=crs).plot(
            ax=ax, linewidth=1.5, color="gray", linestyle="--", zorder=1
        )
        # Plot stop points
        gpd.GeoSeries([removed_pt], crs=crs).plot(
            ax=ax, color="red", markersize=35, zorder=3
        )
        gpd.GeoSeries([kept_pt], crs=crs).plot(
            ax=ax, color="green", markersize=35, zorder=3
        )

        ax.set_aspect("equal")
        ax.set_axis_off()
        dist_txt = rec["network_dist_miles"]
        ax.set_title(f"Deleted stop {sid}  ➜  {tgt_sid}   ({dist_txt} mi)", fontsize=8)
        fig.tight_layout()
        fig.savefig(map_dir / f"{sid}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete analysis."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")

    # 1. Load GTFS stops
    logging.info("Loading GTFS stops …")
    stops = load_gtfs_stops(GTFS_DIR, TARGET_CRS)
    logging.info(f"Total unique stops: {len(stops)}")

    # 2. Clip sidewalk data to an envelope around stops for efficiency
    logging.info("Clipping sidewalk data to search envelope …")
    search_dist = BUFFER_FT * 1.20  # 20% slack
    envelope = stops.unary_union.buffer(search_dist)
    
    # Use a spatial index for efficient clipping
    sidewalks_full = gpd.read_file(SIDEWALK_SHP).to_crs(TARGET_CRS)
    possible_matches_index = list(sidewalks_full.sindex.intersection(envelope.bounds))
    sidewalks_clipped = sidewalks_full.iloc[possible_matches_index].copy()
    sidewalks_clipped = sidewalks_clipped[sidewalks_clipped.intersects(envelope)]

    # 3. Build the physical network graph
    logging.info("Building segmented pedestrian graph …")
    graph, segments_gdf = build_pedestrian_network(sidewalks_clipped)
    logging.info(f"Graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")

    # 4. Snap stops to the graph, splitting edges where necessary
    logging.info("Snapping stops to graph and splitting edges …")
    stop_nodes = snap_stops_and_update_graph(stops, graph, segments_gdf)
    logging.info(f"Snapping complete. Graph nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")

    # 5. Coverage polygons & lost area
    logging.info("Calculating coverage polygons and lost area …")
    orig_cov = unary_union(stops.geometry.buffer(BUFFER_FT))
    kept = stops[~stops.stop_id.isin(DELETED_STOP_IDS)].copy()
    kept_cov = unary_union(kept.geometry.buffer(BUFFER_FT))
    lost_cov_geom = orig_cov.difference(kept_cov)
    lost_area_sqmi = lost_cov_geom.area * SQFT_TO_SQMI
    logging.info(f"Area lost: {lost_area_sqmi:.4f} sq mi")

    # 6. Network-based distance calculations
    logging.info("Calculating network distances to nearest retained stops …")
    results = network_distances(DELETED_STOP_IDS, stops, stop_nodes, graph)

    # 7. Exports (CSV, shapefiles, QA maps)
    logging.info("Exporting results …")
    export_outputs(results, TARGET_CRS, OUTPUT_DIR)
    export_stop_maps(stops, results, TARGET_CRS, OUTPUT_DIR)
    gpd.GeoDataFrame(geometry=[lost_cov_geom], crs=TARGET_CRS).to_file(OUTPUT_DIR / "lost_coverage.shp")

    logging.info("Done ✔")

if __name__ == "__main__":
    main()
