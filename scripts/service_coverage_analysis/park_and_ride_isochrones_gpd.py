"""Generate drive-time isochrones for park-and-ride facilities.

If *REMOVE_OVERLAPS* is ``True``, overlapping polygons are split so the
intersection area is assigned to the facility whose centroid is closest
to the overlap centroid.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

ROADS_PATH: str = r"Path\To\Your\Roadway_Centerlines.shp"
FACILITIES_PATH: str = r"Path\To\Your\park_and_rides.shp"
OUTPUT_DIR: str = r"Path\To\Your\Output_Folder"

DRIVE_TIME_MIN: int = 15  # minutes
REMOVE_OVERLAPS: bool = False  # distance-based assignment when True

SPEED_FIELD: str = "SPEED_LIMI"  # mph
ONEWAY_FIELD: str = "ONEWAY"  # 'Y' or 'N'

BUFFER_SMOOTH_M: float = 75  # smoothing buffer around reached nodes (metres)

# CRS used for routing/buffering
WORK_CRS_EPSG: int = 3857  # metres; keep consistent throughout

# =============================================================================
# FUNCTIONS
# =============================================================================


def build_network(
    roads: gpd.GeoDataFrame, speed_field: str, oneway_field: str
) -> nx.DiGraph:
    """Create a directed graph with travel-time (seconds) weights from road centre-lines."""
    mph_to_mps = 0.44704
    g = nx.DiGraph()

    for _, row in roads.iterrows():
        geom = row.geometry
        try:
            speed = float(row[speed_field])
        except (KeyError, TypeError, ValueError):
            continue
        if speed <= 0 or geom.is_empty:
            continue

        oneway = str(row.get(oneway_field, "N"))

        lines: List[LineString]
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom)
        else:
            continue

        speed_mps = speed * mph_to_mps

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            seg_lens = [
                Point(coords[i]).distance(Point(coords[i + 1]))
                for i in range(len(coords) - 1)
            ]
            for i, seg_len in enumerate(seg_lens):
                if seg_len == 0:
                    continue
                t_sec = seg_len / speed_mps
                u, v = coords[i], coords[i + 1]
                g.add_edge(u, v, weight=t_sec)
                if oneway != "Y":
                    g.add_edge(v, u, weight=t_sec)

    if g.number_of_edges() == 0:
        raise RuntimeError("No usable edges found—check road data and field names.")
    return g


def isochrone_polygon(
    graph: nx.DiGraph, origin: Point, cutoff_sec: int, smooth_m: float
) -> Polygon:
    """Return a single Polygon representing the isochrone."""
    nearest = min(graph.nodes, key=lambda n: origin.distance(Point(n)))
    lengths = nx.single_source_dijkstra_path_length(
        graph, nearest, cutoff_sec, weight="weight"
    )
    if not lengths:  # unreachable
        return Polygon()
    points = [Point(xy) for xy in lengths.keys()]
    poly = unary_union(gpd.GeoSeries(points).buffer(smooth_m))
    return poly


def resolve_overlaps_by_proximity(
    polys: Dict[str, Polygon], centers: Dict[str, Point]
) -> Dict[str, Polygon]:
    """Resolve overlaps by assigning shared areas to the nearest facility.

    The function iteratively makes all polygons in *polys* mutually disjoint.
    Whenever two isochrone polygons intersect, the overlapping area is kept by
    the facility whose centre point is geographically closest to the overlap
    centroid (Euclidean distance in EPSG:3857). The process repeats until no
    intersections remain.

    Args:
        polys: Mapping of facility names to their isochrone polygons.
        centers: Mapping of facility names to their centre points.

    Returns:
        A dictionary of the same structure as *polys*, but with all polygons
        guaranteed to be mutually exclusive.
    """
    changed = True
    while changed:
        changed = False
        names = list(polys.keys())
        for a, b in itertools.combinations(names, 2):
            inter = polys[a].intersection(polys[b])
            if inter.is_empty:
                continue
            # Determine winner
            centroid = inter.centroid
            da = centroid.distance(centers[a])
            db = centroid.distance(centers[b])
            if da <= db:
                # a keeps intersection; remove from b
                polys[b] = polys[b].difference(inter)
            else:
                polys[a] = polys[a].difference(inter)
            changed = True
    return polys


def safe_filename(text: str) -> str:
    """Return *text* converted to a filesystem-safe, lowercase filename.

    All non-alphanumeric characters are replaced with underscores and any
    leading or trailing underscores are removed.
    """
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Generate drive-time isochrones for each park-and-ride facility.

    The function:
      1. Loads and re-projects road and facility layers.
      2. Builds per-facility mini-graphs and runs a travel-time isochrone
         search.
      3. Optionally resolves polygon overlaps.
      4. Writes individual and combined shapefiles to *OUTPUT_DIR*.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load once, project once, build spatial index once
    roads_full = gpd.read_file(ROADS_PATH).to_crs(epsg=WORK_CRS_EPSG)
    roads_index = roads_full.sindex  # STR-tree
    facilities = gpd.read_file(FACILITIES_PATH).to_crs(epsg=WORK_CRS_EPSG)

    # 2. Pre-compute the *absolute* reach radius (worst-case: 70 mph on straight freeway)
    MAX_SPEED_MPS = 70 * 0.44704
    RADIUS_M = DRIVE_TIME_MIN * 60 * MAX_SPEED_MPS  # metres (~28 km for 15 min)

    iso_polys: Dict[str, Polygon] = {}
    fac_points: Dict[str, Point] = {}

    for idx, row in facilities.iterrows():
        name = str(row.get("name", f"facility_{idx}"))
        centre = row.geometry
        print(f"Creating isochrone for {name} …")

        # 3. Clip roads to circular buffer -------------------------------
        buf = centre.buffer(RADIUS_M)
        # rough bbox filter
        cand_idx = list(roads_index.intersection(buf.bounds))
        sub_roads = roads_full.iloc[cand_idx]
        # true intersection test (drops false positives)
        sub_roads = sub_roads[sub_roads.intersects(buf)]

        if sub_roads.empty:
            print(f"  ⚠  No roads in reach buffer for {name}.")
            continue

        # 4. Build *mini* graph and run Dijkstra -------------------------
        sub_graph = build_network(sub_roads, SPEED_FIELD, ONEWAY_FIELD)
        poly = isochrone_polygon(
            sub_graph, centre, DRIVE_TIME_MIN * 60, BUFFER_SMOOTH_M
        )

        if poly.is_empty:
            print(f"  ⚠  Unreachable network for {name}.")
            continue

        iso_polys[name] = poly
        fac_points[name] = centre

    if not iso_polys:
        print("No isochrones generated—exiting.")
        return

    # 5. Resolve overlaps if requested -----------------------------------
    if REMOVE_OVERLAPS:
        iso_polys = resolve_overlaps_by_proximity(iso_polys, fac_points)

    # 6. Write outputs ----------------------------------------------------
    combined_rows = []
    for name, poly in iso_polys.items():
        if poly.is_empty:
            continue
        gdf = gpd.GeoDataFrame(
            {"name": [name]}, geometry=[poly], crs=f"EPSG:{WORK_CRS_EPSG}"
        )
        gdf.to_file(out_dir / f"{safe_filename(name)}_{DRIVE_TIME_MIN}min_iso.shp")
        combined_rows.append(gdf)

    gpd.GeoDataFrame(
        pd.concat(combined_rows, ignore_index=True), crs=f"EPSG:{WORK_CRS_EPSG}"
    ).to_file(out_dir / f"all_facilities_{DRIVE_TIME_MIN}min_iso.shp")

    print("\n✓ Finished (buffer-clipped workflow).  Outputs in", out_dir.resolve())


if __name__ == "__main__":
    main()
