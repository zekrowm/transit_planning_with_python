"""
Export GTFS Stops to Shapefile and Compute a TSP Route for Selected Bus Stops,
using a road network (with one-way directions and variable speeds) as a time network
in the DC area, using EPSG:2283 (NAD83 / Virginia North (ftUS)).

This script:
1. Reads GTFS stops (EPSG:4326) and reprojects them to EPSG:2283.
2. Lets the user specify a starting point in Google Maps DMS format (single string),
   which is also reprojected from EPSG:4326 to EPSG:2283.
3. Builds a directed road network from a shapefile in EPSG:2283.
4. Snaps each bus stop (and the start) to the nearest node in the road network.
5. Computes a TSP route via either an ILP or a Greedy approximation.
6. Exports step-by-step directions and also a route shapefile showing the final path.
"""

import math
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pulp
import pyproj
from shapely.geometry import LineString, Point

# =============================================================================
# CONFIGURATION
# =============================================================================

# Adjust these paths to match your data locations
GTFS_PATH = r"C:\Path\To\Your\GTFS_data"
OUTPUT_DIR = r"C:\Path\To\Your\Output_folder"
ROADWAYS_SHP_PATH = r"C:\Path\To\Your\Roadways.shp"

# GTFS stop IDs for demonstration
SELECTED_STOP_IDS = ["1001", "1002", "1003", "1004", "1005"]

# GTFS column with the stop identifiers
SELECTED_STOP_ID_COL = "stop_id"

# A single string from Google Maps, e.g., "38°51'54.5\"N 77°21'53.6\"W"
GOOGLE_MAPS_COORD_STR = "38°51'54.5\"N 77°21'53.6\"W"

# We assume the road shapefile is in EPSG:2283 (NAD83 / Virginia North (ftUS)).
# This means distances are in feet.
TARGET_ROAD_CRS = "EPSG:2283"

# Road shapefile columns/fields
ONEWAY_COL = "ONEWAY"  # 'Y' for one-way, else two-way
SPEED_COL = "SPEEDLIMI"  # Speed limit in mph
STREET_NAME_COL = "FULLNAME"  # Column containing the street name

# Fallback average speed in ft/s (44 ft/s ~ 30 mph).
AVERAGE_SPEED_FPS = 44.0  # if we fail to parse speed from SPEEDLIMI

# Choose TSP approach: "ilp" (exact) or "greedy" (approx).
OPTIMIZATION_CONFIG = {
    "optimization_approach": "ilp",
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


# Build a transformer from WGS84 (lon/lat) to EPSG:2283 (feet).
PROJECT_4326_TO_2283 = pyproj.Transformer.from_crs(
    "EPSG:4326",  # source (lat/lon)
    TARGET_ROAD_CRS,  # destination (feet)
    always_xy=True,  # treat x=lon, y=lat
)


def reproject_point_4326_to_2283(x_lon, y_lat):
    """
    Reproject a single coordinate from lat/lon (EPSG:4326)
    to EPSG:2283 (x,y in feet).
    Returns (x_2283, y_2283).
    """
    x_2283, y_2283 = PROJECT_4326_TO_2283.transform(x_lon, y_lat)
    return (x_2283, y_2283)


def dms_to_decimal(dms_str):
    """
    Converts a coordinate in DMS format (e.g., "38°50'38.5\"N") to decimal degrees.
    """
    dms_str = dms_str.strip()
    pattern = r"""(?P<degrees>\d+)[°\s]+
                  (?P<minutes>\d+)['\s]+
                  (?P<seconds>\d+(?:\.\d+)?)[\"\s]*
                  (?P<direction>[NSEW])"""
    match = re.match(pattern, dms_str, re.VERBOSE)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    degrees = float(match.group("degrees"))
    minutes = float(match.group("minutes"))
    seconds = float(match.group("seconds"))
    direction = match.group("direction").upper()
    dec = degrees + minutes / 60 + seconds / 3600
    if direction in ["S", "W"]:
        dec = -dec
    return dec


def parse_google_maps_coords(coord_str):
    """
    Given a single string like "38°51'54.5\"N 77°21'53.6\"W",
    split into lat part and lon part, then return decimal lat/lon.
    """
    parts = coord_str.strip().split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Could not split coordinates from: {coord_str}")
    lat_str, lon_str = parts
    lat_dd = dms_to_decimal(lat_str)
    lon_dd = dms_to_decimal(lon_str)
    return lat_dd, lon_dd


def rotate_route(route, start_node):
    """
    Rotates the TSP route so that it starts (and ends) with the given start_node.
    Assumes route is a cycle (i.e. the first and last node are the same).
    """
    if start_node not in route:
        return route
    # Remove duplicate final node if present.
    if route[0] == route[-1]:
        route = route[:-1]
    idx = route.index(start_node)
    rotated = route[idx:] + route[:idx]
    rotated.append(start_node)
    return rotated


def compute_heading(p1, p2):
    """
    Computes the heading (in degrees) from point p1 to point p2.
    p1 and p2 are (x, y) tuples in feet, but that's fine for the angle calculation.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def compute_turn_direction(heading1, heading2, threshold=15):
    """
    Computes a simple turn instruction (Left, Right, or Straight)
    based on the difference between two headings.
    """
    diff = heading2 - heading1
    # Normalize difference to (-180, 180)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    if abs(diff) < threshold:
        return "Straight"
    elif diff > 0:
        return "Left"
    else:
        return "Right"


# -----------------------------------------------------------------------------
# GTFS AND STOP FUNCTIONS
# -----------------------------------------------------------------------------


def export_gtfs_stops(gtfs_path, output_dir, target_crs):
    """
    1) Reads the GTFS stops.txt file (in EPSG:4326).
    2) Reprojects to target_crs (EPSG:2283, ftUS).
    3) Exports as a shapefile in that CRS.
    """
    stops_file = os.path.join(gtfs_path, "stops.txt")
    stops_df = pd.read_csv(stops_file)

    # Original is WGS84 lat/lon
    stops_gdf = gpd.GeoDataFrame(
        stops_df, geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat), crs="EPSG:4326"
    )

    # Now reproject to the road network CRS (feet)
    stops_gdf = stops_gdf.to_crs(target_crs)

    shapefile_path = os.path.join(output_dir, "gtfs_stops.shp")
    stops_gdf.to_file(shapefile_path)
    print(f"Exported GTFS stops to shapefile: {shapefile_path}")
    return stops_gdf


def filter_selected_stops(stops_gdf, selected_stop_ids, stop_id_col):
    """
    Filters the stops GeoDataFrame for the selected stop IDs using the specified column.
    Returns a dict of {stop_id: (x, y)} in EPSG:2283 (feet).
    """
    selected_stops_gdf = stops_gdf[stops_gdf[stop_id_col].isin(selected_stop_ids)]
    if selected_stops_gdf.empty:
        print("No stops found with the specified identifiers. Check your GTFS stops.txt file.")
        exit()
    bus_stops = {
        row[stop_id_col]: (row.geometry.x, row.geometry.y)
        for _, row in selected_stops_gdf.iterrows()
    }
    return bus_stops, selected_stops_gdf


# -----------------------------------------------------------------------------
# ROAD NETWORK FUNCTIONS
# -----------------------------------------------------------------------------


def build_directed_road_network(
    road_shp_path,
    oneway_col=ONEWAY_COL,
    speed_col=SPEED_COL,
    target_crs=TARGET_ROAD_CRS,
    street_name_col=STREET_NAME_COL,
):
    """
    Builds a directed road network from the shapefile, assumed to be in EPSG:2283 (ft).
    Each edge has travel time in seconds (weight), geometry in feet, street name, etc.
    """
    roads_gdf = gpd.read_file(road_shp_path)

    # If your roads are not in EPSG:2283, reproject here:
    # roads_gdf = roads_gdf.to_crs(target_crs)

    G = nx.DiGraph()
    for idx, row in roads_gdf.iterrows():
        oneway = str(row.get(oneway_col, "N")).upper()
        street_name = row.get(street_name_col, "Unnamed Road")

        # Attempt to retrieve speed limit in mph; convert mph -> ft/s
        try:
            speed_val_mph = float(row.get(speed_col, None))
            if speed_val_mph <= 0:
                raise ValueError("Non-positive speed value")
            speed_fps = speed_val_mph * 1.46667  # 1 mph ~ 1.46667 ft/s
        except (TypeError, ValueError):
            speed_fps = AVERAGE_SPEED_FPS

        geom = row.geometry
        # Handle both LineString and MultiLineString
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                seg_line = LineString([start, end])
                length_ft = seg_line.length  # in feet
                travel_time = length_ft / speed_fps  # in seconds
                if oneway == "Y":
                    G.add_edge(
                        start,
                        end,
                        weight=travel_time,
                        geometry=seg_line,
                        length=length_ft,
                        street=street_name,
                    )
                else:
                    G.add_edge(
                        start,
                        end,
                        weight=travel_time,
                        geometry=seg_line,
                        length=length_ft,
                        street=street_name,
                    )
                    G.add_edge(
                        end,
                        start,
                        weight=travel_time,
                        geometry=seg_line,
                        length=length_ft,
                        street=street_name,
                    )

    print(f"Road network graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, roads_gdf


def snap_point_to_network(pt, road_graph):
    """
    Snaps a point (x, y) in EPSG:2283 (feet) to the nearest node in the road network.
    """
    min_dist = float("inf")
    nearest_node = None
    point_geom = Point(pt)
    for node in road_graph.nodes():
        dist = point_geom.distance(Point(node))
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node


def build_complete_graph_from_road_network(road_graph, stops_snapped, bus_stops):
    """
    Builds a complete graph for TSP from the road network, where edge weights
    are the shortest travel time (in seconds) between each pair of stops.
    """
    G = nx.DiGraph()
    # Add nodes with original positions (for plotting).
    for stop, coord in bus_stops.items():
        G.add_node(stop, pos=coord)
    # For each pair of stops, compute the shortest path "weight".
    for stop1 in bus_stops:
        for stop2 in bus_stops:
            if stop1 != stop2:
                node1 = stops_snapped[stop1]
                node2 = stops_snapped[stop2]
                try:
                    travel_time = nx.shortest_path_length(
                        road_graph, source=node1, target=node2, weight="weight"
                    )
                except nx.NetworkXNoPath:
                    travel_time = float("inf")
                G.add_edge(stop1, stop2, weight=travel_time)
    return G


# -----------------------------------------------------------------------------
# TSP SOLVER FUNCTIONS
# -----------------------------------------------------------------------------


def compute_tsp_route_greedy(G):
    """
    Approximate TSP using NetworkX's greedy approach.
    """
    tsp_route = nx.approximation.traveling_salesman_problem(G, weight="weight", cycle=True)
    print("TSP Route (greedy) before rotation:", tsp_route)
    total_travel_time = sum(
        G[tsp_route[i]][tsp_route[i + 1]]["weight"] for i in range(len(tsp_route) - 1)
    )
    print("Total travel time (seconds) (greedy):", total_travel_time)
    return tsp_route, total_travel_time


def compute_tsp_route_ilp(G):
    """
    Optimal TSP using a PuLP ILP (Miller-Tucker-Zemlin) approach.
    Good for smaller stop counts (~15 or fewer).
    """
    nodes = list(G.nodes())
    if "start" in nodes:
        nodes.remove("start")
        nodes.insert(0, "start")
    n = len(nodes)

    index_to_node = {i: nodes[i] for i in range(n)}
    weights = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                u_node = index_to_node[i]
                v_node = index_to_node[j]
                weights[(i, j)] = G[u_node][v_node]["weight"]

    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Subtour elimination variables
    u = {}
    u[0] = pulp.LpVariable("u_0", lowBound=0, upBound=0, cat="Continuous")
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat="Continuous")

    # Objective
    prob += pulp.lpSum(weights[(i, j)] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # Each node has exactly one outgoing edge
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1

    # Each node has exactly one incoming edge
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    # MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = prob.solve(solver)

    if pulp.LpStatus[result_status] != "Optimal":
        print("ILP solver did not find an optimal solution.")
        return None, None

    successor = {}
    for i in range(n):
        for j in range(n):
            if i != j and pulp.value(x[(i, j)]) > 0.5:
                successor[i] = j

    route_indices = [0]
    next_index = successor.get(0)
    while next_index is not None and next_index != 0 and next_index not in route_indices:
        route_indices.append(next_index)
        next_index = successor.get(next_index)
    route_indices.append(0)

    tsp_route = [index_to_node[i] for i in route_indices]

    total_travel_time = 0
    for i in range(len(route_indices) - 1):
        total_travel_time += weights[(route_indices[i], route_indices[i + 1])]

    print("TSP Route (ILP) before rotation:", tsp_route)
    print("Total travel time (seconds) (ILP):", total_travel_time)
    return tsp_route, total_travel_time


# -----------------------------------------------------------------------------
# DIRECTIONS & EXPORTS
# -----------------------------------------------------------------------------


def generate_directions(tsp_route, stops_snapped, road_graph):
    """
    Generates step-by-step driving directions for the entire TSP route,
    referencing actual street names from the road network edges.
    Distances are in feet, since EPSG:2283 is feet-based.
    """
    directions_steps = []
    step_num = 1
    last_heading = None

    for leg in range(len(tsp_route) - 1):
        source_stop = tsp_route[leg]
        dest_stop = tsp_route[leg + 1]

        # Arrival/Departure marker
        if leg == 0:
            directions_steps.append(
                {"Step": step_num, "Instruction": f"Start at stop {source_stop}"}
            )
            step_num += 1
        else:
            directions_steps.append(
                {"Step": step_num, "Instruction": f"Depart from stop {source_stop}"}
            )
            step_num += 1

        source_node = stops_snapped[source_stop]
        dest_node = stops_snapped[dest_stop]
        try:
            path_nodes = nx.shortest_path(
                road_graph, source=source_node, target=dest_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            directions_steps.append(
                {
                    "Step": step_num,
                    "Instruction": f"No path found from {source_stop} to {dest_stop}",
                }
            )
            step_num += 1
            continue

        # Build the list of segments
        segments = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            edge_data = road_graph.get_edge_data(u, v)
            seg_heading = compute_heading(u, v)
            seg_length = edge_data.get("length", 0)  # in feet
            seg_street = edge_data.get("street", "Unnamed Road")
            segments.append({"street": seg_street, "length": seg_length, "heading": seg_heading})

        # Group consecutive segments with the same street
        grouped = []
        if segments:
            current = segments[0].copy()
            for seg in segments[1:]:
                if seg["street"] == current["street"]:
                    current["length"] += seg["length"]
                else:
                    grouped.append(current)
                    current = seg.copy()
            grouped.append(current)

        # Create instructions
        for i, group in enumerate(grouped):
            length_ft = group["length"]
            if i == 0:
                if leg == 0:
                    instruction = f"Proceed on {group['street']} for {length_ft:.0f} ft."
                else:
                    turn = (
                        compute_turn_direction(last_heading, group["heading"])
                        if last_heading is not None
                        else "Straight"
                    )
                    if turn == "Straight":
                        instruction = (
                            f"Continue straight on {group['street']} for {length_ft:.0f} ft."
                        )
                    else:
                        instruction = f"Turn {turn} onto {group['street']} and continue for {length_ft:.0f} ft."
            else:
                turn = compute_turn_direction(grouped[i - 1]["heading"], group["heading"])
                if turn == "Straight":
                    instruction = f"Continue straight on {group['street']} for {length_ft:.0f} ft."
                else:
                    instruction = (
                        f"Turn {turn} onto {group['street']} and continue for {length_ft:.0f} ft."
                    )
            directions_steps.append({"Step": step_num, "Instruction": instruction})
            step_num += 1

        directions_steps.append({"Step": step_num, "Instruction": f"Arrive at stop {dest_stop}"})
        step_num += 1

        if grouped:
            last_heading = grouped[-1]["heading"]

    return directions_steps


def export_directions_excel(directions_steps, output_dir):
    """
    Exports the driving directions as an Excel (.xlsx) file.
    """
    df = pd.DataFrame(directions_steps)
    output_file = os.path.join(output_dir, "directions.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Exported directions to {output_file}")


def export_tsp_route_shapefile(tsp_route, stops_snapped, road_graph, roads_crs, output_dir):
    """
    Creates a shapefile of the entire TSP route by gathering
    every road-segment geometry used by the route.
    """
    route_features = []

    for i in range(len(tsp_route) - 1):
        source_stop = tsp_route[i]
        dest_stop = tsp_route[i + 1]

        source_node = stops_snapped[source_stop]
        dest_node = stops_snapped[dest_stop]

        try:
            path_nodes = nx.shortest_path(
                road_graph, source=source_node, target=dest_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            continue

        # For each consecutive node pair in this path, gather edge geometry
        for j in range(len(path_nodes) - 1):
            u = path_nodes[j]
            v = path_nodes[j + 1]
            edge_data = road_graph.get_edge_data(u, v)
            if not edge_data:
                continue
            geometry = edge_data.get("geometry", None)
            street_name = edge_data.get("street", "Unnamed Road")

            if geometry is not None:
                route_features.append(
                    {
                        "start_stop": source_stop,
                        "end_stop": dest_stop,
                        "street_name": street_name,
                        "geometry": geometry,
                    }
                )

    if not route_features:
        print("No route features found (possibly no valid path). Shapefile not created.")
        return

    route_gdf = gpd.GeoDataFrame(route_features, geometry="geometry", crs=roads_crs)

    out_path = os.path.join(output_dir, "tsp_route.shp")
    route_gdf.to_file(out_path)
    print(f"Exported the TSP route shapefile: {out_path}")


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------


def plot_tsp_route(G, tsp_route):
    """
    Plots the complete graph of stops (in Node space) and highlights the TSP route.
    """
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
    route_edges = list(zip(tsp_route, tsp_route[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color="red", width=2)
    plt.title("TSP Route for Selected Bus Stops (Road Network Time)")
    plt.xlabel("X (feet, EPSG:2283)")
    plt.ylabel("Y (feet, EPSG:2283)")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Executes the entire workflow for exporting GTFS stops, building a directed road network,
    snapping the stops to this network, solving the Traveling Salesman Problem (TSP) on
    selected bus stops, and then exporting directions and the route as a shapefile.

    Steps performed by this function:
        1. Reads GTFS stops from a CSV file in EPSG:4326 and reprojects them to EPSG:2283.
        2. Filters for the selected stop IDs from the GTFS data.
        3. Converts a user-provided Google Maps DMS string into decimal degrees, then reprojects it.
        4. Builds a directed road network from a shapefile (EPSG:2283).
        5. Snaps each selected stop (and the start) to the nearest node in the road network.
        6. Constructs a complete graph of travel times (in seconds) between each stop.
        7. Solves the TSP via either an ILP or a greedy approximation.
        8. Rotates the resulting TSP cycle so it starts/ends at the user-specified 'start' node.
        9. Generates step-by-step directions and exports them to Excel.
       10. Outputs the final route as a shapefile.
       11. (Optional) Plots the TSP route using a simple NetworkX plot of the “bus stops” graph.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Export GTFS stops in EPSG:2283
    stops_gdf = export_gtfs_stops(GTFS_PATH, OUTPUT_DIR, TARGET_ROAD_CRS)

    # 2. Filter the selected stops
    bus_stops, selected_stops_gdf = filter_selected_stops(
        stops_gdf, SELECTED_STOP_IDS, SELECTED_STOP_ID_COL
    )
    # bus_stops now has (x,y) in EPSG:2283 (feet)

    # 3. Parse the user-provided Google Maps DMS into lat/lon
    lat_dd, lon_dd = parse_google_maps_coords(GOOGLE_MAPS_COORD_STR)
    # Reproject from EPSG:4326 -> EPSG:2283 (feet)
    start_coord_2283 = reproject_point_4326_to_2283(lon_dd, lat_dd)
    bus_stops["start"] = start_coord_2283

    # 4. Build the directed road network (assumed EPSG:2283)
    road_graph, roads_gdf = build_directed_road_network(
        ROADWAYS_SHP_PATH,
        oneway_col=ONEWAY_COL,
        speed_col=SPEED_COL,
        target_crs=TARGET_ROAD_CRS,
        street_name_col=STREET_NAME_COL,
    )

    # 5. Snap each stop (and start) to the nearest node
    stops_snapped = {}
    for stop_id, coord in bus_stops.items():
        snapped = snap_point_to_network(coord, road_graph)
        stops_snapped[stop_id] = snapped
    print("Bus stops snapped to road network nodes:")
    for k, v in stops_snapped.items():
        print(f"  {k}: {v}")

    # 6. Build a complete graph from road network travel times
    road_time_complete_graph = build_complete_graph_from_road_network(
        road_graph, stops_snapped, bus_stops
    )

    # 7. Solve TSP using ILP or greedy
    opt = OPTIMIZATION_CONFIG["optimization_approach"].lower()
    if opt == "ilp":
        tsp_route, total_travel_time = compute_tsp_route_ilp(road_time_complete_graph)
    else:
        tsp_route, total_travel_time = compute_tsp_route_greedy(road_time_complete_graph)

    if tsp_route is None:
        print("TSP route computation failed.")
        return

    # 8. Rotate route so it starts/ends at 'start'
    tsp_route = rotate_route(tsp_route, "start")
    print("TSP Route after rotation:", tsp_route)

    # 9. Generate directions and export to Excel
    directions_steps = generate_directions(tsp_route, stops_snapped, road_graph)
    export_directions_excel(directions_steps, OUTPUT_DIR)

    # 10. Export route as a shapefile
    export_tsp_route_shapefile(
        tsp_route,
        stops_snapped,
        road_graph,
        roads_gdf.crs,  # same CRS as roads (EPSG:2283)
        OUTPUT_DIR,
    )

    # 11. Plot TSP route (just on the "bus stops" node graph)
    plot_tsp_route(road_time_complete_graph, tsp_route)


if __name__ == "__main__":
    main()
