"""
Export GTFS Stops to Shapefile and Compute a TSP Route for Selected Bus Stops,
using a road network (with one-way directions and variable speeds) as a time network.

This script performs the following tasks:
1. Reads the GTFS stops.txt file and exports it as a shapefile.
2. Filters a subset of stops based on a user-defined column name (either 'stop_id' or 'stop_code').
3. Lets the user specify a starting point (in DMS format).
4. Builds a directed road network from a shapefile that respects one-way directions,
   computing travel time for each road segment using a speed column.
5. Snaps each bus stop (and the starting point) to the nearest node on the road network.
6. Constructs a complete graph where the weight between two bus stops is the shortest
   travel time (computed via the road network).
7. Computes a TSP route (using either a greedy approximation or a PuLP ILP solver).
8. Rotates the TSP route so that it begins and ends at the user-defined starting point.
9. Exports step-by-step driving directions (using street name, segment length, and a basic turn
   instruction) to an Excel (.xlsx) file.
10. Plots the stops and the computed TSP route.

Notes:
    - Two optimization approaches are available for computing the TSP route:
      "greedy" (using NetworkX's approximation algorithm) and "ilp" (using a PuLP-based ILP solver).
    - The ILP solver can provide an exact solution but may be computationally intensive
      for a larger number of stops (generally more than ~15 stops). For larger problems,
      the greedy method is recommended.
    - This solver is based on speed limit and distance and does not consider congestion or delay related
      to traffic control.
    - The directions export uses the street name (from the configured column, default "FULLNAME"),
      the segment length (in feet), and a turn instruction (Left, Right, or Straight) based on
      the computed segment headings.
"""

import math
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pulp
from shapely.geometry import Point, LineString

# --------------------- CONFIGURATION ---------------------
GTFS_PATH = r'C:\Path\To\Your\GTFS_data'           # Update to your GTFS folder path
OUTPUT_DIR = r'C:\Path\To\Your\Output_folder'         # Update to your desired output folder
ROADWAYS_SHP_PATH = r'C:\Path\To\Your\Roadways.shp'    # File path to roadways.shp

# Selected stop identifiers for the TSP demonstration.
SELECTED_STOP_IDS = ['1001', '1002', '1003', '1004', '1005']

# Specify the name of the column containing stop identifiers in GTFS.
# It can be 'stop_id' (default) or 'stop_code' depending on your data.
SELECTED_STOP_ID_COL = 'stop_id'

# Starting point in DMS format (e.g., as from Google Maps)
START_LAT_DMS = "38°50'38.5\"N"
START_LON_DMS = "77°19'17.5\"W"

# For the road network creation:
ONEWAY_COL = "ONEWAY"         # Expected value 'Y' means one-way; otherwise two-way.
SPEED_COL = "SPEEDLIMI"       # Column with the speed limit (assumed to be in mph).
STREET_NAME_COL = "FULLNAME"   # Default street name field.
TARGET_ROAD_CRS = "EPSG:2263"  # Use a projected CRS (here in US feet).

# Fallback average speed in feet per second (e.g., 44 fps is roughly 30 mph).
AVERAGE_SPEED_FPS = 44  

# Optimization approach configuration.
# Options: "ilp" or "greedy"
OPTIMIZATION_CONFIG = {
    "optimization_approach": "ilp",  # Note: ILP is recommended only for a small number of stops (~15 or fewer)
}

# --------------------- HELPER FUNCTIONS ---------------------

def dms_to_decimal(dms_str):
    """
    Converts a coordinate in DMS format (e.g., "38°50'38.5\"N") to decimal degrees.
    """
    dms_str = dms_str.strip()
    pattern = r"""(?P<degrees>\d+)[°\s]+(?P<minutes>\d+)[\'\s]+(?P<seconds>\d+(?:\.\d+)?)[\"\s]*(?P<direction>[NSEW])"""
    match = re.match(pattern, dms_str, re.VERBOSE)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    degrees = float(match.group("degrees"))
    minutes = float(match.group("minutes"))
    seconds = float(match.group("seconds"))
    direction = match.group("direction").upper()
    dec = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        dec = -dec
    return dec

def rotate_route(route, start_node):
    """
    Rotates the TSP route so that it starts (and ends) with the given start_node.
    Assumes that route is a cycle (i.e. the first and last node are the same).
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
    p1 and p2 are (x, y) tuples.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def compute_turn_direction(heading1, heading2, threshold=15):
    """
    Computes a simple turn instruction (Left, Right, or Straight) based on the difference
    between two headings. If the difference is within the threshold, returns "Straight".
    Otherwise returns "Left" if the change is positive, and "Right" if negative.
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

# --------------------- GTFS AND STOP FUNCTIONS ---------------------

def export_gtfs_stops(gtfs_path, output_dir):
    """
    Reads the GTFS stops.txt file and exports it as a shapefile.
    
    Returns:
        GeoDataFrame: A GeoDataFrame containing all GTFS stops.
    """
    stops_file = os.path.join(gtfs_path, 'stops.txt')
    stops_df = pd.read_csv(stops_file)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs='EPSG:4326'
    )
    shapefile_path = os.path.join(output_dir, 'gtfs_stops.shp')
    stops_gdf.to_file(shapefile_path)
    print(f"Exported GTFS stops to shapefile: {shapefile_path}")
    return stops_gdf

def filter_selected_stops(stops_gdf, selected_stop_ids, stop_id_col):
    """
    Filters the stops GeoDataFrame for the selected stop IDs using the specified column.
    
    Args:
        stops_gdf (GeoDataFrame): GeoDataFrame containing GTFS stops.
        selected_stop_ids (list): List of stop identifiers to select.
        stop_id_col (str): The name of the column to match the identifiers (e.g., 'stop_id' or 'stop_code').
    
    Returns:
        dict: A dictionary mapping stop identifiers to (longitude, latitude) tuples.
        GeoDataFrame: The filtered stops.
    """
    selected_stops_gdf = stops_gdf[stops_gdf[stop_id_col].isin(selected_stop_ids)]
    if selected_stops_gdf.empty:
        print("No stops found with the specified identifiers. Please check your GTFS stops.txt file.")
        exit()
    bus_stops = {row[stop_id_col]: (row.geometry.x, row.geometry.y)
                 for _, row in selected_stops_gdf.iterrows()}
    return bus_stops, selected_stops_gdf

# --------------------- ROAD NETWORK FUNCTIONS ---------------------

def build_directed_road_network(road_shp_path, oneway_col=ONEWAY_COL, speed_col=SPEED_COL,
                                target_crs=TARGET_ROAD_CRS, street_name_col=STREET_NAME_COL):
    """
    Builds a directed road network graph from a shapefile.
    
    1. Reads the road network shapefile and reprojects it to the target CRS.
    2. Iterates over each road segment, splitting geometries into consecutive coordinate pairs.
       - If oneway_col is 'Y', adds an edge in the forward direction only.
       - Otherwise, adds edges in both directions.
    3. For each segment, attempts to read the speed limit from the specified speed_col.
       - If a valid speed limit is found (assumed to be in mph), converts it to feet per second.
       - Otherwise, falls back to the AVERAGE_SPEED_FPS value.
    4. Retrieves the street name from street_name_col.
    5. Computes travel time for each segment as: travel_time = segment_length / speed_fps.
       
    Returns:
        G (DiGraph): A directed NetworkX graph of the road network.
        roads_gdf: The reprojected GeoDataFrame of roads.
    """
    roads_gdf = gpd.read_file(road_shp_path)
    roads_gdf = roads_gdf.to_crs(target_crs)
    
    G = nx.DiGraph()
    for idx, row in roads_gdf.iterrows():
        oneway = str(row.get(oneway_col, "N")).upper()
        # Retrieve the street name from the configured column.
        street_name = row.get(street_name_col, "Unnamed Road")
        
        # Attempt to retrieve the speed limit (assumed to be in mph) from the specified column.
        try:
            speed_val = float(row.get(speed_col, None))
            if speed_val <= 0:
                raise ValueError("Non-positive speed value")
            speed_fps = speed_val * 1.46667  # Convert mph to feet per second.
        except (TypeError, ValueError):
            speed_fps = AVERAGE_SPEED_FPS  # Fallback average speed.
        
        geom = row.geometry
        # Handle both LineString and MultiLineString geometries.
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i+1]
                seg_line = LineString([start, end])
                length = seg_line.length
                travel_time = length / speed_fps  # travel time in seconds
                if oneway == 'Y':
                    # Add only forward edge.
                    G.add_edge(start, end, weight=travel_time, geometry=seg_line, length=length, street=street_name)
                else:
                    # Add edges in both directions.
                    G.add_edge(start, end, weight=travel_time, geometry=seg_line, length=length, street=street_name)
                    G.add_edge(end, start, weight=travel_time, geometry=seg_line, length=length, street=street_name)
    print(f"Road network graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, roads_gdf

def snap_point_to_network(pt, road_graph):
    """
    Snaps a point (tuple (x, y)) to the nearest node in the road network graph.
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
    Builds a complete (directed) graph for TSP based on the road network travel times.
    
    Each edge weight is computed as the shortest travel time (in seconds) between
    the snapped nodes of the bus stops.
    
    Args:
        road_graph (DiGraph): The road network graph.
        stops_snapped (dict): Mapping of bus stop IDs to the snapped road network node.
        bus_stops (dict): Mapping of bus stop IDs to original (lon, lat) coordinates.
    
    Returns:
        DiGraph: A complete directed graph with travel time weights.
    """
    G = nx.DiGraph()
    # Add nodes with original positions (for plotting purposes).
    for stop, coord in bus_stops.items():
        G.add_node(stop, pos=coord)
    # Compute travel time for each pair using the road network.
    for stop1 in bus_stops:
        for stop2 in bus_stops:
            if stop1 != stop2:
                node1 = stops_snapped[stop1]
                node2 = stops_snapped[stop2]
                try:
                    travel_time = nx.shortest_path_length(road_graph, source=node1, target=node2, weight='weight')
                except nx.NetworkXNoPath:
                    travel_time = float('inf')
                G.add_edge(stop1, stop2, weight=travel_time)
    return G

# --------------------- TSP SOLVER FUNCTIONS ---------------------

def compute_tsp_route_greedy(G):
    """
    Computes an approximate TSP route for the given complete graph using NetworkX's
    approximation algorithm (greedy approach).
    
    Args:
        G (Graph): A NetworkX graph with travel time as edge weights.
    
    Returns:
        tuple: The TSP route (list of stops) and the total travel time.
    """
    tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)
    print("TSP Route (greedy) for selected stops (before rotation):", tsp_route)
    total_travel_time = sum(
        G[tsp_route[i]][tsp_route[i+1]]['weight'] 
        for i in range(len(tsp_route) - 1)
    )
    print("Total travel time (seconds) (greedy):", total_travel_time)
    return tsp_route, total_travel_time

def compute_tsp_route_ilp(G):
    """
    Computes the optimal TSP route for the given complete graph using an ILP solver (PuLP).
    
    Args:
        G (DiGraph): A complete directed graph with travel time as edge weights.
    
    Returns:
        tuple: The TSP route (list of stops) and the total travel time.
    
    Note:
        - This ILP approach (using the Miller-Tucker-Zemlin formulation) may be computationally
          intensive for a larger number of stops (generally >15). For such cases, consider using the greedy method.
    """
    # Create a list of nodes, ensuring that the 'start' node is first if present.
    nodes = list(G.nodes())
    if 'start' in nodes:
        nodes.remove('start')
        nodes.insert(0, 'start')
    n = len(nodes)
    
    # Mapping from index to node
    index_to_node = {i: nodes[i] for i in range(n)}
    
    # Create a dictionary for weights.
    weights = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                u_node = index_to_node[i]
                v_node = index_to_node[j]
                weights[(i, j)] = G[u_node][v_node]['weight']
    
    # Define the ILP model.
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)
    
    # Decision variables: x[i,j] = 1 if edge from i to j is in the tour.
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Auxiliary variables for subtour elimination (MTZ formulation).
    u = {}
    u[0] = pulp.LpVariable("u_0", lowBound=0, upBound=0, cat='Continuous')
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat='Continuous')
    
    # Objective: minimize total travel time.
    prob += pulp.lpSum(weights[(i, j)] * x[(i, j)] for i in range(n) for j in range(n) if i != j)
    
    # Constraints: each node has exactly one outgoing edge.
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1, f"outgoing_{i}"
    
    # Constraints: each node has exactly one incoming edge.
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1, f"incoming_{j}"
    
    # Subtour elimination constraints (MTZ).
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1, f"subtour_{i}_{j}"
    
    # Solve the ILP.
    solver = pulp.PULP_CBC_CMD(msg=False)
    result_status = prob.solve(solver)
    
    if pulp.LpStatus[result_status] != "Optimal":
        print("ILP solver did not find an optimal solution.")
        return None, None
    
    # Extract the tour: for each node, determine its successor.
    successor = {}
    for i in range(n):
        for j in range(n):
            if i != j and pulp.value(x[(i, j)]) > 0.5:
                successor[i] = j
    
    # Reconstruct the route starting from node 0 ('start').
    route_indices = [0]
    next_index = successor.get(0)
    while next_index is not None and next_index != 0 and next_index not in route_indices:
        route_indices.append(next_index)
        next_index = successor.get(next_index)
    route_indices.append(0)  # complete the cycle
    
    # Map indices back to node names.
    tsp_route = [index_to_node[i] for i in route_indices]
    
    # Calculate total travel time.
    total_travel_time = 0
    for i in range(len(route_indices) - 1):
        total_travel_time += weights[(route_indices[i], route_indices[i+1])]
    
    print("TSP Route (ILP) for selected stops (before rotation):", tsp_route)
    print("Total travel time (seconds) (ILP):", total_travel_time)
    
    return tsp_route, total_travel_time

# --------------------- DIRECTIONS EXPORT FUNCTIONS ---------------------

def generate_directions(tsp_route, stops_snapped, road_graph):
    """
    Generates step-by-step driving directions for the entire TSP route.
    
    For each leg (between consecutive stops in the TSP route), it:
      - Computes the shortest path on the road network.
      - Breaks the path into segments.
      - Groups consecutive segments with the same street name.
      - Uses segment headings to decide if a turn is required.
    
    Returns:
        List of dictionaries with step number and instruction text.
    """
    directions_steps = []
    step_num = 1
    last_heading = None  # store the last heading from the previous leg
    for leg in range(len(tsp_route) - 1):
        source_stop = tsp_route[leg]
        dest_stop = tsp_route[leg+1]
        # Add an arrival/departure marker.
        if leg == 0:
            directions_steps.append({"Step": step_num, "Instruction": f"Start at stop {source_stop}"})
            step_num += 1
        else:
            directions_steps.append({"Step": step_num, "Instruction": f"Depart from stop {source_stop}"})
            step_num += 1
        
        source_node = stops_snapped[source_stop]
        dest_node = stops_snapped[dest_stop]
        try:
            path_nodes = nx.shortest_path(road_graph, source=source_node, target=dest_node, weight='weight')
        except nx.NetworkXNoPath:
            directions_steps.append({"Step": step_num, "Instruction": f"No path found from {source_stop} to {dest_stop}"})
            step_num += 1
            continue
        
        # Build list of segments from the path.
        segments = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            edge_data = road_graph.get_edge_data(u, v)
            seg_heading = compute_heading(u, v)
            seg_length = edge_data.get("length", 0)
            seg_street = edge_data.get("street", "Unnamed Road")
            segments.append({
                "street": seg_street,
                "length": seg_length,
                "heading": seg_heading
            })
        
        # Group consecutive segments with the same street.
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
        
        # Create instructions from grouped segments.
        for i, group in enumerate(grouped):
            if i == 0:
                if leg == 0:
                    # First leg, first group.
                    instruction = f"Proceed on {group['street']} for {group['length']:.0f} ft."
                else:
                    # For subsequent legs, use the last heading from the previous leg.
                    turn = compute_turn_direction(last_heading, group["heading"]) if last_heading is not None else "Straight"
                    if turn == "Straight":
                        instruction = f"Continue straight on {group['street']} for {group['length']:.0f} ft."
                    else:
                        instruction = f"Turn {turn} onto {group['street']} and continue for {group['length']:.0f} ft."
            else:
                turn = compute_turn_direction(grouped[i-1]["heading"], group["heading"])
                if turn == "Straight":
                    instruction = f"Continue straight on {group['street']} for {group['length']:.0f} ft."
                else:
                    instruction = f"Turn {turn} onto {group['street']} and continue for {group['length']:.0f} ft."
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

# --------------------- PLOTTING FUNCTION ---------------------

def plot_tsp_route(G, tsp_route):
    """
    Plots the complete graph and highlights the computed TSP route.
    
    Args:
        G (Graph): A NetworkX graph of bus stops.
        tsp_route (list): The list of stops representing the TSP route.
    """
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    route_edges = list(zip(tsp_route, tsp_route[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2)
    plt.title("TSP Route for Selected Bus Stops (Road Network Time)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# --------------------- MAIN FUNCTION ---------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Export GTFS stops to a shapefile and get the stops GeoDataFrame.
    stops_gdf = export_gtfs_stops(GTFS_PATH, OUTPUT_DIR)
    
    # 2. Filter the selected stops using the specified stop ID column.
    bus_stops, selected_stops_gdf = filter_selected_stops(stops_gdf, SELECTED_STOP_IDS, SELECTED_STOP_ID_COL)
    
    # 3. Convert the starting point DMS values to decimal degrees and add to bus stops.
    start_lat = dms_to_decimal(START_LAT_DMS)
    start_lon = dms_to_decimal(START_LON_DMS)
    # Note: The bus_stops dictionary stores coordinates as (longitude, latitude).
    start_coord = (start_lon, start_lat)
    bus_stops['start'] = start_coord
    
    # 4. Build the directed road network from the road shapefile.
    road_graph, roads_gdf = build_directed_road_network(
        ROADWAYS_SHP_PATH,
        oneway_col=ONEWAY_COL,
        speed_col=SPEED_COL,
        target_crs=TARGET_ROAD_CRS,
        street_name_col=STREET_NAME_COL
    )
    
    # 5. Snap each bus stop (and the starting point) to the nearest node in the road network.
    stops_snapped = {}
    for stop_id, coord in bus_stops.items():
        snapped = snap_point_to_network(coord, road_graph)
        stops_snapped[stop_id] = snapped
    print("Bus stops snapped to road network nodes:")
    for k, v in stops_snapped.items():
        print(f"  {k}: {v}")
    
    # 6. Build a complete graph where edge weights are computed as the shortest travel times
    #    between stops (via the road network).
    road_time_complete_graph = build_complete_graph_from_road_network(road_graph, stops_snapped, bus_stops)
    
    # 7. Compute the TSP route using the selected optimization approach.
    opt = OPTIMIZATION_CONFIG["optimization_approach"].lower()
    if opt == "ilp":
        tsp_route, total_travel_time = compute_tsp_route_ilp(road_time_complete_graph)
    else:
        tsp_route, total_travel_time = compute_tsp_route_greedy(road_time_complete_graph)
    
    if tsp_route is None:
        print("TSP route computation failed.")
        return
    
    tsp_route = rotate_route(tsp_route, 'start')
    print("TSP Route based on road network travel times (after rotation):", tsp_route)
    
    # 8. Generate and export driving directions as an Excel file.
    directions_steps = generate_directions(tsp_route, stops_snapped, road_graph)
    export_directions_excel(directions_steps, OUTPUT_DIR)
    
    # 9. Plot the computed TSP route.
    plot_tsp_route(road_time_complete_graph, tsp_route)
    
if __name__ == '__main__':
    main()
