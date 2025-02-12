"""
Script that takes bus network, road network, and GTFS data and outputs a route shapefile that
matches the network and is segmented at the stops. It is very useful for checking stop spacing.
"""

import os

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint
from shapely.ops import nearest_points, linemerge, split

# -----------------------------------------------------------------------------
# STEP 0: CONFIGURATION
# -----------------------------------------------------------------------------
GTFS_FOLDER = r'C:\Your\Folder\Path\For\GTFS'
ROAD_NETWORK_FILE = r'C:\Your\File\Path\For\road_network.shp'
ROUTE_SHAPE_FILE = r'C:\Your\File\Path\For\bus_system_network.sh'

OUTPUT_FOLDER = r''
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Choose a projected CRS (units in feet or meters); e.g., EPSG:2263 for the DC area
PROJECTED_CRS = "EPSG:2263"

FILTER_ROUTE = '101'
FILTER_DIRECTION = '0'
FILTER_DEPARTURE = '08:00'
FILTER_CALENDAR = '3'

print("STEP 0: Filtering criteria set:")
print(f"  Route:           {FILTER_ROUTE}")
print(f"  Direction:       {FILTER_DIRECTION}")
print(f"  Departure time:  {FILTER_DEPARTURE}")
print(f"  Calendar:        {FILTER_CALENDAR}")

# =============================================================================
# STEP 1: Load GTFS files, filter trips, and select one trip
# =============================================================================
def load_gtfs(gtfs_folder):
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))
    trips = pd.read_csv(os.path.join(gtfs_folder, "trips.txt"))
    calendar = pd.read_csv(os.path.join(gtfs_folder, "calendar.txt"))
    return stops, stop_times, trips, calendar

stops_df, stop_times_df, trips_df, calendar_df = load_gtfs(GTFS_FOLDER)
print("Loaded GTFS files.")

# Debug: print some unique values
print("Unique route_id in trips:", trips_df["route_id"].unique())
print("Unique direction_id in trips:", trips_df["direction_id"].unique())
print("Unique service_id in trips:", trips_df["service_id"].unique())

# Filter the calendar
try:
    service_filter = int(FILTER_CALENDAR)
except ValueError:
    service_filter = FILTER_CALENDAR
calendar_filtered = calendar_df[calendar_df["service_id"] == service_filter]
valid_service_ids = calendar_filtered["service_id"].unique()
print(f"Calendar filter applied: found {len(valid_service_ids)} valid service_id(s): {valid_service_ids}")

# Filter trips
trips_df["route_id"] = trips_df["route_id"].astype(str)
trips_df["direction_id"] = trips_df["direction_id"].astype(str)
trips_df["service_id"] = trips_df["service_id"].astype(str)

FILTER_ROUTE_str = str(FILTER_ROUTE)
FILTER_DIRECTION_str = str(FILTER_DIRECTION)
FILTER_CALENDAR_str = str(FILTER_CALENDAR)

trips_filtered = trips_df[
    (trips_df["route_id"] == FILTER_ROUTE_str) &
    (trips_df["direction_id"] == FILTER_DIRECTION_str) &
    (trips_df["service_id"].isin([FILTER_CALENDAR_str]))
]
print(f"Trips filtered by route, direction, calendar: {len(trips_filtered)} trips found.")

# Merge stop_times with filtered trips
merged = pd.merge(stop_times_df, trips_filtered, on="trip_id")
print(f"After merging stop_times with filtered trips, {len(merged)} records found.")
if not merged.empty:
    print("Sample original departure_time values:")
    print(merged["departure_time"].head(10).to_string(index=False))
else:
    print("No records after merging stop_times and trips.")

merged["departure_time_wo_sec"] = merged["departure_time"].str[:5]
if not merged.empty:
    print("Sample departure_time without seconds:")
    print(merged["departure_time_wo_sec"].head(10).to_string(index=False))

merged_filtered = merged[merged["departure_time_wo_sec"] >= FILTER_DEPARTURE]
print(f"After filtering by departure time >= {FILTER_DEPARTURE}, {len(merged_filtered)} records remain.")

if merged_filtered.empty:
    print("WARNING: No trips found matching the specified criteria!")
    selected_trip_id = None
else:
    selected_trip_id = merged_filtered["trip_id"].iloc[0]
    print(f"Selected trip: {selected_trip_id}")

# =============================================================================
# STEP 1 (Revised): Create unsnapped stops and chord segments for the selected trip
# =============================================================================

if selected_trip_id is None:
    raise ValueError("No trip selected from STEP 0; cannot proceed with STEP 1.")

# Make sure the output folder is absolute
OUTPUT_FOLDER = os.path.abspath(OUTPUT_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load relevant GTFS files
stops_file = os.path.join(GTFS_FOLDER, "stops.txt")
stop_times_file = os.path.join(GTFS_FOLDER, "stop_times.txt")
stops_df = pd.read_csv(stops_file)
stop_times_df = pd.read_csv(stop_times_file)

# Merge stops with stop_times to get lat/lon, etc.
trip_stops_df = pd.merge(stop_times_df, stops_df, on="stop_id", how="left")
trip_stops_selected = trip_stops_df[trip_stops_df["trip_id"] == selected_trip_id].copy()
print(f"Number of stops for trip {selected_trip_id}: {len(trip_stops_selected)}")
if trip_stops_selected.empty:
    raise ValueError(f"No stops found for trip {selected_trip_id}.")

trip_stops_selected["stop_sequence"] = trip_stops_selected["stop_sequence"].astype(int)
trip_stops_selected.sort_values("stop_sequence", inplace=True)

print("Sample selected stops (trip_id, stop_id, stop_name, stop_sequence, stop_lat, stop_lon):")
print(trip_stops_selected[["trip_id","stop_id","stop_name","stop_sequence","stop_lat","stop_lon"]].head(10))

# Create a GeoDataFrame of unsnapped stops in EPSG:4326, then reproject
unsnapped_stops_gdf = gpd.GeoDataFrame(
    trip_stops_selected,
    geometry=gpd.points_from_xy(trip_stops_selected["stop_lon"], trip_stops_selected["stop_lat"]),
    crs="EPSG:4326"
)
# Reproject to the chosen CRS
unsnapped_stops_gdf = unsnapped_stops_gdf.to_crs(PROJECTED_CRS)

print(f"unsnapped_stops_gdf shape: {unsnapped_stops_gdf.shape}")
print("Sample geometry from unsnapped stops:", unsnapped_stops_gdf.geometry.head(2))

# Export unsnapped stops
unsnapped_stops_out = os.path.join(OUTPUT_FOLDER, f"unsnapped_stops_trip_{selected_trip_id}.shp")
unsnapped_stops_gdf.to_file(unsnapped_stops_out)
print(f"Exported unsnapped stops to {unsnapped_stops_out}")

# Create chord segments between consecutive stops (unsnapped)
sorted_stops = unsnapped_stops_gdf.sort_values("stop_sequence").reset_index(drop=True)
chord_segments = []
for i in range(len(sorted_stops) - 1):
    start_stop = sorted_stops.iloc[i]
    end_stop   = sorted_stops.iloc[i+1]
    segment_line = LineString([start_stop.geometry, end_stop.geometry])
    attrs = {
        "segment_id": i + 1,
        "start_stop_id": start_stop["stop_id"],
        "start_stop_name": start_stop.get("stop_name", None),
        "start_seq": start_stop["stop_sequence"],
        "end_stop_id": end_stop["stop_id"],
        "end_stop_name": end_stop.get("stop_name", None),
        "end_seq": end_stop["stop_sequence"],
        "length": segment_line.length,
        "geometry": segment_line
    }
    chord_segments.append(attrs)

chord_segments_gdf = gpd.GeoDataFrame(chord_segments, crs=unsnapped_stops_gdf.crs)
chord_segments_out = os.path.join(OUTPUT_FOLDER, f"chord_segments_trip_{selected_trip_id}.shp")
chord_segments_gdf.to_file(chord_segments_out)
print(f"Exported chord segments to {chord_segments_out}")

# =============================================================================
# STEP 3: Snap stops to roads, filter road segments, etc.
# =============================================================================
route_shapes_gdf = gpd.read_file(ROUTE_SHAPE_FILE)
# Reproject
route_shapes_gdf = route_shapes_gdf.to_crs(PROJECTED_CRS)

route_shape_selected = route_shapes_gdf[route_shapes_gdf["ROUTE_NUMB"] == FILTER_ROUTE]
if route_shape_selected.empty:
    raise ValueError(f"No route shape found with ROUTE_NUMB = {FILTER_ROUTE}")
bus_route_feature = route_shape_selected.iloc[0]

# Export bus route
bus_route_out = os.path.join(OUTPUT_FOLDER, f"bus_route_{FILTER_ROUTE}.shp")
gpd.GeoDataFrame([bus_route_feature], crs=route_shapes_gdf.crs).to_file(bus_route_out)
print(f"Exported bus route to {bus_route_out}")

# Buffer the route by 25 feet
bus_route_proj = gpd.GeoSeries([bus_route_feature.geometry], crs=route_shapes_gdf.crs)
bus_route_buffer = bus_route_proj.buffer(25)
bus_route_buffer_out = os.path.join(OUTPUT_FOLDER, f"bus_route_{FILTER_ROUTE}_buffer.shp")
gpd.GeoDataFrame(geometry=[bus_route_buffer.unary_union], crs=PROJECTED_CRS).to_file(bus_route_buffer_out)
print("Created and exported bus route buffer (25 feet).")

# Load and reproject the road network
roads_gdf = gpd.read_file(ROAD_NETWORK_FILE)
roads_gdf = roads_gdf.to_crs(PROJECTED_CRS)
buffer_geom = bus_route_buffer.unary_union

# Filter roads that are completely within the buffer
filtered_roads = roads_gdf[roads_gdf.within(buffer_geom)]
print(f"Filtered roads: {len(filtered_roads)} segments within buffer.")
filtered_roads_out = os.path.join(OUTPUT_FOLDER, "roads_within_buffer.shp")
filtered_roads.to_file(filtered_roads_out)
print(f"Exported filtered roads to {filtered_roads_out}")

# Dissolve filtered roads for snapping
if filtered_roads.empty:
    raise ValueError("No roads found within the route buffer!")
all_roads_raw = filtered_roads.unary_union
merged_roads = linemerge(all_roads_raw)
if merged_roads.geom_type == "LineString":
    roads_for_snapping = merged_roads
else:
    roads_for_snapping = all_roads_raw
print("Road geometry for snapping type:", roads_for_snapping.geom_type)

# Snap function using project/interpolate fallback
def snap_point_to_roads(pt, roads_geom):
    try:
        distance_along = roads_geom.project(pt)
        return roads_geom.interpolate(distance_along)
    except Exception as e:
        print("Project/Interpolate failed, using nearest_points. Error:", e)
        _, snapped_pt = nearest_points(pt, roads_geom)
        return snapped_pt

# Snap unsnapped stops
unsnapped_stops_proj = unsnapped_stops_gdf  # already in PROJECTED_CRS
snapped_geoms = unsnapped_stops_proj.geometry.apply(lambda pt: snap_point_to_roads(pt, roads_for_snapping))
snapped_stops_gdf = unsnapped_stops_proj.copy()
snapped_stops_gdf["geometry"] = snapped_geoms

snapped_stops_out = os.path.join(OUTPUT_FOLDER, f"snapped_stops_trip_{selected_trip_id}.shp")
snapped_stops_gdf.to_file(snapped_stops_out)
print(f"Exported snapped stops to {snapped_stops_out}")

# Create chord segments for snapped stops
sorted_snapped = snapped_stops_gdf.sort_values("stop_sequence").reset_index(drop=True)
snapped_chord_segments = []
for i in range(len(sorted_snapped) - 1):
    start_stop = sorted_snapped.iloc[i]
    end_stop   = sorted_snapped.iloc[i+1]
    segment_line = LineString([start_stop.geometry, end_stop.geometry])
    seg_attrs = {
        "segment_id": i + 1,
        "start_stop_id": start_stop["stop_id"],
        "start_stop_name": start_stop.get("stop_name", None),
        "start_seq": start_stop["stop_sequence"],
        "end_stop_id": end_stop["stop_id"],
        "end_stop_name": end_stop.get("stop_name", None),
        "end_seq": end_stop["stop_sequence"],
        "length": segment_line.length,
        "geometry": segment_line
    }
    snapped_chord_segments.append(seg_attrs)

snapped_chord_segments_gdf = gpd.GeoDataFrame(snapped_chord_segments, crs=PROJECTED_CRS)
snapped_chord_segments_out = os.path.join(OUTPUT_FOLDER, f"snapped_chord_segments_trip_{selected_trip_id}.shp")
snapped_chord_segments_gdf.to_file(snapped_chord_segments_out)
print(f"Exported snapped chord segments to {snapped_chord_segments_out}")

# (Optional) Merge them into one feature
merged_snapped_line = linemerge([seg["geometry"] for seg in snapped_chord_segments])
merged_snapped_gdf = gpd.GeoDataFrame({"trip_id": [selected_trip_id]}, geometry=[merged_snapped_line], crs=PROJECTED_CRS)
merged_snapped_out = os.path.join(OUTPUT_FOLDER, f"merged_snapped_chord_trip_{selected_trip_id}.shp")
merged_snapped_gdf.to_file(merged_snapped_out)
print(f"Exported merged snapped chord to {merged_snapped_out}")

# =============================================================================
# STEP 4: Build a directed network that includes mid-segment connections
#         for each snapped stop, then compute shortest paths
# =============================================================================


def build_directed_network_with_stops(roads_gdf, snapped_stops_gdf, oneway_col="ONEWAY"):
    """
    Builds a directed graph from roads_gdf, respecting ONEWAY='Y' or 'N',
    AND connects each snapped stop (mid-segment) to the relevant segment endpoints.

    1) For each road line, add edges for each pair of consecutive endpoints.
       - If oneway='Y', add one forward edge; if 'N', add both directions.
    2) For each snapped stop:
        - Identify the nearest line and its endpoints (A->B).
        - Create small "shortcut" edges from (stop) to A/B so that the stop is
          actually reachable in the graph.
        - If oneway='Y' or 'N', respect direction logic for these small edges.
    """

    # --- 1. Basic directed edges from road endpoints ---
    G = nx.DiGraph()

    for idx, row in roads_gdf.iterrows():
        oneway_value = str(row.get(oneway_col, "N")).upper()
        geom = row.geometry

        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            # Add edges from each pair of consecutive coords
            for i in range(len(coords) - 1):
                start = coords[i]
                end   = coords[i+1]
                seg_line = LineString([start, end])
                length   = seg_line.length

                if oneway_value == 'Y':
                    G.add_edge(start, end, weight=length, geometry=seg_line)
                else:
                    # 'N' or anything else => two-way
                    G.add_edge(start, end, weight=length, geometry=seg_line)
                    G.add_edge(end, start, weight=length, geometry=seg_line)

    # --- 2. Insert each snapped stop as a node + small edges to the nearest segment ---
    # We can store the full geometry for each coordinate, or just keep the weight
    # so that the route is possible in the final path search.

    # We'll need a function to find the single nearest road segment and
    # the coordinates of that segment's endpoints for each stop.
    from shapely.ops import nearest_points

    def find_nearest_segment_and_endpoints(stop_pt, roads_union):
        """
        Returns:
           best_line:  the single linestring from roads that is nearest
           coordsA, coordsB: the line endpoints
        """
        # 1) nearest_points on the big merged geometry
        #    (like you did for snapping).
        nearest_on_line = roads_union.interpolate(roads_union.project(stop_pt))
        # 2) But we also need to figure out which actual linestring
        #    in the multi-geometry that point is on.
        #    Easiest approach: iterate all lines in roads_gdf, find min distance.
        #    For large datasets, that's not the fastest, but it's straightforward.

        min_dist = float("inf")
        best_line = None
        for row in roads_gdf.itertuples():
            geom = row.geometry
            # If it's multi, break it down
            if geom.geom_type == "MultiLineString":
                these_lines = geom.geoms
            else:
                these_lines = [geom]

            for single_line in these_lines:
                dist = single_line.distance(stop_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_line = single_line

        if best_line is not None:
            coords = list(best_line.coords)
            # We'll just treat the entire line as a set of consecutive segments
            # Because a linestring can have multiple vertices. The stop might
            # lie in the middle of e.g. the third sub-segment. We only need
            # the bounding endpoints for the sub-segment on which the stop falls.
            #
            # Let's find the "parameter" along best_line
            param_on_line = best_line.project(stop_pt)
            # We'll walk along the line coords to see which sub-segment it belongs to.
            cumulative = 0.0
            for i in range(len(coords) - 1):
                sub_line = LineString([coords[i], coords[i+1]])
                seg_len  = sub_line.length
                if cumulative + seg_len >= param_on_line:
                    # The stop is on this sub_line
                    return (best_line, coords[i], coords[i+1])
                else:
                    cumulative += seg_len

            # If we get here, it's presumably on the last segment
            return (best_line, coords[-2], coords[-1])

        return (None, None, None)

    # We'll also keep a union of roads for nearest_points:
    roads_union = roads_gdf.unary_union  # merged geometry

    # Add each snapped stop as a node if missing
    for idx, row in snapped_stops_gdf.iterrows():
        stop_pt = row.geometry
        stop_node = (stop_pt.x, stop_pt.y)
        if stop_node not in G:
            G.add_node(stop_node)

        # Find the nearest line and its endpoints
        best_line, A, B = find_nearest_segment_and_endpoints(stop_pt, roads_union)
        if not best_line or not A or not B:
            # fallback: no line found?
            # This would be unusual if everything is correct, but let's handle gracefully
            continue

        # Weight from stop to each endpoint
        distA = stop_pt.distance(Point(A))
        distB = stop_pt.distance(Point(B))

        # Determine oneway logic for that line
        # If your roads_gdf had multiple rows that share best_line geometry,
        # you'd want to fetch the row that actually matched. We'll do a simpler approach:
        # assume oneway is the same for the entire line. You might want a function
        # that truly fetches the correct row's "ONEWAY".
        # For demonstration, we'll just do a naive search:
        oneway_value = 'N'
        for rrow in roads_gdf.itertuples():
            # If the geometry is the same object, or "close enough"
            # a robust approach might be geometry.equals()
            if rrow.geometry.equals(best_line):
                oneway_value = str(getattr(rrow, oneway_col, 'N')).upper()
                break

        # If oneway='Y', we only connect A->B. So if the sub-segment is A->B,
        # does that mean the bus can go from A to B but not B to A?
        # We'll create edges in the direction of sub-line, plus the opposite direction if 'N'.
        sub_line = LineString([A, B])
        # check if param(B) > param(A)
        # We'll do a minimal approach: If 'Y', connect STOP->B if travel is A->B, etc.

        # 2A. If the sub_line is oriented from A->B, let's assume the bus travels that way:
        # Actually, the simpler approach is: we add edges from the STOP to each endpoint
        # and from each endpoint to the STOP, but only if it doesn't violate the direction.

        if oneway_value == 'Y':
            # We have to figure out which direction is the "forward" direction
            # for the entire linestring. In practice, many real road networks
            # store direction in the attribute (like from-node, to-node).
            # For demonstration, let's assume that if the line is from A->B,
            # the "forward" direction is A->B, so:
            #   - allow STOP -> B
            #   - allow A -> STOP
            # but not STOP -> A or B -> STOP
            # Because that would mean going backward along the line.
            #
            # If your data has a separate "from_x, from_y, to_x, to_y", you'd match them.
            # Or you might not strictly need this if your roads are mostly two-way (N).
            # Below is an example logic:

            # distance from A to B
            AB_len = sub_line.length
            # distance from A to STOP
            A_stop_len = Point(A).distance(stop_pt)
            # distance from STOP to B
            stop_B_len = stop_pt.distance(Point(B))

            # If A_stop_len + stop_B_len is approximately AB_len, we interpret that
            # the stop is actually on the segment from A->B (and not an extension).
            # Then do "forward edges" only
            G.add_edge(A, stop_node, weight=distA)          # A -> stop
            G.add_edge(stop_node, B, weight=distB)          # stop -> B

        else:
            # 'N' => two-way
            # So we add edges in both directions to A and B
            G.add_edge(stop_node, A, weight=distA)
            G.add_edge(A, stop_node, weight=distA)
            G.add_edge(stop_node, B, weight=distB)
            G.add_edge(B, stop_node, weight=distB)

    return G

# ---------------------------------------------------------------------
# Now we create the graph:
print("Building directed network (with mid-segment stop connections)...")
road_network = build_directed_network_with_stops(
    roads_gdf=filtered_roads,
    snapped_stops_gdf=snapped_stops_gdf,
    oneway_col="ONEWAY"  # or whatever your oneway column is
)
print(f"Network built: {road_network.number_of_nodes()} nodes, {road_network.number_of_edges()} edges.")

# ---------------------------------------------------------------------
# Then do your nearest-node logic as before:
def find_nearest_node(pt, graph_nodes):
    min_dist = float("inf")
    nearest = None
    for node in graph_nodes:
        dist = Point(node).distance(pt)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

sorted_snapped_stops = snapped_stops_gdf.sort_values("stop_sequence").reset_index(drop=True)
stop_points = sorted_snapped_stops.geometry.tolist()
nodes_list  = list(road_network.nodes())

individual_segments = []
individual_lengths = []

for i in range(len(stop_points) - 1):
    pt_start = stop_points[i]
    pt_end   = stop_points[i+1]

    node_start = find_nearest_node(pt_start, nodes_list)
    node_end   = find_nearest_node(pt_end, nodes_list)

    try:
        node_path = nx.shortest_path(road_network, source=node_start, target=node_end, weight="weight")
        segment_line = LineString(node_path)
        individual_segments.append(segment_line)
        individual_lengths.append(segment_line.length)
        print(f"Segment {i+1}: length = {segment_line.length:.2f}")
    except nx.NetworkXNoPath:
        print(f"No path found between stop {i} and {i+1}. Skipping segment.")

# ---------------------------------------------------------------------
# Finally, export segments
import os
import geopandas as gpd

segments_gdf = gpd.GeoDataFrame({
    "segment_id": list(range(1, len(individual_segments)+1)),
    "length": individual_lengths
}, geometry=individual_segments, crs=PROJECTED_CRS)

segments_out = os.path.join(OUTPUT_FOLDER, f"shortest_path_segments_trip_{selected_trip_id}.shp")

if len(segments_gdf) > 0:
    segments_gdf.to_file(segments_out)
    print(f"Exported {len(segments_gdf)} shortest path segments to {segments_out}")

    # Merge them
    merged_line = linemerge(individual_segments)
    if merged_line.is_empty or merged_line.geom_type == "GeometryCollection":
        print("Merged line is empty/GeometryCollection; skipping shapefile export.")
    else:
        merged_gdf = gpd.GeoDataFrame({"trip_id": [selected_trip_id]}, geometry=[merged_line], crs=PROJECTED_CRS)
        merged_out = os.path.join(OUTPUT_FOLDER, f"merged_shortest_path_trip_{selected_trip_id}.shp")
        merged_gdf.to_file(merged_out)
        print(f"Exported merged shortest path to {merged_out}")
else:
    print("No shortest path segments were created; skipping export.")
