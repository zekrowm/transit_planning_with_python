"""
Script that takes bus network, road network, and GTFS data and outputs a route shapefile that
matches the network and is segmented at the stops. It is very useful for checking stop spacing.
"""

import os

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points, linemerge

# -----------------------------------------------------------------------------
# STEP 0: CONFIGURATION (DO NOT MODIFY)
# -----------------------------------------------------------------------------
GTFS_FOLDER = r'C:\Your\Folder\Path\For\GTFS'
ROAD_NETWORK_FILE = r'C:\Your\File\Path\For\road_network.shp'
ROUTE_SHAPE_FILE = r'C:\Your\File\Path\For\bus_system_network.shp'

OUTPUT_FOLDER = r'C:\Your\Folder\Path\To\Output'
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
# STEP 1: Load GTFS files and filter trips
# =============================================================================
def load_gtfs(gtfs_folder):
    """Loads the GTFS files from the specified folder."""
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))
    trips = pd.read_csv(os.path.join(gtfs_folder, "trips.txt"))
    calendar = pd.read_csv(os.path.join(gtfs_folder, "calendar.txt"))
    return stops, stop_times, trips, calendar


def filter_and_select_trip(trips_df, stop_times_df, calendar_df,
                           filter_route, filter_direction, filter_departure, filter_calendar):
    """Filters the GTFS data based on the provided criteria and selects a trip."""
    print("Unique route_id in trips:", trips_df["route_id"].unique())
    print("Unique direction_id in trips:", trips_df["direction_id"].unique())
    print("Unique service_id in trips:", trips_df["service_id"].unique())

    try:
        service_filter = int(filter_calendar)
    except ValueError:
        service_filter = filter_calendar
    calendar_filtered = calendar_df[calendar_df["service_id"] == service_filter]
    valid_service_ids = calendar_filtered["service_id"].unique()
    print(f"Calendar filter applied: found {len(valid_service_ids)} valid service_id(s): {valid_service_ids}")

    # Ensure columns are strings
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    trips_df["direction_id"] = trips_df["direction_id"].astype(str)
    trips_df["service_id"] = trips_df["service_id"].astype(str)

    filter_route_str = str(filter_route)
    filter_direction_str = str(filter_direction)
    filter_calendar_str = str(filter_calendar)

    trips_filtered = trips_df[
        (trips_df["route_id"] == filter_route_str) &
        (trips_df["direction_id"] == filter_direction_str) &
        (trips_df["service_id"].isin([filter_calendar_str]))
    ]
    print(f"Trips filtered by route, direction, calendar: {len(trips_filtered)} trips found.")

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

    merged_filtered = merged[merged["departure_time_wo_sec"] >= filter_departure]
    print(f"After filtering by departure time >= {filter_departure}, {len(merged_filtered)} records remain.")

    if merged_filtered.empty:
        print("WARNING: No trips found matching the specified criteria!")
        return None
    else:
        selected_trip_id = merged_filtered["trip_id"].iloc[0]
        print(f"Selected trip: {selected_trip_id}")
        return selected_trip_id


# =============================================================================
# STEP 2: Create unsnapped stops and chord segments for the selected trip
# =============================================================================
def create_unsnapped_stops(gtfs_folder, selected_trip_id, projected_crs):
    """Creates a GeoDataFrame of unsnapped stops for the selected trip."""
    stops_file = os.path.join(gtfs_folder, "stops.txt")
    stop_times_file = os.path.join(gtfs_folder, "stop_times.txt")
    stops_df = pd.read_csv(stops_file)
    stop_times_df = pd.read_csv(stop_times_file)

    trip_stops_df = pd.merge(stop_times_df, stops_df, on="stop_id", how="left")
    trip_stops_selected = trip_stops_df[trip_stops_df["trip_id"] == selected_trip_id].copy()
    print(f"Number of stops for trip {selected_trip_id}: {len(trip_stops_selected)}")
    if trip_stops_selected.empty:
        raise ValueError(f"No stops found for trip {selected_trip_id}.")

    trip_stops_selected["stop_sequence"] = trip_stops_selected["stop_sequence"].astype(int)
    trip_stops_selected.sort_values("stop_sequence", inplace=True)
    print("Sample selected stops (trip_id, stop_id, stop_name, stop_sequence, stop_lat, stop_lon):")
    print(trip_stops_selected[["trip_id", "stop_id", "stop_name", "stop_sequence", "stop_lat", "stop_lon"]].head(10))

    unsnapped_stops_gdf = gpd.GeoDataFrame(
        trip_stops_selected,
        geometry=gpd.points_from_xy(trip_stops_selected["stop_lon"], trip_stops_selected["stop_lat"]),
        crs="EPSG:4326"
    )
    unsnapped_stops_gdf = unsnapped_stops_gdf.to_crs(projected_crs)
    print(f"unsnapped_stops_gdf shape: {unsnapped_stops_gdf.shape}")
    print("Sample geometry from unsnapped stops:", unsnapped_stops_gdf.geometry.head(2))
    return unsnapped_stops_gdf


def export_unsnapped_stops(unsnapped_stops_gdf, output_folder, selected_trip_id):
    """Exports the unsnapped stops GeoDataFrame to a shapefile."""
    unsnapped_stops_out = os.path.join(output_folder, f"unsnapped_stops_trip_{selected_trip_id}.shp")
    unsnapped_stops_gdf.to_file(unsnapped_stops_out)
    print(f"Exported unsnapped stops to {unsnapped_stops_out}")


def create_chord_segments_from_stops(stops_gdf):
    """Creates chord segments (lines connecting consecutive stops) from the stops GeoDataFrame."""
    sorted_stops = stops_gdf.sort_values("stop_sequence").reset_index(drop=True)
    segments = []
    for i in range(len(sorted_stops) - 1):
        start_stop = sorted_stops.iloc[i]
        end_stop   = sorted_stops.iloc[i + 1]
        segment_line = LineString([start_stop.geometry, end_stop.geometry])
        segments.append({
            "segment_id": i + 1,
            "start_stop_id": start_stop["stop_id"],
            "start_stop_name": start_stop.get("stop_name", None),
            "start_seq": start_stop["stop_sequence"],
            "end_stop_id": end_stop["stop_id"],
            "end_stop_name": end_stop.get("stop_name", None),
            "end_seq": end_stop["stop_sequence"],
            "length": segment_line.length,
            "geometry": segment_line
        })
    return gpd.GeoDataFrame(segments, crs=stops_gdf.crs)


def export_chord_segments(chord_segments_gdf, output_folder, selected_trip_id, prefix="chord_segments"):
    """Exports chord segments to a shapefile with a given prefix."""
    out_path = os.path.join(output_folder, f"{prefix}_trip_{selected_trip_id}.shp")
    chord_segments_gdf.to_file(out_path)
    print(f"Exported {prefix.replace('_',' ')} to {out_path}")


# =============================================================================
# STEP 3: Process bus route and snap stops to roads
# =============================================================================
def process_bus_route(route_shape_file, filter_route, projected_crs, output_folder, road_network_file):
    """Processes the bus route shapefile, creates a buffer, and filters the road network."""
    # Load and reproject the bus route shapefile
    route_shapes_gdf = gpd.read_file(route_shape_file)
    route_shapes_gdf = route_shapes_gdf.to_crs(projected_crs)
    route_shape_selected = route_shapes_gdf[route_shapes_gdf["ROUTE_NUMB"] == filter_route]
    if route_shape_selected.empty:
        raise ValueError(f"No route shape found with ROUTE_NUMB = {filter_route}")
    bus_route_feature = route_shape_selected.iloc[0]

    # Export bus route
    bus_route_out = os.path.join(output_folder, f"bus_route_{filter_route}.shp")
    gpd.GeoDataFrame([bus_route_feature], crs=route_shapes_gdf.crs).to_file(bus_route_out)
    print(f"Exported bus route to {bus_route_out}")

    # Buffer the route by 25 feet and export
    bus_route_proj = gpd.GeoSeries([bus_route_feature.geometry], crs=route_shapes_gdf.crs)
    bus_route_buffer = bus_route_proj.buffer(25)
    bus_route_buffer_out = os.path.join(output_folder, f"bus_route_{filter_route}_buffer.shp")
    gpd.GeoDataFrame(geometry=[bus_route_buffer.unary_union], crs=projected_crs).to_file(bus_route_buffer_out)
    print("Created and exported bus route buffer (25 feet).")

    # Load and reproject the road network
    roads_gdf = gpd.read_file(road_network_file)
    roads_gdf = roads_gdf.to_crs(projected_crs)
    buffer_geom = bus_route_buffer.unary_union

    # Filter roads that are completely within the buffer
    filtered_roads = roads_gdf[roads_gdf.within(buffer_geom)]
    print(f"Filtered roads: {len(filtered_roads)} segments within buffer.")
    filtered_roads_out = os.path.join(output_folder, "roads_within_buffer.shp")
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
    return bus_route_feature, roads_for_snapping, filtered_roads


def snap_stops(unsnapped_stops_gdf, roads_for_snapping, output_folder, selected_trip_id):
    """Snaps unsnapped stops to the road network geometry."""
    def snap_point_to_roads(pt, roads_geom):
        try:
            distance_along = roads_geom.project(pt)
            return roads_geom.interpolate(distance_along)
        except Exception as e:
            print("Project/Interpolate failed, using nearest_points. Error:", e)
            _, snapped_pt = nearest_points(pt, roads_geom)
            return snapped_pt

    snapped_geoms = unsnapped_stops_gdf.geometry.apply(lambda pt: snap_point_to_roads(pt, roads_for_snapping))
    snapped_stops_gdf = unsnapped_stops_gdf.copy()
    snapped_stops_gdf["geometry"] = snapped_geoms

    snapped_stops_out = os.path.join(output_folder, f"snapped_stops_trip_{selected_trip_id}.shp")
    snapped_stops_gdf.to_file(snapped_stops_out)
    print(f"Exported snapped stops to {snapped_stops_out}")
    return snapped_stops_gdf


def merge_and_export_snapped_chord(snapped_chord_segments_gdf, output_folder, selected_trip_id, projected_crs):
    """Merges snapped chord segments into a single feature and exports it."""
    merged_snapped_line = linemerge([geom for geom in snapped_chord_segments_gdf.geometry])
    merged_snapped_gdf = gpd.GeoDataFrame({"trip_id": [selected_trip_id]}, geometry=[merged_snapped_line], crs=projected_crs)
    merged_out = os.path.join(output_folder, f"merged_snapped_chord_trip_{selected_trip_id}.shp")
    merged_snapped_gdf.to_file(merged_out)
    print(f"Exported merged snapped chord to {merged_out}")


# =============================================================================
# STEP 4: Build a directed network and compute shortest paths
# =============================================================================
def build_directed_network_with_stops(roads_gdf, snapped_stops_gdf, oneway_col="ONEWAY"):
    """
    Builds a directed graph from roads_gdf, respecting ONEWAY='Y' or 'N',
    and connects each snapped stop (mid-segment) to the relevant segment endpoints.
    """
    G = nx.DiGraph()

    # 1. Add edges for each road segment
    for idx, row in roads_gdf.iterrows():
        oneway_value = str(row.get(oneway_col, "N")).upper()
        geom = row.geometry

        # Handle both LineString and MultiLineString geometries
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                seg_line = LineString([start, end])
                length = seg_line.length

                if oneway_value == 'Y':
                    G.add_edge(start, end, weight=length, geometry=seg_line)
                else:
                    G.add_edge(start, end, weight=length, geometry=seg_line)
                    G.add_edge(end, start, weight=length, geometry=seg_line)

    # 2. Connect each snapped stop to the nearest road segment endpoints
    roads_union = roads_gdf.unary_union  # merged geometry

    def find_nearest_segment_and_endpoints(stop_pt, roads_union):
        """
        Returns the nearest linestring from roads and the endpoints of the sub-segment
        where the stop lies.
        """
        nearest_on_line = roads_union.interpolate(roads_union.project(stop_pt))
        min_dist = float("inf")
        best_line = None
        for row in roads_gdf.itertuples():
            geom = row.geometry
            these_lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for single_line in these_lines:
                dist = single_line.distance(stop_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_line = single_line
        if best_line is not None:
            coords = list(best_line.coords)
            param_on_line = best_line.project(stop_pt)
            cumulative = 0.0
            for i in range(len(coords) - 1):
                sub_line = LineString([coords[i], coords[i+1]])
                seg_len = sub_line.length
                if cumulative + seg_len >= param_on_line:
                    return best_line, coords[i], coords[i+1]
                else:
                    cumulative += seg_len
            return best_line, coords[-2], coords[-1]
        return None, None, None

    for idx, row in snapped_stops_gdf.iterrows():
        stop_pt = row.geometry
        stop_node = (stop_pt.x, stop_pt.y)
        if stop_node not in G:
            G.add_node(stop_node)

        best_line, A, B = find_nearest_segment_and_endpoints(stop_pt, roads_union)
        if not best_line or not A or not B:
            continue

        distA = stop_pt.distance(Point(A))
        distB = stop_pt.distance(Point(B))

        # Determine oneway logic (naively)
        oneway_value = 'N'
        for rrow in roads_gdf.itertuples():
            if rrow.geometry.equals(best_line):
                oneway_value = str(getattr(rrow, oneway_col, 'N')).upper()
                break

        if oneway_value == 'Y':
            G.add_edge(A, stop_node, weight=distA)
            G.add_edge(stop_node, B, weight=distB)
        else:
            G.add_edge(stop_node, A, weight=distA)
            G.add_edge(A, stop_node, weight=distA)
            G.add_edge(stop_node, B, weight=distB)
            G.add_edge(B, stop_node, weight=distB)

    return G


def compute_and_export_shortest_paths(road_network, snapped_stops_gdf, output_folder, selected_trip_id, projected_crs):
    """Computes shortest paths between consecutive snapped stops and exports the segments."""
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
    nodes_list = list(road_network.nodes())

    individual_segments = []
    individual_lengths = []

    for i in range(len(stop_points) - 1):
        pt_start = stop_points[i]
        pt_end = stop_points[i + 1]

        node_start = find_nearest_node(pt_start, nodes_list)
        node_end = find_nearest_node(pt_end, nodes_list)

        try:
            node_path = nx.shortest_path(road_network, source=node_start, target=node_end, weight="weight")
            segment_line = LineString(node_path)
            individual_segments.append(segment_line)
            individual_lengths.append(segment_line.length)
            print(f"Segment {i+1}: length = {segment_line.length:.2f}")
        except nx.NetworkXNoPath:
            print(f"No path found between stop {i} and {i+1}. Skipping segment.")

    segments_gdf = gpd.GeoDataFrame({
        "segment_id": list(range(1, len(individual_segments) + 1)),
        "length": individual_lengths
    }, geometry=individual_segments, crs=projected_crs)

    segments_out = os.path.join(output_folder, f"shortest_path_segments_trip_{selected_trip_id}.shp")
    if len(segments_gdf) > 0:
        segments_gdf.to_file(segments_out)
        print(f"Exported {len(segments_gdf)} shortest path segments to {segments_out}")

        merged_line = linemerge(individual_segments)
        if merged_line.is_empty or merged_line.geom_type == "GeometryCollection":
            print("Merged line is empty/GeometryCollection; skipping shapefile export.")
        else:
            merged_gdf = gpd.GeoDataFrame({"trip_id": [selected_trip_id]},
                                          geometry=[merged_line],
                                          crs=projected_crs)
            merged_out = os.path.join(output_folder, f"merged_shortest_path_trip_{selected_trip_id}.shp")
            merged_gdf.to_file(merged_out)
            print(f"Exported merged shortest path to {merged_out}")
    else:
        print("No shortest path segments were created; skipping export.")


# =============================================================================
# MAIN FUNCTION: Orchestrate processing steps
# =============================================================================
def main():
    """Generates a route shapefile from GTFS, road network, and bus route data, segmented at stops."""
    # Ensure output folder is absolute and exists
    output_folder = os.path.abspath(OUTPUT_FOLDER)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 1: Load GTFS files and filter/select a trip
    stops_df, stop_times_df, trips_df, calendar_df = load_gtfs(GTFS_FOLDER)
    selected_trip_id = filter_and_select_trip(
        trips_df, stop_times_df, calendar_df,
        FILTER_ROUTE, FILTER_DIRECTION, FILTER_DEPARTURE, FILTER_CALENDAR
    )
    if selected_trip_id is None:
        raise ValueError("No trip selected from STEP 1; cannot proceed.")

    # Step 2: Create unsnapped stops and chord segments, then export them
    unsnapped_stops_gdf = create_unsnapped_stops(GTFS_FOLDER, selected_trip_id, PROJECTED_CRS)
    export_unsnapped_stops(unsnapped_stops_gdf, output_folder, selected_trip_id)
    chord_segments_gdf = create_chord_segments_from_stops(unsnapped_stops_gdf)
    export_chord_segments(chord_segments_gdf, output_folder, selected_trip_id, prefix="chord_segments")

    # Step 3: Process bus route and snap stops to the road network
    _, roads_for_snapping, filtered_roads = process_bus_route(
        ROUTE_SHAPE_FILE, FILTER_ROUTE, PROJECTED_CRS, output_folder, ROAD_NETWORK_FILE
    )
    snapped_stops_gdf = snap_stops(unsnapped_stops_gdf, roads_for_snapping, output_folder, selected_trip_id)
    snapped_chord_segments_gdf = create_chord_segments_from_stops(snapped_stops_gdf)
    export_chord_segments(snapped_chord_segments_gdf, output_folder, selected_trip_id, prefix="snapped_chord_segments")
    merge_and_export_snapped_chord(snapped_chord_segments_gdf, output_folder, selected_trip_id, PROJECTED_CRS)

    # Step 4: Build a directed network and compute shortest paths between stops
    road_network = build_directed_network_with_stops(filtered_roads, snapped_stops_gdf, oneway_col="ONEWAY")
    print(f"Network built: {road_network.number_of_nodes()} nodes, {road_network.number_of_edges()} edges.")
    compute_and_export_shortest_paths(road_network, snapped_stops_gdf, output_folder, selected_trip_id, PROJECTED_CRS)


if __name__ == "__main__":
    main()
