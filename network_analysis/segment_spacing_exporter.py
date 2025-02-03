
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script that loads GTFS files, filters trips, snaps stops to a road network,
and computes shortest path segments.

Usage:
    python snap_gtfs_stops.py

Author: Your Name
Date: YYYY-MM-DD
"""

import os

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    MultiPoint
)
from shapely.ops import nearest_points, linemerge, split


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

GTFS_FOLDER = r"C:\Your\Folder\Path\For\GTFS"
ROAD_NETWORK_FILE = r"C:\Your\File\Path\For\road_network.shp"
ROUTE_SHAPE_FILE = r"C:\Your\File\Path\For\bus_system_network.shp"

OUTPUT_FOLDER = r"C:\Your\Folder\Path\For\Output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Choose a projected CRS (units in feet or meters)
# e.g., EPSG:2263 for the DC area
PROJECTED_CRS = "EPSG:2263"  # Replace with CRS for your area

FILTER_ROUTE = "101"       # Replace with your bus route name
FILTER_DIRECTION = "0"     # Choose either "0" or "1"
FILTER_DEPARTURE = "08:00" # Replace with desired trip start time
FILTER_CALENDAR = "3"      # Replace with desired calendar code


def main():
    """
    Main workflow for filtering GTFS, snapping stops to roads, and
    computing shortest paths.
    """
    print("STEP 0: Filtering criteria set:")
    print(f"  Route:           {FILTER_ROUTE}")
    print(f"  Direction:       {FILTER_DIRECTION}")
    print(f"  Departure time:  {FILTER_DEPARTURE}")
    print(f"  Calendar:        {FILTER_CALENDAR}")

    # =============================================================================
    # STEP 1: Load GTFS files, filter trips, and select one trip
    # =============================================================================
    stops_df, stop_times_df, trips_df, calendar_df = load_gtfs(GTFS_FOLDER)
    print("Loaded GTFS files.")

    # Debug: print some unique values
    print("Unique route_id in trips:", trips_df["route_id"].unique())
    print("Unique direction_id in trips:", trips_df["direction_id"].unique())
    print("Unique service_id in trips:", trips_df["service_id"].unique())

    selected_trip_id = filter_and_select_trip(
        stops_df,
        stop_times_df,
        trips_df,
        calendar_df,
        FILTER_CALENDAR,
        FILTER_ROUTE,
        FILTER_DIRECTION,
        FILTER_DEPARTURE
    )

    # =============================================================================
    # STEP 2: Create unsnapped stops and chord segments for the selected trip
    # =============================================================================
    unsnapped_stops_gdf, chord_segments_gdf = create_unsnapped(
        GTFS_FOLDER,
        OUTPUT_FOLDER,
        selected_trip_id,
        PROJECTED_CRS
    )

    # =============================================================================
    # STEP 3: Snap stops to roads, filter road segments, etc.
    # =============================================================================
    bus_route_feature = filter_route_shape(
        ROUTE_SHAPE_FILE,
        FILTER_ROUTE,
        OUTPUT_FOLDER,
        PROJECTED_CRS
    )

    # Buffer the route by 25 feet
    create_route_buffer(bus_route_feature, FILTER_ROUTE, OUTPUT_FOLDER)

    filtered_roads = filter_roads_by_buffer(
        ROAD_NETWORK_FILE,
        OUTPUT_FOLDER,
        bus_route_feature,
        buffer_distance=25,
        crs=PROJECTED_CRS
    )

    snapped_stops_gdf, snapped_chord_segments_gdf = snap_stops_create_chords(
        unsnapped_stops_gdf,
        filtered_roads,
        OUTPUT_FOLDER,
        selected_trip_id,
        PROJECTED_CRS
    )

    # =============================================================================
    # STEP 4: Build a directed network that includes mid-segment connections
    #         for each snapped stop, then compute shortest paths
    # =============================================================================
    road_network = build_directed_network_with_stops(
        filtered_roads,
        snapped_stops_gdf,
        oneway_col="ONEWAY"
    )

    compute_and_export_shortest_paths(
        road_network,
        snapped_stops_gdf,
        OUTPUT_FOLDER,
        selected_trip_id,
        PROJECTED_CRS
    )


def load_gtfs(gtfs_folder):
    """
    Loads the GTFS files from the specified folder and returns
    stops, stop_times, trips, and calendar as DataFrames.
    """
    stops_path = os.path.join(gtfs_folder, "stops.txt")
    stop_times_path = os.path.join(gtfs_folder, "stop_times.txt")
    trips_path = os.path.join(gtfs_folder, "trips.txt")
    calendar_path = os.path.join(gtfs_folder, "calendar.txt")

    stops = pd.read_csv(stops_path)
    stop_times = pd.read_csv(stop_times_path)
    trips = pd.read_csv(trips_path)
    calendar = pd.read_csv(calendar_path)

    return stops, stop_times, trips, calendar


def filter_and_select_trip(
    stops_df,
    stop_times_df,
    trips_df,
    calendar_df,
    filter_calendar,
    filter_route,
    filter_direction,
    filter_departure
):
    """
    Filter trips based on route, direction, calendar, and departure time.
    Returns the trip_id of the first trip that meets the criteria.
    """
    # Filter calendar
    try:
        service_filter = int(filter_calendar)
    except ValueError:
        service_filter = filter_calendar

    calendar_filtered = calendar_df[calendar_df["service_id"] == service_filter]
    valid_service_ids = calendar_filtered["service_id"].unique()
    print(
        f"Calendar filter applied: found {len(valid_service_ids)} "
        f"valid service_id(s): {valid_service_ids}"
    )

    # Convert columns to str for safe comparison
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    trips_df["direction_id"] = trips_df["direction_id"].astype(str)
    trips_df["service_id"] = trips_df["service_id"].astype(str)

    filter_route_str = str(filter_route)
    filter_direction_str = str(filter_direction)
    filter_calendar_str = str(filter_calendar)

    trips_filtered = trips_df[
        (trips_df["route_id"] == filter_route_str)
        & (trips_df["direction_id"] == filter_direction_str)
        & (trips_df["service_id"].isin([filter_calendar_str]))
    ]
    print(
        "Trips filtered by route, direction, calendar: "
        f"{len(trips_filtered)} trips found."
    )

    merged = pd.merge(stop_times_df, trips_filtered, on="trip_id")
    print(
        f"After merging stop_times with filtered trips, {len(merged)} records found."
    )
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
    print(
        f"After filtering by departure time >= {filter_departure}, "
        f"{len(merged_filtered)} records remain."
    )

    if merged_filtered.empty:
        print("WARNING: No trips found matching the specified criteria!")
        selected_trip_id = None
    else:
        selected_trip_id = merged_filtered["trip_id"].iloc[0]
        print(f"Selected trip: {selected_trip_id}")

    if selected_trip_id is None:
        raise ValueError(
            "No trip selected from filtering criteria; cannot proceed."
        )

    return selected_trip_id


def create_unsnapped(
    gtfs_folder,
    output_folder,
    selected_trip_id,
    projected_crs
):
    """
    Creates a GeoDataFrame of unsnapped stops, exports them,
    and creates chord segments between consecutive stops.
    """
    # Make sure the output folder is absolute
    abs_output_folder = os.path.abspath(output_folder)
    if not os.path.exists(abs_output_folder):
        os.makedirs(abs_output_folder)

    stops_file = os.path.join(gtfs_folder, "stops.txt")
    stop_times_file = os.path.join(gtfs_folder, "stop_times.txt")

    stops_df = pd.read_csv(stops_file)
    stop_times_df = pd.read_csv(stop_times_file)

    trip_stops_df = pd.merge(
        stop_times_df, stops_df, on="stop_id", how="left"
    )
    trip_stops_selected = trip_stops_df[
        trip_stops_df["trip_id"] == selected_trip_id
    ].copy()

    print(
        f"Number of stops for trip {selected_trip_id}: "
        f"{len(trip_stops_selected)}"
    )
    if trip_stops_selected.empty:
        raise ValueError(f"No stops found for trip {selected_trip_id}.")

    trip_stops_selected["stop_sequence"] = \
        trip_stops_selected["stop_sequence"].astype(int)
    trip_stops_selected.sort_values("stop_sequence", inplace=True)

    print(
        "Sample selected stops "
        "(trip_id, stop_id, stop_name, stop_sequence, stop_lat, stop_lon):"
    )
    print(
        trip_stops_selected[
            [
                "trip_id", "stop_id", "stop_name",
                "stop_sequence", "stop_lat", "stop_lon"
            ]
        ].head(10)
    )

    unsnapped_stops_gdf = gpd.GeoDataFrame(
        trip_stops_selected,
        geometry=gpd.points_from_xy(
            trip_stops_selected["stop_lon"],
            trip_stops_selected["stop_lat"]
        ),
        crs="EPSG:4326"
    )
    unsnapped_stops_gdf = unsnapped_stops_gdf.to_crs(projected_crs)

    print(f"unsnapped_stops_gdf shape: {unsnapped_stops_gdf.shape}")
    print(
        "Sample geometry from unsnapped stops:",
        unsnapped_stops_gdf.geometry.head(2)
    )

    unsnapped_stops_out = os.path.join(
        abs_output_folder,
        f"unsnapped_stops_trip_{selected_trip_id}.shp"
    )
    unsnapped_stops_gdf.to_file(unsnapped_stops_out)
    print(f"Exported unsnapped stops to {unsnapped_stops_out}")

    chord_segments_gdf = create_chord_segments(
        unsnapped_stops_gdf,
        abs_output_folder,
        selected_trip_id,
        prefix="chord_segments"
    )

    return unsnapped_stops_gdf, chord_segments_gdf


def create_chord_segments(stops_gdf, output_folder, selected_trip_id, prefix):
    """
    Creates chord segments between consecutive stops, exports them to file,
    and returns a GeoDataFrame of these segments.
    """
    sorted_stops = stops_gdf.sort_values("stop_sequence").reset_index(drop=True)
    chord_segments = []

    for i in range(len(sorted_stops) - 1):
        start_stop = sorted_stops.iloc[i]
        end_stop = sorted_stops.iloc[i + 1]

        segment_line = LineString(
            [start_stop.geometry, end_stop.geometry]
        )
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

    chord_segments_gdf = gpd.GeoDataFrame(
        chord_segments, crs=stops_gdf.crs
    )
    chord_segments_out = os.path.join(
        output_folder,
        f"{prefix}_trip_{selected_trip_id}.shp"
    )
    chord_segments_gdf.to_file(chord_segments_out)
    print(f"Exported chord segments to {chord_segments_out}")

    return chord_segments_gdf


def filter_route_shape(
    route_shape_file,
    filter_route,
    output_folder,
    projected_crs
):
    """
    Reads the route shape file, filters by ROUTE_NUMB, exports
    the selected route, and returns the single selected feature.
    """
    route_shapes_gdf = gpd.read_file(route_shape_file)
    route_shapes_gdf = route_shapes_gdf.to_crs(projected_crs)

    route_shape_selected = route_shapes_gdf[
        route_shapes_gdf["ROUTE_NUMB"] == filter_route
    ]
    if route_shape_selected.empty:
        raise ValueError(
            f"No route shape found with ROUTE_NUMB = {filter_route}"
        )
    bus_route_feature = route_shape_selected.iloc[0]

    bus_route_out = os.path.join(
        output_folder,
        f"bus_route_{filter_route}.shp"
    )
    gpd.GeoDataFrame(
        [bus_route_feature],
        crs=route_shapes_gdf.crs
    ).to_file(bus_route_out)
    print(f"Exported bus route to {bus_route_out}")

    return bus_route_feature


def create_route_buffer(bus_route_feature, filter_route, output_folder):
    """
    Creates and exports a 25-foot buffer around the bus route.
    """
    bus_route_proj = gpd.GeoSeries(
        [bus_route_feature.geometry], crs=bus_route_feature.crs
    )
    bus_route_buffer = bus_route_proj.buffer(25)

    bus_route_buffer_out = os.path.join(
        output_folder, f"bus_route_{filter_route}_buffer.shp"
    )
    gpd.GeoDataFrame(
        geometry=[bus_route_buffer.unary_union],
        crs=bus_route_feature.crs
    ).to_file(bus_route_buffer_out)
    print("Created and exported bus route buffer (25 feet).")


def filter_roads_by_buffer(
    road_network_file,
    output_folder,
    bus_route_feature,
    buffer_distance,
    crs
):
    """
    Loads the road network, reprojects it, filters roads
    within the buffer around the bus_route_feature, exports them,
    and returns the filtered roads.
    """
    roads_gdf = gpd.read_file(road_network_file)
    roads_gdf = roads_gdf.to_crs(crs)

    bus_route_proj = gpd.GeoSeries(
        [bus_route_feature.geometry],
        crs=crs
    )
    buffer_geom = bus_route_proj.buffer(buffer_distance).unary_union

    filtered_roads = roads_gdf[roads_gdf.within(buffer_geom)]
    print(
        f"Filtered roads: {len(filtered_roads)} segments within buffer."
    )

    filtered_roads_out = os.path.join(
        output_folder, "roads_within_buffer.shp"
    )
    filtered_roads.to_file(filtered_roads_out)
    print(f"Exported filtered roads to {filtered_roads_out}")

    if filtered_roads.empty:
        raise ValueError("No roads found within the route buffer!")

    return filtered_roads


def snap_stops_create_chords(
    unsnapped_stops_gdf,
    filtered_roads,
    output_folder,
    selected_trip_id,
    projected_crs
):
    """
    Snap unsnapped stops to the filtered roads, export snapped stops,
    create chord segments, and return both GDFs.
    """
    all_roads_raw = filtered_roads.unary_union
    merged_roads = linemerge(all_roads_raw)
    if merged_roads.geom_type == "LineString":
        roads_for_snapping = merged_roads
    else:
        roads_for_snapping = all_roads_raw
    print(
        "Road geometry for snapping type:",
        roads_for_snapping.geom_type
    )

    snapped_geoms = unsnapped_stops_gdf.geometry.apply(
        lambda pt: snap_point_to_roads(pt, roads_for_snapping)
    )
    snapped_stops_gdf = unsnapped_stops_gdf.copy()
    snapped_stops_gdf["geometry"] = snapped_geoms

    snapped_stops_out = os.path.join(
        output_folder,
        f"snapped_stops_trip_{selected_trip_id}.shp"
    )
    snapped_stops_gdf.to_file(snapped_stops_out)
    print(f"Exported snapped stops to {snapped_stops_out}")

    # Create chord segments for snapped stops
    snapped_chord_segments_gdf = create_chord_segments(
        snapped_stops_gdf,
        output_folder,
        selected_trip_id,
        prefix="snapped_chord_segments"
    )

    # Merge them into one feature (optional)
    merged_snapped_line = linemerge(
        [
            seg["geometry"] for seg in snapped_chord_segments_gdf.to_dict(
                "records"
            )
        ]
    )
    merged_snapped_gdf = gpd.GeoDataFrame(
        {"trip_id": [selected_trip_id]},
        geometry=[merged_snapped_line],
        crs=projected_crs
    )
    merged_snapped_out = os.path.join(
        output_folder,
        f"merged_snapped_chord_trip_{selected_trip_id}.shp"
    )
    merged_snapped_gdf.to_file(merged_snapped_out)
    print(f"Exported merged snapped chord to {merged_snapped_out}")

    return snapped_stops_gdf, snapped_chord_segments_gdf


def snap_point_to_roads(pt, roads_geom):
    """
    Snap a point to the road geometry using .project() and .interpolate()
    as a first attempt; falls back to nearest_points if it fails.
    """
    try:
        distance_along = roads_geom.project(pt)
        return roads_geom.interpolate(distance_along)
    except Exception as exc:
        print("Project/Interpolate failed, using nearest_points. Error:", exc)
        _, snapped_pt = nearest_points(pt, roads_geom)
        return snapped_pt


def build_directed_network_with_stops(
    roads_gdf,
    snapped_stops_gdf,
    oneway_col="ONEWAY"
):
    """
    Builds a directed graph from roads_gdf, respecting ONEWAY='Y' or 'N',
    and connects each snapped stop (mid-segment) to the relevant segment endpoints.
    """
    graph = nx.DiGraph()

    # --- 1. Basic directed edges from road endpoints ---
    for _, row in roads_gdf.iterrows():
        oneway_value = str(row.get(oneway_col, "N")).upper()
        geom = row.geometry

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
                length = seg_line.length

                if oneway_value == "Y":
                    graph.add_edge(start, end, weight=length, geometry=seg_line)
                else:
                    graph.add_edge(start, end, weight=length, geometry=seg_line)
                    graph.add_edge(end, start, weight=length, geometry=seg_line)

    # --- 2. Insert each snapped stop as a node + edges to the nearest segment ---
    roads_union = roads_gdf.unary_union

    for _, row in snapped_stops_gdf.iterrows():
        stop_pt = row.geometry
        stop_node = (stop_pt.x, stop_pt.y)
        if stop_node not in graph:
            graph.add_node(stop_node)

        best_line, coord_a, coord_b = find_nearest_segment_and_endpoints(
            stop_pt,
            roads_gdf,
            roads_union
        )
        if not best_line or not coord_a or not coord_b:
            # fallback: no line found?
            continue

        dist_a = stop_pt.distance(Point(coord_a))
        dist_b = stop_pt.distance(Point(coord_b))

        oneway_value = get_oneway_for_line(best_line, roads_gdf, oneway_col)
        if oneway_value == "Y":
            graph.add_edge(coord_a, stop_node, weight=dist_a)
            graph.add_edge(stop_node, coord_b, weight=dist_b)
        else:
            graph.add_edge(stop_node, coord_a, weight=dist_a)
            graph.add_edge(coord_a, stop_node, weight=dist_a)
            graph.add_edge(stop_node, coord_b, weight=dist_b)
            graph.add_edge(coord_b, stop_node, weight=dist_b)

    return graph


def find_nearest_segment_and_endpoints(stop_pt, roads_gdf, roads_union):
    """
    Returns a tuple of (best_line, coordA, coordB):
    - best_line: the single LineString from roads that is nearest
    - coordA, coordB: the line endpoints bounding the sub-segment
                      where the stop lies
    """
    # 1) Snap to roads_union
    nearest_on_line = roads_union.interpolate(roads_union.project(stop_pt))

    # 2) Iterate over roads to find the actual line with minimal distance
    min_dist = float("inf")
    best_line = None

    for row in roads_gdf.itertuples():
        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            lines = [geom]

        for single_line in lines:
            dist = single_line.distance(stop_pt)
            if dist < min_dist:
                min_dist = dist
                best_line = single_line

    if best_line is not None:
        coords = list(best_line.coords)
        param_on_line = best_line.project(stop_pt)
        cumulative = 0.0
        for i in range(len(coords) - 1):
            sub_line = LineString([coords[i], coords[i + 1]])
            seg_len = sub_line.length
            if cumulative + seg_len >= param_on_line:
                return best_line, coords[i], coords[i + 1]
            cumulative += seg_len
        # If it exceeds, presumably it's on the last segment
        return best_line, coords[-2], coords[-1]

    return None, None, None


def get_oneway_for_line(best_line, roads_gdf, oneway_col):
    """
    Tries to fetch the ONEWAY value from roads_gdf for the geometry
    matching best_line. Fallback: 'N'.
    """
    for row in roads_gdf.itertuples():
        geom = row.geometry
        if geom.equals(best_line):
            return str(getattr(row, oneway_col, "N")).upper()
        elif geom.geom_type == "MultiLineString":
            # If geometry is multi, check each sub-line
            for sub_line in geom.geoms:
                if sub_line.equals(best_line):
                    return str(getattr(row, oneway_col, "N")).upper()

    return "N"


def compute_and_export_shortest_paths(
    road_network,
    snapped_stops_gdf,
    output_folder,
    selected_trip_id,
    projected_crs
):
    """
    Given a DiGraph and snapped stops, compute the shortest path
    between consecutive stops, export them to shapefile.
    """
    sorted_snapped_stops = snapped_stops_gdf.sort_values("stop_sequence")
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
            node_path = nx.shortest_path(
                road_network,
                source=node_start,
                target=node_end,
                weight="weight"
            )
            segment_line = LineString(node_path)
            individual_segments.append(segment_line)
            individual_lengths.append(segment_line.length)
            print(f"Segment {i + 1}: length = {segment_line.length:.2f}")
        except nx.NetworkXNoPath:
            print(f"No path found between stop {i} and {i + 1}. Skipping.")

    segments_gdf = gpd.GeoDataFrame(
        {
            "segment_id": list(range(1, len(individual_segments) + 1)),
            "length": individual_lengths
        },
        geometry=individual_segments,
        crs=projected_crs
    )

    segments_out = os.path.join(
        output_folder, f"shortest_path_segments_trip_{selected_trip_id}.shp"
    )

    if len(segments_gdf) > 0:
        segments_gdf.to_file(segments_out)
        print(
            f"Exported {len(segments_gdf)} shortest path segments "
            f"to {segments_out}"
        )

        merged_line = linemerge(individual_segments)
        if (merged_line.is_empty
                or merged_line.geom_type == "GeometryCollection"):
            print("Merged line is empty/GeometryCollection; skipping export.")
        else:
            merged_gdf = gpd.GeoDataFrame(
                {"trip_id": [selected_trip_id]},
                geometry=[merged_line],
                crs=projected_crs
            )
            merged_out = os.path.join(
                output_folder,
                f"merged_shortest_path_trip_{selected_trip_id}.shp"
            )
            merged_gdf.to_file(merged_out)
            print(f"Exported merged shortest path to {merged_out}")
    else:
        print("No shortest path segments were created; skipping export.")


def find_nearest_node(pt, graph_nodes):
    """
    Finds the nearest node in graph_nodes to the given point pt.
    """
    min_dist = float("inf")
    nearest = None
    for node in graph_nodes:
        dist = Point(node).distance(pt)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


if __name__ == "__main__":
    main()
