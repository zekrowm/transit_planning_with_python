#!/usr/bin/env python
# coding: utf-8

"""
Park and Ride Jobs Residents Served Script
==========================================

Overview:
---------
This script evaluates the accessibility of Park and Ride facilities by analyzing the number of jobs
and residents served within specified time frames using public transit and road networks. It processes
GTFS data to build transit networks, identifies reachable stops, creates accessible areas around
facilities, and integrates census demographic data to estimate the population and employment
characteristics served.

Key Features:
-------------
- **Transit Network Construction**: Parses GTFS data to create a transit network graph, including transfer times.
- **Accessible Area Identification**: Determines reachable bus stops and generates accessible area polygons around each facility.
- **Demographic Integration**: Overlays accessible areas with census data to calculate the number of jobs and residents served.
- **Road Network Analysis**: Builds a road network to create isochrones representing areas reachable by car within a specified driving time.
- **Output Generation**: Produces shapefiles for accessible areas, reachable stops, and demographic data for visualization and further analysis.
"""

import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# ----------------- CONFIGURATION SECTION -----------------
GTFS_PATH = r'C:\Path\To\Your\GTFS_data'  # Replace with your folder path

# Define time windows (in seconds since midnight)
TIME_WINDOWS = [
    {'start': 6 * 3600, 'end': 9 * 3600},   # AM Peak: 6:00 AM to 9:00 AM
    {'start': 16 * 3600, 'end': 19 * 3600}, # PM Peak: 4:00 PM to 7:00 PM
]

# Define facility locations with correct decimal degrees
LOCATIONS = [
    {"name": "East Falls Church", "latitude": 38.899333, "longitude": -77.189972},
    {"name": "Monument Drive", "latitude": 38.8585, "longitude": -77.360944},
    # Add other locations with correct coordinates here
]

OUTPUT_DIR = r"C:\Path\To\Your\Output_folder"  # Replace with your folder path

TRANSFER_TIME_MINUTES = 45  # Imposes transfer time penalty
TRANSFER_TIME_SECONDS = TRANSFER_TIME_MINUTES * 60

# -- New configuration parameters --
MAX_TRANSIT_TIME_MINUTES = 45   # Adjust as needed for transit isochrones
MAX_DRIVING_TIME_MINUTES = 10   # Adjust as needed for driving isochrones
# Convert minutes to seconds
MAX_TRANSIT_TIME_SECONDS = MAX_TRANSIT_TIME_MINUTES * 60
MAX_DRIVING_TIME_SECONDS = MAX_DRIVING_TIME_MINUTES * 60
# -----------------------------------

CENSUS_BLOCKS_PATH = (
    r"C:\Path\To\Your\Census_Blocks.shp" # Replace with your file path
)
ACCESSIBLE_AREAS_DIR = (
    r"C:\Path\To\Your\Output_folder" # Replace with your folder path
)
OUTPUT_DIR_ANALYSIS = (
    r"C:\Path\To\Your\Output_folder" # Replace with your folder path
)

ROAD_SHP_PATH = (
    r"C:\Path\To\Your\Roadway_Centerlines.shp" # Replace with your file path
)
# Check build_road_network function to ensure speed limit and oneway field names are correct

# --------------- END CONFIGURATION SECTION ---------------


def parse_gtfs_time(time_str):
    """
    Parse a GTFS time string into total seconds since midnight.
    """
    hours, mins, secs = map(int, time_str.split(':'))
    return hours * 3600 + mins * 60 + secs


def in_time_windows(seconds):
    """
    Check if a time (in seconds) falls within any of the defined time windows.
    """
    return any(window['start'] <= seconds <= window['end'] for window in TIME_WINDOWS)


# Load GTFS data
STOPS = pd.read_csv(os.path.join(GTFS_PATH, 'stops.txt'))
STOP_TIMES = pd.read_csv(os.path.join(GTFS_PATH, 'stop_times.txt'))
TRIPS = pd.read_csv(os.path.join(GTFS_PATH, 'trips.txt'))
ROUTES = pd.read_csv(os.path.join(GTFS_PATH, 'routes.txt'))

STOP_TIMES['departure_seconds'] = STOP_TIMES['departure_time'].apply(parse_gtfs_time)
FILTERED_STOP_TIMES = STOP_TIMES[STOP_TIMES['departure_seconds'].apply(in_time_windows)]
FILTERED_TRIP_IDS = FILTERED_STOP_TIMES['trip_id'].unique()
FILTERED_TRIPS = TRIPS[TRIPS['trip_id'].isin(FILTERED_TRIP_IDS)]

STOPS['geometry'] = gpd.points_from_xy(STOPS['stop_lon'], STOPS['stop_lat'])
STOPS_GDF = gpd.GeoDataFrame(STOPS, geometry='geometry', crs='EPSG:4326')


FACILITIES_GDF = gpd.GeoDataFrame(
    LOCATIONS,
    geometry=gpd.points_from_xy(
        [loc['longitude'] for loc in LOCATIONS],
        [loc['latitude'] for loc in LOCATIONS]
    ),
    crs='EPSG:4326'
)


# pylint: disable=too-many-locals, redefined-outer-name, unused-argument
def create_transit_network_with_transfers(
    stop_times_df, trips_df, xfer_time_seconds=300
):
    """
    Create a transit network graph with transfer times.
    """
    stop_times_merged = stop_times_df.merge(
        trips_df[['trip_id', 'route_id']], on='trip_id', how='left'
    )
    graph_obj = nx.DiGraph()

    # Build edges based on sequential stops in a trip
    for _, trip_group in stop_times_merged.groupby('trip_id'):
        route_id = trip_group['route_id'].iloc[0]
        trip_group = trip_group.sort_values('stop_sequence')
        trip_group['arrival_seconds'] = trip_group['arrival_time'].apply(parse_gtfs_time)
        trip_group['departure_seconds'] = trip_group['departure_time'].apply(parse_gtfs_time)
        stops_list = trip_group[['stop_id', 'arrival_seconds',
                                 'departure_seconds']].values

        for i, _ in enumerate(stops_list[:-1]):
            start_stop_id, _, start_departure = stops_list[i]
            end_stop_id, end_arrival, _ = stops_list[i + 1]
            travel_time_local = end_arrival - start_departure
            if travel_time_local > 0:
                graph_obj.add_edge(
                    (start_stop_id, route_id),
                    (end_stop_id, route_id),
                    weight=travel_time_local
                )

    # Add transfer edges between different routes at the same stop
    stop_routes_df = stop_times_merged.groupby('stop_id')['route_id'].unique()
    for stop_id, routes_lst in stop_routes_df.items():
        for route1 in routes_lst:
            for route2 in routes_lst:
                if route1 != route2:
                    graph_obj.add_edge(
                        (stop_id, route1),
                        (stop_id, route2),
                        weight=xfer_time_seconds
                    )
    return graph_obj


def find_reachable_stops_with_times(graph_obj, origin_nodes, max_time_minutes=45):
    """
    Find reachable stops within a given time (minutes) and their travel times.
    """
    max_time_seconds = max_time_minutes * 60
    reachable_stops = {}
    for origin_node in origin_nodes:
        lengths_local = nx.single_source_dijkstra_path_length(
            graph_obj, origin_node, cutoff=max_time_seconds, weight='weight'
        )
        for node, length_val in lengths_local.items():
            if length_val <= max_time_seconds:
                stop_id_val, _ = node
                if (stop_id_val not in reachable_stops or
                        length_val < reachable_stops[stop_id_val]):
                    reachable_stops[stop_id_val] = length_val
    return reachable_stops


# pylint: disable=too-many-locals, unused-argument
def find_accessible_area_with_transfers(
    stops_gdf_local,
    graph_obj,
    facility_point_gdf,
    xfer_time_seconds,
    max_time_minutes=45
):
    """
    Return an accessible area polygon and reachable stops for a given facility.
    """
    # Distance from facility to consider initial walk to a stop
    buffer_distance_miles = 0.5
    buffer_distance_meters = buffer_distance_miles * 1609.34

    stops_proj = stops_gdf_local.to_crs(epsg=3857)
    facility_proj = facility_point_gdf.to_crs(epsg=3857)

    buffer_geom = facility_proj.geometry.buffer(buffer_distance_meters).iloc[0]
    nearby_stops = stops_proj[stops_proj.geometry.within(buffer_geom)]

    if nearby_stops.empty:
        print(f"No stops within {buffer_distance_miles} miles of "
              f"{facility_point_gdf['name'].iloc[0]}")
        return None, None

    nearby_stop_ids = nearby_stops['stop_id'].tolist()
    origin_nodes = []

    # Merge to find routes for each stop
    stop_routes_local = FILTERED_STOP_TIMES[
        FILTERED_STOP_TIMES['stop_id'].isin(nearby_stop_ids)
    ].merge(
        FILTERED_TRIPS[['trip_id', 'route_id']], on='trip_id', how='left'
    ).groupby('stop_id')['route_id'].unique()

    for stop_id_val, routes_lst in stop_routes_local.items():
        for route_val in routes_lst:
            origin_nodes.append((stop_id_val, route_val))

    if not origin_nodes:
        print(f"No routes found for stops near {facility_point_gdf['name'].iloc[0]}")
        return None, None

    # Use the maximum transit time from the config
    reachable_stop_times = find_reachable_stops_with_times(
        graph_obj, origin_nodes, max_time_minutes
    )

    if not reachable_stop_times:
        print(f"No stops reachable within {max_time_minutes} minutes from "
              f"{facility_point_gdf['name'].iloc[0]}")
        return None, None

    reachable_stops_df = pd.DataFrame({
        'stop_id': list(reachable_stop_times.keys()),
        'travel_time_seconds': list(reachable_stop_times.values())
    })
    reachable_stops_gdf = stops_gdf_local.merge(reachable_stops_df, on='stop_id')

    total_time_seconds = max_time_minutes * 60
    reachable_stops_gdf['remaining_time_seconds'] = (
        total_time_seconds - reachable_stops_gdf['travel_time_seconds']
    )

    # Walking speed in meters per second
    walking_speed_mps = 1.34112

    reachable_stops_gdf['walking_distance_meters'] = (
        reachable_stops_gdf['remaining_time_seconds'] * walking_speed_mps
    )

    # Cap walking distance at 0.25 miles (about 402 meters) or adjust as needed
    max_walking_dist = 0.25 * 1609.34
    reachable_stops_gdf['walking_distance_meters'] = (
        reachable_stops_gdf['walking_distance_meters'].clip(upper=max_walking_dist)
    )

    reachable_stops_proj = reachable_stops_gdf.to_crs(epsg=3857)
    buffers_local = reachable_stops_proj.geometry.buffer(
        reachable_stops_gdf['walking_distance_meters']
    )
    accessible_area_local = buffers_local.unary_union
    accessible_area_local = gpd.GeoSeries(
        [accessible_area_local], crs='EPSG:3857'
    ).to_crs(epsg=4326).iloc[0]

    reachable_stops_gdf = reachable_stops_proj.to_crs(epsg=4326)

    return accessible_area_local, reachable_stops_gdf


os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG, AX = plt.subplots(figsize=(10, 10))
STOPS_GDF.plot(ax=AX, color='blue', markersize=1, label='Bus Stops')
FACILITIES_GDF.plot(ax=AX, color='red', markersize=50, label='Facilities')
plt.legend()
plt.title('Bus Stops and Facility Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

FACILITIES_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'facilities.shp')
FACILITIES_GDF.to_file(FACILITIES_OUTPUT_PATH)
print(f"Saved all facility locations to {FACILITIES_OUTPUT_PATH}")

TRANSIT_GRAPH = create_transit_network_with_transfers(
    FILTERED_STOP_TIMES, FILTERED_TRIPS, TRANSFER_TIME_SECONDS
)

for _, facility in FACILITIES_GDF.iterrows():
    print(f"Processing: {facility['name']}")
    fac_gdf = gpd.GeoDataFrame(
        [facility], geometry=[facility.geometry], crs='EPSG:4326'
    )

    # Pass in the new config variable: MAX_TRANSIT_TIME_MINUTES
    accessible_area, reachable_stops_gdf = find_accessible_area_with_transfers(
        STOPS_GDF, TRANSIT_GRAPH, fac_gdf, TRANSFER_TIME_SECONDS, max_time_minutes=MAX_TRANSIT_TIME_MINUTES
    )

    if accessible_area is None or accessible_area.is_empty:
        print(f"No accessible area found for {facility['name']}")
        continue

    safe_name = re.sub(r'\W+', '_', facility['name'].lower())
    area_output_filename = os.path.join(
        OUTPUT_DIR, f"{safe_name}_accessible_area.shp"
    )

    accessible_area_gdf = gpd.GeoDataFrame(
        {'name': [facility['name']]},
        geometry=[accessible_area],
        crs='EPSG:4326'
    )
    accessible_area_gdf.to_file(area_output_filename)
    print(f"Saved accessible area to {area_output_filename}")

    stops_output_filename = os.path.join(
        OUTPUT_DIR, f"{safe_name}_reachable_stops.shp"
    )
    reachable_stops_gdf.to_file(stops_output_filename)
    print(f"Saved reachable bus stops to {stops_output_filename}")

all_stops_output_path = os.path.join(OUTPUT_DIR, 'all_bus_stops.shp')
STOPS_GDF.to_file(all_stops_output_path)
print(f"Saved all bus stops to {all_stops_output_path}")


os.makedirs(OUTPUT_DIR_ANALYSIS, exist_ok=True)
CENSUS_GDF = gpd.read_file(CENSUS_BLOCKS_PATH)
CENSUS_GDF = CENSUS_GDF.to_crs(epsg=3395)
CENSUS_GDF['area_ac_og'] = CENSUS_GDF.geometry.area

UNIQUE_ID = 'GEOID20'
if UNIQUE_ID not in CENSUS_GDF.columns:
    raise ValueError(
        f"The census data must contain a unique identifier column named '{UNIQUE_ID}'."
    )

accessible_area_files = [
    f for f in os.listdir(ACCESSIBLE_AREAS_DIR) if f.endswith('_accessible_area.shp')
]
JOB_FIELDS = ["tot_empl", "low_wage", "mid_wage", "high_wage"]

for area_file in accessible_area_files:
    facility_name_shp = (
        area_file.replace('_accessible_area.shp', '').replace('_', ' ').title()
    )
    print(f"Processing facility: {facility_name_shp}")
    accessible_area_path = os.path.join(ACCESSIBLE_AREAS_DIR, area_file)
    accessible_area_df = gpd.read_file(accessible_area_path).to_crs(epsg=3395)

    clipped_census_gdf = gpd.overlay(CENSUS_GDF, accessible_area_df, how='intersection')
    clipped_census_gdf['area_ac_cl'] = clipped_census_gdf.geometry.area
    clipped_census_gdf['area_perc'] = (
        clipped_census_gdf['area_ac_cl'] / clipped_census_gdf['area_ac_og']
    )

    area_perc_sums = clipped_census_gdf.groupby(UNIQUE_ID)['area_perc'].sum().reset_index()
    area_perc_sums = area_perc_sums.rename(columns={'area_perc': 'area_perc_total'})
    clipped_census_gdf = clipped_census_gdf.merge(area_perc_sums, on=UNIQUE_ID)
    overlapping_blocks = clipped_census_gdf['area_perc_total'] > 1

    clipped_census_gdf.loc[overlapping_blocks, 'area_perc'] = (
        clipped_census_gdf.loc[overlapping_blocks, 'area_perc'] /
        clipped_census_gdf.loc[overlapping_blocks, 'area_perc_total']
    )

    for job_field in JOB_FIELDS:
        if job_field in clipped_census_gdf.columns:
            clipped_census_gdf[job_field] = pd.to_numeric(
                clipped_census_gdf[job_field], errors='coerce'
            ).fillna(0)
            clipped_census_gdf[f'syn_{job_field}'] = (
                clipped_census_gdf['area_perc'] * clipped_census_gdf[job_field]
            )
        else:
            clipped_census_gdf[f'syn_{job_field}'] = 0

    totals = clipped_census_gdf[[f'syn_{field}' for field in JOB_FIELDS]].sum().round(0)
    print(f"Job characteristics accessible from {facility_name_shp}:")
    for field, value in totals.items():
        field_clean = field.replace('syn_', '').replace('_', ' ').title()
        print(f"  Total {field_clean}: {int(value)}")

    output_path = os.path.join(
        OUTPUT_DIR_ANALYSIS,
        f"{facility_name_shp.replace(' ', '_').lower()}_accessible_jobs.shp"
    )
    clipped_census_gdf.to_file(output_path)
    print(f"Saved accessible job data to {output_path}\n")


# pylint: disable=too-many-locals
def build_road_network():
    """
    Build a directed road network graph from the road centerlines.
    """
    roads_df = gpd.read_file(ROAD_SHP_PATH)
    roads_df = roads_df.to_crs(epsg=3857)
    graph_road = nx.DiGraph()

    print("Building road network with time-based edges...")
    for _, row in roads_df.iterrows():
        geom = row.geometry
        speed_limit = row['SPEED_LIMI']
        oneway_flag = row['ONEWAY']
        shape_len_val = row['Shape__Len']

        try:
            speed_limit = float(speed_limit)
            if speed_limit <= 0:
                continue
        except Exception:
            continue

        speed_mps = speed_limit * 0.44704
        travel_time_segment = shape_len_val / speed_mps

        if isinstance(geom, LineString):
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            coords = []
            for line_obj in geom:
                coords.extend(list(line_obj.coords))
        else:
            continue

        total_length_local = 0
        segment_lengths = []
        for i in range(len(coords) - 1):
            start_pt = coords[i]
            end_pt = coords[i + 1]
            seg_len = Point(start_pt).distance(Point(end_pt))
            segment_lengths.append(seg_len)
            total_length_local += seg_len

        if total_length_local == 0:
            continue

        for i, _ in enumerate(coords[:-1]):
            start_pt = coords[i]
            end_pt = coords[i + 1]
            seg_len = segment_lengths[i]
            travel_time_local = (seg_len / total_length_local) * travel_time_segment

            if oneway_flag == 'Y':
                graph_road.add_edge(start_pt, end_pt, weight=travel_time_local)
            elif oneway_flag == 'N':
                graph_road.add_edge(start_pt, end_pt, weight=travel_time_local)
                graph_road.add_edge(end_pt, start_pt, weight=travel_time_local)
            else:
                graph_road.add_edge(start_pt, end_pt, weight=travel_time_local)
                graph_road.add_edge(end_pt, start_pt, weight=travel_time_local)

    print("Road network built successfully.")
    return graph_road


road_graph = build_road_network()

# Use the new config variable
DRIVING_TIME_SECONDS = MAX_DRIVING_TIME_SECONDS

FACILITIES_GDF = FACILITIES_GDF.to_crs(epsg=3857)
CENSUS_GDF = gpd.read_file(CENSUS_BLOCKS_PATH)
CENSUS_GDF = CENSUS_GDF.to_crs(epsg=3857)
CENSUS_GDF['area_ac_og'] = CENSUS_GDF.geometry.area

if UNIQUE_ID not in CENSUS_GDF.columns:
    raise ValueError(
        f"The census data must contain a unique identifier column named '{UNIQUE_ID}'."
    )

DEMOGRAPHIC_FIELDS = [
    "total_pop", "total_hh", "est_minori", "est_lep", "est_lo_veh",
    "est_lo_v_1", "est_youth", "est_elderl"
]

for _, facility in FACILITIES_GDF.iterrows():
    facility_name = facility['name']
    print(f"Processing facility: {facility_name}")

    closest_node = min(
        road_graph.nodes,
        key=lambda nd: facility.geometry.distance(Point(nd))
    )

    lengths = nx.single_source_dijkstra_path_length(
        road_graph, source=closest_node, cutoff=DRIVING_TIME_SECONDS, weight='weight'
    )

    if not lengths:
        print(f"No nodes reachable within {DRIVING_TIME_SECONDS / 60} minutes from "
              f"{facility_name}")
        continue

    reachable_nodes_set = set(lengths.keys())
    reachable_points = [Point(node) for node in reachable_nodes_set]
    reachable_points_gdf = gpd.GeoDataFrame(geometry=reachable_points, crs='EPSG:3857')

    # Buffer by 100 meters to create a smoother isochrone
    buffers_local = reachable_points_gdf.geometry.buffer(100)
    isochrone = unary_union(buffers_local)
    isochrone = isochrone.simplify(tolerance=50)
    isochrone_gdf = gpd.GeoDataFrame(geometry=[isochrone], crs='EPSG:3857')

    clipped_census_gdf = gpd.overlay(CENSUS_GDF, isochrone_gdf, how='intersection')
    clipped_census_gdf['area_ac_cl'] = clipped_census_gdf.geometry.area

    original_areas = CENSUS_GDF[[UNIQUE_ID, 'area_ac_og']]
    clipped_census_gdf = clipped_census_gdf.merge(
        original_areas,
        on=UNIQUE_ID,
        how='left',
        suffixes=('', '_orig')
    )
    clipped_census_gdf['area_perc'] = (
        clipped_census_gdf['area_ac_cl'] / clipped_census_gdf['area_ac_og']
    )
    area_perc_sums = clipped_census_gdf.groupby(UNIQUE_ID)['area_perc'].sum().reset_index()
    area_perc_sums = area_perc_sums.rename(columns={'area_perc': 'area_perc_total'})
    clipped_census_gdf = clipped_census_gdf.merge(area_perc_sums, on=UNIQUE_ID)
    overlaps = clipped_census_gdf['area_perc_total'] > 1
    clipped_census_gdf.loc[overlaps, 'area_perc'] = (
        clipped_census_gdf.loc[overlaps, 'area_perc'] /
        clipped_census_gdf.loc[overlaps, 'area_perc_total']
    )

    # Recalculate area for the final overlay portion
    clipped_census_gdf['area_ac_og'] = clipped_census_gdf['geometry'].area
    clipped_census_gdf['area_ac_cl'] = clipped_census_gdf.geometry.area
    clipped_census_gdf['area_perc'] = (
        clipped_census_gdf['area_ac_cl'] / clipped_census_gdf['area_ac_og']
    )

    for field in DEMOGRAPHIC_FIELDS:
        if field in clipped_census_gdf.columns:
            clipped_census_gdf[field] = pd.to_numeric(
                clipped_census_gdf[field], errors='coerce'
            ).fillna(0)
            clipped_census_gdf[f'syn_{field}'] = (
                clipped_census_gdf['area_perc'] * clipped_census_gdf[field]
            )
        else:
            clipped_census_gdf[f'syn_{field}'] = 0

    totals = clipped_census_gdf[
        [f'syn_{field}' for field in DEMOGRAPHIC_FIELDS]
    ].sum().round(0)

    print(f"Demographics within {MAX_DRIVING_TIME_MINUTES}-minute drive of {facility_name}:")
    for fld, val in totals.items():
        clean_field = fld.replace('syn_', '').replace('_', ' ').title()
        print(f"  Total {clean_field}: {int(val)}")

    iso_output_path = os.path.join(
        OUTPUT_DIR_ANALYSIS,
        f"{facility_name.replace(' ', '_').lower()}_{MAX_DRIVING_TIME_MINUTES}min_drive_isochrone.shp"
    )
    isochrone_gdf.to_file(iso_output_path)

    census_output_path = os.path.join(
        OUTPUT_DIR_ANALYSIS,
        f"{facility_name.replace(' ', '_').lower()}_{MAX_DRIVING_TIME_MINUTES}min_drive_demographics.shp"
    )
    clipped_census_gdf.to_file(census_output_path)
    print(f"Saved {MAX_DRIVING_TIME_MINUTES}-minute drive isochrone to {iso_output_path}")
    print(f"Saved demographic data to {census_output_path}\n")
