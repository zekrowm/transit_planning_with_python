#!/usr/bin/env python
# coding: utf-8

"""
gtfs_stop_roadway_shp_intersection_check.py

This script processes GTFS stop data and roadway shapefiles to identify
intersecting stops. It performs spatial joins and buffers to determine
the depth of conflict between stops and roadways, then outputs the results
as shapefiles and CSV files.
"""

import os

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

# Paths to input files
ROADWAYS_PATH = (
    r'path\to\your\roadways.shp'  # Replace with your roadways shapefile path
)
GTFS_FOLDER = (
    r'path\to\your\GTFS\folder'     # Replace with your GTFS folder path
)
STOPS_PATH = os.path.join(GTFS_FOLDER, 'stops.txt')
OUTPUT_DIR = (
    r'path\to\output\directory'       # Replace with your desired output directory
)

# Coordinate Reference Systems
STOPS_CRS = 'EPSG:4326'   # WGS84 Latitude/Longitude
TARGET_CRS = 'EPSG:2283'  # NAD83 / Virginia North

# Negative buffer distances in feet
BUFFER_DISTANCES = [-1, -5, -10]  # Adjust buffer distances as needed

# Output file names
OUTPUT_SHP_NAME = 'intersecting_stops.shp'
OUTPUT_CSV_NAME = 'intersecting_stops.csv'

# ==============================
# END OF CONFIGURATION SECTION
# ==============================

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Verify that stops.txt exists in the provided GTFS folder
if not os.path.isfile(STOPS_PATH):
    raise FileNotFoundError(
        f"'stops.txt' not found in the GTFS folder: {GTFS_FOLDER}"
    )

# Read roadways shapefile
roadways_gdf = gpd.read_file(ROADWAYS_PATH)

# Read GTFS stops.txt into DataFrame
stops_df = pd.read_csv(STOPS_PATH)

# Create GeoDataFrame from stops_df
geometry = [Point(xy) for xy in zip(stops_df.stop_lon, stops_df.stop_lat)]
stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry)

# Set CRS for stops_gdf
stops_gdf.set_crs(STOPS_CRS, inplace=True)

# Reproject both datasets to the target CRS
roadways_gdf = roadways_gdf.to_crs(TARGET_CRS)
stops_gdf = stops_gdf.to_crs(TARGET_CRS)

# Perform spatial join to find stops that intersect roadways
intersecting_stops = gpd.sjoin(
    stops_gdf,
    roadways_gdf,
    how='inner',
    predicate='intersects'
)

# Optional: keep only columns from stops_gdf
intersecting_stops = intersecting_stops[stops_gdf.columns]

# Add 'x' and 'y' columns for coordinate reference
intersecting_stops['x'] = intersecting_stops.geometry.x
intersecting_stops['y'] = intersecting_stops.geometry.y

def determine_conflict_depth(stops_gdf, roadways_gdf, buffer_distances):
    """
    Determines the depth of conflict between stops and roadways based on buffer distances.

    For each buffer distance, buffers the roadways and performs a spatial join
    to identify stops that intersect the buffered roadways. Updates the stops_gdf
    with conflict indicators for each buffer distance.

    Args:
        stops_gdf (GeoDataFrame): GeoDataFrame containing stop locations.
        roadways_gdf (GeoDataFrame): GeoDataFrame containing roadway geometries.
        buffer_distances (list): List of negative buffer distances in feet.

    Returns:
        GeoDataFrame: Updated stops_gdf with conflict depth indicators.
    """
    for buffer_distance in buffer_distances:
        # Buffer the roadways by the negative distance
        roadways_buffered = roadways_gdf.copy()
        roadways_buffered['geometry'] = roadways_buffered.geometry.buffer(buffer_distance)

        # Remove invalid or empty geometries
        roadways_buffered = roadways_buffered[~roadways_buffered.is_empty]
        roadways_buffered = roadways_buffered[roadways_buffered.is_valid]

        # Spatial join to find stops that intersect the buffered roadways
        buffered_join = gpd.sjoin(
            stops_gdf,
            roadways_buffered[['geometry']],
            how='left',
            predicate='intersects'
        )

        # Create a column to indicate whether the stop intersects the buffered roadways
        column_name = f'conflict_{-buffer_distance}ft'
        stops_gdf[column_name] = ~buffered_join['index_right'].isnull()
    return stops_gdf

# Determine depth of conflict and update intersecting_stops
intersecting_stops = determine_conflict_depth(
    intersecting_stops,
    roadways_gdf,
    BUFFER_DISTANCES
)

# Sort by conflict depth columns in descending order
conflict_columns = [f'conflict_{-bd}ft' for bd in BUFFER_DISTANCES]
intersecting_stops = intersecting_stops.sort_values(
    by=conflict_columns,
    ascending=[False]*len(conflict_columns)
)

# Save to shapefile
output_shp_path = os.path.join(OUTPUT_DIR, OUTPUT_SHP_NAME)
intersecting_stops.to_file(output_shp_path)

# Save to CSV
output_csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
intersecting_stops.to_csv(output_csv_path, index=False)

print("Processing complete. Output saved to:", OUTPUT_DIR)
