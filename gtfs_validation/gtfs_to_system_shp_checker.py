"""
gtfs_to_system_shp_checker.py

This script compares GTFS route data with a transit system's shapefile to identify discrepancies.
It calculates similarity scores for route names and identifies stops that are not within
a specified distance allowance from their serving routes.
The results are exported as CSV and Shapefile for further analysis.
"""

import os

import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz
from shapely.geometry import Point

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

# FILE PATHS

# Path to your GTFS data folder, which contains the .txt files (e.g., routes.txt, stops.txt)
GTFS_DIR = (
    r'\\your_project_folder\system_gtfs'
)

# Full path to your transit system's shapefile (.shp)
SHAPEFILE_PATH = (
    r'\\your_project_folder\your_transit_system\your_transit_system.shp'
)

# Path to your desired output folder for results
OUTPUT_DIR = (
    r'\\your_project_folder\output'
)


# COLUMN NAMES FOR MATCHING

# Column name in the shapefile that corresponds to 'route_short_name' in the GTFS data
ROUTE_NUMBER_COLUMN = 'ROUTE_NUMB'

# Column name in the shapefile that corresponds to 'route_long_name' in the GTFS data
ROUTE_NAME_COLUMN = 'ROUTE_NAME'


# DISTANCE SETTINGS

# Maximum allowed distance between a bus stop and its serving route (in feet)
DISTANCE_ALLOWANCE = 100  # Modify this value as needed


# COORDINATE REFERENCE SYSTEM (CRS) SETTINGS

# Input CRS for data (e.g., stops and shapefiles)
# WGS84 ('EPSG:4326') is the standard geographic coordinate system and is commonly used.
# It is recommended to keep this setting as is unless you have a specific reason to change it.
INPUT_CRS = 'EPSG:4326'  # WGS84

# Projection CRS for accurate distance calculations
# A projected CRS is necessary for measuring distances accurately.
# 'EPSG:26918' corresponds to NAD83 / UTM zone 18N, suitable for certain regions.
# Choose a projected CRS appropriate for your geographic area if different.
PROJECTED_CRS = 'EPSG:26918'  # NAD83 / UTM zone 18N

# Output CRS for the resulting data
# Typically set to 'EPSG:4326' to maintain compatibility with most GIS applications.
# It is recommended to keep this setting as is unless you need the output in a different CRS.
OUTPUT_CRS = 'EPSG:4326'  # WGS84

# ==============================
# END OF CONFIGURATION SECTION
# ==============================

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read GTFS data
routes_txt_path = os.path.join(
    GTFS_DIR,
    'routes.txt'
)
stops_txt_path = os.path.join(
    GTFS_DIR,
    'stops.txt'
)
trips_txt_path = os.path.join(
    GTFS_DIR,
    'trips.txt'
)
stop_times_txt_path = os.path.join(
    GTFS_DIR,
    'stop_times.txt'
)

# Verify GTFS files exist
for file_path in [
    routes_txt_path,
    stops_txt_path,
    trips_txt_path,
    stop_times_txt_path
]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Required GTFS file not found: {file_path}")

routes_df = pd.read_csv(routes_txt_path)
stops_df = pd.read_csv(stops_txt_path)
trips_df = pd.read_csv(trips_txt_path)
stop_times_df = pd.read_csv(stop_times_txt_path)

# Read shapefile
if not os.path.isfile(SHAPEFILE_PATH):
    raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

shp_df = gpd.read_file(SHAPEFILE_PATH)

# Set CRS for shapefile if not already set
if shp_df.crs is None:
    shp_df.set_crs(INPUT_CRS, inplace=True)
else:
    shp_df = shp_df.to_crs(INPUT_CRS)

# Ensure columns are strings and strip spaces
routes_df['route_short_name_str'] = (
    routes_df['route_short_name']
    .astype(str)
    .str.strip()
)
shp_df[ROUTE_NUMBER_COLUMN + '_str'] = (
    shp_df[ROUTE_NUMBER_COLUMN]
    .astype(str)
    .str.strip()
    .str.replace(" ", "")
)

# Merge DataFrames on route numbers
merged_df = pd.merge(
    routes_df,
    shp_df,
    left_on='route_short_name_str',
    right_on=ROUTE_NUMBER_COLUMN + '_str',
    how='outer',
    suffixes=('_gtfs', '_shp')
)

# Compute variation scores
merged_df['short_name_score'] = merged_df.apply(
    lambda x: fuzz.ratio(
        str(x['route_short_name_str']),
        str(x[ROUTE_NUMBER_COLUMN + '_str'])
    ),
    axis=1
)
merged_df['long_name_score'] = merged_df.apply(
    lambda x: fuzz.ratio(
        str(x['route_long_name']),
        str(x[ROUTE_NAME_COLUMN])
    ),
    axis=1
)

# Determine exact matches
merged_df['short_name_exact_match'] = merged_df['short_name_score'] == 100
merged_df['long_name_exact_match'] = merged_df['long_name_score'] == 100

# Calculate percentages
total_short_names = len(merged_df)
exact_short_name_matches = merged_df['short_name_exact_match'].sum()
percentage_short_name_matches = (
    (exact_short_name_matches / total_short_names) * 100
    if total_short_names > 0 else 0
)

total_long_names = len(merged_df)
exact_long_name_matches = merged_df['long_name_exact_match'].sum()
percentage_long_name_matches = (
    (exact_long_name_matches / total_long_names) * 100
    if total_long_names > 0 else 0
)

# Print percentages
print(f"\nPercentage of exact matches for short names: {percentage_short_name_matches:.2f}%")
print(f"Percentage of exact matches for long names: {percentage_long_name_matches:.2f}%")

# Export merged DataFrame to CSV
output_csv_path = os.path.join(
    OUTPUT_DIR,
    'gtfs_shp_comparison.csv'
)
merged_df.to_csv(output_csv_path, index=False)
print(f"Route comparison exported to {output_csv_path}")

# Proceed with problem stops identification
# Filter matched routes where short names are an exact match
matched_routes = merged_df[merged_df['short_name_exact_match']].copy()

# Get trips for matched routes
matched_trips = trips_df[trips_df['route_id'].isin(matched_routes['route_id'])]

# Get stop_times for matched trips
matched_stop_times = stop_times_df[stop_times_df['trip_id'].isin(matched_trips['trip_id'])]

# Get stops used in matched routes
matched_stops = stops_df[stops_df['stop_id'].isin(matched_stop_times['stop_id'])].copy()

# Convert stops to GeoDataFrame
matched_stops['stop_lat'] = pd.to_numeric(
    matched_stops['stop_lat'],
    errors='coerce'
)
matched_stops['stop_lon'] = pd.to_numeric(
    matched_stops['stop_lon'],
    errors='coerce'
)
matched_stops = matched_stops.dropna(subset=['stop_lat', 'stop_lon'])
matched_stops['geometry'] = [
    Point(xy) for xy in zip(matched_stops['stop_lon'], matched_stops['stop_lat'])
]
matched_stops_gdf = gpd.GeoDataFrame(
    matched_stops,
    geometry='geometry',
    crs=INPUT_CRS
)

# Convert to projected CRS for accurate distance calculation
matched_stops_gdf = matched_stops_gdf.to_crs(PROJECTED_CRS)
shp_df = shp_df.to_crs(PROJECTED_CRS)

# Rename geometry column in shapefile to 'route_geometry' (if not already)
if 'route_geometry' not in shp_df.columns:
    shp_df = shp_df.rename(columns={'geometry': 'route_geometry'})

# Merge matched_routes with route geometries
matched_routes = matched_routes.merge(
    shp_df[[ROUTE_NUMBER_COLUMN + '_str', 'route_geometry']],
    on=ROUTE_NUMBER_COLUMN + '_str',
    how='left'
)

# Merge stop_times with trips to get route IDs for stops
matched_stop_times_trips = matched_stop_times.merge(
    matched_trips[['trip_id', 'route_id']],
    on='trip_id',
    how='left'
)

# Get stop-route pairs
stop_route_pairs = matched_stop_times_trips[['stop_id', 'route_id']].drop_duplicates()

# Merge with stops GeoDataFrame to get geometry
stop_route_pairs = stop_route_pairs.merge(
    matched_stops_gdf[['stop_id', 'geometry']],
    on='stop_id',
    how='left'
)

# Merge with matched_routes to get route geometries
stop_route_pairs = stop_route_pairs.merge(
    matched_routes[['route_id', 'route_geometry']],
    on='route_id',
    how='left'
)

# Calculate distance between stop and route
def calculate_distance(row):
    """
    Calculate the distance between a stop and its route geometry in meters.

    Parameters:
        row (pd.Series): A row from the DataFrame containing 'geometry' and 'route_geometry'.

    Returns:
        float or None: Distance in meters if geometries are present, else None.
    """
    if pd.notnull(row['geometry']) and pd.notnull(row['route_geometry']):
        return row['geometry'].distance(row['route_geometry'])
    return None

# Apply the distance calculation
stop_route_pairs['distance_to_route_meters'] = stop_route_pairs.apply(
    calculate_distance,
    axis=1
)

# Convert distance to feet
stop_route_pairs['distance_to_route_feet'] = (
    stop_route_pairs['distance_to_route_meters'] * 3.28084
)

# Determine if within allowance
stop_route_pairs['within_allowance'] = (
    stop_route_pairs['distance_to_route_feet'] <= DISTANCE_ALLOWANCE
)

# Identify stops not within the distance allowance
stops_not_within_allowance = stop_route_pairs[~stop_route_pairs['within_allowance']].copy()

# Add reason column and select relevant columns
stops_not_within_allowance = stops_not_within_allowance.merge(
    matched_stops_gdf[['stop_id', 'stop_name']],
    on='stop_id',
    how='left'
)
stops_not_within_allowance['reason'] = f'Not within {DISTANCE_ALLOWANCE} feet'
stops_not_within_allowance = stops_not_within_allowance.rename(
    columns={'distance_to_route_feet': 'distance_to_route_feet'}
)

# Identify stops without a matching route
all_route_ids = routes_df['route_id'].unique()
matched_route_ids = matched_routes['route_id'].unique()
unmatched_route_ids = set(all_route_ids) - set(matched_route_ids)

# Get stops from unmatched routes
unmatched_trips = trips_df[trips_df['route_id'].isin(unmatched_route_ids)]
unmatched_stop_times = stop_times_df[stop_times_df['trip_id'].isin(unmatched_trips['trip_id'])]
unmatched_stops = stops_df[stops_df['stop_id'].isin(unmatched_stop_times['stop_id'])].copy()

# Convert unmatched stops to GeoDataFrame
unmatched_stops['stop_lat'] = pd.to_numeric(
    unmatched_stops['stop_lat'],
    errors='coerce'
)
unmatched_stops['stop_lon'] = pd.to_numeric(
    unmatched_stops['stop_lon'],
    errors='coerce'
)
unmatched_stops = unmatched_stops.dropna(subset=['stop_lat', 'stop_lon'])
unmatched_stops['geometry'] = [
    Point(xy) for xy in zip(unmatched_stops['stop_lon'], unmatched_stops['stop_lat'])
]
unmatched_stops_gdf = gpd.GeoDataFrame(
    unmatched_stops,
    geometry='geometry',
    crs=INPUT_CRS
)
unmatched_stops_gdf = unmatched_stops_gdf.to_crs(PROJECTED_CRS)

# Add reason and distance columns
unmatched_stops_gdf['reason'] = 'No matching route'
unmatched_stops_gdf['distance_to_route_feet'] = None  # No route to calculate distance

# Combine stop_times with trips and routes to get route_short_name for each stop
stop_times_trips = stop_times_df[['stop_id', 'trip_id']].merge(
    trips_df[['trip_id', 'route_id']],
    on='trip_id',
    how='left'
)
stop_times_trips_routes = stop_times_trips.merge(
    routes_df[['route_id', 'route_short_name']],
    on='route_id',
    how='left'
)

# Group by stop_id and aggregate route_short_names
stop_routes_df = stop_times_trips_routes.groupby('stop_id')['route_short_name'].unique().reset_index()
stop_routes_df['routes_serving_stop'] = stop_routes_df['route_short_name'].apply(
    lambda x: ', '.join(map(str, x))
)
stop_routes_df = stop_routes_df[['stop_id', 'routes_serving_stop']]

# Merge routes_serving_stop into problem stops
stops_not_within_allowance = stops_not_within_allowance.merge(
    stop_routes_df,
    on='stop_id',
    how='left'
)
unmatched_stops_gdf = unmatched_stops_gdf.merge(
    stop_routes_df,
    on='stop_id',
    how='left'
)

# Combine the two GeoDataFrames
problem_stops_gdf = pd.concat(
    [
        stops_not_within_allowance[
            [
                'stop_id',
                'stop_name',
                'geometry',
                'reason',
                'distance_to_route_feet',
                'routes_serving_stop'
            ]
        ],
        unmatched_stops_gdf[
            [
                'stop_id',
                'stop_name',
                'geometry',
                'reason',
                'distance_to_route_feet',
                'routes_serving_stop'
            ]
        ],
    ],
    ignore_index=True,
)

# Remove duplicates
problem_stops_gdf = problem_stops_gdf.drop_duplicates(subset='stop_id')

# Ensure it's a GeoDataFrame
problem_stops_gdf = gpd.GeoDataFrame(
    problem_stops_gdf,
    geometry='geometry',
    crs=PROJECTED_CRS
)

# Reproject back to output CRS for compatibility (optional)
problem_stops_gdf = problem_stops_gdf.to_crs(OUTPUT_CRS)

# Export problem stops to shapefile
OUTPUT_SHP_PATH = os.path.join(
    OUTPUT_DIR,
    'problem_stops.shp'
)
problem_stops_gdf.to_file(OUTPUT_SHP_PATH)
print(f"Problem stops were exported to {OUTPUT_SHP_PATH}!")
