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

def load_gtfs_data(gtfs_dir):
    """Load GTFS files and return DataFrames for routes, stops, trips, and stop_times."""
    routes_txt_path = os.path.join(gtfs_dir, 'routes.txt')
    stops_txt_path = os.path.join(gtfs_dir, 'stops.txt')
    trips_txt_path = os.path.join(gtfs_dir, 'trips.txt')
    stop_times_txt_path = os.path.join(gtfs_dir, 'stop_times.txt')

    # Verify required files exist
    for file_path in [routes_txt_path, stops_txt_path, trips_txt_path, stop_times_txt_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required GTFS file not found: {file_path}")

    routes_df = pd.read_csv(routes_txt_path)
    stops_df = pd.read_csv(stops_txt_path)
    trips_df = pd.read_csv(trips_txt_path)
    stop_times_df = pd.read_csv(stop_times_txt_path)

    return routes_df, stops_df, trips_df, stop_times_df

def load_shapefile(shp_path, input_crs):
    """Load the shapefile and set its CRS if needed."""
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    shp_df = gpd.read_file(shp_path)
    if shp_df.crs is None:
        shp_df.set_crs(input_crs, inplace=True)
    else:
        shp_df = shp_df.to_crs(input_crs)
    return shp_df

def preprocess_data(routes_df, shp_df, route_number_col):
    """Ensure route columns are strings and cleaned for merging."""
    routes_df['route_short_name_str'] = routes_df['route_short_name'].astype(str).str.strip()
    shp_df[route_number_col + '_str'] = (
        shp_df[route_number_col]
        .astype(str)
        .str.strip()
        .str.replace(" ", "")
    )
    return routes_df, shp_df

def merge_and_score(routes_df, shp_df, route_number_col, route_name_col):
    """Merge the GTFS routes with the shapefile data and compute similarity scores."""
    merged_df = pd.merge(
        routes_df,
        shp_df,
        left_on='route_short_name_str',
        right_on=route_number_col + '_str',
        how='outer',
        suffixes=('_gtfs', '_shp')
    )

    merged_df['short_name_score'] = merged_df.apply(
        lambda x: fuzz.ratio(str(x['route_short_name_str']), str(x[route_number_col + '_str'])),
        axis=1
    )
    merged_df['long_name_score'] = merged_df.apply(
        lambda x: fuzz.ratio(str(x['route_long_name']), str(x[route_name_col])),
        axis=1
    )
    merged_df['short_name_exact_match'] = merged_df['short_name_score'] == 100
    merged_df['long_name_exact_match'] = merged_df['long_name_score'] == 100
    return merged_df

def export_comparison(merged_df, output_dir):
    """Print match percentages and export the merged DataFrame as CSV."""
    total_short_names = len(merged_df)
    exact_short_name_matches = merged_df['short_name_exact_match'].sum()
    percentage_short_name_matches = (exact_short_name_matches / total_short_names * 100) if total_short_names > 0 else 0

    total_long_names = len(merged_df)
    exact_long_name_matches = merged_df['long_name_exact_match'].sum()
    percentage_long_name_matches = (exact_long_name_matches / total_long_names * 100) if total_long_names > 0 else 0

    print(f"\nPercentage of exact matches for short names: {percentage_short_name_matches:.2f}%")
    print(f"Percentage of exact matches for long names: {percentage_long_name_matches:.2f}%")

    output_csv_path = os.path.join(output_dir, 'gtfs_shp_comparison.csv')
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Route comparison exported to {output_csv_path}")
    return merged_df

def convert_stops_to_gdf(stops_df, input_crs):
    """Convert the stops DataFrame to a GeoDataFrame with proper geometry."""
    stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'], errors='coerce')
    stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'], errors='coerce')
    stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])
    stops_df['geometry'] = [Point(xy) for xy in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry='geometry', crs=input_crs)
    return stops_gdf

def prepare_geometries(stops_gdf, shp_df, projected_crs):
    """Reproject GeoDataFrames to the projected CRS and rename shapefile geometry."""
    stops_gdf = stops_gdf.to_crs(projected_crs)
    shp_df = shp_df.to_crs(projected_crs)
    if 'route_geometry' not in shp_df.columns:
        shp_df = shp_df.rename(columns={'geometry': 'route_geometry'})
    return stops_gdf, shp_df

def calculate_distance(row):
    """
    Calculate the distance between a stop and its route geometry.
    Returns distance in meters if both geometries are present.
    """
    if pd.notnull(row['geometry']) and pd.notnull(row['route_geometry']):
        return row['geometry'].distance(row['route_geometry'])
    return None

def identify_problem_stops(routes_df, stops_df, trips_df, stop_times_df, matched_routes,
                           shp_df, input_crs, projected_crs, output_crs, distance_allowance, route_number_col):
    """Identify stops that are problematic based on distance allowance and missing route matches."""
    # Process matched routes
    matched_trips = trips_df[trips_df['route_id'].isin(matched_routes['route_id'])]
    matched_stop_times = stop_times_df[stop_times_df['trip_id'].isin(matched_trips['trip_id'])]
    matched_stops = stops_df[stops_df['stop_id'].isin(matched_stop_times['stop_id'])].copy()
    matched_stops_gdf = convert_stops_to_gdf(matched_stops, input_crs)
    matched_stops_gdf, shp_df = prepare_geometries(matched_stops_gdf, shp_df, projected_crs)

    # Merge matched_routes with route geometries
    matched_routes = matched_routes.merge(
        shp_df[[route_number_col + '_str', 'route_geometry']],
        on=route_number_col + '_str',
        how='left'
    )

    matched_stop_times_trips = matched_stop_times.merge(
        matched_trips[['trip_id', 'route_id']],
        on='trip_id',
        how='left'
    )

    stop_route_pairs = matched_stop_times_trips[['stop_id', 'route_id']].drop_duplicates()
    stop_route_pairs = stop_route_pairs.merge(
        matched_stops_gdf[['stop_id', 'geometry']],
        on='stop_id',
        how='left'
    )
    stop_route_pairs = stop_route_pairs.merge(
        matched_routes[['route_id', 'route_geometry']],
        on='route_id',
        how='left'
    )

    stop_route_pairs['distance_to_route_meters'] = stop_route_pairs.apply(calculate_distance, axis=1)
    stop_route_pairs['distance_to_route_feet'] = stop_route_pairs['distance_to_route_meters'] * 3.28084
    stop_route_pairs['within_allowance'] = stop_route_pairs['distance_to_route_feet'] <= distance_allowance
    stops_not_within_allowance = stop_route_pairs[~stop_route_pairs['within_allowance']].copy()
    stops_not_within_allowance = stops_not_within_allowance.merge(
        matched_stops_gdf[['stop_id', 'stop_name']],
        on='stop_id',
        how='left'
    )
    stops_not_within_allowance['reason'] = f'Not within {distance_allowance} feet'

    # Process unmatched stops (stops without a matching route)
    all_route_ids = routes_df['route_id'].unique()
    matched_route_ids = matched_routes['route_id'].unique()
    unmatched_route_ids = set(all_route_ids) - set(matched_route_ids)
    unmatched_trips = trips_df[trips_df['route_id'].isin(unmatched_route_ids)]
    unmatched_stop_times = stop_times_df[stop_times_df['trip_id'].isin(unmatched_trips['trip_id'])]
    unmatched_stops = stops_df[stops_df['stop_id'].isin(unmatched_stop_times['stop_id'])].copy()
    unmatched_stops_gdf = convert_stops_to_gdf(unmatched_stops, input_crs)
    unmatched_stops_gdf = unmatched_stops_gdf.to_crs(projected_crs)
    unmatched_stops_gdf['reason'] = 'No matching route'
    unmatched_stops_gdf['distance_to_route_feet'] = None  # No route to calculate distance

    # Aggregate routes serving each stop
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
    stop_routes_df = stop_times_trips_routes.groupby('stop_id')['route_short_name'].unique().reset_index()
    stop_routes_df['routes_serving_stop'] = stop_routes_df['route_short_name'].apply(lambda x: ', '.join(map(str, x)))
    stop_routes_df = stop_routes_df[['stop_id', 'routes_serving_stop']]

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

    problem_stops_gdf = pd.concat(
        [
            stops_not_within_allowance[['stop_id', 'stop_name', 'geometry', 'reason',
                                          'distance_to_route_feet', 'routes_serving_stop']],
            unmatched_stops_gdf[['stop_id', 'stop_name', 'geometry', 'reason',
                                 'distance_to_route_feet', 'routes_serving_stop']]
        ],
        ignore_index=True,
    )

    problem_stops_gdf = problem_stops_gdf.drop_duplicates(subset='stop_id')
    problem_stops_gdf = gpd.GeoDataFrame(problem_stops_gdf, geometry='geometry', crs=projected_crs)
    problem_stops_gdf = problem_stops_gdf.to_crs(output_crs)
    return problem_stops_gdf

def main():
    """Load data, compare GTFS routes to the system shapefile, identify discrepancies, and export results."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    routes_df, stops_df, trips_df, stop_times_df = load_gtfs_data(GTFS_DIR)
    shp_df = load_shapefile(SHAPEFILE_PATH, INPUT_CRS)

    # Preprocess for merging
    routes_df, shp_df = preprocess_data(routes_df, shp_df, ROUTE_NUMBER_COLUMN)

    # Merge data and compute similarity scores
    merged_df = merge_and_score(routes_df, shp_df, ROUTE_NUMBER_COLUMN, ROUTE_NAME_COLUMN)

    # Print percentages and export merged comparison
    merged_df = export_comparison(merged_df, OUTPUT_DIR)

    # Filter matched routes based on exact short name match
    matched_routes = merged_df[merged_df['short_name_exact_match']].copy()

    # Identify problem stops
    problem_stops_gdf = identify_problem_stops(
        routes_df, stops_df, trips_df, stop_times_df, matched_routes,
        shp_df, INPUT_CRS, PROJECTED_CRS, OUTPUT_CRS, DISTANCE_ALLOWANCE, ROUTE_NUMBER_COLUMN
    )

    # Export problem stops to a shapefile
    output_shp_path = os.path.join(OUTPUT_DIR, 'problem_stops.shp')
    problem_stops_gdf.to_file(output_shp_path)
    print(f"Problem stops were exported to {output_shp_path}!")

if __name__ == '__main__':
    main()
