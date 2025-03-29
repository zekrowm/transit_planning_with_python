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
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

# FILE PATHS
GTFS_DIR = r'\\your_project_folder\system_gtfs'
SHAPEFILE_PATH = r'\\your_project_folder\your_transit_system\your_transit_system.shp'
OUTPUT_DIR = r'\\your_project_folder\output'

# COLUMN NAMES FOR MATCHING
ROUTE_NUMBER_COLUMN = 'ROUTE_NUMB'
ROUTE_NAME_COLUMN = 'ROUTE_NAME'

# DISTANCE SETTINGS
DISTANCE_ALLOWANCE = 100  # in feet

# COORDINATE REFERENCE SYSTEM (CRS) SETTINGS
INPUT_CRS = 'EPSG:4326'     # WGS84
PROJECTED_CRS = 'EPSG:26918' # NAD83 / UTM zone 18N (adjust as appropriate)
OUTPUT_CRS = 'EPSG:4326'    # WGS84

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
    routes_df['route_short_name_str'] = (
        routes_df['route_short_name'].astype(str).str.strip()
    )
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

    # Similarity scores
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
    if total_short_names > 0:
        percentage_short_name_matches = (exact_short_name_matches / total_short_names) * 100
    else:
        percentage_short_name_matches = 0

    total_long_names = len(merged_df)
    exact_long_name_matches = merged_df['long_name_exact_match'].sum()
    if total_long_names > 0:
        percentage_long_name_matches = (exact_long_name_matches / total_long_names) * 100
    else:
        percentage_long_name_matches = 0

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
    """Calculate distance between a stop and its route geometry (in meters)."""
    if pd.notnull(row['geometry']) and pd.notnull(row['route_geometry']):
        return row['geometry'].distance(row['route_geometry'])
    return None

def identify_problem_stops(
    routes_df,
    stops_df,
    trips_df,
    stop_times_df,
    matched_routes,
    shp_df,
    input_crs,
    projected_crs,
    output_crs,
    distance_allowance,
    route_number_col
):
    """
    Identify stops that are problematic based on distance allowance and
    missing route matches. Also determine which routes have out-of-buffer stops.

    Returns:
      1) problem_stops_gdf (GeoDataFrame) with columns:
           stop_id, stop_name, geometry, reason, distance_to_route_feet, routes_serving_stop,
           route_id
      2) routes_with_stops_outside_buffer (list of route_id's that had at least one
           out-of-buffer stop)
      3) A reference table (DataFrame) with route_id + route_short_name + route_geometry,
           so we can plot route by route.
    """
    # Process matched routes
    matched_trips = trips_df[trips_df['route_id'].isin(matched_routes['route_id'])]
    matched_stop_times = stop_times_df[stop_times_df['trip_id'].isin(matched_trips['trip_id'])]
    matched_stops = stops_df[stops_df['stop_id'].isin(matched_stop_times['stop_id'])].copy()
    matched_stops_gdf = convert_stops_to_gdf(matched_stops, input_crs)
    matched_stops_gdf, shp_df = prepare_geometries(matched_stops_gdf, shp_df, projected_crs)

    # Merge matched_routes with route geometries
    route_geometry_table = matched_routes.merge(
        shp_df[[route_number_col + '_str', 'route_geometry']],
        on=route_number_col + '_str',
        how='left'
    )
    # Keep only columns needed for reference
    route_geometry_table = route_geometry_table[['route_id', 'route_short_name', 'route_geometry']].drop_duplicates()

    matched_stop_times_trips = matched_stop_times.merge(
        matched_trips[['trip_id', 'route_id']],
        on='trip_id',
        how='left'
    )

    stop_route_pairs = matched_stop_times_trips[['stop_id', 'route_id']].drop_duplicates()
    stop_route_pairs = stop_route_pairs.merge(
        matched_stops_gdf[['stop_id', 'geometry', 'stop_name']],
        on='stop_id',
        how='left'
    )
    stop_route_pairs = stop_route_pairs.merge(
        route_geometry_table[['route_id', 'route_geometry']],
        on='route_id',
        how='left'
    )

    # Distance calculations
    stop_route_pairs['distance_to_route_meters'] = stop_route_pairs.apply(calculate_distance, axis=1)
    stop_route_pairs['distance_to_route_feet'] = stop_route_pairs['distance_to_route_meters'] * 3.28084
    stop_route_pairs['within_allowance'] = stop_route_pairs['distance_to_route_feet'] <= distance_allowance

    # Identify stops not within allowance
    stops_not_within_allowance = stop_route_pairs[~stop_route_pairs['within_allowance']].copy()
    stops_not_within_allowance['reason'] = f'Not within {distance_allowance} feet'

    # Which route_ids have out-of-buffer stops?
    # We'll filter out rows with route_id == NaN just in case
    stops_outside_allowance = stops_not_within_allowance.dropna(subset=['route_id'])
    routes_with_stops_outside_buffer = stops_outside_allowance['route_id'].unique()

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

    # Build a DataFrame of all stops vs the routes they serve (by short name)
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

    # Merge route-serving info into out-of-buffer stops
    stops_not_within_allowance = stops_not_within_allowance.merge(
        stop_routes_df,
        on='stop_id',
        how='left'
    )

    # Merge route-serving info into unmatched stops
    unmatched_stops_gdf = unmatched_stops_gdf.merge(
        stop_routes_df,
        on='stop_id',
        how='left'
    )

    # Combine out-of-buffer stops + unmatched stops
    problem_stops_gdf = pd.concat(
        [
            stops_not_within_allowance[
                ['stop_id', 'stop_name', 'geometry', 'reason',
                 'distance_to_route_feet', 'routes_serving_stop', 'route_id']
            ],
            unmatched_stops_gdf[
                ['stop_id', 'stop_name', 'geometry', 'reason',
                 'distance_to_route_feet', 'routes_serving_stop']
            ].assign(route_id=None)  # Unmatched => no route_id
        ],
        ignore_index=True,
    )

    # Drop duplicates
    problem_stops_gdf = problem_stops_gdf.drop_duplicates(subset=['stop_id', 'route_id'])

    # Convert to a GeoDataFrame
    problem_stops_gdf = gpd.GeoDataFrame(problem_stops_gdf, geometry='geometry', crs=projected_crs)
    problem_stops_gdf = problem_stops_gdf.to_crs(output_crs)

    return problem_stops_gdf, routes_with_stops_outside_buffer, route_geometry_table

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

    # Identify problem stops and gather route geometry for plotting
    problem_stops_gdf, routes_with_stops_outside_buffer, route_geometry_table = identify_problem_stops(
        routes_df, stops_df, trips_df, stop_times_df,
        matched_routes, shp_df, INPUT_CRS, PROJECTED_CRS, OUTPUT_CRS,
        DISTANCE_ALLOWANCE, ROUTE_NUMBER_COLUMN
    )

    # --------------------------------------------------------------------
    # 1) Export problem stops as a shapefile
    # --------------------------------------------------------------------
    output_shp_path = os.path.join(OUTPUT_DIR, 'problem_stops.shp')
    problem_stops_gdf.to_file(output_shp_path)
    print(f"Problem stops were exported to {output_shp_path}!")

    # --------------------------------------------------------------------
    # 2) Export updated CSV with flags
    # --------------------------------------------------------------------
    if 'route_id' in merged_df.columns:
        merged_df['has_stops_outside_buffer'] = merged_df['route_id'].isin(routes_with_stops_outside_buffer)
        updated_comparison_csv = os.path.join(OUTPUT_DIR, 'gtfs_shp_comparison_with_flags.csv')
        merged_df.to_csv(updated_comparison_csv, index=False)
        print(f"Updated comparison with buffer flags exported to {updated_comparison_csv}")

    # --------------------------------------------------------------------
    # 3) Export an Excel file of problem stops (with route_short_name and distance)
    # --------------------------------------------------------------------
    # Merge to get route_short_name for each route_id
    # (Unmatched stops will have route_id == None => route_short_name is NaN)
    problem_stops_export = problem_stops_gdf.merge(
        route_geometry_table[['route_id', 'route_short_name']],
        on='route_id',
        how='left'
    )

    # Reorder columns for clarity
    problem_stops_export = problem_stops_export[[
        'stop_id',
        'stop_name',
        'route_short_name',
        'distance_to_route_feet',
        'reason'
    ]]

    xlsx_path = os.path.join(OUTPUT_DIR, 'problem_stops.xlsx')
    problem_stops_export.to_excel(xlsx_path, index=False)
    print(f"Problem stops exported (with distance) to {xlsx_path}")

    # --------------------------------------------------------------------
    # 4) Create a SEPARATE PLOT FOR EACH ROUTE with out-of-buffer stops
    # --------------------------------------------------------------------
    if len(routes_with_stops_outside_buffer) > 0:
        for rid in routes_with_stops_outside_buffer:
            # Get route geometry + route short name from our reference
            route_row = route_geometry_table[route_geometry_table['route_id'] == rid]
            if route_row.empty:
                continue  # Safety check

            route_short_name = route_row['route_short_name'].values[0]
            route_geom = route_row['route_geometry'].values[0]

            # Filter problem stops for this route_id
            route_problem_stops = problem_stops_gdf[problem_stops_gdf['route_id'] == rid]

            # Print route-specific warning
            warning_msg = (
                f"WARNING: Route {route_short_name!r} has stops outside the "
                f"{DISTANCE_ALLOWANCE} ft buffer allowance!"
            )
            print(warning_msg)

            # Convert single geometry to a GeoDataFrame for plotting
            if route_geom is None:
                continue

            route_gdf = gpd.GeoDataFrame(
                route_row[['route_id', 'route_short_name']].copy(),
                geometry=[route_geom],
                crs=PROJECTED_CRS
            ).to_crs(OUTPUT_CRS)

            # Make a figure for THIS route
            fig, ax = plt.subplots(figsize=(10, 10))
            # Plot route geometry
            route_gdf.plot(ax=ax, edgecolor='black', facecolor='none')
            # Plot problem stops
            if not route_problem_stops.empty:
                route_problem_stops.plot(ax=ax, markersize=5)

            ax.set_title(f"Route {route_short_name} — Problem Stops")
            # Save with "warning" in the file name
            warning_jpeg_path = os.path.join(
                OUTPUT_DIR,
                f"problem_stops_{route_short_name}_warning.jpeg"
            )
            plt.savefig(warning_jpeg_path, dpi=300)
            plt.close()

            print(f"WARNING: Visualization for Route {route_short_name} exported to {warning_jpeg_path}")
    else:
        print("No stops found outside the buffer allowance. No per-route warning plots created.")

if __name__ == '__main__':
    main()
