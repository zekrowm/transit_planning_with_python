#!/usr/bin/env python
# coding: utf-8

"""
Module for identifying and analyzing GTFS routes in proximity to specified manual locations.
"""

import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

# Define GTFS data directory
GTFS_INPUT_PATH = r'\\your_file_path\here'

# Define output directory (if needed)
OUTPUT_PATH = r'\\your_file_path\here'

# GTFS files to load
GTFS_FILES = {
    'stops': 'stops.txt',
    'stop_times': 'stop_times.txt',
    'trips': 'trips.txt',
    'routes': 'routes.txt'
}

# Define manual locations
MANUAL_LOCATIONS = [
    {"name": "Ballston", "latitude": 38.881724, "longitude": -77.111615},
    {"name": "Braddock", "latitude": 38.813545, "longitude": -77.053864},
    {"name": "Crystal City", "latitude": 38.85835, "longitude": -77.051232}
]

# Buffer radius configuration
# Specify the buffer distance and its unit ('miles' or 'feet')
BUFFER_DISTANCE = 0.5  # e.g., 0.5
BUFFER_UNIT = 'miles'  # options: 'miles', 'feet'

# Projected CRS (NAD83 / DC State Plane (US Feet))
PROJECTED_CRS = "EPSG:2232"  # Replace with your desired CRS

# ==============================
# END OF CONFIGURATION SECTION
# ==============================


def check_input_files(base_path, files_dict):
    """
    Verify that all required GTFS files exist in the specified directory.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The input directory {base_path} does not exist.")
    for file_name in files_dict.values():
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The required GTFS file {file_name} does not exist in {base_path}."
            )


def load_gtfs_data(base_path, files_dict):
    """
    Load GTFS data files into Pandas DataFrames.
    """
    data = {}
    for file_name in files_dict.values():
        file_path = os.path.join(base_path, file_name)
        try:
            data[file_name.split('.')[0]] = pd.read_csv(file_path)
            print(f"Loaded {file_name} with {len(data[file_name.split('.')[0]])} records.")
        except Exception as error:
            raise Exception(f"Error loading {file_name}: {error}") from error
    return data


def create_geodataframe_locations(locations, crs="EPSG:4326"):
    """
    Convert a list of location dictionaries to a GeoDataFrame.
    """
    gdf = gpd.GeoDataFrame(
        locations,
        geometry=[Point(loc['longitude'], loc['latitude']) for loc in locations],
        crs=crs
    )
    return gdf


def create_geodataframe_stops(stops_df, crs="EPSG:4326"):
    """
    Convert stops DataFrame to a GeoDataFrame with point geometries.
    """
    gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs=crs
    )
    return gdf


def convert_buffer_distance(distance, unit):
    """
    Convert buffer distance to feet based on the specified unit.
    """
    if unit.lower() == 'miles':
        return distance * 5280  # 1 mile = 5280 feet
    elif unit.lower() == 'feet':
        return distance
    else:
        raise ValueError("Unsupported buffer unit. Please use 'miles' or 'feet'.")


def reproject_geodataframes(gdf_locations, stops_gdf, target_crs):
    """
    Reproject GeoDataFrames to the target CRS.
    """
    print(f"Reprojecting GeoDataFrames to {target_crs}...")
    gdf_locations_proj = gdf_locations.to_crs(target_crs)
    stops_gdf_proj = stops_gdf.to_crs(target_crs)
    print("Reprojection completed.\n")
    return gdf_locations_proj, stops_gdf_proj


def find_nearby_routes(gdf_locations, stops_gdf, stop_times_trips_routes, buffer_distance_feet):
    """
    For each location, find unique routes within the buffer distance.
    """
    results = []
    for _, location in gdf_locations.iterrows():
        # Create a buffer around the location
        location_buffer = location.geometry.buffer(buffer_distance_feet)

        # Find stops within the buffer
        nearby_stops = stops_gdf[stops_gdf.geometry.within(location_buffer)]

        # Get stop_ids of nearby stops
        nearby_stop_ids = nearby_stops['stop_id'].unique()

        # Find the routes associated with these stops
        nearby_routes = stop_times_trips_routes[
            stop_times_trips_routes['stop_id'].isin(nearby_stop_ids)
        ]

        # Get unique route short names
        unique_routes = nearby_routes['route_short_name'].unique()

        # Prepare the result
        if unique_routes.size > 0:
            routes_str = ', '.join(unique_routes)
        else:
            routes_str = 'No routes'

        results.append({
            'Location': location['name'],
            'Routes': routes_str
        })
    return results


def save_results_to_csv(results, output_file):
    """
    Save the results to a CSV file.
    """
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results.to_csv(output_file, index=False)
    print(f"Results successfully saved to {output_file}")


def main():
    """
    Main function to execute the GTFS nearby routes analysis.
    """
    try:
        print("Checking input files...")
        check_input_files(GTFS_INPUT_PATH, GTFS_FILES)
        print("All input files are present.\n")

        print("Loading GTFS data...")
        data = load_gtfs_data(GTFS_INPUT_PATH, GTFS_FILES)
        print("GTFS data loaded successfully.\n")

        print("Creating GeoDataFrame for manual locations...")
        gdf_locations = create_geodataframe_locations(MANUAL_LOCATIONS)
        print("GeoDataFrame for locations created.\n")

        print("Creating GeoDataFrame for stops...")
        stops_gdf = create_geodataframe_stops(data['stops'])
        print("GeoDataFrame for stops created.\n")

        print("Reprojecting GeoDataFrames to projected CRS...")
        gdf_locations_proj, stops_gdf_proj = reproject_geodataframes(
            gdf_locations, stops_gdf, PROJECTED_CRS
        )

        # Convert buffer distance to feet
        buffer_distance_feet = convert_buffer_distance(BUFFER_DISTANCE, BUFFER_UNIT)
        print(
            f"Buffer distance set to {buffer_distance_feet} feet ({BUFFER_DISTANCE} {BUFFER_UNIT}).\n"
        )

        print("Merging stop_times with trips...")
        stop_times_trips = pd.merge(
            data['stop_times'], data['trips'], on='trip_id'
        )
        print(f"Merged stop_times with trips: {len(stop_times_trips)} records.\n")

        print("Merging with routes to associate routes with trips and stop times...")
        stop_times_trips_routes = pd.merge(
            stop_times_trips, data['routes'], on='route_id'
        )
        print(f"Merged with routes: {len(stop_times_trips_routes)} records.\n")

        print("Finding nearby routes for each location...")
        results = find_nearby_routes(
            gdf_locations_proj,
            stops_gdf_proj,
            stop_times_trips_routes,
            buffer_distance_feet
        )
        print("Nearby routes found for all locations.\n")

        # Define output file path
        output_file = os.path.join(OUTPUT_PATH, "nearby_routes.csv")
        print(f"Saving results to {output_file}...")
        save_results_to_csv(results, output_file)
        print("Process completed successfully!")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except ValueError as val_error:
        print(f"Value error: {val_error}")
    except Exception as error:
        print(f"An unexpected error occurred: {error}")


if __name__ == "__main__":
    main()
