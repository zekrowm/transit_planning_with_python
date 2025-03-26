"""
Module for identifying GTFS routes near specified manual locations and determining the nearest stop for each unique
route-direction pair within a given buffer radius.

This module:
- Loads GTFS data (stops, trips, routes, stop_times).
- Creates GeoDataFrames from manual locations and GTFS stops.
- Reprojects spatial data to a specified projected coordinate reference system.
- Identifies nearby GTFS routes and the closest stops for each route-direction combination.
- Outputs results including both route identifiers and associated nearest stop IDs to a CSV file for further analysis.

The script is customizable via a configuration section for paths, buffer size, and spatial references.
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
    {"name": "Braddock", "latitude": 38.813545, "longitude": -77.053864},
    {"name": "Crystal City", "latitude": 38.85835, "longitude": -77.051232}
]

# Buffer radius configuration
# Specify the buffer distance and its unit ('miles' or 'feet')
BUFFER_DISTANCE = 0.25 # Replace with your desired buffer distance
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


def find_nearby_routes_with_nearest_stops(gdf_locations, stops_gdf, stop_times_trips_routes, buffer_distance_feet):
    """
    For each location, identify all unique (route_short_name + direction_id) pairs
    within the buffer, then find the single nearest stop to the location for each pair.
    Finally, return both the list of routes (e.g., '598, 599') and the list of
    associated nearest stop IDs (e.g., '1056, 2869').
    """
    results = []
    for _, location in gdf_locations.iterrows():
        # Create a buffer around the location
        location_buffer = location.geometry.buffer(buffer_distance_feet)

        # Filter stops to only those within the buffer
        nearby_stops = stops_gdf[stops_gdf.geometry.within(location_buffer)]
        if nearby_stops.empty:
            # No stops at all in this location's buffer
            results.append({
                'Location': location['name'],
                'Routes': 'No routes',
                'Stops': 'No stops'
            })
            continue

        # Subset stop_times_trips_routes to only those with stop_ids in nearby_stops
        nearby_stop_ids = nearby_stops['stop_id'].unique()
        df_nearby_routes = stop_times_trips_routes[
            stop_times_trips_routes['stop_id'].isin(nearby_stop_ids)
        ]

        if df_nearby_routes.empty:
            # No routes found
            results.append({
                'Location': location['name'],
                'Routes': 'No routes',
                'Stops': 'No stops'
            })
            continue

        # Merge the subset of stops to get geometry for distance calculations
        merged_stops = pd.merge(
            nearby_stops[['stop_id', 'geometry']],
            df_nearby_routes[['stop_id', 'route_short_name', 'direction_id']],
            on='stop_id'
        ).drop_duplicates()

        # Compute distance from this location's geometry to each stop
        merged_stops['distance'] = merged_stops['geometry'].distance(location.geometry)

        # Group by route_short_name + direction_id, pick the stop with minimum distance
        # for each (route, direction).
        # As an example, we handle direction_id to ensure we get unique route+direction pairs.
        grouped = merged_stops.groupby(['route_short_name', 'direction_id'], as_index=False)
        nearest_stops = grouped.apply(lambda x: x.loc[x['distance'].idxmin()])

        # Build final lists for routes and stops
        # (One stop ID per route+direction pair)
        route_list = nearest_stops['route_short_name'].astype(str).unique().tolist()
        stop_list = nearest_stops['stop_id'].astype(str).unique().tolist()

        # Prepare string output (or you could store as lists, etc.)
        if len(route_list) > 0:
            routes_str = ', '.join(route_list)
            stops_str = ', '.join(stop_list)
        else:
            routes_str = 'No routes'
            stops_str = 'No stops'

        results.append({
            'Location': location['name'],
            'Routes': routes_str,
            'Stops': stops_str
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
            f"Buffer distance set to {buffer_distance_feet} feet "
            f"({BUFFER_DISTANCE} {BUFFER_UNIT}).\n"
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

        print("Finding nearby routes and nearest stops for each location...")
        results = find_nearby_routes_with_nearest_stops(
            gdf_locations_proj,
            stops_gdf_proj,
            stop_times_trips_routes,
            buffer_distance_feet
        )
        print("Nearby routes found for all locations.\n")

        # Convert to DataFrame and display or save
        df_results = pd.DataFrame(results)
        print(df_results)

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
