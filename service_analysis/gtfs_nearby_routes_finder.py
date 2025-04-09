"""
Module for identifying GTFS routes near specified manual locations OR for user-specified stop_codes.
Supports two modes:
  1) 'location': Buffers a provided lat/lon coordinate (or multiple points), finds all stops within that buffer,
     and retrieves the associated routes/directions.
  2) 'stop_code': Directly uses provided stop_codes to retrieve route/direction information (no spatial processing).
"""
import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

GTFS_FOLDER = r"\\your_file_path\here"
OUTPUT_FOLDER = r"\\your_file_path\here"

# Choose "location" OR "stop_code"
INPUT_MODE = "location"

# For 'location' mode
MANUAL_LOCATIONS = [
    {"name": "Braddock", "latitude": 38.813545, "longitude": -77.053864},
    {"name": "Crystal City", "latitude": 38.85835, "longitude": -77.051232}
]
BUFFER_DISTANCE = 0.25  # 0.25 miles (example)
BUFFER_UNIT = "miles"   # or 'feet'
PROJECTED_CRS = "EPSG:2232"  # e.g. NAD83 / DC State Plane (US Feet)

# For 'stop_code' mode
STOP_CODE_FILTER = ["1001", "1002", "1003"]  # example stop_codes

# Output file name
OUTPUT_FILE_NAME = "results.csv"

# ==============================
# END CONFIGURATION
# ==============================

def check_input_files(base_path):
    """
    Verify that the standard GTFS files exist in the specified directory.
    Required files: stops.txt, stop_times.txt, trips.txt, routes.txt.
    """
    required_files = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The input directory {base_path} does not exist.")
    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The required GTFS file {file_name} does not exist in {base_path}."
            )

def load_gtfs_data(base_path):
    """
    Load the standard GTFS data files into Pandas DataFrames from the specified folder.
    Returns a dictionary with keys: "stops", "stop_times", "trips", "routes".
    """
    required_files = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]
    data = {}

    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        # Use the file name minus extension as the dict key, e.g. "stops", "trips", etc.
        dict_key = file_name.split(".")[0]
        try:
            data[dict_key] = pd.read_csv(file_path, dtype=str)
            print(f"Loaded {file_name} with {len(data[dict_key])} records.")
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
    stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
    stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)
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
    """
    results = []
    for _, location in gdf_locations.iterrows():
        loc_name = location.get('name', '(unnamed)')

        # Create a buffer around the location
        location_buffer = location.geometry.buffer(buffer_distance_feet)

        # Filter stops to only those within the buffer
        nearby_stops = stops_gdf[stops_gdf.geometry.within(location_buffer)]
        if nearby_stops.empty:
            results.append({
                'Location': loc_name,
                'Routes': 'No routes',
                'Stops': 'No stops'
            })
            continue

        # Subset relevant stop_ids
        nearby_stop_ids = nearby_stops['stop_id'].unique()
        df_nearby_routes = stop_times_trips_routes[
            stop_times_trips_routes['stop_id'].isin(nearby_stop_ids)
        ]

        if df_nearby_routes.empty:
            results.append({
                'Location': loc_name,
                'Routes': 'No routes',
                'Stops': 'No stops'
            })
            continue

        # Merge geometry back so we can compute distance
        merged_stops = pd.merge(
            nearby_stops[['stop_id', 'geometry']],
            df_nearby_routes[['stop_id', 'route_short_name', 'direction_id']],
            on='stop_id'
        ).drop_duplicates()

        # Compute distance from this location's geometry to each stop
        merged_stops['distance'] = merged_stops['geometry'].distance(location.geometry)

        # Group by route_short_name + direction_id, pick the stop with minimum distance
        grouped = merged_stops.groupby(['route_short_name', 'direction_id'], as_index=False)
        nearest_stops = grouped.apply(lambda x: x.loc[x['distance'].idxmin()])

        # Build final lists for routes and stops
        route_list = nearest_stops['route_short_name'].astype(str).unique().tolist()
        stop_list = nearest_stops['stop_id'].astype(str).unique().tolist()

        routes_str = ', '.join(route_list) if route_list else 'No routes'
        stops_str = ', '.join(stop_list) if stop_list else 'No stops'

        results.append({
            'Location': loc_name,
            'Routes': routes_str,
            'Stops': stops_str
        })

    return results

# ---------------------------------------------------------------------
#  HELPER FUNCTIONS FOR STOP_CODE-BASED LOOKUPS
# ---------------------------------------------------------------------
def get_stop_ids_for_stop_codes(stops_df, stop_code_filter):
    """
    Filter the stops DataFrame to include only those stops with a stop_code
    in the filter and return the matching stop_ids.
    """
    if 'stop_code' not in stops_df.columns:
        raise ValueError("stops.txt does not have a 'stop_code' column.")

    filtered_stops = stops_df[stops_df["stop_code"].isin(stop_code_filter)]
    return filtered_stops["stop_id"].unique().tolist()

def find_routes_by_stop_ids(stops_df, stop_ids, stop_times_df, trips_df, routes_df):
    """
    Return a list of dictionaries, each describing the stop_code, stop_name,
    and associated route/direction pairs.
    """
    if not stop_ids:
        return []

    # Filter stop_times to only include the specified stop_ids
    filtered_stop_times = stop_times_df[stop_times_df["stop_id"].isin(stop_ids)]

    # Merge stop_times with trips to get route_id and direction_id
    st_trips = pd.merge(filtered_stop_times, trips_df, on="trip_id", how="inner")

    # Merge with routes to get route_short_name
    st_trips_routes = pd.merge(st_trips, routes_df, on="route_id", how="inner")

    # Build a dictionary: stop_id -> set of (route_short_name, direction_id)
    routes_by_stop_id = {}
    for _, row in st_trips_routes.iterrows():
        sid = row["stop_id"]
        route_short_name = row["route_short_name"]
        direction_id = row["direction_id"]
        routes_by_stop_id.setdefault(sid, set()).add((route_short_name, direction_id))

    # Create a small index for stop_code / stop_name by stop_id
    stops_index = stops_df.set_index("stop_id").to_dict("index")

    results = []
    for sid, route_pairs in routes_by_stop_id.items():
        row_info = stops_index.get(sid, {})
        scode = row_info.get("stop_code", "N/A")
        sname = row_info.get("stop_name", "N/A")
        # Convert the route/direction pairs into sorted lists
        route_list = []
        for (route_sn, dir_id) in sorted(route_pairs):
            route_list.append(f"{route_sn} (direction {dir_id})")

        results.append({
            "Stop_ID": sid,
            "Stop_Code": scode,
            "Stop_Name": sname,
            "Routes": "; ".join(route_list) if route_list else "No routes"
        })
    return results

def save_results_to_csv(results, output_file):
    """
    Save the results to a CSV file.
    """
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Results successfully saved to {output_file}")

def main():
    """
    Main function to execute either:
      - 'location' mode: Identify nearby routes based on lat/lon buffers, or
      - 'stop_code' mode: Retrieve routes/directions for specific stops by code.
    """
    try:
        print("Checking input files...")
        check_input_files(GTFS_FOLDER)
        print("All input files are present.\n")

        print("Loading GTFS data...")
        data = load_gtfs_data(GTFS_FOLDER)
        print("GTFS data loaded successfully.\n")

        print("Merging stop_times with trips, then with routes...")
        stop_times_trips = pd.merge(data['stop_times'], data['trips'], on='trip_id', how='inner')
        stop_times_trips_routes = pd.merge(stop_times_trips, data['routes'], on='route_id', how='inner')
        print(f"Combined stop_times+trips+routes: {len(stop_times_trips_routes)} records.\n")

        # Decide which approach to run
        if INPUT_MODE == "location":
            print("== LOCATION MODE SELECTED ==\n")

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
            print(f"Buffer distance set to {buffer_distance_feet} feet "
                  f"({BUFFER_DISTANCE} {BUFFER_UNIT}).\n")

            print("Finding nearby routes and nearest stops for each location...")
            results = find_nearby_routes_with_nearest_stops(
                gdf_locations_proj, stops_gdf_proj, stop_times_trips_routes, buffer_distance_feet
            )
            print("Nearby routes found for all locations.\n")

        elif INPUT_MODE == "stop_code":
            print("== STOP_CODE MODE SELECTED ==\n")

            print(f"Filtering stops for stop_codes: {STOP_CODE_FILTER}...")
            stops_df = data["stops"].copy()
            matched_stop_ids = get_stop_ids_for_stop_codes(stops_df, STOP_CODE_FILTER)
            if not matched_stop_ids:
                print("No stops matched the provided stop_code filter.")
                return

            print(f"Found {len(matched_stop_ids)} matching stop(s) for the given stop_codes.\n")

            print("Finding routes for provided stop_codes...")
            results = find_routes_by_stop_ids(
                stops_df=stops_df,
                stop_ids=matched_stop_ids,
                stop_times_df=data["stop_times"],
                trips_df=data["trips"],
                routes_df=data["routes"]
            )
            print("Routes/directions lookup completed.\n")

        else:
            raise ValueError("Invalid INPUT_MODE. Use 'location' or 'stop_code'.")

        if not results:
            print("No results found.")
        else:
            output_file = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_NAME)
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
