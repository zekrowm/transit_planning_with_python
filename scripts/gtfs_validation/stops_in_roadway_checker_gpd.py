"""Identifies GTFS stops that intersect with roadway features and evaluates conflict depth.

This script performs a spatial analysis to find stops that fall within roadway geometries.
It uses negative buffering to assess how deeply stops are embedded in roadways and
outputs the results for review or remediation.

Inputs:
    - GTFS 'stops.txt' file
    - Roadway shapefile (.shp)
    - Input and analysis CRS strings
    - List of negative buffer distances (in feet)

Outputs:
    - Shapefile of intersecting stops with conflict depth flags
    - CSV with stop attributes and conflict indicators
"""

import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to input files
ROADWAYS_PATH = r"path\to\your\roadways.shp"  # Replace with your roadways shapefile path

GTFS_FOLDER = r"path\to\your\GTFS\folder"  # Replace with your GTFS folder path
STOPS_PATH = os.path.join(GTFS_FOLDER, "stops.txt")
OUTPUT_DIR = r"path\to\output\directory"  # Replace with your desired output directory

# Coordinate Reference Systems
STOPS_CRS = "EPSG:4326"  # WGS84 Latitude/Longitude
TARGET_CRS = "EPSG:2283"  # NAD83 / Virginia North

# Negative buffer distances in feet
BUFFER_DISTANCES = [-1, -5, -10]  # Adjust buffer distances as needed

# Output file names
OUTPUT_SHP_NAME = "intersecting_stops.shp"
OUTPUT_CSV_NAME = "intersecting_stops.csv"

# =============================================================================
# FUNCTIONS
# =============================================================================


def create_output_directory(output_dir):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def validate_stops_file_exists(stops_path):
    """Raises a FileNotFoundError if the stops.txt file is missing."""
    if not os.path.isfile(stops_path):
        raise FileNotFoundError(f"'stops.txt' not found at: {stops_path}")


def load_roadways(roadways_path):
    """Reads and returns the roadway shapefile as a GeoDataFrame."""
    return gpd.read_file(roadways_path)


def load_stops(stops_path):
    """Reads the stops.txt file into a DataFrame and converts it into a GeoDataFrame."""
    stops_df = pd.read_csv(stops_path)
    geometry = [Point(xy) for xy in zip(stops_df.stop_lon, stops_df.stop_lat)]
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry)
    return stops_gdf


def reproject_data(gdf, target_crs):
    """Reprojects a GeoDataFrame to the specified target CRS."""
    return gdf.to_crs(target_crs)


def find_intersecting_stops(stops_gdf, roadways_gdf):
    """Performs a spatial join to find stops that intersect with roadways.

    Returns only columns from stops_gdf.
    """
    intersecting = gpd.sjoin(stops_gdf, roadways_gdf, how="inner", predicate="intersects")
    return intersecting[stops_gdf.columns]


def add_xy_columns(gdf):
    """Adds 'x' and 'y' columns to a GeoDataFrame from its geometry."""
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def determine_conflict_depth(stops_gdf, roadways_gdf, buffer_distances):
    """Determines the depth of conflict between stops and roadways based on buffer distances.

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
        roadways_buffered["geometry"] = roadways_buffered.geometry.buffer(buffer_distance)

        # Remove invalid or empty geometries
        roadways_buffered = roadways_buffered[~roadways_buffered.is_empty]
        roadways_buffered = roadways_buffered[roadways_buffered.is_valid]

        # Spatial join to find stops that intersect the buffered roadways
        buffered_join = gpd.sjoin(
            stops_gdf,
            roadways_buffered[["geometry"]],
            how="left",
            predicate="intersects",
        )

        # Create a column to indicate whether the stop intersects the buffered roadways
        column_name = f"conflict_{-buffer_distance}ft"
        stops_gdf[column_name] = ~buffered_join["index_right"].isnull()

    return stops_gdf


def sort_stops_by_conflict_depth(stops_gdf, buffer_distances):
    """Sorts the stops by conflict depth columns in descending order."""
    conflict_columns = [f"conflict_{-bd}ft" for bd in buffer_distances]
    return stops_gdf.sort_values(by=conflict_columns, ascending=[False] * len(conflict_columns))


def save_shapefile(gdf, output_dir, shp_name):
    """Saves the GeoDataFrame to a shapefile in the specified output directory."""
    output_shp_path = os.path.join(output_dir, shp_name)
    gdf.to_file(output_shp_path)


def save_csv(gdf, output_dir, csv_name):
    """Saves the GeoDataFrame to a CSV file in the specified output directory."""
    output_csv_path = os.path.join(output_dir, csv_name)
    gdf.to_csv(output_csv_path, index=False)


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point for running the GTFS stop–roadway shapefile intersection checks."""
    # Create output directory if it doesn't exist
    create_output_directory(OUTPUT_DIR)

    # Validate stops.txt file existence
    validate_stops_file_exists(STOPS_PATH)

    # Read roadways shapefile
    roadways_gdf = load_roadways(ROADWAYS_PATH)

    # Read GTFS stops and create GeoDataFrame
    stops_gdf = load_stops(STOPS_PATH)
    stops_gdf.set_crs(STOPS_CRS, inplace=True)

    # Reproject both datasets to the target CRS
    roadways_gdf = reproject_data(roadways_gdf, TARGET_CRS)
    stops_gdf = reproject_data(stops_gdf, TARGET_CRS)

    # Find intersecting stops
    intersecting_stops = find_intersecting_stops(stops_gdf, roadways_gdf)

    # Add X and Y columns
    intersecting_stops = add_xy_columns(intersecting_stops)

    # Determine depth of conflict
    intersecting_stops = determine_conflict_depth(
        intersecting_stops, roadways_gdf, BUFFER_DISTANCES
    )

    # Sort by conflict depth
    intersecting_stops = sort_stops_by_conflict_depth(intersecting_stops, BUFFER_DISTANCES)

    # Save outputs
    save_shapefile(intersecting_stops, OUTPUT_DIR, OUTPUT_SHP_NAME)
    save_csv(intersecting_stops, OUTPUT_DIR, OUTPUT_CSV_NAME)

    print("Processing complete. Output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
