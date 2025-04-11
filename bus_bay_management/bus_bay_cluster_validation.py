"""
GTFS Bus Bay Cluster Validation

This module validates GTFS bus arrival data by checking clusters of bus stops
for consistency. Planners can use this to check proposed field check bus stop
clusters or bus bay conflict check clusters. It identifies potential errors
like similar stop names, nearby excluded stops, distant included stops,
and stops with different names.

Results are exported as Excel files and shapefiles for further analysis
and visual inspection.
"""

import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rapidfuzz import fuzz, process

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input directory for GTFS files
BASE_INPUT_PATH = r'\\your_input_path\here\\'

# Output directory for results
BASE_OUTPUT_PATH = r'\\your_output_path\here\\'

# Define clusters with provided stop IDs
# Format: {'Cluster Name': [stop_id1, stop_id2, ...]}
# You can leave the clusters dictionary empty if you're not ready to define clusters.
# The script will still run and export all stops for visual inspection.
clusters = {
    # 'Downtown Bus Station': [1, 2, 3],
    # 'Airport Terminal': [4, 5, 6],
    # Add more clusters as needed
}

# Thresholds for analysis
SIMILARITY_THRESHOLD = 85       # For finding similar stop names (0-100)
DISTANCE_THRESHOLD_NEARBY = 200 # Distance in feet for nearby stops
DISTANCE_THRESHOLD_DISTANT = 1000  # Distance in feet for distant included stops
SIMILARITY_THRESHOLD_NAMES = 70  # For included stops with different names (0-100)

# CRS for distance calculations (Projected CRS).
# NOTE:
#  - Choose a CRS appropriate for your city/region for accurate distance calculations.
#  - Avoid using Web Mercator (EPSG:3857). Example used here: NAD83 / Metro DC area (feet).
DISTANCE_CRS_EPSG = 2248  # NAD83 / Maryland (ft)

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------

# Create output directory if it doesn't exist
if not os.path.exists(BASE_OUTPUT_PATH):
    os.makedirs(BASE_OUTPUT_PATH)

# Load GTFS stops.txt file
stops_file = os.path.join(BASE_INPUT_PATH, 'stops.txt')
if not os.path.exists(stops_file):
    raise FileNotFoundError(
        f"stops.txt not found in {BASE_INPUT_PATH}"
    )

stops = pd.read_csv(stops_file)

# Convert stop_id to string
stops['stop_id'] = stops['stop_id'].astype(str)

# Convert stops to GeoDataFrame
stops['geometry'] = stops.apply(
    lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1
)
stops_gdf = gpd.GeoDataFrame(stops, geometry='geometry', crs='EPSG:4326')

# Reproject to a projected CRS for accurate distance calculations
stops_gdf = stops_gdf.to_crs(epsg=DISTANCE_CRS_EPSG)

# Initialize clusters if not provided
if not clusters:
    print("No clusters defined. Proceeding without cluster analysis.")
    # Create empty GeoDataFrames for included and excluded stops
    included_stops_global = gpd.GeoDataFrame(columns=stops_gdf.columns)
    excluded_stops_global = stops_gdf.copy()
else:
    # Convert cluster stop IDs to strings
    clusters = {
        cluster_name: [str(stop_id) for stop_id in stop_ids]
        for cluster_name, stop_ids in clusters.items()
    }

    # Create a list of included stop IDs
    included_stop_ids = [
        stop_id for ids in clusters.values() for stop_id in ids
    ]
    included_stops_global = stops_gdf[
        stops_gdf['stop_id'].isin(included_stop_ids)
    ].copy()
    included_stops_global['cluster'] = None

    # Assign cluster names to included stops
    for cluster_name, stop_ids in clusters.items():
        included_stops_global.loc[
            included_stops_global['stop_id'].isin(stop_ids), 'cluster'
        ] = cluster_name

    # Create a GeoDataFrame for excluded stops
    excluded_stops_global = stops_gdf[
        ~stops_gdf['stop_id'].isin(included_stop_ids)
    ].copy()

    # Reset index to ensure a clean DataFrame
    excluded_stops_global = excluded_stops_global.reset_index(drop=True)


def find_similar_stop_names(
    inc_stops, exc_stops, threshold=SIMILARITY_THRESHOLD
):
    """
    Find excluded stops whose names are similar to included stops, above a threshold.

    Parameters
    ----------
    inc_stops : GeoDataFrame
        GeoDataFrame of included stops.
    exc_stops : GeoDataFrame
        GeoDataFrame of excluded stops.
    threshold : int, optional
        Similarity threshold (default = SIMILARITY_THRESHOLD).

    Returns
    -------
    pandas.DataFrame
        DataFrame listing pairs of included stop names, excluded stop names,
        and similarity scores above the threshold.
    """
    if inc_stops.empty or exc_stops.empty:
        return pd.DataFrame(
            columns=[
                'included_stop_name', 'excluded_stop_name', 'similarity_score',
                'stop_id', 'stop_lat', 'stop_lon'
            ]
        )
    # Convert stop names to lowercase for case-insensitive comparison
    inc_stops['stop_name_lower'] = inc_stops['stop_name'].str.lower()
    exc_stops['stop_name_lower'] = exc_stops['stop_name'].str.lower()

    similar_stops = []
    included_names = inc_stops['stop_name_lower'].unique()

    for name in included_names:
        matches = process.extract(
            name,
            exc_stops['stop_name_lower'],
            scorer=fuzz.token_sort_ratio,
            limit=None
        )
        for _, score, idx in matches:
            if score >= threshold:
                similar_stop = exc_stops.iloc[idx]
                original_included_name = inc_stops[
                    inc_stops['stop_name_lower'] == name
                ]['stop_name'].iloc[0]
                similar_stops.append({
                    'included_stop_name': original_included_name,
                    'excluded_stop_name': similar_stop['stop_name'],
                    'similarity_score': score,
                    'stop_id': similar_stop['stop_id'],
                    'stop_lat': similar_stop['stop_lat'],
                    'stop_lon': similar_stop['stop_lon']
                })

    columns = [
        'included_stop_name', 'excluded_stop_name', 'similarity_score',
        'stop_id', 'stop_lat', 'stop_lon'
    ]
    return pd.DataFrame(similar_stops, columns=columns)


def find_nearby_excluded_stops(
    inc_stops, exc_stops, distance_threshold=DISTANCE_THRESHOLD_NEARBY
):
    """
    Find excluded stops that are physically close to included stops, within a threshold.

    Parameters
    ----------
    inc_stops : GeoDataFrame
        GeoDataFrame of included stops.
    exc_stops : GeoDataFrame
        GeoDataFrame of excluded stops.
    distance_threshold : float, optional
        Distance threshold in feet (default = DISTANCE_THRESHOLD_NEARBY).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing nearby excluded stops with distance to the included stop.
    """
    if inc_stops.empty or exc_stops.empty:
        return pd.DataFrame(
            columns=[
                'included_stop_id', 'included_stop_name',
                'excluded_stop_id', 'excluded_stop_name', 'distance_m'
            ]
        )

    nearby_stops = []
    for _, included_stop in inc_stops.iterrows():
        # Buffer the included stop by the distance threshold
        buffer_geom = included_stop.geometry.buffer(distance_threshold)
        # Find excluded stops within that buffer
        nearby = exc_stops[exc_stops.geometry.within(buffer_geom)]
        for _, nearby_stop in nearby.iterrows():
            distance = included_stop.geometry.distance(nearby_stop.geometry)
            nearby_stops.append({
                'included_stop_id': included_stop['stop_id'],
                'included_stop_name': included_stop['stop_name'],
                'excluded_stop_id': nearby_stop['stop_id'],
                'excluded_stop_name': nearby_stop['stop_name'],
                'distance_m': distance
            })

    columns = [
        'included_stop_id', 'included_stop_name',
        'excluded_stop_id', 'excluded_stop_name', 'distance_m'
    ]
    return pd.DataFrame(nearby_stops, columns=columns)


def find_distant_included_stops(
    inc_stops, distance_threshold=DISTANCE_THRESHOLD_DISTANT
):
    """
    Find included stops that are far from all other stops in the same cluster.

    Parameters
    ----------
    inc_stops : GeoDataFrame
        GeoDataFrame of included stops.
    distance_threshold : float, optional
        Distance threshold in feet (default = DISTANCE_THRESHOLD_DISTANT).

    Returns
    -------
    pandas.DataFrame
        DataFrame listing included stops whose minimum distance to cluster
        siblings exceeds the threshold.
    """
    if inc_stops.empty:
        return pd.DataFrame(
            columns=[
                'stop_id', 'stop_name', 'cluster', 'min_distance_to_cluster_m'
            ]
        )

    distant_stops = []
    clusters_list = inc_stops['cluster'].unique()

    for cluster in clusters_list:
        cluster_stops = inc_stops[inc_stops['cluster'] == cluster]
        for _, stop in cluster_stops.iterrows():
            other_stops = cluster_stops.drop(stop.name)
            if other_stops.empty:
                continue
            distances = other_stops.geometry.distance(stop.geometry)
            if distances.min() > distance_threshold:
                distant_stops.append({
                    'stop_id': stop['stop_id'],
                    'stop_name': stop['stop_name'],
                    'cluster': cluster,
                    'min_distance_to_cluster_m': distances.min()
                })

    columns = [
        'stop_id', 'stop_name', 'cluster', 'min_distance_to_cluster_m'
    ]
    return pd.DataFrame(distant_stops, columns=columns)


def find_different_named_included_stops(
    inc_stops, similarity_threshold=SIMILARITY_THRESHOLD_NAMES
):
    """
    Find included stops in the same cluster whose name similarity is below a threshold.

    Parameters
    ----------
    inc_stops : GeoDataFrame
        GeoDataFrame of included stops.
    similarity_threshold : int, optional
        Name similarity threshold (default = SIMILARITY_THRESHOLD_NAMES).

    Returns
    -------
    pandas.DataFrame
        DataFrame of included stops whose maximum name similarity in the cluster
        is below the threshold.
    """
    if inc_stops.empty:
        return pd.DataFrame(
            columns=[
                'stop_id', 'stop_name', 'cluster', 'max_similarity_score'
            ]
        )

    different_named_stops = []
    clusters_list = inc_stops['cluster'].unique()

    for cluster in clusters_list:
        cluster_stops = inc_stops[inc_stops['cluster'] == cluster]
        names = cluster_stops['stop_name'].unique()

        for _, stop in cluster_stops.iterrows():
            similarities = [
                fuzz.token_sort_ratio(stop['stop_name'], name)
                for name in names if name != stop['stop_name']
            ]
            if similarities and max(similarities) < similarity_threshold:
                different_named_stops.append({
                    'stop_id': stop['stop_id'],
                    'stop_name': stop['stop_name'],
                    'cluster': cluster,
                    'max_similarity_score': max(similarities)
                })

    columns = [
        'stop_id', 'stop_name', 'cluster', 'max_similarity_score'
    ]
    return pd.DataFrame(different_named_stops, columns=columns)


# Perform analysis only if clusters are defined
if clusters:
    # Find excluded stops with similar names
    similar_name_stops = find_similar_stop_names(
        included_stops_global, excluded_stops_global,
        threshold=SIMILARITY_THRESHOLD
    )

    print(f"Number of similar name stops found: {len(similar_name_stops)}")

    # Find excluded stops that are physically close to included stops
    nearby_excluded_stops = find_nearby_excluded_stops(
        included_stops_global, excluded_stops_global,
        distance_threshold=DISTANCE_THRESHOLD_NEARBY
    )
    print(f"Number of nearby excluded stops found: {len(nearby_excluded_stops)}")

    # Find included stops that are distant from their cluster
    distant_included_stops = find_distant_included_stops(
        included_stops_global, distance_threshold=DISTANCE_THRESHOLD_DISTANT
    )
    print(f"Number of distant included stops found: {len(distant_included_stops)}")

    # Find included stops with different names
    different_named_included_stops = find_different_named_included_stops(
        included_stops_global, similarity_threshold=SIMILARITY_THRESHOLD_NAMES
    )
    print(
        "Number of different named included stops found: "
        f"{len(different_named_included_stops)}"
    )
else:
    # If clusters are not defined, create empty DataFrames for outputs
    similar_name_stops = pd.DataFrame()
    nearby_excluded_stops = pd.DataFrame()
    distant_included_stops = pd.DataFrame()
    different_named_included_stops = pd.DataFrame()

output_directory = os.path.join(BASE_OUTPUT_PATH, 'stops_check')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def save_to_excel(data_frame, filename):
    """
    Save a DataFrame to Excel, ensuring headers are written even if DataFrame is empty.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The DataFrame to be saved.
    filename : str
        Name of the output Excel file (without any path).
    """
    file_path = os.path.join(output_directory, filename)

    # If empty, create a DataFrame with relevant columns so headers appear
    if data_frame.empty:
        if 'similar_names' in filename:
            columns = [
                'included_stop_name', 'excluded_stop_name',
                'similarity_score', 'stop_id', 'stop_lat', 'stop_lon'
            ]
        elif 'nearby' in filename:
            columns = [
                'included_stop_id', 'included_stop_name',
                'excluded_stop_id', 'excluded_stop_name', 'distance_m'
            ]
        elif 'distant' in filename:
            columns = [
                'stop_id', 'stop_name', 'cluster',
                'min_distance_to_cluster_m'
            ]
        elif 'different_names' in filename:
            columns = [
                'stop_id', 'stop_name', 'cluster', 'max_similarity_score'
            ]
        else:
            columns = data_frame.columns
        data_frame = pd.DataFrame(columns=columns)

    data_frame.to_excel(file_path, index=False, header=True)


# Save the DataFrames
save_to_excel(similar_name_stops, 'excluded_stops_similar_names.xlsx')
save_to_excel(nearby_excluded_stops, 'excluded_stops_nearby.xlsx')
save_to_excel(distant_included_stops, 'included_stops_distant.xlsx')
save_to_excel(different_named_included_stops, 'included_stops_different_names.xlsx')

# Export included and excluded stops as shapefiles for visual inspection
included_stops_shp = os.path.join(output_directory, 'included_stops.shp')
excluded_stops_shp = os.path.join(output_directory, 'excluded_stops.shp')
all_stops_shp = os.path.join(output_directory, 'all_stops.shp')

if not included_stops_global.empty:
    included_stops_global.to_file(included_stops_shp)
else:
    print("No included stops to export.")

if not excluded_stops_global.empty:
    excluded_stops_global.to_file(excluded_stops_shp)
else:
    print("No excluded stops to export.")

# Export all stops
stops_gdf.to_file(all_stops_shp)

print("Stops check completed. Results have been saved to the 'stops_check' directory.")
