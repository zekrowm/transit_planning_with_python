"""
Script Name:
    bus_bay_cluster_validation.py

Purpose:
    Validates predefined clusters of GTFS bus stops for accuracy.
    Identifies potential issues like nearby excluded stops, distant
    included stops, and stops with dissimilar names within clusters.

Inputs:
    1. GTFS 'stops.txt' file (path configured in GTFS_FOLDER_PATH).
    2. 'clusters' dictionary defining stop groupings (in script config).
    3. Configuration constants (thresholds, CRS) in script config.

Outputs:
    1. Excel files detailing potential validation issues found.
    2. Shapefiles ('included_stops.shp', 'excluded_stops.shp',
       'all_stops.shp') for spatial analysis.
    3. (Outputs saved to a 'stops_check' subdirectory in BASE_OUTPUT_PATH)

Dependencies:
    1. Libraries: os, geopandas, pandas, rapidfuzz, shapely
"""

import os

import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz, process
from shapely.geometry import Point

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

GTFS_FOLDER_PATH = r"\\your_GTFS_folder_path\here\\"

BASE_OUTPUT_PATH = r"\\your_output_folder_path\here\\"

clusters = {
    # Example:
    # 'Downtown Bus Station': ['1', '2', '3'], # Your stop_id codes here
    # 'Airport Terminal': ['4', '5', '6'],
}

SIMILARITY_THRESHOLD = 85
DISTANCE_THRESHOLD_NEARBY = 200
DISTANCE_THRESHOLD_DISTANT = 1000
SIMILARITY_THRESHOLD_NAMES = 70

DISTANCE_CRS_EPSG = 2248  # NAD83 / Maryland (ft)

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------------------------------------------------------


def load_gtfs_data(files=None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from a path defined externally
    (GTFS_FOLDER_PATH).

    Parameters:
        files (list[str], optional): GTFS filenames to load. Default is all
            standard GTFS files:
            [
                "agency.txt",
                "stops.txt",
                "routes.txt",
                "trips.txt",
                "stop_times.txt",
                "calendar.txt",
                "calendar_dates.txt",
                "fare_attributes.txt",
                "fare_rules.txt",
                "feed_info.txt",
                "frequencies.txt",
                "shapes.txt",
                "transfers.txt"
            ]
        dtype (str or dict, optional): Pandas dtype to use. Default is str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        FileNotFoundError: If GTFS_FOLDER_PATH doesn't exist or if any required file is missing.
        ValueError: If a file is empty or there's a parsing error.
        RuntimeError: For any unexpected error during loading.
    """
    if not os.path.exists(GTFS_FOLDER_PATH):
        raise FileNotFoundError(f"The directory '{GTFS_FOLDER_PATH}' does not exist.")

    if files is None:
        files = [
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
            "fare_attributes.txt",
            "fare_rules.txt",
            "feed_info.txt",
            "frequencies.txt",
            "shapes.txt",
            "transfers.txt",
        ]

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(GTFS_FOLDER_PATH, file_name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing GTFS files in '{GTFS_FOLDER_PATH}': {', '.join(missing)}"
        )

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(GTFS_FOLDER_PATH, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype)
            data[key] = df
            print(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(f"Parser error in '{file_name}': {exc}") from exc

        except Exception as exc:
            # Use a more specific error class than bare Exception (e.g., RuntimeError)
            raise RuntimeError(f"Error loading '{file_name}': {exc}") from exc

    return data


# --------------------------------------------------------------------------------------------------
# REGULAR FUNCTIONS
# --------------------------------------------------------------------------------------------------


def prepare_stops_gdf(crs_epsg):
    """
    Load GTFS 'stops.txt' as a DataFrame using load_gtfs_data, then convert
    to a GeoDataFrame (and reproject to crs_epsg).
    Returns the GeoDataFrame.
    """
    # Use the new function to load stops.txt
    gtfs_data = load_gtfs_data(files=["stops.txt"], dtype=str)
    stops_df = gtfs_data["stops"]

    # Convert stop_id to string, build geometry, and reproject
    stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    stops_df["geometry"] = stops_df.apply(
        lambda row: Point(float(row["stop_lon"]), float(row["stop_lat"])), axis=1
    )
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry="geometry", crs="EPSG:4326")
    stops_gdf = stops_gdf.to_crs(epsg=crs_epsg)

    return stops_gdf


def initialize_clusters(stops_gdf, clusters_dict):
    """
    Splits stops into included and excluded sets based on cluster definitions.

    Returns:
        included_stops_global (GeoDataFrame),
        excluded_stops_global (GeoDataFrame),
        updated_clusters_dict (dict, with stop IDs as strings).
    """
    if not clusters_dict:
        print("No clusters defined. Proceeding without cluster analysis.")
        included_stops_global = gpd.GeoDataFrame(columns=stops_gdf.columns)
        excluded_stops_global = stops_gdf.copy()
    else:
        # Convert cluster stop IDs to strings
        clusters_dict = {
            cluster_name: [str(stop_id) for stop_id in stop_ids]
            for cluster_name, stop_ids in clusters_dict.items()
        }

        included_stop_ids = [
            stop_id
            for stop_id_list in clusters_dict.values()
            for stop_id in stop_id_list
        ]

        included_stops_global = stops_gdf[
            stops_gdf["stop_id"].isin(included_stop_ids)
        ].copy()
        included_stops_global["cluster"] = None

        # Assign cluster names
        for cluster_name, stop_ids in clusters_dict.items():
            included_stops_global.loc[
                included_stops_global["stop_id"].isin(stop_ids), "cluster"
            ] = cluster_name

        excluded_stops_global = stops_gdf[
            ~stops_gdf["stop_id"].isin(included_stop_ids)
        ].copy()
        excluded_stops_global.reset_index(drop=True, inplace=True)

    return included_stops_global, excluded_stops_global, clusters_dict


def find_similar_stop_names(inc_stops, exc_stops, threshold):
    """
    Find excluded stops whose names are similar to included stops above a threshold.
    """
    if inc_stops.empty or exc_stops.empty:
        return pd.DataFrame(
            columns=[
                "included_stop_name",
                "excluded_stop_name",
                "similarity_score",
                "stop_id",
                "stop_lat",
                "stop_lon",
            ]
        )

    # Convert stop names to lowercase for case-insensitive comparison
    inc_stops["stop_name_lower"] = inc_stops["stop_name"].str.lower()
    exc_stops["stop_name_lower"] = exc_stops["stop_name"].str.lower()

    similar_stops = []
    included_names = inc_stops["stop_name_lower"].unique()

    for name in included_names:
        matches = process.extract(
            name, exc_stops["stop_name_lower"], scorer=fuzz.token_sort_ratio, limit=None
        )
        for _, score, idx in matches:
            if score >= threshold:
                similar_stop = exc_stops.iloc[idx]
                original_included_name = inc_stops[
                    inc_stops["stop_name_lower"] == name
                ]["stop_name"].iloc[0]
                similar_stops.append(
                    {
                        "included_stop_name": original_included_name,
                        "excluded_stop_name": similar_stop["stop_name"],
                        "similarity_score": score,
                        "stop_id": similar_stop["stop_id"],
                        "stop_lat": similar_stop["stop_lat"],
                        "stop_lon": similar_stop["stop_lon"],
                    }
                )

    columns = [
        "included_stop_name",
        "excluded_stop_name",
        "similarity_score",
        "stop_id",
        "stop_lat",
        "stop_lon",
    ]
    return pd.DataFrame(similar_stops, columns=columns)


def find_nearby_excluded_stops(inc_stops, exc_stops, distance_threshold):
    """
    Find excluded stops that are within distance_threshold from included stops.
    """
    if inc_stops.empty or exc_stops.empty:
        return pd.DataFrame(
            columns=[
                "included_stop_id",
                "included_stop_name",
                "excluded_stop_id",
                "excluded_stop_name",
                "distance_m",
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
            nearby_stops.append(
                {
                    "included_stop_id": included_stop["stop_id"],
                    "included_stop_name": included_stop["stop_name"],
                    "excluded_stop_id": nearby_stop["stop_id"],
                    "excluded_stop_name": nearby_stop["stop_name"],
                    "distance_m": distance,
                }
            )

    columns = [
        "included_stop_id",
        "included_stop_name",
        "excluded_stop_id",
        "excluded_stop_name",
        "distance_m",
    ]
    return pd.DataFrame(nearby_stops, columns=columns)


def find_distant_included_stops(inc_stops, distance_threshold):
    """
    Find included stops that are far from all other stops in their cluster.
    """
    if inc_stops.empty:
        return pd.DataFrame(
            columns=["stop_id", "stop_name", "cluster", "min_distance_to_cluster_m"]
        )

    distant_stops = []
    clusters_list = inc_stops["cluster"].unique()

    for cluster in clusters_list:
        cluster_stops = inc_stops[inc_stops["cluster"] == cluster]
        for _, stop in cluster_stops.iterrows():
            other_stops = cluster_stops.drop(stop.name)
            if other_stops.empty:
                continue
            distances = other_stops.geometry.distance(stop.geometry)
            if distances.min() > distance_threshold:
                distant_stops.append(
                    {
                        "stop_id": stop["stop_id"],
                        "stop_name": stop["stop_name"],
                        "cluster": cluster,
                        "min_distance_to_cluster_m": distances.min(),
                    }
                )

    columns = ["stop_id", "stop_name", "cluster", "min_distance_to_cluster_m"]
    return pd.DataFrame(distant_stops, columns=columns)


def find_different_named_included_stops(inc_stops, similarity_threshold):
    """
    Find included stops in the same cluster whose name similarity is below similarity_threshold.
    """
    if inc_stops.empty:
        return pd.DataFrame(
            columns=["stop_id", "stop_name", "cluster", "max_similarity_score"]
        )

    different_named_stops = []
    clusters_list = inc_stops["cluster"].unique()

    for cluster in clusters_list:
        cluster_stops = inc_stops[inc_stops["cluster"] == cluster]
        names = cluster_stops["stop_name"].unique()

        for _, stop in cluster_stops.iterrows():
            similarities = [
                fuzz.token_sort_ratio(stop["stop_name"], name)
                for name in names
                if name != stop["stop_name"]
            ]
            if similarities and max(similarities) < similarity_threshold:
                different_named_stops.append(
                    {
                        "stop_id": stop["stop_id"],
                        "stop_name": stop["stop_name"],
                        "cluster": cluster,
                        "max_similarity_score": max(similarities),
                    }
                )

    columns = ["stop_id", "stop_name", "cluster", "max_similarity_score"]
    return pd.DataFrame(different_named_stops, columns=columns)


def save_to_excel(data_frame, filename, output_directory):
    """
    Save a DataFrame to Excel with headers, even if empty.
    """
    file_path = os.path.join(output_directory, filename)

    if data_frame.empty:
        # Provide columns so headers appear
        if "similar_names" in filename:
            columns = [
                "included_stop_name",
                "excluded_stop_name",
                "similarity_score",
                "stop_id",
                "stop_lat",
                "stop_lon",
            ]
        elif "nearby" in filename:
            columns = [
                "included_stop_id",
                "included_stop_name",
                "excluded_stop_id",
                "excluded_stop_name",
                "distance_m",
            ]
        elif "distant" in filename:
            columns = ["stop_id", "stop_name", "cluster", "min_distance_to_cluster_m"]
        elif "different_names" in filename:
            columns = ["stop_id", "stop_name", "cluster", "max_similarity_score"]
        else:
            columns = data_frame.columns
        data_frame = pd.DataFrame(columns=columns)

    data_frame.to_excel(file_path, index=False, header=True)


# ==================================================================================================
# MAIN
# ==================================================================================================


def main():
    """
    Main entry point for the GTFS Bus Bay Cluster Validation script.
    Creates output directories, prepares data, runs checks, and saves outputs.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)
    output_directory = os.path.join(BASE_OUTPUT_PATH, "stops_check")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Prepare stops using the new function (no reference to BASE_INPUT_PATH)
    stops_gdf = prepare_stops_gdf(DISTANCE_CRS_EPSG)

    # Split stops into included and excluded sets
    included_stops_global, excluded_stops_global, updated_clusters = (
        initialize_clusters(stops_gdf, clusters)
    )

    # Perform analysis only if clusters are defined
    if updated_clusters:
        similar_name_stops = find_similar_stop_names(
            included_stops_global, excluded_stops_global, SIMILARITY_THRESHOLD
        )
        nearby_excluded_stops = find_nearby_excluded_stops(
            included_stops_global, excluded_stops_global, DISTANCE_THRESHOLD_NEARBY
        )
        distant_included_stops = find_distant_included_stops(
            included_stops_global, DISTANCE_THRESHOLD_DISTANT
        )
        different_named_included_stops = find_different_named_included_stops(
            included_stops_global, SIMILARITY_THRESHOLD_NAMES
        )
    else:
        # If clusters are not defined, create empty DataFrames
        similar_name_stops = pd.DataFrame()
        nearby_excluded_stops = pd.DataFrame()
        distant_included_stops = pd.DataFrame()
        different_named_included_stops = pd.DataFrame()

    # Print summary (line kept under 100 chars)
    print(f"Number of similar name stops found: {len(similar_name_stops)}")
    print(f"Number of nearby excluded stops found: {len(nearby_excluded_stops)}")
    print(f"Number of distant included stops found: {len(distant_included_stops)}")
    print(
        f"Number of different named included stops found: {len(different_named_included_stops)}"
    )

    # Save outputs to Excel
    save_to_excel(
        similar_name_stops, "excluded_stops_similar_names.xlsx", output_directory
    )
    save_to_excel(nearby_excluded_stops, "excluded_stops_nearby.xlsx", output_directory)
    save_to_excel(
        distant_included_stops, "included_stops_distant.xlsx", output_directory
    )
    save_to_excel(
        different_named_included_stops,
        "included_stops_different_names.xlsx",
        output_directory,
    )

    # Export included, excluded, and all stops to shapefiles
    included_stops_shp = os.path.join(output_directory, "included_stops.shp")
    excluded_stops_shp = os.path.join(output_directory, "excluded_stops.shp")
    all_stops_shp = os.path.join(output_directory, "all_stops.shp")

    if not included_stops_global.empty:
        included_stops_global.to_file(included_stops_shp)
    else:
        print("No included stops to export.")

    if not excluded_stops_global.empty:
        excluded_stops_global.to_file(excluded_stops_shp)
    else:
        print("No excluded stops to export.")

    stops_gdf.to_file(all_stops_shp)

    print("Stops check completed. See the 'stops_check' folder for results.")


if __name__ == "__main__":
    main()
