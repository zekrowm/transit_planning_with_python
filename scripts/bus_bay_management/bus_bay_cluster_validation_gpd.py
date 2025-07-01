"""Validates GTFS stop clusters by checking spatial and naming consistency.

Analyzes predefined bus bay clusters for errors such as nearby excluded stops,
distant included stops, and inconsistent stop names. Useful for creating bus
stop cluster lists for subsequent analysis.

Typical Usage:
    ArcPro or standalone Python notebooks

Inputs:
    - GTFS 'stops.txt', 'trips.txt', and 'stop_times.txt' from GTFS_FOLDER_PATH
    - Cluster definitions in the `clusters` dictionary
    - Configuration constants (CRS, distance and similarity thresholds)

Outputs:
    - Excel files summarizing validation issues
    - Shapefiles of included, excluded, and all stops
    - Log file and messages in BASE_OUTPUT_PATH/stops_check
"""

import logging
import os
import sys

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
    # 'Airport Terminal': ['55', '72', '2304', '3277'],
}

SIMILARITY_THRESHOLD = 85
DISTANCE_THRESHOLD_NEARBY = 200
DISTANCE_THRESHOLD_DISTANT = 1000
SIMILARITY_THRESHOLD_NAMES = 70

DISTANCE_CRS_EPSG = 2248  # NAD83 / Maryland (ft)

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

def prepare_stops_gdf(crs_epsg: int, service_id: str = "3"):
    """Loads stops, trips, and stop_times with `load_gtfs_data()`.

    It filters the data to the specified `service_id` and returns a re-projected
    GeoDataFrame of *active* stops only.

    Args:
        crs_epsg (int): Target CRS for distance calculations (e.g. 2248).
        service_id (str, optional): GTFS service_id to analyse. Defaults to "3".

    Returns:
        geopandas.GeoDataFrame: Stops used by the chosen service, projected to `crs_epsg`.
    """
    # ------------------------------------------------------------------
    # Load the three core files we need
    # ------------------------------------------------------------------
    gtfs = load_gtfs_data(
        gtfs_folder_path=GTFS_FOLDER_PATH,
        files=["stops.txt", "trips.txt", "stop_times.txt"],
        dtype=str,
    )

    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    stops_df = gtfs["stops"]

    # ------------------------------------------------------------------
    # Ensure a service_id column exists and filter
    # ------------------------------------------------------------------
    if "service_id" not in trips.columns:
        logging.warning(
            "'service_id' missing in trips.txt – assigning '%s' to all rows", service_id
        )
        trips["service_id"] = service_id

    trips_filtered = trips.loc[trips["service_id"] == service_id]
    logging.info(
        "Filtered trips to service_id=%s → %d trips", service_id, len(trips_filtered)
    )

    # Keep only stop_times for those trips
    stop_times_filtered = stop_times.loc[
        stop_times["trip_id"].isin(trips_filtered["trip_id"])
    ]
    logging.info(
        "Remaining stop_times after filter → %d records", len(stop_times_filtered)
    )

    # Keep only stops referenced by those stop_times
    active_stop_ids = stop_times_filtered["stop_id"].unique()
    stops_active = stops_df.loc[stops_df["stop_id"].isin(active_stop_ids)].copy()
    logging.info(
        "Active stops for service_id=%s → %d stops", service_id, len(stops_active)
    )

    # ------------------------------------------------------------------
    # Build GeoDataFrame
    # ------------------------------------------------------------------
    stops_active["geometry"] = [
        Point(float(lon), float(lat))
        for lon, lat in zip(stops_active.stop_lon, stops_active.stop_lat)
    ]
    stops_gdf = gpd.GeoDataFrame(
        stops_active, geometry="geometry", crs="EPSG:4326"
    ).to_crs(epsg=crs_epsg)

    return stops_gdf


def initialize_clusters(stops_gdf, clusters_dict):
    """Splits stops into included and excluded sets based on cluster definitions.

    Args:
        stops_gdf (geopandas.GeoDataFrame): GeoDataFrame of all stops.
        clusters_dict (dict): Dictionary defining clusters.

    Returns:
        tuple: A tuple containing:
            - included_stops_global (geopandas.GeoDataFrame): Stops included in clusters.
            - excluded_stops_global (geopandas.GeoDataFrame): Stops not included in clusters.
            - updated_clusters_dict (dict): The clusters dictionary with stop IDs as strings.
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
    """Find excluded stops whose names are similar to included stops above a threshold.

    Args:
        inc_stops (geopandas.GeoDataFrame): GeoDataFrame of included stops.
        exc_stops (geopandas.GeoDataFrame): GeoDataFrame of excluded stops.
        threshold (int): Similarity threshold (0-100).

    Returns:
        pd.DataFrame: DataFrame of excluded stops with similar names.
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
    """Find excluded stops that are within distance_threshold from included stops.

    Args:
        inc_stops (geopandas.GeoDataFrame): GeoDataFrame of included stops.
        exc_stops (geopandas.GeoDataFrame): GeoDataFrame of excluded stops.
        distance_threshold (float): Distance threshold in meters.

    Returns:
        pd.DataFrame: DataFrame of excluded stops found nearby.
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
    """Find included stops that are far from all other stops in their cluster.

    Args:
        inc_stops (geopandas.GeoDataFrame): GeoDataFrame of included stops.
        distance_threshold (float): Distance threshold in meters.

    Returns:
        pd.DataFrame: DataFrame of included stops found to be distant.
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
    """Find included stops in the same cluster whose name similarity is below similarity_threshold.

    Args:
        inc_stops (geopandas.GeoDataFrame): GeoDataFrame of included stops.
        similarity_threshold (int): Similarity threshold (0-100).

    Returns:
        pd.DataFrame: DataFrame of included stops with different names.
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
    """Save a DataFrame to Excel with headers, even if empty.

    Args:
        data_frame (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the Excel file.
        output_directory (str): The directory to save the file in.
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


# --------------------------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------------------------------------------------------

def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """Loads GTFS files into pandas DataFrames from the specified directory.

    This function uses the logging module for output.

    Args:
        gtfs_folder_path (str): Path to the directory containing GTFS files.
        files (list[str], optional): GTFS filenames to load. Defaults to all
            standard GTFS files.
        dtype (str or dict, optional): Pandas dtype to use. Defaults to str.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by file name without extension.

    Raises:
        OSError: If gtfs_folder_path doesn't exist or if any required file is missing.
        ValueError: If a file is empty or there's a parsing error.
        RuntimeError: For OS errors during file reading.
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

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
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(
            f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}"
        )

    data = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info(f"Loaded {file_name} ({len(df)} records).")

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"File '{file_name}' in '{gtfs_folder_path}' is empty."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data

# ==================================================================================================
# MAIN
# ==================================================================================================


def main(service_id: str = "3"):
    """Main entry point for the GTFS Bus-Bay Cluster Validation script.

    Steps:
    1. Configure logging (console + file) exactly once.
    2. Create required output folders.
    3. Load and filter GTFS data, build GeoDataFrames.
    4. Run cluster checks and store Excel / shapefile outputs.

    Args:
        service_id (str, optional): GTFS service_id to analyze. Defaults to "3".
    """
    # ------------------------------------------------------------------
    # Logging ─ single centralised configuration
    # ------------------------------------------------------------------
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    log_file = os.path.join(BASE_OUTPUT_PATH, "stops_check.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # replace any handlers left over from notebooks etc.
    )
    logging.info("Bus-bay cluster validation started (service_id=%s)", service_id)

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    output_directory = os.path.join(BASE_OUTPUT_PATH, "stops_check")
    os.makedirs(output_directory, exist_ok=True)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    stops_gdf = prepare_stops_gdf(DISTANCE_CRS_EPSG, service_id)

    included_stops_global, excluded_stops_global, updated_clusters = (
        initialize_clusters(stops_gdf, clusters)
    )

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    if updated_clusters:
        logging.info("Running checks on %d clusters", len(updated_clusters))
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
        logging.warning("No clusters defined — skipping cluster-based checks")
        similar_name_stops = nearby_excluded_stops = distant_included_stops = (
            different_named_included_stops
        ) = pd.DataFrame()

    logging.info(
        "Summary → similar:%d  nearby:%d  distant:%d  diff-names:%d",
        len(similar_name_stops),
        len(nearby_excluded_stops),
        len(distant_included_stops),
        len(different_named_included_stops),
    )

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
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

    # Export shapefiles
    included_stops_shp = os.path.join(output_directory, "included_stops.shp")
    excluded_stops_shp = os.path.join(output_directory, "excluded_stops.shp")
    all_stops_shp = os.path.join(output_directory, "all_stops.shp")

    if not included_stops_global.empty:
        included_stops_global.to_file(included_stops_shp)
    else:
        logging.info("No included stops to export")

    if not excluded_stops_global.empty:
        excluded_stops_global.to_file(excluded_stops_shp)
    else:
        logging.info("No excluded stops to export")

    stops_gdf.to_file(all_stops_shp)
    logging.info("Cluster validation complete — outputs in %s", output_directory)


if __name__ == "__main__":
    main()
