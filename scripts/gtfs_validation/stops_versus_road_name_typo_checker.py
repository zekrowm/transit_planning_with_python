"""
Script Name:
        stops_versus_road_name_typo_checker.py

Purpose:
        Identifies potential typos in GTFS stop names by spatially
        joining them with roadway centerlines from a shapefile and
        comparing names using fuzzy matching.

Inputs:
        1. GTFS 'stops.txt' file.
        2. Roadway centerline shapefile (.shp).
        3. Configuration constants within the script (paths, CRS,
           similarity threshold, buffer distance).
        4. User input for mapping roadway shapefile columns if
           standard names are not found.

Outputs:
        1. CSV file (default: 'potential_typos.csv') listing potential
           typos, including stop details, the street segment from the
           stop name, the similar roadway name, and the similarity score.

Dependencies:
        geopandas, pandas, pyproj, rapidfuzz
        logging (standard library), os (standard library), re (standard library)

Module: gtfs_stop_road_shp_typo_finder
Description: Identifies potential typos in GTFS stop names by comparing them
to roadway shapefile names using fuzzy matching.
"""

import logging
import os
import re

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from rapidfuzz import fuzz, process

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to input files
GTFS_FOLDER = r"path\to\your\GTFS\folder"  # Replace with your GTFS folder path
STOPS_FILENAME = "stops.txt"
STOPS_PATH = os.path.join(GTFS_FOLDER, STOPS_FILENAME)

ROADWAYS_PATH = r"path\to\your\roadways.shp"  # Replace with your roadways shapefile path

# Output settings
OUTPUT_DIR = r"path\to\output\directory"  # Replace with your desired output directory
OUTPUT_CSV_NAME = "potential_typos.csv"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)

# Coordinate Reference Systems
STOPS_CRS = "EPSG:4326"  # WGS84 Latitude/Longitude. Typically standard for GTFS stops.
TARGET_CRS = "EPSG:2248"  # Projected CRS for spatial analysis (adjust as needed).

# Processing parameters
SIMILARITY_THRESHOLD = 80  # 0-100, higher number yields fewer results

# Buffer distance configuration
BUFFER_DISTANCE_VALUE = 50
BUFFER_DISTANCE_UNIT = "feet"  # 'feet' or 'meters'

# Roadway Shapefile Column Configuration
REQUIRED_COLUMNS_ROADWAY = ["RW_PREFIX", "RW_TYPE_US", "RW_SUFFIX", "RW_SUFFIX_", "FULLNAME"]

DESCRIPTIONS_ROADWAY = {
    "RW_PREFIX": "Directional prefix (e.g., 'N' in 'N Washington St')",
    "RW_TYPE_US": "Street type (e.g., 'St' in 'N Washington St')",
    "RW_SUFFIX": "Directional suffix (e.g., 'SE' in 'Park St SE')",
    "RW_SUFFIX_": "Additional suffix (e.g., 'EB' in 'RT267 EB')",
    "FULLNAME": "Full street name",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_crs_unit(crs_code):
    """
    Determine the linear unit of a CRS.
    """
    try:
        crs = CRS.from_user_input(crs_code)
        if crs.axis_info:
            return crs.axis_info[0].unit_name
        logging.error("CRS has no axis information.")
        return None
    except ValueError as err:
        logging.error("Error determining CRS unit: %s", err)
        return None


def convert_buffer_distance(value, from_unit, to_unit):
    """
    Convert buffer distance from `from_unit` to `to_unit` using known conversion factors.
    """
    conversion_factors = {
        ("feet", "meters"): 0.3048,
        ("meters", "feet"): 3.28084,
        ("metre", "feet"): 3.28084,
        ("us survey foot", "meters"): 0.3048006096012192,
        ("meters", "us survey foot"): 3.280833333333333,
        ("feet", "us survey foot"): 0.999998,
        ("us survey foot", "feet"): 1.000002,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversion_factors:
        return value * conversion_factors[key]
    raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported.")


# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------


def load_stops(stops_path):
    """
    Load and validate the GTFS stops file, and return a GeoDataFrame.
    """
    required_columns_stops = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    stops_df = pd.read_csv(stops_path, dtype=str)
    missing_cols = [col for col in required_columns_stops if col not in stops_df.columns]
    if missing_cols:
        raise ValueError(
            "The following required columns are missing in stops.txt: %s" % missing_cols
        )
    stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
    stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=STOPS_CRS,
    )
    return stops_gdf


def load_roadways(roadways_path):
    """
    Load the roadway shapefile and return a GeoDataFrame.
    """
    return gpd.read_file(roadways_path)


# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------


def map_roadway_columns(roadways_gdf):
    """
    Map the required roadway columns. Prompts the user to input the correct
    column names if missing.
    """
    column_mapping = {}
    for col in REQUIRED_COLUMNS_ROADWAY:
        if col in roadways_gdf.columns:
            column_mapping[col] = col
        else:
            logging.warning("The column '%s' is missing from the roadway shapefile.", col)
            logging.info("Description: %s", DESCRIPTIONS_ROADWAY[col])
            logging.info("Available columns: %s", roadways_gdf.columns.tolist())
            new_col = input(
                f"Please enter the correct column name for '{col}' " "(or leave blank to skip): "
            ).strip()
            while new_col and new_col not in roadways_gdf.columns:
                logging.warning(
                    "'%s' is not among the available columns: %s",
                    new_col,
                    roadways_gdf.columns.tolist(),
                )
                new_col = input(
                    f"Please enter the correct column name for '{col}' "
                    "(or leave blank to skip): "
                ).strip()
            if new_col:
                column_mapping[col] = new_col
                logging.info("Mapped '%s' to '%s'", col, new_col)
            else:
                logging.info("Skipped mapping for '%s'", col)
    return {k: v for k, v in column_mapping.items() if v is not None}


def extract_modifiers(roadways_gdf, column_mapping_roadway):
    """
    Extract unique modifier values (e.g., street types) from the roadway
    GeoDataFrame.
    """
    modifiers_fields = ["RW_TYPE_US"]
    modifiers = set()
    for field in modifiers_fields:
        mapped_field = column_mapping_roadway.get(field)
        if mapped_field and mapped_field in roadways_gdf.columns:
            unique_vals = roadways_gdf[mapped_field].dropna().unique()
            modifiers.update(unique_vals)
    modifiers = set(
        str(mod).lower().strip() for mod in modifiers if pd.notnull(mod) and str(mod).strip()
    )
    return modifiers


def normalize_street_name(name, modifiers_set):
    """
    Normalize street name by removing known modifiers, punctuation, and spaces.
    """
    if pd.isnull(name) or not isinstance(name, str):
        return ""
    if modifiers_set:
        pattern = r"\b(" + "|".join(re.escape(m) for m in modifiers_set) + r")\b"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def create_buffered_stops(stops_gdf, buffer_distance):
    """
    Create a buffered geometry for each stop.
    """
    stops_gdf["buffered_geometry"] = stops_gdf.geometry.buffer(buffer_distance)
    return stops_gdf.set_geometry("buffered_geometry")


def spatial_join_stops_roadways(stops_buffered_gdf, roadways_gdf):
    """
    Spatially join the buffered stops with the roadways.
    """
    return gpd.sjoin(
        stops_buffered_gdf[["stop_id", "stop_name", "buffered_geometry"]],
        roadways_gdf[["FULLNAME", "FULLNAME_clean", "geometry"]],
        how="left",
        predicate="intersects",
    )


def extract_street_names(stop_name, modifiers):
    """
    Extract potential street names from a stop name using common separators.
    """
    if pd.isnull(stop_name) or not isinstance(stop_name, str):
        return []
    separators = [" @ ", " and ", " & ", "/", " intersection of "]
    pattern = "|".join(map(re.escape, separators))
    streets = re.split(pattern, stop_name, flags=re.IGNORECASE)
    return [normalize_street_name(street, modifiers) for street in streets if street]


def compare_stop_to_roads(stop_id, stop_name, stop_streets, road_names, roads_gdf, threshold):
    """
    Compare each portion of the stop name to known road names via fuzzy matching.
    """
    potential_typos_list = []
    for street in stop_streets:
        if street in road_names:
            continue
        match_tuples = process.extract(street, road_names, scorer=fuzz.token_set_ratio, limit=3)
        for match_clean, score, _ in match_tuples:
            if threshold <= score < 100:
                original_matches = roads_gdf.loc[
                    roads_gdf["FULLNAME_clean"] == match_clean, "FULLNAME"
                ].unique()
                for original_match in original_matches:
                    potential_typos_list.append(
                        {
                            "stop_id": stop_id,
                            "stop_name": stop_name,
                            "street_in_stop_name": street,
                            "similar_road_name_clean": match_clean,
                            "similar_road_name_original": original_match,
                            "similarity_score": score,
                        }
                    )
    return potential_typos_list


def process_typos(stops_gdf, roadways_gdf, modifiers, road_names_clean, threshold):
    """
    Process each stop, perform fuzzy matching to identify potential typos,
    and return a deduplicated DataFrame.
    """
    potential_typos = []
    for _, stop in stops_gdf.iterrows():
        s_id = stop["stop_id"]
        s_name = stop["stop_name"]
        s_streets = extract_street_names(s_name, modifiers)
        typos = compare_stop_to_roads(
            s_id, s_name, s_streets, road_names_clean, roadways_gdf, threshold
        )
        potential_typos.extend(typos)

    logging.info("Total potential typos found before deduplication: %d", len(potential_typos))
    typos_df = pd.DataFrame(potential_typos)
    typos_df_sorted = typos_df.sort_values(by="similarity_score", ascending=False).drop_duplicates()
    return typos_df_sorted


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main entry point of the GTFS stop/road shapefile typo-finding script.

    Steps:
      1. Validates and loads the GTFS stops data from a CSV file.
      2. Loads and prepares the roadway shapefile data.
      3. Converts both datasets to the specified target CRS.
      4. Maps required columns in the roadways data, prompting for user input
         if necessary.
      5. Extracts and normalizes modifiers (e.g., street type abbreviations).
      6. Buffers the stops’ geometries in the target CRS and performs a spatial
         join with the roadway data.
      7. Uses fuzzy matching to compare stop names to roadway names,
         identifying potential typos.
      8. Exports any potential typos to a CSV file for further review.

    Raises:
        FileNotFoundError: If 'stops.txt' is not found.
        ValueError: If required columns are missing, or if there's a CRS/unit issue.
    """
    logging.info("Starting processing...")

    # Ensure the output directory exists.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info("Created output directory: %s", OUTPUT_DIR)

    # Validate the existence of the GTFS stops file.
    if not os.path.isfile(STOPS_PATH):
        raise FileNotFoundError("'stops.txt' not found in the GTFS folder: %s" % GTFS_FOLDER)

    # Determine CRS unit and compute the appropriate buffer distance.
    crs_unit = get_crs_unit(TARGET_CRS)
    if crs_unit is None:
        raise ValueError("Unable to determine the CRS unit. Please check the TARGET_CRS.")

    logging.info("Target CRS (%s) uses '%s' as its linear unit.", TARGET_CRS, crs_unit)
    supported_units = ["feet", "meters", "metre", "us survey foot"]
    if BUFFER_DISTANCE_UNIT.lower() not in supported_units:
        raise ValueError(
            "Unsupported buffer distance unit '%s'. Supported units are: %s"
            % (BUFFER_DISTANCE_UNIT, supported_units)
        )

    try:
        if BUFFER_DISTANCE_UNIT.lower() != crs_unit.lower():
            buffer_distance = convert_buffer_distance(
                BUFFER_DISTANCE_VALUE, BUFFER_DISTANCE_UNIT, crs_unit
            )
            logging.info(
                "Buffer distance converted from %s to %s: %.6f %s",
                BUFFER_DISTANCE_UNIT,
                crs_unit,
                buffer_distance,
                crs_unit,
            )
        else:
            buffer_distance = BUFFER_DISTANCE_VALUE
            logging.info("Buffer distance: %f %s", buffer_distance, crs_unit)
    except ValueError as ve:
        logging.error("Conversion error: %s", ve)
        raise

    # Load GTFS stops and roadway data.
    stops_gdf = load_stops(STOPS_PATH)
    roadways_gdf = load_roadways(ROADWAYS_PATH)

    # Convert both datasets to the target CRS.
    stops_gdf = stops_gdf.to_crs(TARGET_CRS)
    roadways_gdf = roadways_gdf.to_crs(TARGET_CRS)

    # Map the roadway shapefile columns.
    column_mapping_roadway = map_roadway_columns(roadways_gdf)
    if "FULLNAME" not in column_mapping_roadway or not column_mapping_roadway["FULLNAME"]:
        raise ValueError("The 'FULLNAME' column is required in the roadway shapefile.")
    roadways_gdf = roadways_gdf.rename(columns=column_mapping_roadway)

    # Extract and log unique modifiers.
    modifiers = extract_modifiers(roadways_gdf, column_mapping_roadway)
    logging.info("Extracted Modifiers (%d): %s", len(modifiers), modifiers)

    # Normalize roadway names.
    roadways_gdf["FULLNAME_clean"] = roadways_gdf["FULLNAME"].apply(
        lambda x: normalize_street_name(x, modifiers)
    )

    # Create buffered stops and perform spatial join.
    stops_buffered_gdf = create_buffered_stops(stops_gdf, buffer_distance)
    joined_gdf = spatial_join_stops_roadways(stops_buffered_gdf, roadways_gdf)
    logging.info("Total stops processed: %d", len(stops_gdf))
    logging.info("Total spatial join matches: %d", joined_gdf.shape[0])

    # Obtain unique, cleaned roadway names.
    road_names_clean = set(roadways_gdf["FULLNAME_clean"].dropna().unique())

    # Process potential typos.
    typos_df_sorted = process_typos(
        stops_gdf, roadways_gdf, modifiers, road_names_clean, SIMILARITY_THRESHOLD
    )
    logging.info("Total potential typos after deduplication: %d", typos_df_sorted.shape[0])

    if typos_df_sorted.empty:
        logging.info("No potential typos found.")
    else:
        typos_df_sorted.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info("Potential typos saved to %s", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
