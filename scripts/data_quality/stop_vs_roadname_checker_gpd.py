"""Detects potential typos in GTFS stop names using spatial and fuzzy matching.

This script buffers GTFS stops, spatially joins them with nearby roadway
centerlines, and uses fuzzy string comparison to flag discrepancies between
stop names and adjacent road names.

Inputs:
    - GTFS 'stops.txt' file
    - Roadway centerline shapefile
    - Configuration parameters (paths, CRS, buffer distance, similarity threshold)
    - Optional user input for mapping non-standard roadway field names

Outputs:
    - CSV listing potential stop name typos and similarity scores
"""

import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Set

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from rapidfuzz import fuzz, process

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to input files
GTFS_FOLDER = r"path\to\your\GTFS\folder"  # Replace with your GTFS folder path

ROADWAYS_PATH = r"path\to\your\roadways.shp"  # Replace with your roadways centerline shapefile path

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
REQUIRED_COLUMNS_ROADWAY = [
    "RW_PREFIX",
    "RW_TYPE_US",
    "RW_SUFFIX",
    "RW_SUFFIX_",
    "FULLNAME",
]

DESCRIPTIONS_ROADWAY = {
    "RW_PREFIX": "Directional prefix (e.g., 'N' in 'N Washington St')",
    "RW_TYPE_US": "Street type (e.g., 'St' in 'N Washington St')",
    "RW_SUFFIX": "Directional suffix (e.g., 'SE' in 'Park St SE')",
    "RW_SUFFIX_": "Additional suffix (e.g., 'EB' in 'RT267 EB')",
    "FULLNAME": "Full street name",
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_crs_unit(crs_code: str) -> Optional[str]:
    """Determine the linear unit of a CRS.

    Args:
        crs_code: The CRS code (e.g., "EPSG:4326").

    Returns:
        str or None: The unit name if found, otherwise None.
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


def convert_buffer_distance(value: float, from_unit: str, to_unit: str) -> float:
    """Convert buffer distance from `from_unit` to `to_unit` using known conversion factors.

    Args:
        value (float): The distance value to convert.
        from_unit (str): The unit of the input value (e.g., "feet", "meters").
        to_unit (str): The desired unit for the output value (e.g., "feet", "meters").

    Returns:
        float: The converted distance value.

    Raises:
        ValueError: If the conversion from `from_unit` to `to_unit` is not supported.
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


def load_stops(stops_df: pd.DataFrame, crs: str = STOPS_CRS) -> gpd.GeoDataFrame:
    """Validate an in-memory GTFS stops DataFrame and return a GeoDataFrame.

    Args:
        stops_df (pandas.DataFrame): Frame created by `load_gtfs_data(..., files=["stops.txt"])`.
        crs (str, optional): CRS to assign to the resulting GeoDataFrame.
            Defaults to STOPS_CRS.

    Returns:
        geopandas.GeoDataFrame: Stops with point geometries in the requested CRS.

    Raises:
        ValueError: If required columns are missing or lat/lon cannot be cast to float.
    """
    required_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    missing = [c for c in required_cols if c not in stops_df.columns]
    if missing:
        raise ValueError(f"Required columns missing from stops.txt: {', '.join(missing)}")

    # Ensure numeric latitude / longitude
    stops_df = stops_df.copy()
    stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
    stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)

    gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs=crs,
    )
    return gdf


def load_roadways(roadways_path: str) -> gpd.GeoDataFrame:
    """Load the roadway shapefile and return a GeoDataFrame.

    Args:
        roadways_path (str): The file path to the roadway shapefile.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the roadway data.
    """
    return gpd.read_file(roadways_path)


# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------


def map_roadway_columns(roadways_gdf: gpd.GeoDataFrame) -> Dict[str, str]:
    """Map the required roadway columns.

    Prompts the user to input the correct column names if missing.

    Args:
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame containing roadway data.

    Returns:
        dict: A dictionary mapping required column names to their actual names in the GeoDataFrame.
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
                f"Please enter the correct column name for '{col}' (or leave blank to skip): "
            ).strip()
            while new_col and new_col not in roadways_gdf.columns:
                logging.warning(
                    "'%s' is not among the available columns: %s",
                    new_col,
                    roadways_gdf.columns.tolist(),
                )
                new_col = input(
                    f"Please enter the correct column name for '{col}' (or leave blank to skip): "
                ).strip()
            if new_col:
                column_mapping[col] = new_col
                logging.info("Mapped '%s' to '%s'", col, new_col)
            else:
                logging.info("Skipped mapping for '%s'", col)
    return {k: v for k, v in column_mapping.items() if v is not None}


def extract_modifiers(
    roadways_gdf: gpd.GeoDataFrame, column_mapping_roadway: Dict[str, str]
) -> Set[str]:
    """Extract unique modifier values (e.g., street types) from the roadway GeoDataFrame.

    Args:
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame containing roadway data.
        column_mapping_roadway (dict): A dictionary mapping required column names to
            their actual names.

    Returns:
        set: A set of unique, normalized modifier strings.
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


def normalize_street_name(name: str, modifiers_set: Set[str]) -> str:
    """Normalize a street name by removing known modifiers, punctuation, and extra spaces.

    Args:
        name (str): The street name to normalize.
        modifiers_set (set): A set of known modifiers to remove from the name.

    Returns:
        str: The normalized street name.
    """
    if pd.isnull(name) or not isinstance(name, str):
        return ""
    if modifiers_set:
        pattern = r"\b(" + "|".join(re.escape(m) for m in modifiers_set) + r")\b"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def create_buffered_stops(stops_gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
    """Create a buffered geometry for each stop.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        buffer_distance (float): The distance to buffer the stops by.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with a new 'buffered_geometry' column.
    """
    stops_gdf["buffered_geometry"] = stops_gdf.geometry.buffer(buffer_distance)
    return stops_gdf.set_geometry("buffered_geometry")  # type: ignore[no-any-return]


def spatial_join_stops_roadways(
    stops_buffered_gdf: gpd.GeoDataFrame, roadways_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatially join the buffered stops with the roadways.

    Args:
        stops_buffered_gdf (gpd.GeoDataFrame): The GeoDataFrame of buffered stops.
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    return gpd.sjoin(
        stops_buffered_gdf[["stop_id", "stop_name", "buffered_geometry"]],
        roadways_gdf[["FULLNAME", "FULLNAME_clean", "geometry"]],
        how="left",
        predicate="intersects",
    )


def extract_street_names(stop_name: str, modifiers: Set[str]) -> List[str]:
    """Extract potential street names from a stop name using common separators.

    Args:
        stop_name (str): The name of the stop.
        modifiers (set): A set of known modifiers to assist in normalization.

    Returns:
        list: A list of normalized street names extracted from the stop name.
    """
    if pd.isnull(stop_name) or not isinstance(stop_name, str):
        return []
    separators = [" @ ", " and ", " & ", "/", " intersection of "]
    pattern = "|".join(map(re.escape, separators))
    streets = re.split(pattern, stop_name, flags=re.IGNORECASE)
    return [normalize_street_name(street, modifiers) for street in streets if street]


def compare_stop_to_roads(
    stop_id: str,
    stop_name: str,
    stop_streets: List[str],
    road_names: Set[str],
    roads_gdf: gpd.GeoDataFrame,
    threshold: int,
) -> List[Dict[str, Any]]:
    """Compare each portion of the stop name to known road names via fuzzy matching.

    Args:
        stop_id (str): The ID of the stop.
        stop_name (str): The original name of the stop.
        stop_streets (list): A list of potential street names extracted from the stop
            name.
        road_names (list): A list of normalized road names for comparison.
        roads_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways, used to retrieve
            original road names.
        threshold (int): The similarity score threshold (0-100) for considering a
            match.

    Returns:
        list[dict]: A list of dictionaries, each representing a potential typo.
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


def process_typos(
    stops_gdf: gpd.GeoDataFrame,
    roadways_gdf: gpd.GeoDataFrame,
    modifiers: Set[str],
    road_names_clean: Set[str],
    threshold: int,
) -> pd.DataFrame:
    """Process each stop and perform fuzzy matching to identify potential typos.

    Args:
        stops_gdf (gpd.GeoDataFrame): The GeoDataFrame of stops.
        roadways_gdf (gpd.GeoDataFrame): The GeoDataFrame of roadways.
        modifiers (set): A set of known street name modifiers.
        road_names_clean (set): A set of normalized roadway names.
        threshold (int): The similarity score threshold for fuzzy matching.

    Returns:
        pd.DataFrame: A deduplicated DataFrame of potential typos, sorted by similarity score.
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


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
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
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point for the GTFS stop-vs-road typo-checker script."""
    # ------------------------------------------------------------------
    # 1. Configure logging *inside* main so importing this module is silent
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Starting processing …")

    # ------------------------------------------------------------------
    # 2. Ensure the output directory exists
    # ------------------------------------------------------------------
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info("Created output directory %s", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 3. Load GTFS data (only stops.txt is required for this task)
    # ------------------------------------------------------------------
    gtfs_data = load_gtfs_data(GTFS_FOLDER, files=["stops.txt"])
    stops_df = gtfs_data["stops"]  # key name = file name w/o ".txt"
    stops_gdf = load_stops(stops_df)  # validate and convert to GDF

    # 4. Load roadway shapefile
    roadways_gdf = load_roadways(ROADWAYS_PATH)

    # 5. Re-project both layers to TARGET_CRS
    stops_gdf = stops_gdf.to_crs(TARGET_CRS)
    roadways_gdf = roadways_gdf.to_crs(TARGET_CRS)

    # ------------------------------------------------------------------
    # 6. Map roadway columns (prompting user if needed)
    # ------------------------------------------------------------------
    column_mapping = map_roadway_columns(roadways_gdf)
    if not column_mapping.get("FULLNAME"):
        raise ValueError("The 'FULLNAME' column is required in the roadway data.")
    roadways_gdf = roadways_gdf.rename(columns=column_mapping)

    # 7. Extract modifiers and normalise roadway names
    modifiers = extract_modifiers(roadways_gdf, column_mapping)
    logging.info("Extracted modifiers (%d): %s", len(modifiers), modifiers)
    roadways_gdf["FULLNAME_clean"] = roadways_gdf["FULLNAME"].apply(
        lambda x: normalize_street_name(x, modifiers)
    )

    # ------------------------------------------------------------------
    # 8. Compute buffer distance in target CRS units
    # ------------------------------------------------------------------
    crs_unit = get_crs_unit(TARGET_CRS)
    if crs_unit is None:
        raise ValueError("Unable to determine the unit for TARGET_CRS.")
    buffer_distance = (
        convert_buffer_distance(BUFFER_DISTANCE_VALUE, BUFFER_DISTANCE_UNIT, crs_unit)
        if BUFFER_DISTANCE_UNIT.lower() != crs_unit.lower()
        else BUFFER_DISTANCE_VALUE
    )

    # 9. Buffer stops, spatial-join with roadways
    stops_buffered = create_buffered_stops(stops_gdf, buffer_distance)
    join_gdf = spatial_join_stops_roadways(stops_buffered, roadways_gdf)
    logging.info("Spatial join produced %d candidate matches", join_gdf.shape[0])

    # ------------------------------------------------------------------
    # 10. Fuzzy-match street names to find potential typos
    # ------------------------------------------------------------------
    road_names_clean = set(roadways_gdf["FULLNAME_clean"].dropna().unique())
    typos_df = process_typos(
        stops_gdf,
        roadways_gdf,
        modifiers,
        road_names_clean,
        SIMILARITY_THRESHOLD,
    )

    # 11. Save or report results
    if typos_df.empty:
        logging.info("No potential typos found.")
    else:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
        typos_df.to_csv(out_path, index=False)
        logging.info("Potential typos saved to %s", out_path)


if __name__ == "__main__":
    main()
