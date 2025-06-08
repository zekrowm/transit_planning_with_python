"""
Generates a matrix of GTFS transit routes by district coverage.

Buffers GTFS stops, intersects them with district polygons, and determines
which routes serve which districts. Outputs a route-vs-district matrix
to an Excel file.

Inputs:
    - DISTRICTS_SHP: Shapefile of district boundaries.
    - GTFS_DIR: Folder containing GTFS files: routes.txt, stops.txt,
      trips.txt, and stop_times.txt.
    - Configuration constants for buffer size, EPSG, and district ID field.

Outputs:
    - Excel file with a matrix of route_short_name vs. district (y/n).
"""

import os
import logging
import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Shapefile of Districts (already in or to be projected to EPSG:2248, for example)
DISTRICTS_SHP = r"Path\To\Your\Districts.shp"

# Path to GTFS folder on G: drive
GTFS_DIR = r"Path\To\Your\GTFS_data"
GTFS_FILES = [
    "routes.txt",
    "stops.txt",
    "trips.txt",
    "stop_times.txt",
]

# Distance in feet (example: 1320 ~ 0.25 miles in a feet-based projection)
BUFFER_DISTANCE = 1320

# Final Excel output
OUTPUT_EXCEL = r"Path\To\Your\Excel_File.xlsx"

# Workspace folder (if you need to write intermediate shapefiles, place them here)
WORKSPACE_FOLDER = r"Path\To\Your\Temp_folder"

# Example CRS for Maryland State Plane, in feet
# EPSG:2248 => NAD83 / Maryland (ftUS)
TARGET_EPSG = 2248  # Adjust if your region uses a different EPSG code

# Specify the district column name in your shapefile
DISTRICT_FIELD = "DISTRICT"

# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(gtfs_folder_path: str, files: list[str] = None, dtype=str):
    """
    Loads GTFS files into pandas DataFrames from the specified directory.
    This function uses the logging module for output.

    Parameters:
        gtfs_folder_path (str): Path to the directory containing GTFS files.
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


def create_projected_stops_gdf(stops_df, epsg_out):
    """
    Convert GTFS stops (lat/lon in WGS84) into a GeoDataFrame,
    then project to EPSG `epsg_out`.
    Returns a projected GeoDataFrame of stops.
    """
    # 1) Convert lat/lon columns to float
    stops_df["stop_lat"] = stops_df["stop_lat"].astype(float)
    stops_df["stop_lon"] = stops_df["stop_lon"].astype(float)

    # 2) Create a GeoDataFrame using WGS84
    wgs84_crs = "EPSG:4326"
    geometry = gpd.points_from_xy(x=stops_df["stop_lon"], y=stops_df["stop_lat"])
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs=wgs84_crs)

    # 3) Reproject to the target CRS
    stops_projected = stops_gdf.to_crs(epsg=epsg_out)
    return stops_projected


def buffer_stops_gdf(stops_gdf, buffer_dist):
    """
    Buffer the stops by `buffer_dist` (feet or meters, depending on CRS).
    Returns a new GeoDataFrame with the buffered geometries.
    """
    # Copy so we don't overwrite the original geometry
    stops_buffered = stops_gdf.copy()
    # .buffer() distance is in the same units as the layer's CRS
    stops_buffered["geometry"] = stops_buffered.geometry.buffer(buffer_dist)
    return stops_buffered


def intersect_districts_gdf(stops_buffer_gdf, districts_gdf):
    """
    Use geopandas overlay (intersection) between the buffered stops polygons
    and the districts polygons.
    Returns a GeoDataFrame that includes columns from both layers.
    """
    # Use gpd.overlay with how='intersection'
    intersected = gpd.overlay(stops_buffer_gdf, districts_gdf, how="intersection")
    return intersected


def build_route_district_matrix(gtfs_data, intersect_gdf, district_field="DISTRICT"):
    """
    Build a DataFrame that shows route_short_name vs. district coverage (y/n).

    We rely on the 'stop_id' being in `intersect_gdf` from the stops layer,
    and the `district_field` coming from the District shapefile.

    Steps:
      1) Build route->trip->stop relationships from GTFS
      2) For each stop_id, see which districts it intersects
      3) Combine to create route vs. district coverage
    """
    routes_df = gtfs_data["routes"]
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    # Build quick lookups
    route_id_to_name = dict(zip(routes_df["route_id"], routes_df["route_short_name"]))
    trip_id_to_route_id = dict(zip(trips_df["trip_id"], trips_df["route_id"]))

    # Build stop_id -> set of route_ids
    stop_id_to_route_ids = {}
    for row in stop_times_df.itertuples(index=False):
        trip_id = row.trip_id
        stop_id = row.stop_id
        route_id = trip_id_to_route_id.get(trip_id)
        if route_id:
            stop_id_to_route_ids.setdefault(stop_id, set()).add(route_id)

    # Gather district info for each stop_id from `intersect_gdf`
    # `intersect_gdf` must have columns: "stop_id" (from stops), and the district field
    stop_id_to_districts = {}
    for idx, row in intersect_gdf.iterrows():
        stop_id_val = row["stop_id"]
        dist_val = row[district_field]
        stop_id_to_districts.setdefault(stop_id_val, set()).add(dist_val)

    # Build route -> set of districts
    route_to_districts = {}
    for stop_id, route_ids in stop_id_to_route_ids.items():
        if stop_id in stop_id_to_districts:
            these_districts = stop_id_to_districts[stop_id]
            for rt_id in route_ids:
                route_to_districts.setdefault(rt_id, set()).update(these_districts)

    # Build final matrix with route_short_name as rows, each district as a column
    all_route_ids = sorted(
        route_to_districts.keys(), key=lambda rid: route_id_to_name.get(rid, "zzz")
    )
    all_districts = sorted(
        {dist for dset in route_to_districts.values() for dist in dset}
    )

    matrix_data = []
    for route_id in all_route_ids:
        short_name = route_id_to_name.get(route_id, route_id)
        covered_districts = route_to_districts[route_id]
        row_dict = {"route_short_name": short_name}
        for dist in all_districts:
            row_dict[dist] = "y" if dist in covered_districts else "n"
        matrix_data.append(row_dict)

    df_out = pd.DataFrame(matrix_data)
    return df_out


def write_dataframe_to_excel(df, excel_path, sheet_name="districts_vs_routes"):
    """
    Write the DataFrame to an Excel file (using openpyxl).
    """
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """
    End-to-end workflow:

      1. Configure logging.
      2. Load GTFS data (using the new loader).
      3. Read / re-project district polygons.
      4. Build a projected GeoDataFrame of stops.
      5. Buffer the stops and intersect with districts.
      6. Build the route-vs-district matrix.
      7. Write the result to Excel.
    """
    # ------------------------------------------------------------------ 1 — LOGGING
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    # ------------------------------------------------------------------ 2 — GTFS
    try:
        gtfs_data = load_gtfs_data(
            GTFS_DIR,
            files=GTFS_FILES,  # ["routes.txt", "stops.txt", "trips.txt", "stop_times.txt"]
            dtype=str,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Failed to load GTFS data: %s", exc)
        raise

    # ------------------------------------------------------------------ 3 — DISTRICTS
    districts_gdf = gpd.read_file(DISTRICTS_SHP)
    if districts_gdf.crs is None or districts_gdf.crs.to_epsg() != TARGET_EPSG:
        logging.info("Re-projecting districts to EPSG:%s", TARGET_EPSG)
        districts_gdf = districts_gdf.to_crs(epsg=TARGET_EPSG)

    # ------------------------------------------------------------------ 4 — STOPS → GDF
    stops_projected_gdf = create_projected_stops_gdf(
        stops_df=gtfs_data["stops"],
        epsg_out=TARGET_EPSG,
    )

    # ------------------------------------------------------------------ 5 — BUFFER + INTERSECT
    stops_buffer_gdf = buffer_stops_gdf(stops_projected_gdf, BUFFER_DISTANCE)
    intersect_gdf = intersect_districts_gdf(stops_buffer_gdf, districts_gdf)

    # ------------------------------------------------------------------ 6 — MATRIX
    df_matrix = build_route_district_matrix(
        gtfs_data=gtfs_data,
        intersect_gdf=intersect_gdf,
        district_field=DISTRICT_FIELD,
    )

    # ------------------------------------------------------------------ 7 — OUTPUT
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    write_dataframe_to_excel(df_matrix, OUTPUT_EXCEL)
    logging.info("Done! Excel written to: %s", OUTPUT_EXCEL)


if __name__ == "__main__":
    main()
