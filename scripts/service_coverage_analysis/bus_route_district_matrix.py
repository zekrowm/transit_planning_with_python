"""
Script Name:
        bus_route_district_matrix.py

Purpose:
        Processes GTFS (General Transit Feed Specification) data and
        district-level geographic data to determine which transit routes
        operate within each district. It involves loading GTFS feeds,
        projecting and buffering transit stops, intersecting these stops
        with district boundaries (jurisdiction or political boundaries),
        and generating a report.

Inputs:
        1. Path to a shapefile defining district boundaries (DISTRICTS_SHP).
        2. Path to a directory containing GTFS files (GTFS_DIR):
           - routes.txt (ROUTES_FILE)
           - stops.txt (STOPS_FILE)
           - trips.txt (TRIPS_FILE)
           - stop_times.txt (STOP_TIMES_FILE)
        3. Configuration parameters defined in the script:
           - BUFFER_DISTANCE: Distance in feet for buffering stops.
           - TARGET_EPSG: EPSG code for spatial projection.
           - DISTRICT_FIELD: Name of the field in the district
             shapefile that identifies districts.

Outputs:
        1. An Excel file (OUTPUT_EXCEL) containing a matrix that indicates
           which transit routes serve which districts ('y' for yes, 'n' for no).

Dependencies: geopandas, pandas, openpyxl
"""

import os

import geopandas as gpd
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Shapefile of Districts (already in or to be projected to EPSG:2248, for example)
DISTRICTS_SHP = r"Path\To\Your\Districts.shp"

# Path to GTFS folder on G: drive
GTFS_DIR = r"Path\To\Your\GTFS_data"
ROUTES_FILE = "routes.txt"
STOPS_FILE = "stops.txt"
TRIPS_FILE = "trips.txt"
STOP_TIMES_FILE = "stop_times.txt"

# Distance in feet (example: 1320 ~ 0.25 miles in a feet-based projection)
BUFFER_DISTANCE = 1320

# Final Excel output
OUTPUT_EXCEL = r"Path\To\Your\Output_folder"

# Workspace folder (if you need to write intermediate shapefiles, place them here)
WORKSPACE_FOLDER = r"Path\To\Your\Temp_folder"

# Example CRS for Maryland State Plane, in feet
# EPSG:2248 => NAD83 / Maryland (ftUS)
TARGET_EPSG = 2248  # Adjust if your region uses a different EPSG code

# Specify the district column name in your shapefile
DISTRICT_FIELD = "DISTRICT"

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(gtfs_dir):
    """
    Load GTFS data into pandas DataFrames.
    Returns a dict with keys: 'routes', 'stops', 'trips', 'stop_times'.
    """
    data = {}
    routes_path = os.path.join(gtfs_dir, ROUTES_FILE)
    stops_path = os.path.join(gtfs_dir, STOPS_FILE)
    trips_path = os.path.join(gtfs_dir, TRIPS_FILE)
    stop_times_path = os.path.join(gtfs_dir, STOP_TIMES_FILE)

    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"Could not find {routes_path}")
    if not os.path.exists(stops_path):
        raise FileNotFoundError(f"Could not find {stops_path}")
    if not os.path.exists(trips_path):
        raise FileNotFoundError(f"Could not find {trips_path}")
    if not os.path.exists(stop_times_path):
        raise FileNotFoundError(f"Could not find {stop_times_path}")

    data["routes"] = pd.read_csv(routes_path, dtype=str)
    data["stops"] = pd.read_csv(stops_path, dtype=str)
    data["trips"] = pd.read_csv(trips_path, dtype=str)
    data["stop_times"] = pd.read_csv(stop_times_path, dtype=str)

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


def main():
    """
    End-to-end workflow using GeoPandas:
      1) Load GTFS data into pandas
      2) Read District shapefile (geopandas)
      3) Create stops GDF (lat/lon -> points), reproject
      4) Buffer stops
      5) Intersect with District polygons
      6) Build route vs. district coverage matrix
      7) Output to Excel
    """
    # 1) Load GTFS data
    gtfs_data = load_gtfs_data(GTFS_DIR)

    # 2) Read District shapefile, reproject if not in EPSG:2248
    districts_gdf = gpd.read_file(DISTRICTS_SHP)
    if districts_gdf.crs is None or districts_gdf.crs.to_epsg() != TARGET_EPSG:
        districts_gdf = districts_gdf.to_crs(epsg=TARGET_EPSG)

    # 3) Create a projected GeoDataFrame of stops
    stops_projected_gdf = create_projected_stops_gdf(
        stops_df=gtfs_data["stops"], epsg_out=TARGET_EPSG
    )

    # 4) Buffer the stops
    stops_buffer_gdf = buffer_stops_gdf(stops_projected_gdf, BUFFER_DISTANCE)

    # 5) Intersect with District polygons
    intersect_gdf = intersect_districts_gdf(stops_buffer_gdf, districts_gdf)

    # 6) Build route vs. district matrix using the configured district field
    df_matrix = build_route_district_matrix(
        gtfs_data=gtfs_data, intersect_gdf=intersect_gdf, district_field=DISTRICT_FIELD
    )

    # 7) Write to Excel
    write_dataframe_to_excel(df_matrix, OUTPUT_EXCEL)
    print(f"Done! Excel written to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
