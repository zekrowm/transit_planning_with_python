"""Joins ridership data to bus stop features, optionally with a spatial join to polygon areas.

Designed for ArcGIS Pro workflows, this script merges stop-level ridership data from
an Excel file with stop locations (from a shapefile or GTFS stops.txt), and optionally
joins to a polygon layer (e.g., Census Blocks) for geographic aggregation.

Outputs include shapefiles of stops with ridership attributes, CSV summaries, and,
if a polygon layer is provided, shapefiles and CSVs with aggregated ridership by area.

Typical use:
    Configure paths and options at the top of the script, then run inside ArcGIS Pro
    or as a standalone Python script with access to the ArcPy environment.
"""

import csv
import logging
import os
from typing import List, Tuple

import arcpy
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# INPUTS --------------------------------------------------------------------
# Bus stops can be either a .shp or GTFS stops.txt
BUS_STOPS_INPUT = r"Your\File\Path\To\stops.txt"  # Replace as needed

# Path to Excel with ridership data.
EXCEL_FILE = r"Your\File\Path\To\STOP_USAGE_(BY_STOP_ID).XLSX"

# Optional: Filter your Excel data for certain routes. If empty, no filter.
# Example: ROUTE_FILTER_LIST = ["101", "202"]
ROUTE_FILTER_LIST: list[str] = []

# Set to False to create one shapefile for all stops,
# or True to create a separate shapefile per unique route.
SPLIT_BY_ROUTE = False

# OUTPUTS -------------------------------------------------------------------
OUTPUT_FOLDER = r"Your\Folder\Path\To\Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Optional: Polygon features to join ridership data to.
# If empty, the spatial-join and aggregation steps will be skipped.
POLYGON_LAYER = r"Your\File\Path\To\census_blocks.shp"

# File paths for intermediate & final outputs:
GTFS_STOPS_FC = os.path.join(OUTPUT_FOLDER, "bus_stops_generated.shp")
JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_JoinedPolygon.shp")
MATCHED_JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_Matched_JoinedPolygon.shp")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "bus_stops_with_polygon.csv")
POLYGON_WITH_RIDERSHIP_SHP = os.path.join(OUTPUT_FOLDER, "polygon_with_ridership.shp")

# FIELDS & JOIN KEYS -------------------------------------------------------
# 1. Key fields in the bus stops data:
GTFS_KEY_FIELD = "stop_code"  # GTFS "unique" stop identifier
SHAPE_KEY_FIELD = "StopId"  # Shapefile "unique" stop identifier

# 2. Additional useful fields in GTFS or shapefile:
GTFS_SECONDARY_ID_FIELD = "stop_id"  # For reference, e.g. "stop_id" in stops.txt
SHAPE_SECONDARY_ID_FIELD = "StopNum"  # For reference, e.g. "StopNum" in shapefile

# 3. Polygon fields to export (and optional join field):
POLYGON_JOIN_FIELD = "GEOID"  # e.g., Census GEOID
POLYGON_FIELDS_TO_KEEP = ["NAME", "GEOID", "GEOIDFQ"]  # Must include the join field

# ENVIRONMENT & FLAGS ------------------------------------------------------
IS_GTFS_INPUT = BUS_STOPS_INPUT.lower().endswith(".txt")
arcpy.env.overwriteOutput = True

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================


def create_bus_stops_feature_class() -> Tuple[str, List[str]]:
    """Create or identify the bus stops feature class.

    If input is a GTFS stops.txt file, convert it to a point feature class.
    Otherwise, assume we have a shapefile. Returns:
      - bus_stops_fc: path to the resulting feature class
      - fields_to_export: list of fields to export (including the polygon fields)
    """
    if POLYGON_LAYER.strip():
        extra_fields = POLYGON_FIELDS_TO_KEEP
    else:
        extra_fields = []

    if IS_GTFS_INPUT:
        # Convert GTFS stops.txt to point feature class
        arcpy.management.XYTableToPoint(
            in_table=BUS_STOPS_INPUT,
            out_feature_class=GTFS_STOPS_FC,
            x_field="stop_lon",
            y_field="stop_lat",
            coordinate_system=arcpy.SpatialReference(4326),  # WGS84
        )
        logger.info("GTFS stops feature class created at: %s", GTFS_STOPS_FC)
        bus_stops_fc = GTFS_STOPS_FC

        # Fields to export to CSV: the GTFS key field, secondary field, plus polygon fields.
        fields_to_export = [
            GTFS_KEY_FIELD,
            GTFS_SECONDARY_ID_FIELD,
            "stop_name",
        ] + extra_fields

    else:
        # Using an existing shapefile of bus stops directly
        logger.info("Using existing bus stops shapefile: %s", BUS_STOPS_INPUT)
        bus_stops_fc = BUS_STOPS_INPUT

        # Fields to export to CSV: the shapefile key field, secondary field, plus polygon fields.
        fields_to_export = [
            SHAPE_KEY_FIELD,
            SHAPE_SECONDARY_ID_FIELD,
        ] + extra_fields

    return bus_stops_fc, fields_to_export


def spatial_join_bus_stops_to_polygons(bus_stops_fc: str, fields_to_export: List[str]) -> str:
    """Perform a spatial join of bus stops to polygon features (if provided)."""
    polygon_layer_str = POLYGON_LAYER.strip()
    if polygon_layer_str:
        arcpy.SpatialJoin_analysis(
            target_features=bus_stops_fc,
            join_features=polygon_layer_str,
            out_feature_class=JOINED_FC,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_ALL",
            match_option="INTERSECT",
        )
        logger.info(
            "Spatial join completed. Joined feature class created at: %s",
            JOINED_FC,
        )

        # Export joined data to CSV
        with (
            arcpy.da.SearchCursor(JOINED_FC, fields_to_export) as cursor,
            open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile,
        ):
            writer = csv.writer(csvfile)
            writer.writerow(fields_to_export)
            for row in cursor:
                writer.writerow(row)

        logger.info("CSV export completed. CSV file created at: %s", OUTPUT_CSV)
        current_fc = JOINED_FC
    else:
        logger.info("POLYGON_LAYER is empty. Skipping spatial join.")
        # Export the bus stops feature class to CSV so that merge can still work.
        with (
            arcpy.da.SearchCursor(bus_stops_fc, fields_to_export) as cursor,
            open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile,
        ):
            writer = csv.writer(csvfile)
            writer.writerow(fields_to_export)
            for row in cursor:
                writer.writerow(row)
        logger.info("CSV export completed. CSV file created at: %s", OUTPUT_CSV)
        current_fc = bus_stops_fc

    return current_fc


def read_and_filter_ridership_data() -> pd.DataFrame:
    """Read ridership data from EXCEL_FILE and optionally filter by routes.

    Return a DataFrame with aggregated totals.
    """
    df_excel = pd.read_excel(EXCEL_FILE)

    # Optional route filter
    if ROUTE_FILTER_LIST:
        initial_count = len(df_excel)
        df_excel = df_excel[df_excel["ROUTE_NAME"].isin(ROUTE_FILTER_LIST)]
        logger.info("Filtered Excel data to routes in %s.", ROUTE_FILTER_LIST)
        logger.info(
            "Records reduced from %d to %d.",
            initial_count,
            len(df_excel),
        )
    else:
        logger.info("No route filter applied.")

    # Calculate TOTAL
    df_excel["TOTAL"] = df_excel["XBOARDINGS"] + df_excel["XALIGHTINGS"]
    return df_excel


def merge_ridership_and_csv(
    df_excel: pd.DataFrame, fields_to_export: List[str]
) -> Tuple[pd.DataFrame, str]:
    """Merge ridership data (df_excel) with the CSV from the spatial join.

    Raises error if no polygon layer was provided and CSV does not exist.

    Returns:
      - df_joined: merged DataFrame
      - key_field: which field was used as the merge key (GTFS_KEY_FIELD or SHAPE_KEY_FIELD)
    """
    # Read from the CSV we created in the spatial join (or direct bus stops).
    df_csv = pd.read_csv(OUTPUT_CSV)

    # Merge on appropriate key (GTFS vs. shapefile)
    if IS_GTFS_INPUT:
        df_excel["STOP_ID"] = df_excel["STOP_ID"].astype(str)
        df_csv[GTFS_KEY_FIELD] = df_csv[GTFS_KEY_FIELD].astype(str)
        df_joined = pd.merge(
            df_excel, df_csv, left_on="STOP_ID", right_on=GTFS_KEY_FIELD, how="inner"
        )
        key_field = GTFS_KEY_FIELD
    else:
        df_excel["STOP_ID"] = df_excel["STOP_ID"].astype(str)
        df_csv[SHAPE_KEY_FIELD] = df_csv[SHAPE_KEY_FIELD].astype(str)
        df_joined = pd.merge(
            df_excel, df_csv, left_on="STOP_ID", right_on=SHAPE_KEY_FIELD, how="inner"
        )
        key_field = SHAPE_KEY_FIELD

    logger.info(
        "Data merged successfully. Number of matched bus stops: %d",
        len(df_joined),
    )
    return df_joined, key_field


def filter_matched_bus_stops(current_fc: str, df_joined: pd.DataFrame, key_field: str) -> str:
    """Filter the joined feature class to include only matched bus stops.

    Returns the path to the filtered shapefile.
    """
    matched_keys = df_joined[key_field].dropna().unique().tolist()
    if not matched_keys:
        logger.error("No matched bus stops found in Excel data. Exiting script.")
        exit()

    arcpy.MakeFeatureLayer_management(current_fc, "joined_lyr")
    fields = arcpy.ListFields(current_fc, key_field)
    if not fields:
        logger.error("Field '%s' not found in '%s'. Exiting.", key_field, current_fc)
        exit()

    field_type = fields[0].type
    field_delimited = arcpy.AddFieldDelimiters(current_fc, key_field)

    # Prepare values for WHERE clause based on field type
    if field_type in ["String", "Guid", "Date"]:
        formatted_keys = []
        for k in matched_keys:
            escaped = k.replace("'", "''")
            formatted_keys.append(f"'{escaped}'")
    elif field_type in ["Integer", "SmallInteger", "Double", "Single", "OID"]:
        formatted_keys = [str(k) for k in matched_keys]
    else:
        logger.error(
            "Unsupported field type '%s' for field '%s'. Exiting.",
            field_type,
            key_field,
        )
        exit()

    # Due to potential large number of keys, split into chunks
    chunk_size = 999
    where_clauses = []
    for i in range(0, len(formatted_keys), chunk_size):
        chunk = formatted_keys[i : i + chunk_size]
        clause = f"{field_delimited} IN ({', '.join(chunk)})"
        where_clauses.append(clause)

    full_where_clause = " OR ".join(where_clauses)
    logger.debug(
        "Constructed WHERE clause (first 200 chars): %s",
        full_where_clause[:200],
    )

    try:
        arcpy.SelectLayerByAttribute_management("joined_lyr", "NEW_SELECTION", full_where_clause)
    except arcpy.ExecuteError:
        logger.error("Failed SelectLayerByAttribute. Check WHERE clause syntax.")
        logger.error("WHERE clause attempted: %s", full_where_clause)
        raise

    selected_count = int(arcpy.GetCount_management("joined_lyr").getOutput(0))
    if selected_count == 0:
        logger.error("No features matched the WHERE clause. Exiting script.")
        exit()
    else:
        logger.info("Number of features selected: %d", selected_count)

    arcpy.CopyFeatures_management("joined_lyr", MATCHED_JOINED_FC)
    logger.info("Filtered joined feature class created at: %s", MATCHED_JOINED_FC)

    return MATCHED_JOINED_FC


def update_bus_stops_ridership(current_fc: str, df_joined: pd.DataFrame, key_field: str) -> None:
    """Add ridership fields to the bus stops shapefile and update them with data from df_joined."""
    ridership_fields = [
        ("XBOARD", "DOUBLE"),
        ("XALIGHT", "DOUBLE"),
        ("XTOTAL", "DOUBLE"),
    ]

    existing_fields = [f.name for f in arcpy.ListFields(current_fc)]
    for f_name, f_type in ridership_fields:
        if f_name not in existing_fields:
            arcpy.management.AddField(current_fc, f_name, f_type)

    logger.info("Ridership fields added (if not existing).")

    # Build dictionary from the joined DataFrame
    stop_ridership_dict = {}
    for _, row in df_joined.iterrows():
        code = row[key_field] if not pd.isna(row[key_field]) else None
        if code is not None:
            stop_ridership_dict[str(code)] = {
                "XBOARD": row["XBOARDINGS"],
                "XALIGHT": row["XALIGHTINGS"],
                "XTOTAL": row["TOTAL"],
            }

    with arcpy.da.UpdateCursor(current_fc, [key_field, "XBOARD", "XALIGHT", "XTOTAL"]) as cursor:
        for r in cursor:
            code_val = str(r[0])
            if code_val in stop_ridership_dict:
                r[1] = stop_ridership_dict[code_val]["XBOARD"]
                r[2] = stop_ridership_dict[code_val]["XALIGHT"]
                r[3] = stop_ridership_dict[code_val]["XTOTAL"]
            else:
                # Should not occur if we've filtered for matched features
                r[1], r[2], r[3] = 0, 0, 0
            cursor.updateRow(r)

    logger.info("Bus stops shapefile updated with ridership data at: %s", current_fc)


def aggregate_ridership(df_joined: pd.DataFrame) -> None:
    """Aggregate ridership by the polygon join field and update the polygon layer shapefile.

    Also exports the aggregated data to CSV for verification.
    """
    if not POLYGON_LAYER.strip():
        logger.info("POLYGON_LAYER is empty, so aggregation steps have been skipped.")
        return

    # Group by the designated polygon join field, e.g. "GEOID"
    df_agg = df_joined.groupby(POLYGON_JOIN_FIELD, as_index=False).agg(
        {"XBOARDINGS": "sum", "XALIGHTINGS": "sum", "TOTAL": "sum"}
    )
    logger.info("Ridership data aggregated by %s.", POLYGON_JOIN_FIELD)

    # ─── Export aggregated ridership spreadsheet ───
    agg_polygon_csv = os.path.join(OUTPUT_FOLDER, "agg_ridership_by_polygon.csv")
    df_agg.to_csv(agg_polygon_csv, index=False)
    logger.info("Aggregated ridership by polygon exported to: %s", agg_polygon_csv)

    # Copy the source polygons so we can add fields without touching the original
    arcpy.management.CopyFeatures(POLYGON_LAYER, POLYGON_WITH_RIDERSHIP_SHP)

    agg_fields = [
        ("XBOARD_SUM", "DOUBLE"),
        ("XALITE_SUM", "DOUBLE"),
        ("TOTAL_SUM", "DOUBLE"),
    ]

    existing_fields_blocks = [f.name for f in arcpy.ListFields(POLYGON_WITH_RIDERSHIP_SHP)]
    for f_name, f_type in agg_fields:
        if f_name not in existing_fields_blocks:
            arcpy.management.AddField(POLYGON_WITH_RIDERSHIP_SHP, f_name, f_type)

    logger.info(
        "Aggregation fields added to polygon shapefile (if not existing).",
    )

    # Build lookup dictionary for fast updates
    agg_dict = {
        row[POLYGON_JOIN_FIELD]: {
            "XBOARD_SUM": row["XBOARDINGS"],
            "XALITE_SUM": row["XALIGHTINGS"],
            "TOTAL_SUM": row["TOTAL"],
        }
        for _, row in df_agg.iterrows()
    }

    with arcpy.da.UpdateCursor(
        POLYGON_WITH_RIDERSHIP_SHP,
        [POLYGON_JOIN_FIELD, "XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"],
    ) as cursor:
        for rec in cursor:
            geoid = rec[0]
            if geoid in agg_dict:
                rec[1] = agg_dict[geoid]["XBOARD_SUM"]
                rec[2] = agg_dict[geoid]["XALITE_SUM"]
                rec[3] = agg_dict[geoid]["TOTAL_SUM"]
            else:
                rec[1], rec[2], rec[3] = 0, 0, 0
            cursor.updateRow(rec)

    logger.info(
        "Polygon shapefile updated with aggregated ridership data at: %s",
        POLYGON_WITH_RIDERSHIP_SHP,
    )


def process_stops_for_single_run() -> None:
    """(Helper) Original single-run flow (no splitting by route).

    Creates one shapefile for the entire network of bus stops,
    and now also exports an intermediate aggregated ridership CSV.
    """
    # Step 1: Create or identify the bus-stops feature class
    bus_stops_fc, fields_to_export = create_bus_stops_feature_class()

    # Step 2: Spatial Join (optional) → also exports CSV of bus stops (+ polygons)
    current_fc = spatial_join_bus_stops_to_polygons(bus_stops_fc, fields_to_export)

    # Step 3: Read ridership data from Excel & optionally filter by routes
    df_excel = read_and_filter_ridership_data()

    # ─── AGGREGATE PER STOP (network-wide) ───
    # Collapse any multi-route rows down to one row per STOP_ID
    df_excel = df_excel.groupby("STOP_ID", as_index=False).agg(
        {"XBOARDINGS": "sum", "XALIGHTINGS": "sum", "TOTAL": "sum"}
    )

    # Export the intermediate aggregated ridership spreadsheet
    agg_per_stop_csv = os.path.join(OUTPUT_FOLDER, "agg_ridership_per_stop.csv")
    df_excel.to_csv(agg_per_stop_csv, index=False)
    logger.info("Aggregated ridership per stop exported to: %s", agg_per_stop_csv)

    # Step 4: Merge ridership data with CSV from spatial join
    df_joined, key_field = merge_ridership_and_csv(df_excel, fields_to_export)

    # Step 4a: Filter to matched bus stops
    filtered_fc = filter_matched_bus_stops(current_fc, df_joined, key_field)

    # Step 5: Update the bus-stops shapefile with ridership fields
    update_bus_stops_ridership(filtered_fc, df_joined, key_field)

    # Steps 6 & 7: Aggregate ridership (optional, by polygon)
    aggregate_ridership(df_joined)

    logger.info("Single-run process complete.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point for the script.

    Either processes all routes at once (creating a single shapefile) or splits
    by route (creating multiple shapefiles), depending on SPLIT_BY_ROUTE.
    """
    # >>>>> NEW BRANCHING LOGIC <<<<<
    if not SPLIT_BY_ROUTE:
        # ---- Original single-run approach ----
        logger.info("SPLIT_BY_ROUTE = False. Running single shapefile process.")
        process_stops_for_single_run()

    else:
        # ---- Per-route approach (from the second script) ----
        logger.info("SPLIT_BY_ROUTE = True. Creating one shapefile per route.")

        # Step 1: Create or identify the bus stops feature class
        bus_stops_fc, fields_to_export = create_bus_stops_feature_class()

        # Step 2: Spatial Join (Optional) -> also exports CSV
        current_fc = spatial_join_bus_stops_to_polygons(bus_stops_fc, fields_to_export)

        # Step 3: Read ridership data from Excel & optionally filter by routes
        df_excel = read_and_filter_ridership_data()

        # Identify unique routes
        unique_routes = df_excel["ROUTE_NAME"].unique()
        logger.info("Found the following unique routes: %s", unique_routes)

        # For each route, merge, filter, and export a shapefile
        for route in unique_routes:
            logger.info("=== Processing route: %s ===", route)
            df_route = df_excel[df_excel["ROUTE_NAME"] == route].copy()
            if df_route.empty:
                logger.warning("No ridership data for route %s. Skipping.", route)
                continue

            # Merge data
            df_joined, key_field = merge_ridership_and_csv(df_route, fields_to_export)
            if df_joined.empty:
                logger.warning(
                    "No matched bus stops found for route %s. Skipping.",
                    route,
                )
                continue

            # Create a route-specific feature class path
            route_output_fc = os.path.join(OUTPUT_FOLDER, f"BusStops_{route}.shp")

            matched_keys = df_joined[key_field].dropna().unique().tolist()
            if not matched_keys:
                logger.warning("No matched bus stops found. Skipping route %s.", route)
                continue

            arcpy.MakeFeatureLayer_management(current_fc, "joined_lyr_route")
            field_delimited = arcpy.AddFieldDelimiters(current_fc, key_field)

            # We have to chunk keys to avoid 'IN' clause limit
            chunk_size = 999
            where_clauses = []
            for i in range(0, len(matched_keys), chunk_size):
                chunk = matched_keys[i : i + chunk_size]
                # Quote or not quote based on field type if needed.
                # Here we'll assume string type for simplicity:
                chunk_str = ", ".join(f"'{k}'" for k in chunk)
                where_clauses.append(f"{field_delimited} IN ({chunk_str})")

            route_where_clause = " OR ".join(where_clauses)
            arcpy.SelectLayerByAttribute_management(
                "joined_lyr_route", "NEW_SELECTION", route_where_clause
            )

            selected_count = int(arcpy.GetCount_management("joined_lyr_route").getOutput(0))
            if selected_count == 0:
                logger.warning(
                    "No bus stops found in FC for route %s. Skipping.",
                    route,
                )
                continue

            arcpy.CopyFeatures_management("joined_lyr_route", route_output_fc)
            logger.info("Route-specific shapefile created at: %s", route_output_fc)

            # Now update ridership fields
            update_bus_stops_ridership(route_output_fc, df_joined, key_field)

        # After all routes are processed, optionally aggregate if you want
        # aggregated polygon results across *all* stops (regardless of route).
        # If you only want polygons by route, you'd do a per-route polygon join.
        # For simplicity, we show it once for the entire dataset:
        aggregate_ridership(df_excel)

        logger.info("Per-route process complete.")


if __name__ == "__main__":
    main()
