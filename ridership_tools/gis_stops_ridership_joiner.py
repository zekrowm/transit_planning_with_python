"""
This script processes bus stop data by optionally performing a spatial join with a polygon layer
(e.g., Census Blocks, Tracts, Places, etc.), merging with ridership data from an Excel file, and
filtering out bus stops that do not have corresponding ridership data. The final outputs include
updated shapefiles with ridership information and (optionally) aggregated data by the polygon layer.
"""

import csv
import os

import arcpy
import pandas as pd

# --------------------------------------------------------------------------
# User-defined variables
# --------------------------------------------------------------------------

# This can be either a .shp or a .txt (GTFS stops.txt)
BUS_STOPS_INPUT = r"C:\Path\To\Your\GTFS\stops.txt"  # Replace with your GTFS .txt or .shp file path

EXCEL_FILE      = r"C:\Path\To\Your\STOP_USAGE_(BY_STOP_ID).xlsx" # Replace with your ridership data file path

# Optional: Filter your Excel data for certain routes. If left empty, no filtering is applied.
# Use route name from EXCEL_FILE
ROUTE_FILTER_LIST = []  # Example: ["101", "202"]

OUTPUT_FOLDER   = r"C:\Path\To\Your\Output_folder"  # Replace with your desired output folder path
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Optional: Polygon features to join ridership data to. If empty, the spatial-join and aggregation steps will be skipped.
POLYGON_LAYER   = r"C:\Path\To\Your\PolygonLayer.shp" # Replace with your .shp file path or leave empty


# Intermediate and final outputs
# If using GTFS, we will create a feature class from the stops.txt.
# Otherwise, if using a shapefile, we use it directly.
GTFS_STOPS_FC = os.path.join(OUTPUT_FOLDER, "bus_stops_generated.shp")
JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_JoinedPolygon.shp")
MATCHED_JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_Matched_JoinedPolygon.shp")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "bus_stops_with_polygon.csv")

# If aggregation is performed, create a new polygon shapefile with aggregated ridership data.
POLYGON_WITH_RIDERSHIP_SHP = os.path.join(OUTPUT_FOLDER, "polygon_with_ridership.shp")

# Field configuration:
# For GTFS input: fields are assumed to be "stop_code", "stop_id", "stop_name", "stop_lat", "stop_lon"
# For shapefile input: fields are assumed to "StopId", "StopNum", etc.
# Adjust as needed.

# For ridership data, Excel contains STOP_ID, STOP_NAME, XBOARDINGS, XALIGHTINGS.
# The final output expects a consistent set of fields. We'll standardize to "stop_code" for GTFS
# and "StopId" for shapefile. The Excel uses STOP_ID, so we'll map accordingly.

# Decide which approach to take based on file type
IS_GTFS_INPUT = BUS_STOPS_INPUT.lower().endswith(".txt")

# Overwrite outputs
arcpy.env.overwriteOutput = True


# --------------------------------------------------------------------------
# Step 1: Create or identify the bus stops feature class
# --------------------------------------------------------------------------
if IS_GTFS_INPUT:
    # We have a GTFS stops.txt file. Convert it to a point feature class.
    arcpy.management.XYTableToPoint(
        in_table=BUS_STOPS_INPUT,
        out_feature_class=GTFS_STOPS_FC,
        x_field="stop_lon",
        y_field="stop_lat",
        coordinate_system=arcpy.SpatialReference(4326)  # WGS84
    )
    print("GTFS stops feature class created at:\n{}".format(GTFS_STOPS_FC))
    BUS_STOPS_FC = GTFS_STOPS_FC

    # Fields to export to CSV after join (GTFS scenario)
    FIELDS_TO_EXPORT = ["stop_code", "stop_id", "stop_name", "GEOID20", "GEOIDFQ20"]

else:
    # We have a shapefile of bus stops directly
    BUS_STOPS_FC = BUS_STOPS_INPUT
    print("Using existing bus stops shapefile:\n{}".format(BUS_STOPS_FC))

    # Fields to export to CSV after join (Shapefile scenario)
    FIELDS_TO_EXPORT = ["StopId", "StopNum", "GEOID20", "GEOIDFQ20"]


# --------------------------------------------------------------------------
# Step 2 (Optional): Spatial Join - Join bus stops to the polygon layer
#         Only if POLYGON_LAYER is not empty
# --------------------------------------------------------------------------
if POLYGON_LAYER.strip():
    arcpy.SpatialJoin_analysis(
        target_features=BUS_STOPS_FC,
        join_features=POLYGON_LAYER,
        out_feature_class=JOINED_FC,
        join_operation="JOIN_ONE_TO_ONE",
        join_type="KEEP_ALL",
        match_option="INTERSECT"
    )
    print("Spatial join completed. Joined feature class created at:\n{}".format(JOINED_FC))

    # Step 3: Export joined data to CSV
    with arcpy.da.SearchCursor(JOINED_FC, FIELDS_TO_EXPORT) as cursor, open(
        OUTPUT_CSV, 'w', newline='', encoding='utf-8'
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(FIELDS_TO_EXPORT)
        for row in cursor:
            writer.writerow(row)

    print("CSV export completed. CSV file created at:\n{}".format(OUTPUT_CSV))

    # For the next steps, we rely on the joined shapefile:
    CURRENT_FC = JOINED_FC

else:
    # If no polygon layer is provided, we skip the spatial join and CSV export,
    # but we can still proceed with route filtering and ridership updates
    print("POLYGON_LAYER is empty. Skipping spatial join.")
    CURRENT_FC = BUS_STOPS_FC
    # If there's no join, we won't have GEOID fields, but we can still attempt merges below.


# --------------------------------------------------------------------------
# Step 4: Read ridership data from Excel and (optionally) filter by route
# --------------------------------------------------------------------------
df_excel = pd.read_excel(EXCEL_FILE)

# Optional route filter
if ROUTE_FILTER_LIST:
    initial_excel_count = len(df_excel)
    df_excel = df_excel[df_excel['ROUTE_NAME'].isin(ROUTE_FILTER_LIST)]
    print(f"Filtered Excel data to routes in {ROUTE_FILTER_LIST}.")
    print(f"Records reduced from {initial_excel_count} to {len(df_excel)}.")
else:
    print("No route filter applied.")

# Recalculate TOTAL
df_excel['TOTAL'] = df_excel['XBOARDINGS'] + df_excel['XALIGHTINGS']

# If we performed a spatial join and exported CSV, read that CSV.
# Otherwise, if we skipped it (no polygon layer), we attempt to read from the original stops data.
if POLYGON_LAYER.strip():
    df_csv = pd.read_csv(OUTPUT_CSV)
else:
    # If not joined, you'd have to define how to read or parse your shapefile or GTFS
    # to get a DataFrame. For simplicity here, we'll read from a CSV if you already have one
    # or skip. Adjust to your use-case.
    raise ValueError("No polygon layer given; define how to handle bus stops for ridership merge.")


# We need to merge these dataframes. The Excel uses STOP_ID as the key.
# For GTFS scenario: we have stop_code and stop_id in the CSV.
# For shapefile scenario: we have StopId in the CSV.
if IS_GTFS_INPUT:
    # GTFS scenario: We'll join on stop_code <-> STOP_ID
    df_excel['STOP_ID'] = df_excel['STOP_ID'].astype(str)
    df_csv['stop_code'] = df_csv['stop_code'].astype(str)
    df_joined = pd.merge(
        df_excel, df_csv, left_on='STOP_ID', right_on='stop_code', how='inner'
    )
    KEY_FIELD = 'stop_code'
else:
    # Shapefile scenario: We'll join on StopId <-> STOP_ID
    df_excel['STOP_ID'] = df_excel['STOP_ID'].astype(str)
    df_csv['StopId'] = df_csv['StopId'].astype(str)
    df_joined = pd.merge(
        df_excel, df_csv, left_on='STOP_ID', right_on='StopId', how='inner'
    )
    KEY_FIELD = 'StopId'

print("Data merged successfully. Number of matched bus stops: {}".format(len(df_joined)))


# --------------------------------------------------------------------------
# Step 4a: Filter the joined feature class to include only matched bus stops
# --------------------------------------------------------------------------
MATCHED_KEYS = df_joined[KEY_FIELD].dropna().unique().tolist()

if MATCHED_KEYS:
    # Create a feature layer for the previously joined FC
    arcpy.MakeFeatureLayer_management(CURRENT_FC, "joined_lyr")

    # Determine the field type
    fields = arcpy.ListFields(CURRENT_FC, KEY_FIELD)
    if not fields:
        print(f"Error: Field '{KEY_FIELD}' not found in '{CURRENT_FC}'. Exiting script.")
        exit()

    FIELD_TYPE = fields[0].type  # e.g., 'String', 'Integer', etc.
    FIELD_DELIMITED = arcpy.AddFieldDelimiters(CURRENT_FC, KEY_FIELD)

    # Prepare the values for WHERE clause based on field type
    if FIELD_TYPE in ['String', 'Guid', 'Date']:
        FORMATTED_KEYS = [f"'{k.replace(\"'\", \"''\")}'" for k in MATCHED_KEYS]
    elif FIELD_TYPE in ['Integer', 'SmallInteger', 'Double', 'Single', 'OID']:
        FORMATTED_KEYS = [str(k) for k in MATCHED_KEYS]
    else:
        print(f"Unsupported field type '{FIELD_TYPE}' for field '{KEY_FIELD}'. Exiting script.")
        exit()

    # Due to potential large number of keys, split into chunks
    CHUNK_SIZE = 999
    WHERE_CLAUSES = []
    for i in range(0, len(FORMATTED_KEYS), CHUNK_SIZE):
        chunk = FORMATTED_KEYS[i:i + CHUNK_SIZE]
        clause = "{} IN ({})".format(FIELD_DELIMITED, ", ".join(chunk))
        WHERE_CLAUSES.append(clause)
    FULL_WHERE_CLAUSE = " OR ".join(WHERE_CLAUSES)

    print(f"Constructed WHERE clause (first 200 chars): {FULL_WHERE_CLAUSE[:200]}...")

    # Select features that match
    try:
        arcpy.SelectLayerByAttribute_management("joined_lyr", "NEW_SELECTION", FULL_WHERE_CLAUSE)
    except arcpy.ExecuteError:
        print("Failed SelectLayerByAttribute. Check WHERE clause syntax.")
        print(f"WHERE clause attempted: {FULL_WHERE_CLAUSE}")
        raise

    SELECTED_COUNT = int(arcpy.GetCount_management("joined_lyr").getOutput(0))
    if SELECTED_COUNT == 0:
        print("No features matched the WHERE clause. Exiting script.")
        exit()
    else:
        print(f"Number of features selected: {SELECTED_COUNT}")

    # Export the selected features to a new shapefile
    arcpy.CopyFeatures_management("joined_lyr", MATCHED_JOINED_FC)
    print("Filtered joined feature class created at:\n{}".format(MATCHED_JOINED_FC))

    CURRENT_FC = MATCHED_JOINED_FC

else:
    print("No matched bus stops found in Excel data. Exiting script.")
    exit()


# --------------------------------------------------------------------------
# Step 5: Update the Bus Stops Shapefile with Ridership Data
# --------------------------------------------------------------------------
# Add fields for ridership: XBOARD, XALIGHT, XTOTAL
RIDERSHIP_FIELDS = [
    ("XBOARD", "DOUBLE"),
    ("XALIGHT", "DOUBLE"),
    ("XTOTAL", "DOUBLE")
]

EXISTING_FIELDS = [f.name for f in arcpy.ListFields(CURRENT_FC)]
for F_NAME, F_TYPE in RIDERSHIP_FIELDS:
    if F_NAME not in EXISTING_FIELDS:
        arcpy.management.AddField(CURRENT_FC, F_NAME, F_TYPE)

print("Ridership fields added (if not existing).")

# Create a dictionary of keys to ridership
STOP_RIDERSHIP_DICT = {}
for _, row in df_joined.iterrows():
    code = row[KEY_FIELD] if not pd.isna(row[KEY_FIELD]) else None
    if code is not None:
        STOP_RIDERSHIP_DICT[str(code)] = {
            'XBOARD': row['XBOARDINGS'],
            'XALIGHT': row['XALIGHTINGS'],
            'XTOTAL': row['TOTAL']
        }

with arcpy.da.UpdateCursor(CURRENT_FC, [KEY_FIELD, "XBOARD", "XALIGHT", "XTOTAL"]) as cursor:
    for r in cursor:
        code_val = str(r[0])
        if code_val in STOP_RIDERSHIP_DICT:
            r[1] = STOP_RIDERSHIP_DICT[code_val]['XBOARD']
            r[2] = STOP_RIDERSHIP_DICT[code_val]['XALIGHT']
            r[3] = STOP_RIDERSHIP_DICT[code_val]['XTOTAL']
        else:
            # Should not occur because we've filtered for matched features
            r[1], r[2], r[3] = 0, 0, 0
        cursor.updateRow(r)

print("Bus stops shapefile updated with ridership data at:\n{}".format(CURRENT_FC))


# --------------------------------------------------------------------------
# Steps 6 & 7: Aggregate ridership (Optional) - only if we have a polygon
# --------------------------------------------------------------------------
if POLYGON_LAYER.strip():
    # Because we used "GEOID20" in the join, we can group by GEOID20
    df_agg = df_joined.groupby('GEOID20', as_index=False).agg({
        'XBOARDINGS': 'sum',
        'XALIGHTINGS': 'sum',
        'TOTAL': 'sum'
    })
    print("Ridership data aggregated by GEOID20.")

    # Create a new polygon layer shapefile with aggregated ridership
    arcpy.management.CopyFeatures(POLYGON_LAYER, POLYGON_WITH_RIDERSHIP_SHP)

    AGG_FIELDS = [
        ("XBOARD_SUM", "DOUBLE"),
        ("XALITE_SUM", "DOUBLE"),
        ("TOTAL_SUM", "DOUBLE")
    ]

    EXISTING_FIELDS_BLOCKS = [f.name for f in arcpy.ListFields(POLYGON_WITH_RIDERSHIP_SHP)]
    for F_NAME, F_TYPE in AGG_FIELDS:
        if F_NAME not in EXISTING_FIELDS_BLOCKS:
            arcpy.management.AddField(POLYGON_WITH_RIDERSHIP_SHP, F_NAME, F_TYPE)

    print("Aggregation fields added to polygon shapefile (if not existing).")

    # Build a dictionary from aggregated data
    AGG_DICT = {}
    for _, row in df_agg.iterrows():
        geoid = row['GEOID20']
        AGG_DICT[geoid] = {
            'XBOARD_SUM': row['XBOARDINGS'],
            'XALITE_SUM': row['XALIGHTINGS'],
            'TOTAL_SUM': row['TOTAL']
        }

    with arcpy.da.UpdateCursor(
        POLYGON_WITH_RIDERSHIP_SHP,
        ["GEOID20", "XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"]
    ) as cursor:
        for r in cursor:
            geoid = r[0]
            if geoid in AGG_DICT:
                r[1] = AGG_DICT[geoid]['XBOARD_SUM']
                r[2] = AGG_DICT[geoid]['XALITE_SUM']
                r[3] = AGG_DICT[geoid]['TOTAL_SUM']
            else:
                r[1] = 0
                r[2] = 0
                r[3] = 0
            cursor.updateRow(r)

    print("Polygon shapefile updated with aggregated ridership data at:\n{}".format(
        POLYGON_WITH_RIDERSHIP_SHP
    ))
else:
    print("POLYGON_LAYER is empty, so aggregation steps have been skipped.")

print("Process complete.")
