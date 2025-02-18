"""
This script processes bus stop data by performing a spatial join with census blocks,
merging with ridership data from an Excel file, and filtering out bus stops that do
not have corresponding ridership data. The final outputs include updated shapefiles
with ridership information and aggregated data by census block.
"""

import csv
import os

import arcpy
import pandas as pd

# --------------------------------------------------------------------------
# User-defined variables
# --------------------------------------------------------------------------
CENSUS_BLOCKS = (
    r"C:\Your\Path\To\census_tabblock20_folder" # Replace with your folder path
    r"\tl_2024_50_tabblock20.shp"               # Replace with your .shp file name
)
# This can be either a .shp or a .txt (GTFS stops.txt)
BUS_STOPS_INPUT = (
    r"C:\Your\Path\To\bus_stops_folder" # Replace with your shapefile or GTFS folder path
    r"\stops.txt"                       # Rrepalce with your .shp or .txt file name
)
EXCEL_FILE = (
    r"C:\Your\Path\To\census_tabblock20_folder" # Replace with your ridership data folder path
    r"\STOP_USAGE_(BY_STOP_ID)_2024_12_23.xlsx" # Replace with your ridership data file name
)

OUTPUT_FOLDER = r"C:\Your\Path\To\Output_folder" # Replace with your desired output folder path
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Intermediate and final outputs
# If using GTFS, we will create a feature class from the stops.txt.
# Otherwise, if using a shapefile, we use it directly.
GTFS_STOPS_FC = os.path.join(OUTPUT_FOLDER, "bus_stops_generated.shp")
JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_JoinedBlocks.shp")
MATCHED_JOINED_FC = os.path.join(OUTPUT_FOLDER, "BusStops_Matched_JoinedBlocks.shp")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "bus_stops_with_census_blocks.csv")
BLOCKS_WITH_RIDERSHIP_SHP = os.path.join(
    OUTPUT_FOLDER, "census_blocks_with_ridership.shp"
)

# Field configuration:
# For GTFS input: fields are assumed to be "stop_code", "stop_id", "stop_name", "stop_lat", "stop_lon"
# For shapefile input: fields are assumed to "StopId", "StopNum", etc.
# Adjust as needed.

# For ridership data, Excel contains STOP_ID, STOP_NAME, XBOARDINGS, XALIGHTINGS.
# The final output expects a consistent set of fields. We'll standardize to "stop_code" for GTFS
# and "StopId" for shapefile. Ultimately, we need a common join key.
# For this example, let's assume:
# - GTFS: We'll join on stop_code
# - Shapefile: We'll join on StopId
#
# We'll unify this by internally standardizing to "stop_code" for GTFS and "StopId" for shapefile.
# The Excel uses STOP_ID, so we'll map accordingly.

# Decide which approach to take based on file type
IS_GTFS_INPUT = BUS_STOPS_INPUT.lower().endswith(".txt")

# Overwrite outputs
arcpy.env.overwriteOutput = True

# TODO: Make unique Census feature ID into a constant - GEOID, GEOIDFQ, GEOID20, GEOIDFQ20 are common

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

    # We'll export fields from this FC and also rename fields for consistency.
    # Fields to export to CSV after join:
    # We know GTFS stops have: stop_code, stop_id, stop_name, and after spatial join: GEOID20, GEOIDFQ20
    FIELDS_TO_EXPORT = ["stop_code", "stop_id", "stop_name", "GEOID20", "GEOIDFQ20"]

else:
    # We have a shapefile of bus stops directly
    BUS_STOPS_FC = BUS_STOPS_INPUT
    print("Using existing bus stops shapefile:\n{}".format(BUS_STOPS_FC))

    # Fields to export to CSV after join for shapefile scenario:
    # Assuming fields: StopId, StopNum, and after join: GEOID20, GEOIDFQ20
    FIELDS_TO_EXPORT = ["StopId", "StopNum", "GEOID20", "GEOIDFQ20"]

# --------------------------------------------------------------------------
# Step 2: Spatial Join - Join bus stops to census blocks
# --------------------------------------------------------------------------
arcpy.SpatialJoin_analysis(
    target_features=BUS_STOPS_FC,
    join_features=CENSUS_BLOCKS,
    out_feature_class=JOINED_FC,
    join_operation="JOIN_ONE_TO_ONE",
    join_type="KEEP_ALL",
    match_option="INTERSECT"
)
print("Spatial join completed. Joined feature class created at:\n{}".format(JOINED_FC))

# --------------------------------------------------------------------------
# Step 3: Export joined data to CSV
# --------------------------------------------------------------------------
with arcpy.da.SearchCursor(JOINED_FC, FIELDS_TO_EXPORT) as cursor, open(
    OUTPUT_CSV, 'w', newline='', encoding='utf-8'
) as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(FIELDS_TO_EXPORT)
    for row in cursor:
        writer.writerow(row)

print("CSV export completed. CSV file created at:\n{}".format(OUTPUT_CSV))

# --------------------------------------------------------------------------
# Step 4: Read ridership data from Excel and merge
# --------------------------------------------------------------------------
df_excel = pd.read_excel(EXCEL_FILE)

# Recalculate TOTAL
df_excel['TOTAL'] = df_excel['XBOARDINGS'] + df_excel['XALIGHTINGS']

# Read the joined CSV
df_csv = pd.read_csv(OUTPUT_CSV)

# We need to merge these dataframes. The Excel uses STOP_ID as the key.
# For GTFS scenario: we have stop_code and stop_id in the CSV.
# For shapefile scenario: we have StopId in the CSV.
#
# We'll handle each scenario separately:

if IS_GTFS_INPUT:
    # GTFS scenario: We'll join on stop_code <-> STOP_ID (since Excel STOP_ID matches GTFS stop_code typically)
    df_excel['STOP_ID'] = df_excel['STOP_ID'].astype(str)
    df_csv['stop_code'] = df_csv['stop_code'].astype(str)
    df_joined = pd.merge(
        df_excel, df_csv, left_on='STOP_ID', right_on='stop_code', how='inner'
    )
else:
    # Shapefile scenario: We'll join on StopId <-> STOP_ID
    df_excel['STOP_ID'] = df_excel['STOP_ID'].astype(str)
    df_csv['StopId'] = df_csv['StopId'].astype(str)
    df_joined = pd.merge(
        df_excel, df_csv, left_on='STOP_ID', right_on='StopId', how='inner'
    )

print("Data merged successfully. Number of matched bus stops: {}".format(len(df_joined)))

# --------------------------------------------------------------------------
# Step 4a: Filter joined_fc to include only matched bus stops
# --------------------------------------------------------------------------
# Define the key field based on input type
KEY_FIELD = 'stop_code' if IS_GTFS_INPUT else 'StopId'

# Extract unique keys from the joined dataframe
MATCHED_KEYS = df_joined[KEY_FIELD].dropna().unique().tolist()

if MATCHED_KEYS:
    # Determine the field type
    fields = arcpy.ListFields(JOINED_FC, KEY_FIELD)
    if not fields:
        print(f"Error: Field '{KEY_FIELD}' not found in '{JOINED_FC}'. Exiting script.")
        exit()
    FIELD_TYPE = fields[0].type  # e.g., 'String', 'Integer', etc.

    # Prepare the SQL where clause based on field type
    FIELD_DELIMITED = arcpy.AddFieldDelimiters(JOINED_FC, KEY_FIELD)

    if FIELD_TYPE in ['String', 'Guid', 'Date']:
        # String-based field types require values to be quoted
        FORMATTED_KEYS = [
            "'{}'".format(k.replace("'", "''")) for k in MATCHED_KEYS
        ]
    elif FIELD_TYPE in ['Integer', 'SmallInteger', 'Double', 'Single', 'OID']:
        # Numeric field types do not require quotes
        FORMATTED_KEYS = [str(k) for k in MATCHED_KEYS]
    else:
        print(f"Unsupported field type '{FIELD_TYPE}' for field '{KEY_FIELD}'. Exiting script.")
        exit()

    # Due to potential large number of keys, split into manageable chunks
    CHUNK_SIZE = 999  # Adjust based on database limitations
    WHERE_CLAUSES = []
    for i in range(0, len(FORMATTED_KEYS), CHUNK_SIZE):
        chunk = FORMATTED_KEYS[i:i + CHUNK_SIZE]
        clause = "{} IN ({})".format(FIELD_DELIMITED, ", ".join(chunk))
        WHERE_CLAUSES.append(clause)
    # Combine clauses with OR
    FULL_WHERE_CLAUSE = " OR ".join(WHERE_CLAUSES)

    print(
        f"Constructed WHERE clause for filtering: {FULL_WHERE_CLAUSE[:200]}..."
    )  # Print a snippet for verification

    # Create a feature layer
    arcpy.MakeFeatureLayer_management(JOINED_FC, "joined_lyr")

    # Select features that match the where clause
    try:
        arcpy.SelectLayerByAttribute_management(
            "joined_lyr", "NEW_SELECTION", FULL_WHERE_CLAUSE
        )
    except arcpy.ExecuteError:
        print("Failed to execute SelectLayerByAttribute. Please check the WHERE clause syntax.")
        print(f"WHERE clause attempted: {FULL_WHERE_CLAUSE}")
        raise

    # Check if any features were selected
    SELECTED_COUNT = int(arcpy.GetCount_management("joined_lyr").getOutput(0))
    if SELECTED_COUNT == 0:
        print("No features matched the WHERE clause. Exiting script.")
        exit()
    else:
        print(f"Number of features selected: {SELECTED_COUNT}")

    # Export the selected features to a new shapefile
    arcpy.CopyFeatures_management("joined_lyr", MATCHED_JOINED_FC)
    print(
        "Filtered joined feature class with matched bus stops created at:\n{}".format(
            MATCHED_JOINED_FC
        )
    )

    # Update joined_fc to point to the filtered feature class
    JOINED_FC = MATCHED_JOINED_FC
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

EXISTING_FIELDS = [f.name for f in arcpy.ListFields(JOINED_FC)]
for F_NAME, F_TYPE in RIDERSHIP_FIELDS:
    if F_NAME not in EXISTING_FIELDS:
        arcpy.management.AddField(JOINED_FC, F_NAME, F_TYPE)

print("Ridership fields added (if not existing).")

# Create a dictionary of keys to ridership
STOP_RIDERSHIP_DICT = {}
for idx, row in df_joined.iterrows():
    code = row[KEY_FIELD] if not pd.isna(row[KEY_FIELD]) else None
    if code is not None:
        STOP_RIDERSHIP_DICT[str(code)] = {
            'XBOARD': row['XBOARDINGS'],
            'XALIGHT': row['XALIGHTINGS'],
            'XTOTAL': row['TOTAL']
        }

with arcpy.da.UpdateCursor(
    JOINED_FC, [KEY_FIELD, "XBOARD", "XALIGHT", "XTOTAL"]
) as cursor:
    for r in cursor:
        CODE_VAL = str(r[0])
        if CODE_VAL in STOP_RIDERSHIP_DICT:
            r[1] = STOP_RIDERSHIP_DICT[CODE_VAL]['XBOARD']
            r[2] = STOP_RIDERSHIP_DICT[CODE_VAL]['XALIGHT']
            r[3] = STOP_RIDERSHIP_DICT[CODE_VAL]['XTOTAL']
            cursor.updateRow(r)
        else:
            # This should not occur as we've filtered matched features
            r[1] = 0
            r[2] = 0
            r[3] = 0
            cursor.updateRow(r)

print("Bus stops shapefile updated with ridership data at:\n{}".format(JOINED_FC))

# --------------------------------------------------------------------------
# Step 6: Aggregate ridership by GEOID20
# --------------------------------------------------------------------------
df_agg = df_joined.groupby('GEOID20', as_index=False).agg({
    'XBOARDINGS': 'sum',
    'XALIGHTINGS': 'sum',
    'TOTAL': 'sum'
})

print("Ridership data aggregated by GEOID20.")

# --------------------------------------------------------------------------
# Step 7: Create a new Census Blocks Shapefile with aggregated ridership
# --------------------------------------------------------------------------
arcpy.management.CopyFeatures(CENSUS_BLOCKS, BLOCKS_WITH_RIDERSHIP_SHP)

AGG_FIELDS = [
    ("XBOARD_SUM", "DOUBLE"),
    ("XALITE_SUM", "DOUBLE"),
    ("TOTAL_SUM", "DOUBLE")
]

EXISTING_FIELDS_BLOCKS = [f.name for f in arcpy.ListFields(BLOCKS_WITH_RIDERSHIP_SHP)]
for F_NAME, F_TYPE in AGG_FIELDS:
    if F_NAME not in EXISTING_FIELDS_BLOCKS:
        arcpy.management.AddField(BLOCKS_WITH_RIDERSHIP_SHP, F_NAME, F_TYPE)

print("Aggregation fields added to census blocks shapefile (if not existing).")

AGG_DICT = {}
for idx, row in df_agg.iterrows():
    geoid = row['GEOID20']
    AGG_DICT[geoid] = {
        'XBOARD_SUM': row['XBOARDINGS'],
        'XALITE_SUM': row['XALIGHTINGS'],
        'TOTAL_SUM': row['TOTAL']
    }

with arcpy.da.UpdateCursor(
    BLOCKS_WITH_RIDERSHIP_SHP,
    ["GEOID20", "XBOARD_SUM", "XALITE_SUM", "TOTAL_SUM"]
) as cursor:
    for r in cursor:
        geoid = r[0]
        if geoid in AGG_DICT:
            r[1] = AGG_DICT[geoid]['XBOARD_SUM']
            r[2] = AGG_DICT[geoid]['XALITE_SUM']
            r[3] = AGG_DICT[geoid]['TOTAL_SUM']
            cursor.updateRow(r)
        else:
            r[1] = 0
            r[2] = 0
            r[3] = 0
            cursor.updateRow(r)

print(
    "Census blocks shapefile updated with aggregated ridership data at:\n{}".format(
        BLOCKS_WITH_RIDERSHIP_SHP
    )
)
print("Process complete.")
