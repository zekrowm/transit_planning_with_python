"""Script to automate the calculation of service population demographics for EIA documents."""

import os

import arcpy
import pandas as pd

# Allow overwriting outputs in the ArcPy environment.
arcpy.env.overwriteOutput = True

# -----------------------------------------------------------------------------
# CONFIGURATION SECTION
# -----------------------------------------------------------------------------

ANALYSIS_MODE = "network"  # Options: "network", "route", or "stop"

# Paths
GTFS_DATA_PATH = r"Path\To\Your\GTFS_Folder"
SHAPEFILE_PATH = r""

# Shapefile filtering
SHAPEFILE_FILTER_FIELD = "ROUTE_NUMB"
SHAPEFILE_FILTER_VALUES = []

DEMOGRAPHICS_FC_PATH = (r"Path\To\Your\Census_Demographics.shp")
OUTPUT_DIRECTORY = (r"Path\To\Your\Output_Folder")

ROUTES_TO_INCLUDE = ["101", "202", "303"]
ROUTES_TO_EXCLUDE = ["9999A", "9999B", "9999C"]

STOP_IDS_TO_INCLUDE = []
STOP_IDS_TO_EXCLUDE = []

BUFFER_DISTANCE = 0.25  # miles
LARGE_BUFFER_DISTANCE = 2.0  # miles
STOP_IDS_LARGE_BUFFER = []

# Filter for specific State/County
STATEFP_FILTER = ["51"]
COUNTYFP_FILTER = ["059"]

# Fields in your demographic data to be partially weighted
SYNTHETIC_FIELDS = ["Tot_Pop", "Total_Whit", "Minority", "TotHH", "Non_Low_in", "Lowincome_"]

CRS_EPSG_CODE = 3395

REQUIRED_GTFS_FILES = ["trips.txt", "stop_times.txt", "routes.txt", "stops.txt", "calendar.txt"]

# Name for the file geodatabase we'll create (or use if it already exists).
GDB_NAME = "analysis.gdb"


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def ensure_gdb_exists(output_dir, gdb_name):
    """Create a file geodatabase if it does not already exist.

    Return the path to the GDB.
    """
    gdb_path = os.path.join(output_dir, gdb_name)
    if not arcpy.Exists(gdb_path):
        os.makedirs(output_dir, exist_ok=True)
        arcpy.management.CreateFileGDB(output_dir, gdb_name)
    return gdb_path


def load_gtfs_data(gtfs_path):
    """Load GTFS CSVs into pandas DataFrames."""
    for filename in REQUIRED_GTFS_FILES:
        full_path = os.path.join(gtfs_path, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Missing GTFS file: {filename}")

    trips = pd.read_csv(os.path.join(gtfs_path, "trips.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"))
    routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"))
    stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"))
    calendar = pd.read_csv(os.path.join(gtfs_path, "calendar.txt"))

    return trips, stop_times, routes_df, stops_df, calendar


def filter_weekday_service(calendar_df):
    """Return service IDs that run Monday–Friday."""
    weekday_filter = (
        (calendar_df["monday"] == 1)
        & (calendar_df["tuesday"] == 1)
        & (calendar_df["wednesday"] == 1)
        & (calendar_df["thursday"] == 1)
        & (calendar_df["friday"] == 1)
    )
    return calendar_df[weekday_filter]["service_id"]


def apply_county_state_filter(demog_fc, statefp_filter, countyfp_filter):
    """Filter a demographics feature class by optional STATEFP and COUNTYFP code lists.

    Return the name of an in-memory feature class containing the selection.
    """
    if not statefp_filter and not countyfp_filter:
        print("No STATEFP/COUNTYFP filter applied; processing all features.")
        return demog_fc

    demog_lyr = "demog_lyr"
    if arcpy.Exists(demog_lyr):
        arcpy.management.Delete(demog_lyr)
    arcpy.management.MakeFeatureLayer(demog_fc, demog_lyr)

    where_clauses = []
    if statefp_filter:
        statefp_strlist = ", ".join(f"'{val}'" for val in statefp_filter)
        where_clauses.append(f"STATEFP IN ({statefp_strlist})")
    if countyfp_filter:
        countyfp_strlist = ", ".join(f"'{val}'" for val in countyfp_filter)
        where_clauses.append(f"COUNTYFP IN ({countyfp_strlist})")

    final_where = " AND ".join(where_clauses)
    before_count = int(arcpy.management.GetCount(demog_lyr).getOutput(0))
    arcpy.management.SelectLayerByAttribute(demog_lyr, "NEW_SELECTION", final_where)
    after_count = int(arcpy.management.GetCount(demog_lyr).getOutput(0))
    print(
        f"Applied STATEFP/COUNTYFP filter (State: {statefp_filter}, County: {countyfp_filter}). "
        f"Reduced from {before_count} to {after_count} records."
    )

    filtered_fc = r"in_memory\demog_filtered"
    if arcpy.Exists(filtered_fc):
        arcpy.management.Delete(filtered_fc)
    arcpy.management.CopyFeatures(demog_lyr, filtered_fc)
    return filtered_fc


def get_included_routes(routes_df, routes_to_include, routes_to_exclude):
    """Apply route filters to routes_df and return the filtered DataFrame."""
    filtered = routes_df.copy()
    if routes_to_include:
        filtered = filtered[filtered["route_short_name"].isin(routes_to_include)]
    if routes_to_exclude:
        filtered = filtered[~filtered["route_short_name"].isin(routes_to_exclude)]

    final_count = len(filtered)
    print(f"Including {final_count} routes after applying include/exclude lists.")
    included_names = ", ".join(sorted(filtered["route_short_name"].unique()))
    if included_names:
        print(f"  Included Routes: {included_names}")
    else:
        print("  Included Routes: None")
    return filtered


def get_included_stops(stops_df, stop_ids_to_include, stop_ids_to_exclude):
    """Apply stop filters to a DataFrame of stops.

    The DataFrame may be the result of merging GTFS trips, stop_times, and stops.
    """
    filtered = stops_df.copy()

    if stops_df["stop_id"].dtype == "O":
        stop_ids_to_include = [str(s) for s in stop_ids_to_include]
        stop_ids_to_exclude = [str(s) for s in stop_ids_to_exclude]
    else:
        filtered["stop_id"] = filtered["stop_id"].astype(int)
        stop_ids_to_include = [int(s) for s in stop_ids_to_include]
        stop_ids_to_exclude = [int(s) for s in stop_ids_to_exclude]

    if stop_ids_to_include:
        filtered = filtered[filtered["stop_id"].isin(stop_ids_to_include)]
    if stop_ids_to_exclude:
        filtered = filtered[~filtered["stop_id"].isin(stop_ids_to_exclude)]

    final_count = len(filtered)
    print(f"Including {final_count} stops after applying stop include/exclude lists.")
    return filtered


def pick_buffer_distance(stop_id, normal_buffer, large_buffer, large_buffer_ids):
    """Return the normal or large buffer distance in miles depending on stop_id."""
    str_stop_id = str(stop_id)
    large_list_str = [str(s) for s in large_buffer_ids]
    if str_stop_id in large_list_str:
        return large_buffer
    return normal_buffer


def create_feature_class_from_stops(stops_df, out_fc, spatial_ref):
    """Create a point feature class from stops.

    Given a DataFrame of stops with 'stop_id', 'stop_lat', and 'stop_lon' columns,
    add an 'dist_m' field for buffer distance in meters.
    """
    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(out_fc),
        out_name=os.path.basename(out_fc),
        geometry_type="POINT",
        spatial_reference=spatial_ref,
    )
    arcpy.management.AddField(out_fc, "stop_id", "TEXT")
    arcpy.management.AddField(out_fc, "dist_m", "DOUBLE")

    with arcpy.da.InsertCursor(out_fc, ["stop_id", "dist_m", "SHAPE@"]) as cur:
        for _, row in stops_df.iterrows():
            stop_id = row["stop_id"]
            lat = float(row["stop_lat"])
            lon = float(row["stop_lon"])
            dist_meters = (
                pick_buffer_distance(
                    stop_id,
                    normal_buffer=BUFFER_DISTANCE,
                    large_buffer=LARGE_BUFFER_DISTANCE,
                    large_buffer_ids=STOP_IDS_LARGE_BUFFER,
                )
                * 1609.34
            )

            pt = arcpy.Point(lon, lat)
            pt_geom = arcpy.PointGeometry(pt, arcpy.SpatialReference(4326))
            pt_geom_projected = pt_geom.projectAs(spatial_ref)

            cur.insertRow([str(stop_id), dist_meters, pt_geom_projected])
    return out_fc


def buffer_feature_class_with_variable_distance(in_fc, out_fc, dist_field="dist_m"):
    """Buffer the input feature class using a field-based distance.

    Use the buffer distance values stored in dist_field.
    """
    arcpy.analysis.Buffer(
        in_features=in_fc,
        out_feature_class=out_fc,
        buffer_distance_or_field=dist_field,
        line_side="FULL",
        line_end_type="ROUND",
        dissolve_option="ALL",
    )
    return out_fc


def intersect_and_partial_weight(demog_fc, buffer_fc, synthetic_fields, out_fc):
    """Intersect the buffer feature class with demographics and compute partial-weighted synthetic fields.

    Ensure that demog_fc has an 'area_ac_og' field representing original area in acres.
    Intersect buffer_fc with demog_fc to produce out_fc.
    Add 'area_ac_cl', 'area_perc', and synthetic_<field> columns.
    Return a dict mapping each synthetic_<field> to its summed value.
    """
    temp_demog = r"in_memory\demog_temp_area"
    if arcpy.Exists(temp_demog):
        arcpy.management.Delete(temp_demog)
    arcpy.management.CopyFeatures(demog_fc, temp_demog)

    existing_fields = [f.name.lower() for f in arcpy.ListFields(temp_demog)]
    if "area_ac_og" not in existing_fields:
        arcpy.management.AddField(temp_demog, "area_ac_og", "DOUBLE")
        arcpy.management.CalculateGeometryAttributes(
            temp_demog, [["area_ac_og", "AREA_GEODESIC"]], area_unit="ACRES"
        )

    arcpy.analysis.Intersect(
        in_features=[temp_demog, buffer_fc],
        out_feature_class=out_fc,
    )

    arcpy.management.AddField(out_fc, "area_ac_cl", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        out_fc, [["area_ac_cl", "AREA_GEODESIC"]], area_unit="ACRES"
    )

    arcpy.management.AddField(out_fc, "area_perc", "DOUBLE")
    expression = "!area_ac_cl! / !area_ac_og!"
    arcpy.management.CalculateField(out_fc, "area_perc", expression, "PYTHON3")

    sum_dict = {}
    for fld in synthetic_fields:
        syn_fld = f"synthetic_{fld}"
        arcpy.management.AddField(out_fc, syn_fld, "DOUBLE")
        calc_expr = f"!area_perc! * !{fld}!"
        arcpy.management.CalculateField(out_fc, syn_fld, calc_expr, "PYTHON3")
        sum_dict[syn_fld] = 0.0

    stats_table = r"in_memory\stats_table"
    if arcpy.Exists(stats_table):
        arcpy.management.Delete(stats_table)

    stats_fields = [[f"synthetic_{fld}", "SUM"] for fld in synthetic_fields]
    arcpy.analysis.Statistics(out_fc, stats_table, stats_fields)

    with arcpy.da.SearchCursor(
        stats_table, [f"SUM_synthetic_{f}" for f in synthetic_fields]
    ) as cur:
        for row in cur:
            for i, fld in enumerate(synthetic_fields):
                sum_dict[f"synthetic_{fld}"] = round(row[i], 0)

    return sum_dict


def compute_demographics_totals(demog_fc, fields):
    """Compute total sums for specified fields in the demographic feature class.

    Return a dict mapping each field name to its unweighted sum value.
    """
    stats_table = r"in_memory\demog_totals"
    if arcpy.Exists(stats_table):
        arcpy.management.Delete(stats_table)

    stat_fields = [[fld, "SUM"] for fld in fields]
    arcpy.analysis.Statistics(demog_fc, stats_table, stat_fields)

    totals = {fld: 0.0 for fld in fields}
    sum_field_names = [f"SUM_{fld}" for fld in fields]
    with arcpy.da.SearchCursor(stats_table, sum_field_names) as cur:
        for row in cur:
            for i, fld in enumerate(fields):
                totals[fld] = float(row[i])
    return totals


def build_output_dict(partial_sums_dict, total_sums_dict, synthetic_fields):
    """Build a dictionary of partial and percentage values for synthetic fields.

    Given partial_sums_dict mapping synthetic_<field> names to summed values and
    total_sums_dict mapping field names to total values, return a dict that
    includes each partial value and its percentage of the total (with keys
    '<field>' and 'Pct_<field>').
    """
    result = {}
    for fld in synthetic_fields:
        part_val = partial_sums_dict[f"synthetic_{fld}"]
        tot_val = total_sums_dict[fld]
        pct_val = (part_val / tot_val * 100.0) if tot_val else 0.0

        result[fld] = round(part_val, 0)
        result[f"Pct_{fld}"] = round(pct_val, 2)

    return result


def export_summary_to_excel(data_dict, output_path):
    """Export a dictionary of scalar values to a one-row Excel file.

    Write the keys of data_dict as column names and their values in the first
    row to output_path.
    """
    summary_data = {k: [v] for k, v in data_dict.items()}
    summary_df = pd.DataFrame(summary_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_excel(output_path, index=False)
    print(f"Exported Excel summary: {output_path}")


# -----------------------------------------------------------------------------
# ANALYSIS FUNCTIONS (GTFS)
# -----------------------------------------------------------------------------


def do_network_analysis(
    trips, stop_times, routes_df, stops_df, demog_fc, all_demog_totals, gdb_path
):
    """Perform network-wide analysis of service buffers.

    Create a single dissolved buffer from all included stops, intersect with
    demographics, and output summary results.
    """
    print("\n=== Network-wide Analysis ===")
    final_routes_df = get_included_routes(routes_df, ROUTES_TO_INCLUDE, ROUTES_TO_EXCLUDE)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting network analysis.")
        return

    trips_merged = pd.merge(trips, final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    final_stops_df = get_included_stops(merged_data, STOP_IDS_TO_INCLUDE, STOP_IDS_TO_EXCLUDE)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting network analysis.")
        return

    sr_proj = arcpy.SpatialReference(CRS_EPSG_CODE)
    inmem_stops_fc = r"in_memory\final_stops"
    if arcpy.Exists(inmem_stops_fc):
        arcpy.management.Delete(inmem_stops_fc)
    create_feature_class_from_stops(final_stops_df, inmem_stops_fc, sr_proj)

    network_buffer_fc = os.path.join(gdb_path, "network_buffer")
    if arcpy.Exists(network_buffer_fc):
        arcpy.management.Delete(network_buffer_fc)
    buffer_feature_class_with_variable_distance(inmem_stops_fc, network_buffer_fc)

    clipped_fc = os.path.join(gdb_path, "all_routes_service_buffer_data")
    if arcpy.Exists(clipped_fc):
        arcpy.management.Delete(clipped_fc)

    sums = intersect_and_partial_weight(
        demog_fc=demog_fc,
        buffer_fc=network_buffer_fc,
        synthetic_fields=SYNTHETIC_FIELDS,
        out_fc=clipped_fc,
    )

    final_dict = build_output_dict(sums, all_demog_totals, SYNTHETIC_FIELDS)

    print("Network-wide totals (partial vs. % of total):")
    for fld in SYNTHETIC_FIELDS:
        print(f"  {fld}: {int(final_dict[fld])},  Pct_{fld}: {final_dict[f'Pct_{fld}']:.2f}%")

    xlsx_path = os.path.join(OUTPUT_DIRECTORY, "all_routes_service_buffer_data.xlsx")
    export_summary_to_excel(final_dict, xlsx_path)


def do_route_by_route_analysis(
    trips, stop_times, routes_df, stops_df, demog_fc, all_demog_totals, gdb_path
):
    """Perform route-by-route analysis of service buffers.

    Create dissolved buffers by route_short_name, intersect with demographics,
    and output summary results for each route.
    """
    print("\n=== Route-by-Route Analysis ===")
    final_routes_df = get_included_routes(routes_df, ROUTES_TO_INCLUDE, ROUTES_TO_EXCLUDE)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting route-by-route analysis.")
        return

    trips_merged = pd.merge(trips, final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    final_stops_df = get_included_stops(merged_data, STOP_IDS_TO_INCLUDE, STOP_IDS_TO_EXCLUDE)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting route-by-route analysis.")
        return

    sr_proj = arcpy.SpatialReference(CRS_EPSG_CODE)

    inmem_stops_fc = r"in_memory\final_stops_route"
    if arcpy.Exists(inmem_stops_fc):
        arcpy.management.Delete(inmem_stops_fc)
    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(inmem_stops_fc),
        out_name=os.path.basename(inmem_stops_fc),
        geometry_type="POINT",
        spatial_reference=sr_proj,
    )
    arcpy.management.AddField(inmem_stops_fc, "stop_id", "TEXT")
    arcpy.management.AddField(inmem_stops_fc, "route_short_name", "TEXT")
    arcpy.management.AddField(inmem_stops_fc, "dist_m", "DOUBLE")

    with arcpy.da.InsertCursor(
        inmem_stops_fc, ["stop_id", "route_short_name", "dist_m", "SHAPE@"]
    ) as cur:
        for _, row in final_stops_df.iterrows():
            stop_id = row["stop_id"]
            route_name = row["route_short_name"]
            lat = float(row["stop_lat"])
            lon = float(row["stop_lon"])
            dist_meters = (
                pick_buffer_distance(
                    stop_id, BUFFER_DISTANCE, LARGE_BUFFER_DISTANCE, STOP_IDS_LARGE_BUFFER
                )
                * 1609.34
            )

            pt = arcpy.Point(lon, lat)
            pt_geom = arcpy.PointGeometry(pt, arcpy.SpatialReference(4326))
            pt_geom_projected = pt_geom.projectAs(sr_proj)

            cur.insertRow([str(stop_id), str(route_name), dist_meters, pt_geom_projected])

    route_buffer_fc = os.path.join(gdb_path, "route_buffers")
    if arcpy.Exists(route_buffer_fc):
        arcpy.management.Delete(route_buffer_fc)
    arcpy.analysis.Buffer(
        in_features=inmem_stops_fc,
        out_feature_class=route_buffer_fc,
        buffer_distance_or_field="dist_m",
        line_side="FULL",
        line_end_type="ROUND",
        dissolve_option="LIST",
        dissolve_field=["route_short_name"],
    )

    route_field = "route_short_name"
    unique_route_names = set()
    with arcpy.da.SearchCursor(route_buffer_fc, [route_field]) as cur:
        for row in cur:
            unique_route_names.add(row[0])

    for route_name in sorted(unique_route_names):
        print(f"\nProcessing route: {route_name}")
        route_lyr = "route_lyr"
        if arcpy.Exists(route_lyr):
            arcpy.management.Delete(route_lyr)
        arcpy.management.MakeFeatureLayer(route_buffer_fc, route_lyr)
        where_clause = f"{route_field} = '{route_name}'"
        arcpy.management.SelectLayerByAttribute(route_lyr, "NEW_SELECTION", where_clause)

        count_sel = int(arcpy.management.GetCount(route_lyr).getOutput(0))
        if count_sel == 0:
            print(f"No stops found for route '{route_name}' - skipping.")
            continue

        out_fc_name = f"route_{route_name}_service_buffer_data".replace(" ", "_").replace("-", "_")
        route_out_fc = os.path.join(gdb_path, out_fc_name)
        if arcpy.Exists(route_out_fc):
            arcpy.management.Delete(route_out_fc)

        sums = intersect_and_partial_weight(demog_fc, route_lyr, SYNTHETIC_FIELDS, route_out_fc)
        final_dict = build_output_dict(sums, all_demog_totals, SYNTHETIC_FIELDS)

        for fld in SYNTHETIC_FIELDS:
            print(f"  {fld}: {int(final_dict[fld])},  Pct_{fld}: {final_dict[f'Pct_{fld}']:.2f}%")

        xlsx_path = os.path.join(OUTPUT_DIRECTORY, f"{route_name}_service_buffer_data.xlsx")
        export_summary_to_excel(final_dict, xlsx_path)


def do_stop_by_stop_analysis(
    trips, stop_times, routes_df, stops_df, demog_fc, all_demog_totals, gdb_path
):
    """Perform stop-by-stop analysis of service buffers.

    For each included stop, create an individual buffer, intersect with demographics,
    and export feature classes and Excel summaries with partial-weighted sums and percentages.
    """
    print("\n=== Stop-by-Stop Analysis ===")
    final_routes_df = get_included_routes(routes_df, ROUTES_TO_INCLUDE, ROUTES_TO_EXCLUDE)
    if final_routes_df.empty:
        print("No routes remain after route filters. Aborting stop-by-stop analysis.")
        return

    trips_merged = pd.merge(trips, final_routes_df[["route_id", "route_short_name"]], on="route_id")
    merged_data = pd.merge(stop_times, trips_merged, on="trip_id")
    merged_data = pd.merge(merged_data, stops_df, on="stop_id")

    final_stops_df = get_included_stops(merged_data, STOP_IDS_TO_INCLUDE, STOP_IDS_TO_EXCLUDE)
    if final_stops_df.empty:
        print("No stops remain after stop filters. Aborting stop-by-stop analysis.")
        return

    sr_proj = arcpy.SpatialReference(CRS_EPSG_CODE)
    unique_stop_ids = sorted(set(final_stops_df["stop_id"].tolist()))

    for sid in unique_stop_ids:
        single_stop_df = final_stops_df[final_stops_df["stop_id"] == sid]
        if single_stop_df.empty:
            continue

        stop_id_str = str(sid)

        single_fc = r"in_memory\single_stop"
        if arcpy.Exists(single_fc):
            arcpy.management.Delete(single_fc)
        arcpy.management.CreateFeatureclass(
            out_path=os.path.dirname(single_fc),
            out_name=os.path.basename(single_fc),
            geometry_type="POINT",
            spatial_reference=sr_proj,
        )
        arcpy.management.AddField(single_fc, "stop_id", "TEXT")
        arcpy.management.AddField(single_fc, "dist_m", "DOUBLE")

        with arcpy.da.InsertCursor(single_fc, ["stop_id", "dist_m", "SHAPE@"]) as cur:
            for _, row in single_stop_df.iterrows():
                lat = float(row["stop_lat"])
                lon = float(row["stop_lon"])
                dist_meters = (
                    pick_buffer_distance(
                        sid, BUFFER_DISTANCE, LARGE_BUFFER_DISTANCE, STOP_IDS_LARGE_BUFFER
                    )
                    * 1609.34
                )

                pt = arcpy.Point(lon, lat)
                pt_geom = arcpy.PointGeometry(pt, arcpy.SpatialReference(4326))
                pt_geom_projected = pt_geom.projectAs(sr_proj)
                cur.insertRow([stop_id_str, dist_meters, pt_geom_projected])

        stop_buffer_fc = r"in_memory\single_stop_buffer"
        if arcpy.Exists(stop_buffer_fc):
            arcpy.management.Delete(stop_buffer_fc)
        arcpy.analysis.Buffer(
            in_features=single_fc,
            out_feature_class=stop_buffer_fc,
            buffer_distance_or_field="dist_m",
            line_side="FULL",
            line_end_type="ROUND",
            dissolve_option="ALL",
        )

        out_fc_name = f"stop_{stop_id_str}_service_buffer_data".replace(" ", "_").replace("-", "_")
        output_fc = os.path.join(gdb_path, out_fc_name)
        if arcpy.Exists(output_fc):
            arcpy.management.Delete(output_fc)

        sums = intersect_and_partial_weight(demog_fc, stop_buffer_fc, SYNTHETIC_FIELDS, output_fc)
        final_dict = build_output_dict(sums, all_demog_totals, SYNTHETIC_FIELDS)

        print(f"\nProcessing stop: {stop_id_str}")
        for fld in SYNTHETIC_FIELDS:
            print(f"  {fld}: {int(final_dict[fld])},  Pct_{fld}: {final_dict[f'Pct_{fld}']:.2f}%")

        xlsx_path = os.path.join(OUTPUT_DIRECTORY, f"stop_{stop_id_str}_service_buffer_data.xlsx")
        export_summary_to_excel(final_dict, xlsx_path)


def do_shapefile_analysis(
    shp_path, shp_filter_field, filter_values, demog_fc, all_demog_totals, gdb_path
):
    """Perform shapefile-based analysis of service buffers.

    Make a buffer around features in shp_path (filtered by shp_filter_field and
    filter_values if provided), intersect with demographics, and export a summary
    of partial-weighted results.
    """
    print("\n=== Shapefile-based Analysis ===")

    shp_lyr = "shp_lyr"
    if arcpy.Exists(shp_lyr):
        arcpy.management.Delete(shp_lyr)
    arcpy.management.MakeFeatureLayer(shp_path, shp_lyr)

    before_count = int(arcpy.management.GetCount(shp_lyr).getOutput(0))

    if filter_values:
        vals_quoted = ", ".join(f"'{val}'" for val in filter_values)
        where_clause = f"{shp_filter_field} IN ({vals_quoted})"
        arcpy.management.SelectLayerByAttribute(shp_lyr, "NEW_SELECTION", where_clause)
        after_count = int(arcpy.management.GetCount(shp_lyr).getOutput(0))
        print(
            f"Filtered shapefile from {before_count} to {after_count} records based on field '{shp_filter_field}'."
        )

        if after_count == 0:
            print("No matching features found in shapefile. Aborting shapefile analysis.")
            return
    else:
        print(f"No shapefile filter values specified; using all {before_count} features.")

    shapefile_buffer_fc = os.path.join(gdb_path, "shapefile_buffer")
    if arcpy.Exists(shapefile_buffer_fc):
        arcpy.management.Delete(shapefile_buffer_fc)

    arcpy.analysis.Buffer(
        in_features=shp_lyr,
        out_feature_class=shapefile_buffer_fc,
        buffer_distance_or_field=f"{BUFFER_DISTANCE} Miles",
        line_side="FULL",
        line_end_type="ROUND",
        dissolve_option="ALL",
    )

    out_fc = os.path.join(gdb_path, "shapefile_service_buffer_data")
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)

    sums = intersect_and_partial_weight(
        demog_fc=demog_fc,
        buffer_fc=shapefile_buffer_fc,
        synthetic_fields=SYNTHETIC_FIELDS,
        out_fc=out_fc,
    )

    final_dict = build_output_dict(sums, all_demog_totals, SYNTHETIC_FIELDS)

    print("Shapefile-based totals (partial vs. % of total):")
    for fld in SYNTHETIC_FIELDS:
        print(f"  {fld}: {int(final_dict[fld])},  Pct_{fld}: {final_dict[f'Pct_{fld}']:.2f}%")

    xlsx_path = os.path.join(OUTPUT_DIRECTORY, "shapefile_service_buffer_data.xlsx")
    export_summary_to_excel(final_dict, xlsx_path)


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        use_shapefile = bool(SHAPEFILE_PATH)
        use_gtfs = bool(GTFS_DATA_PATH)

        gdb_path = ensure_gdb_exists(OUTPUT_DIRECTORY, GDB_NAME)

        demog_fc_filtered = apply_county_state_filter(
            DEMOGRAPHICS_FC_PATH, STATEFP_FILTER, COUNTYFP_FILTER
        )
        all_demog_totals = compute_demographics_totals(demog_fc_filtered, SYNTHETIC_FIELDS)

        if use_shapefile and not use_gtfs:
            do_shapefile_analysis(
                shp_path=SHAPEFILE_PATH,
                shp_filter_field=SHAPEFILE_FILTER_FIELD,
                filter_values=SHAPEFILE_FILTER_VALUES,
                demog_fc=demog_fc_filtered,
                all_demog_totals=all_demog_totals,
                gdb_path=gdb_path,
            )

        elif use_gtfs and not use_shapefile:
            trips, stop_times, routes_df, stops_df, calendar = load_gtfs_data(GTFS_DATA_PATH)
            weekday_service_ids = filter_weekday_service(calendar)

            if ANALYSIS_MODE.lower() == "network":
                do_network_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            elif ANALYSIS_MODE.lower() == "route":
                do_route_by_route_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            elif ANALYSIS_MODE.lower() == "stop":
                do_stop_by_stop_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            else:
                print(
                    f"Unknown ANALYSIS_MODE: {ANALYSIS_MODE}. Please set it to 'network', 'route', or 'stop'."
                )

        elif use_shapefile and use_gtfs:
            print("Both shapefile and GTFS are provided; proceeding with GTFS analysis.")
            trips, stop_times, routes_df, stops_df, calendar = load_gtfs_data(GTFS_DATA_PATH)
            weekday_service_ids = filter_weekday_service(calendar)

            if ANALYSIS_MODE.lower() == "network":
                do_network_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            elif ANALYSIS_MODE.lower() == "route":
                do_route_by_route_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            elif ANALYSIS_MODE.lower() == "stop":
                do_stop_by_stop_analysis(
                    trips,
                    stop_times,
                    routes_df,
                    stops_df,
                    demog_fc_filtered,
                    all_demog_totals,
                    gdb_path,
                )
            else:
                print(
                    f"Unknown ANALYSIS_MODE: {ANALYSIS_MODE}. Please set it to 'network', 'route', or 'stop'."
                )

        else:
            print("No valid input paths provided (neither shapefile nor GTFS). Aborting.")

    except Exception as e:
        print(f"An error occurred: {e}")
