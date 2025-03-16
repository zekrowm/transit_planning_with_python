"""
This module extracts unique stop patterns from GTFS data and exports them as Excel workbooks.
It supports:
 - Route-based filtering
 - Calendar-based filtering (via service_id) with separate files per service_id
 - Distance conversions
 - Timepoint-only exports
 - Optional distance validation for timepoint segments
"""

import logging
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = r'\\Folder\Path\To\Your\GTFS_data'
OUTPUT_DIR = r'\\Folder\Path\To\Your\Output'

TRIPS_FILE = 'trips.txt'
STOP_TIMES_FILE = 'stop_times.txt'
STOPS_FILE = 'stops.txt'
ROUTES_FILE = 'routes.txt'

FILTER_IN_ROUTE_SHORT_NAMES = []
FILTER_OUT_ROUTE_SHORT_NAMES = ['9999A', '9999B', '9999C']
FILTER_IN_CALENDAR_IDS = []

SIGNUP_NAME = "January2025Signup"

INPUT_DISTANCE_UNIT = "meters"
CONVERT_TO_MILES = True
EXPORT_TIMEPOINTS_ONLY = True
VALIDATE_TIMEPOINT_DISTANCE = True


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def is_number(value):
    """Return True if 'value' can be converted to a float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def convert_dist_to_miles(distance, input_unit):
    """
    Convert distance to miles if CONVERT_TO_MILES is True.
    Otherwise, return the original distance.
    """
    if not CONVERT_TO_MILES or pd.isna(distance):
        return distance

    if input_unit.lower() == "feet":
        conv_factor = 5280.0
    elif input_unit.lower() == "meters":
        conv_factor = 1609.34
    else:
        logging.warning("Unknown input distance unit '%s'. No conversion applied.", input_unit)
        conv_factor = 1.0

    return distance / conv_factor


# ------------------------------------------------------------
# LOAD AND FILTER
# ------------------------------------------------------------

def load_gtfs_files(input_dir):
    """
    Load the necessary GTFS files from the input directory.
    Returns a dictionary with keys: stops, trips, stop_times, and routes.
    Raises FileNotFoundError or pd.errors.ParserError on problems.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    stops = pd.read_csv(os.path.join(input_dir, STOPS_FILE))
    trips = pd.read_csv(os.path.join(input_dir, TRIPS_FILE))
    stop_times = pd.read_csv(os.path.join(input_dir, STOP_TIMES_FILE))
    routes = pd.read_csv(os.path.join(input_dir, ROUTES_FILE))

    logging.info("Successfully loaded GTFS files.")
    return {
        'stops': stops,
        'trips': trips,
        'stop_times': stop_times,
        'routes': routes
    }


def filter_trips(trips_df, routes_df, calendar_ids=None):
    """
    Filter trips based on:
      - FILTER_IN_ROUTE_SHORT_NAMES
      - FILTER_OUT_ROUTE_SHORT_NAMES
      - (Optional) calendar_ids (service_id in GTFS)
    Returns a filtered DataFrame of trips joined to route information.
    """
    trips_routes = pd.merge(
        trips_df,
        routes_df[['route_id', 'route_short_name']],
        on='route_id',
        how='left'
    )

    # 1) Filter IN route_short_names if provided
    if FILTER_IN_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[
            trips_routes['route_short_name'].isin(FILTER_IN_ROUTE_SHORT_NAMES)
        ]

    # 2) Filter OUT route_short_names if provided
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[
            ~trips_routes['route_short_name'].isin(FILTER_OUT_ROUTE_SHORT_NAMES)
        ]

    # 3) Filter by calendar_ids (service_id) if provided
    if calendar_ids and 'service_id' in trips_routes.columns:
        trips_routes = trips_routes[trips_routes['service_id'].isin(calendar_ids)]
    elif calendar_ids:
        logging.warning("service_id column not found in the trips data.")

    if trips_routes.empty:
        logging.warning("No trips remaining after filtering.")

    return trips_routes


# ------------------------------------------------------------
# GENERATE PATTERNS
# ------------------------------------------------------------

def merge_trips_stop_times(stop_times_df, filtered_trips_df):
    """
    Merge stop_times with filtered trips to get route_id, direction_id, service_id in each row.
    Ensures a shape_dist_traveled column exists.
    """
    merged = pd.merge(
        stop_times_df,
        filtered_trips_df[['trip_id', 'route_id', 'direction_id', 'service_id']],
        on='trip_id',
        how='inner'
    )

    # Add shape_dist_traveled column if missing
    if 'shape_dist_traveled' not in merged.columns:
        merged['shape_dist_traveled'] = np.nan

    return merged


def add_stop_names(merged_stop_times, stops_df):
    """
    Merge the stops' names into the combined DataFrame.
    """
    with_names = pd.merge(
        merged_stop_times,
        stops_df[['stop_id', 'stop_name']],
        on='stop_id',
        how='left'
    )
    with_names.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)
    return with_names


def compute_original_distances(stop_times_w_names):
    """
    For each trip, compute the original full-trip distance (if needed)
    for distance validation. Returns a dict {trip_id: distance_in_miles_or_none}.
    """
    original_trip_distances = {}
    grouped = stop_times_w_names.groupby('trip_id')

    for trip_id, grp in grouped:
        grp = grp.dropna(subset=['shape_dist_traveled'])
        if grp.empty:
            original_trip_distances[trip_id] = None
            continue

        first_val = grp.iloc[0]['shape_dist_traveled']
        last_val = grp.iloc[-1]['shape_dist_traveled']
        dist_val = last_val - first_val if pd.notnull(last_val) else None
        dist_val = convert_dist_to_miles(dist_val, INPUT_DISTANCE_UNIT)
        original_trip_distances[trip_id] = dist_val

    return original_trip_distances


def build_trip_pattern(trip_id, group):
    """
    Build the ordered list of stops (timepoint-only if configured)
    with distance differences between consecutive stops.
    """
    if EXPORT_TIMEPOINTS_ONLY:
        group = group[group['timepoint'] == 1]

    if group.empty:
        return None

    stops_list = []
    prev_dist_val = None

    for _, row in group.iterrows():
        stop_name = row.get('stop_name', 'Unknown')
        stop_id = row.get('stop_id', 'Unknown')
        current_dist = row.get('shape_dist_traveled', np.nan)

        if prev_dist_val is None:
            distance_str = "-"
        else:
            if pd.notnull(current_dist) and pd.notnull(prev_dist_val):
                diff = convert_dist_to_miles(float(current_dist) - float(prev_dist_val),
                                             INPUT_DISTANCE_UNIT)
                distance_str = f"{diff:.2f}" if diff is not None else ""
            else:
                distance_str = ""

        stops_list.append((stop_name, stop_id, distance_str))
        prev_dist_val = current_dist if pd.notnull(current_dist) else None

    return stops_list


def validate_timepoint_distance(trip_id, stops_list, original_trip_distances):
    """
    If configured, check that the sum of timepoint segment distances
    is close to the original full-trip distance.
    """
    orig_dist = original_trip_distances.get(trip_id, None)
    if orig_dist is None:
        return

    sum_of_segments = 0.0
    for (_, _, dist_str) in stops_list:
        if dist_str not in ("-", "", None) and is_number(dist_str):
            sum_of_segments += float(dist_str)

    if abs(sum_of_segments - orig_dist) > 0.02:
        logging.warning(
            "Trip %s: sum of timepoint distances %.2f differs from "
            "full trip distance %.2f by more than 0.02.",
            trip_id,
            sum_of_segments,
            orig_dist
        )


def generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df):
    """
    Integrates the smaller helper functions to produce unique patterns.
    Returns a dictionary keyed by (route_id, direction_id, service_id, pattern).
    """
    merged = merge_trips_stop_times(stop_times_df, filtered_trips_df)
    stop_times_w_names = add_stop_names(merged, stops_df)

    # Precompute original distances for validation if needed
    original_trip_distances = {}
    if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
        original_trip_distances = compute_original_distances(stop_times_w_names)

    trip_patterns = []
    for trip_id, group in stop_times_w_names.groupby('trip_id'):
        group = group.sort_values('stop_sequence')
        stops_list = build_trip_pattern(trip_id, group)
        if not stops_list:
            continue

        # Validate if configured
        if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
            validate_timepoint_distance(trip_id, stops_list, original_trip_distances)

        first_row = group.iloc[0]
        trip_patterns.append({
            'trip_id': trip_id,
            'route_id': first_row.get('route_id', 'Unknown'),
            'direction_id': first_row.get('direction_id', 'Unknown'),
            'service_id': first_row.get('service_id', 'Unknown'),
            'pattern': tuple(stops_list)
        })

    # Build a dictionary of unique patterns
    patterns_dict = {}
    for rec in trip_patterns:
        key = (rec['route_id'], rec['direction_id'], rec['service_id'], rec['pattern'])
        if key not in patterns_dict:
            patterns_dict[key] = {
                'route_id': rec['route_id'],
                'direction_id': rec['direction_id'],
                'service_id': rec['service_id'],
                'pattern': rec['pattern'],
                'trip_count': 0
            }
        patterns_dict[key]['trip_count'] += 1

    logging.info("Generated %d unique patterns.", len(patterns_dict))
    return patterns_dict


def assign_pattern_ids(patterns_dict):
    """
    Assign a sequential pattern_id for each (route_id, service_id, direction_id).
    Returns a list of records with route_id, direction_id, service_id, pattern_id, trip_count, pattern.
    """
    patterns_by_rsd = {}
    for _, rec in patterns_dict.items():
        route_id = rec['route_id']
        direction_id = rec['direction_id']
        service_id = rec['service_id']
        patterns_by_rsd.setdefault((route_id, service_id, direction_id), []).append(rec)

    pattern_records = []
    for (route_id, service_id, direction_id), recs in patterns_by_rsd.items():
        # Sort records by pattern (arbitrary but stable)
        recs = sorted(recs, key=lambda r: r['pattern'])
        for i, pattern_rec in enumerate(recs, start=1):
            pattern_rec['pattern_id'] = i
            pattern_records.append({
                'route_id': route_id,
                'direction_id': direction_id,
                'service_id': service_id,
                'pattern_id': i,
                'trip_count': pattern_rec['trip_count'],
                'pattern': pattern_rec['pattern']
            })

    logging.info("Assigned pattern IDs to each unique pattern.")
    return pattern_records


# ------------------------------------------------------------
# EXPORTING TO EXCEL
# ------------------------------------------------------------

def create_route_workbook():
    """
    Create a fresh Workbook with its default sheet removed.
    """
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    return workbook


def fill_worksheet(worksheet, route_short_name, direction_id, service_id, stops):
    """
    Fill a single Excel worksheet with distances and summary row.
    """
    # Write header
    header = ["Route", "Direction", "Calendar (service_id)", "Distance"]
    stop_headers = [f"{stop_name} ({stop_id})" for stop_name, stop_id, _ in stops]
    header.extend(stop_headers)
    worksheet.append(header)

    # Compute total distance
    total_distance = 0.0
    for _, _, dist_str in stops:
        if dist_str not in ("-", "", None) and is_number(dist_str):
            total_distance += float(dist_str)

    total_distance_str = f"{total_distance:.2f}"

    # Write data row
    row = [route_short_name, direction_id, service_id, total_distance_str]
    row.extend([stop[2] for stop in stops])
    worksheet.append(row)

    # Set column widths
    for i, _ in enumerate(header, start=1):
        col_letter = get_column_letter(i)
        worksheet.column_dimensions[col_letter].width = 20


def export_patterns_to_excel(pattern_records, routes_df):
    """
    Export pattern records to separate Excel files for each (route_id, service_id).
    """
    file_groups = {}
    for rec in pattern_records:
        route_id = rec['route_id']
        service_id = rec['service_id']
        file_groups.setdefault((route_id, service_id), []).append(rec)

    for (route_id, service_id), group_records in file_groups.items():
        # Attempt to get route_short_name
        route_info = routes_df[routes_df['route_id'] == route_id]
        if not route_info.empty:
            route_short_name = route_info.iloc[0].get('route_short_name', f"Route_{route_id}")
        else:
            route_short_name = f"Route_{route_id}"

        workbook = create_route_workbook()
        # Sort records by direction_id and then pattern_id
        group_records = sorted(group_records, key=lambda r: (r['direction_id'], r['pattern_id']))

        for record in group_records:
            direction_id = record.get('direction_id', 'Unknown')
            pattern_id = record.get('pattern_id', 'Unknown')
            stops = record.get('pattern', [])
            sheet_title = f"Dir{direction_id}_Pat{pattern_id}"

            try:
                worksheet = workbook.create_sheet(title=sheet_title)
            except ValueError as err:  # e.g., invalid sheet title
                logging.error("Error creating worksheet '%s': %s", sheet_title, err)
                continue

            fill_worksheet(worksheet, route_short_name, direction_id, service_id, stops)

        output_filename = f"{route_short_name}_{service_id}_{SIGNUP_NAME}.xlsx"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            workbook.save(output_path)
            logging.info("Workbook saved: %s", output_path)
        except (OSError, PermissionError) as err:
            logging.error("Error saving workbook '%s': %s", output_filename, err)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    """Main function to load GTFS data, filter, generate patterns, and export to Excel."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logging.info("Created output directory: %s", OUTPUT_DIR)
        except (OSError, PermissionError) as err:
            logging.error("Unable to create output directory '%s': %s", OUTPUT_DIR, err)
            return

    # Load GTFS files
    try:
        gtfs_data = load_gtfs_files(INPUT_DIR)
    except (FileNotFoundError, pd.errors.ParserError) as err:
        logging.error("Failed to load GTFS files: %s", err)
        return

    stops_df = gtfs_data['stops']
    trips_df = gtfs_data['trips']
    stop_times_df = gtfs_data['stop_times']
    routes_df = gtfs_data['routes']

    # Filter trips by route & calendar
    filtered_trips_df = filter_trips(trips_df, routes_df, calendar_ids=FILTER_IN_CALENDAR_IDS)
    if filtered_trips_df.empty:
        logging.error("No trips to process after filtering. Exiting.")
        return

    # Generate unique patterns
    patterns_dict = generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df)
    if not patterns_dict:
        logging.warning("No unique patterns found. Exiting.")
        return

    pattern_records = assign_pattern_ids(patterns_dict)
    if not pattern_records:
        logging.warning("No pattern records to export. Exiting.")
        return

    # Export patterns to Excel
    export_patterns_to_excel(pattern_records, routes_df)


if __name__ == "__main__":
    main()
