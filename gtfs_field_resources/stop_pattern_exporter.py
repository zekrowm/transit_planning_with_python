"""
This module extracts unique stop patterns from GTFS data and exports them as Excel workbooks.
It supports route-based filtering, distance conversions, timepoint-only exports, and optional
distance validation (comparing total trip distance to timepoint segments).
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
# Input directory where GTFS files are stored.
INPUT_DIR = r'\\Folder\Path\To\Your\GTFS_data'

# Output directory where the Excel files will be saved.
OUTPUT_DIR = r'\\Folder\Path\To\Your\Output'

# GTFS filenames
TRIPS_FILE = 'trips.txt'
STOP_TIMES_FILE = 'stop_times.txt'
STOPS_FILE = 'stops.txt'
ROUTES_FILE = 'routes.txt'

# Optional filtering based on route_short_name.
FILTER_IN_ROUTE_SHORT_NAMES = []  # e.g., ['10', '20']
FILTER_OUT_ROUTE_SHORT_NAMES = ['9999A', '9999B', '9999C']  # e.g., ['30']

# Optional signup name to append to output filenames.
SIGNUP_NAME = "January2025Signup"  # e.g., "January2025Signup"

# ============================================================
# DISTANCE CONVERSION CONFIGURATION
# ============================================================
# Specify the unit of the shape_dist_traveled values in the GTFS data.
# Options: "feet" or "meters"
INPUT_DISTANCE_UNIT = "meters"  # Change to "meters" if your GTFS uses meters.

# Set to True to convert distances to miles.
CONVERT_TO_MILES = True

# ============================================================
# TIMEPOINT CONFIGURATION
# ============================================================
# NEW: Set to True if you only want to export stops with timepoint=1.
EXPORT_TIMEPOINTS_ONLY = True

# OPTIONAL: If True, we'll compare the total distance from the first to last stop
# with the sum of segment distances for timepoint stops only, and log a warning if they differ.
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

# ------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------

def load_gtfs_files(input_dir):
    """
    Load the necessary GTFS files from the input directory.
    Returns a dictionary with keys: stops, trips, stop_times, and routes.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    try:
        stops = pd.read_csv(os.path.join(input_dir, STOPS_FILE))
        trips = pd.read_csv(os.path.join(input_dir, TRIPS_FILE))
        stop_times = pd.read_csv(os.path.join(input_dir, STOP_TIMES_FILE))
        routes = pd.read_csv(os.path.join(input_dir, ROUTES_FILE))
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(
            f"One or more GTFS files not found in {input_dir}: {fnf_error}"
        ) from fnf_error
    except pd.errors.ParserError as parse_err:
        raise Exception(
            f"Error parsing one of the GTFS files: {parse_err}"
        ) from parse_err
    except Exception as err:
        raise Exception(
            f"Unexpected error loading GTFS files: {err}"
        ) from err

    logging.info("Successfully loaded GTFS files.")
    return {
        'stops': stops,
        'trips': trips,
        'stop_times': stop_times,
        'routes': routes
    }

def filter_trips_by_route(trips_df, routes_df):
    """
    Filter trips based on FILTER_IN_ROUTE_SHORT_NAMES and FILTER_OUT_ROUTE_SHORT_NAMES.
    Returns a filtered DataFrame of trips.
    """
    try:
        trips_routes = pd.merge(
            trips_df,
            routes_df[['route_id', 'route_short_name']],
            on='route_id',
            how='left'
        )
    except Exception as err:  # pylint: disable=broad-except
        raise Exception(
            f"Error merging trips and routes: {err}"
        ) from err

    if FILTER_IN_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[
            trips_routes['route_short_name'].isin(FILTER_IN_ROUTE_SHORT_NAMES)
        ]
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[
            ~trips_routes['route_short_name'].isin(FILTER_OUT_ROUTE_SHORT_NAMES)
        ]

    if trips_routes.empty:
        logging.warning("No trips remaining after filtering by route_short_name.")

    return trips_routes

def generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df):
    """
    For each trip, merge with stop_times and stops; sort by stop_sequence;
    and build a pattern as an ordered tuple of stops.

    If EXPORT_TIMEPOINTS_ONLY=True, only keep stops where timepoint=1.

    Each stop in the final pattern is represented as a tuple:
        (stop_name, stop_id, distance)
    where 'distance' is the difference in shape_dist_traveled from the previous
    included stop. For the first stop, distance is "-".

    Returns:
        A dictionary keyed by (route_id, direction_id, pattern) containing:
            - route_id
            - direction_id
            - pattern (tuple of (stop_name, stop_id, distance))
            - trip_count (number of trips with that pattern)
    """
    try:
        trips_stop_times = pd.merge(
            stop_times_df,
            filtered_trips_df[['trip_id', 'route_id', 'direction_id']],
            on='trip_id',
            how='inner'
        )
    except Exception as err:  # pylint: disable=broad-except
        raise Exception(
            f"Error merging stop_times with filtered trips: {err}"
        ) from err

    # Add a shape_dist_traveled column if missing
    if 'shape_dist_traveled' not in trips_stop_times.columns:
        trips_stop_times['shape_dist_traveled'] = np.nan

    # Merge in stop names
    try:
        trips_stop_times = pd.merge(
            trips_stop_times,
            stops_df[['stop_id', 'stop_name']],
            on='stop_id',
            how='left'
        )
    except Exception as err:  # pylint: disable=broad-except
        raise Exception(
            f"Error merging stop_times with stops: {err}"
        ) from err

    trips_stop_times.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)

    original_trip_distances = {}
    if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
        for trip_id, grp in trips_stop_times.groupby('trip_id'):
            grp = grp.dropna(subset=['shape_dist_traveled'])
            if grp.empty:
                original_trip_distances[trip_id] = None
                continue
            first_val = grp.iloc[0]['shape_dist_traveled']
            last_val = grp.iloc[-1]['shape_dist_traveled']
            dist_val = last_val - first_val
            if pd.notnull(dist_val) and CONVERT_TO_MILES:
                if INPUT_DISTANCE_UNIT.lower() == "feet":
                    conv_factor = 5280.0
                elif INPUT_DISTANCE_UNIT.lower() == "meters":
                    conv_factor = 1609.34
                else:
                    conv_factor = 1.0
                dist_val = dist_val / conv_factor
            original_trip_distances[trip_id] = dist_val

    trip_patterns = []
    for trip_id, group in trips_stop_times.groupby('trip_id'):
        group = group.sort_values('stop_sequence')

        if EXPORT_TIMEPOINTS_ONLY:
            group = group[group['timepoint'] == 1]

        if group.empty:
            continue

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
                    try:
                        diff = float(current_dist) - float(prev_dist_val)
                        if CONVERT_TO_MILES:
                            if INPUT_DISTANCE_UNIT.lower() == "feet":
                                conv_factor = 5280.0
                            elif INPUT_DISTANCE_UNIT.lower() == "meters":
                                conv_factor = 1609.34
                            else:
                                logging.warning(
                                    "Unknown input distance unit '%s'. No conversion applied.",
                                    INPUT_DISTANCE_UNIT
                                )
                                conv_factor = 1.0
                            diff = diff / conv_factor
                        distance_str = f"{diff:.2f}"
                    except (ValueError, TypeError) as distance_err:  # pylint: disable=broad-except
                        logging.error(
                            "Error calculating distance difference for trip %s: %s",
                            trip_id,
                            distance_err
                        )
                        distance_str = ""
                else:
                    distance_str = ""

            stops_list.append((stop_name, stop_id, distance_str))
            if pd.notnull(current_dist):
                prev_dist_val = current_dist
            else:
                prev_dist_val = None

        first_row = group.iloc[0]
        trip_patterns.append({
            'trip_id': trip_id,
            'route_id': first_row.get('route_id', 'Unknown'),
            'direction_id': first_row.get('direction_id', 'Unknown'),
            'pattern': tuple(stops_list)
        })

        if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
            sum_of_segments = 0.0
            for (_, _, dist_str) in stops_list:
                if dist_str not in ("-", "", None) and is_number(dist_str):
                    sum_of_segments += float(dist_str)

            orig_dist = original_trip_distances.get(trip_id, None)
            if orig_dist is not None:
                if abs(sum_of_segments - orig_dist) > 0.01:
                    logging.warning(
                        "Trip %s: sum of timepoint distances %.2f differs from full trip "
                        "distance %.2f by more than 0.01.",
                        trip_id,
                        sum_of_segments,
                        orig_dist
                    )

    patterns_dict = {}
    for rec in trip_patterns:
        key = (rec['route_id'], rec['direction_id'], rec['pattern'])
        if key not in patterns_dict:
            patterns_dict[key] = {
                'route_id': rec['route_id'],
                'direction_id': rec['direction_id'],
                'pattern': rec['pattern'],
                'trip_count': 0
            }
        patterns_dict[key]['trip_count'] += 1

    logging.info("Generated %d unique patterns.", len(patterns_dict))
    return patterns_dict

def assign_pattern_ids(patterns_dict):
    """
    Given a dictionary of patterns (keyed by (route_id, direction_id, pattern)),
    assign a sequential pattern_id for each (route_id, direction_id) group.

    Returns a list of records, each with:
      route_id, direction_id, pattern_id, trip_count, pattern
    """
    patterns_by_route_dir = {}
    for _, rec in patterns_dict.items():
        route_id = rec['route_id']
        direction_id = rec['direction_id']
        patterns_by_route_dir.setdefault((route_id, direction_id), []).append(rec)

    pattern_records = []
    for (route_id, direction_id), recs in patterns_by_route_dir.items():
        recs = sorted(recs, key=lambda r: r['pattern'])
        for i, pattern_rec in enumerate(recs, start=1):
            pattern_rec['pattern_id'] = i
            pattern_records.append({
                'route_id': route_id,
                'direction_id': direction_id,
                'pattern_id': i,
                'trip_count': pattern_rec['trip_count'],
                'pattern': pattern_rec['pattern']
            })

    logging.info("Assigned pattern IDs.")
    return pattern_records

def export_patterns_to_excel(pattern_records, routes_df):
    """
    Export pattern records to Excel files, one file per route.
    Each Excel file contains worksheets for each unique stop pattern.
    Uses route_short_name for file names and the "Route" column values.
    """
    route_groups = {}
    for rec in pattern_records:
        route_groups.setdefault(rec['route_id'], []).append(rec)

    for route_id, records in route_groups.items():
        try:
            route_info = routes_df[routes_df['route_id'] == route_id]
        except Exception as err:  # pylint: disable=broad-except
            logging.error(
                "Error retrieving route info for route_id %s: %s",
                route_id,
                err
            )
            route_info = None

        if route_info is not None and not route_info.empty:
            route_short_name = route_info.iloc[0].get('route_short_name', f"Route_{route_id}")
        else:
            route_short_name = f"Route_{route_id}"

        workbook = Workbook()
        default_sheet = workbook.active
        workbook.remove(default_sheet)

        for record in records:
            direction_id = record.get('direction_id', 'Unknown')
            pattern_id = record.get('pattern_id', 'Unknown')
            worksheet_title = f"Dir{direction_id}_Pat{pattern_id}"
            try:
                worksheet = workbook.create_sheet(title=worksheet_title)
            except Exception as err:  # pylint: disable=broad-except
                logging.error(
                    "Error creating worksheet '%s' for route %s: %s",
                    worksheet_title,
                    route_id,
                    err
                )
                continue

            header = ["Route", "Direction", "Distance"]
            stops = record.get('pattern', [])
            stop_headers = [f"{stop_name} ({stop_id})" for stop_name, stop_id, _ in stops]
            header.extend(stop_headers)

            try:
                worksheet.append(header)
            except Exception as err:  # pylint: disable=broad-except
                logging.error(
                    "Error writing header in worksheet '%s' for route %s: %s",
                    worksheet_title,
                    route_id,
                    err
                )
                continue

            total_distance = 0.0
            for stop in stops:
                dist_str = stop[2]
                if dist_str not in ("-", "", None) and is_number(dist_str):
                    try:
                        total_distance += float(dist_str)
                    except Exception as dist_err:  # pylint: disable=broad-except
                        logging.error(
                            "Error converting stop distance '%s' in route %s: %s",
                            dist_str,
                            route_id,
                            dist_err
                        )

            total_distance_str = f"{total_distance:.2f}"
            row = [route_short_name, direction_id, total_distance_str]
            row.extend([stop[2] for stop in stops])

            try:
                worksheet.append(row)
            except Exception as err:  # pylint: disable=broad-except
                logging.error(
                    "Error writing data row in worksheet '%s' for route %s: %s",
                    worksheet_title,
                    route_id,
                    err
                )

            for i, _ in enumerate(header, start=1):
                col_letter = get_column_letter(i)
                worksheet.column_dimensions[col_letter].width = 20

        output_filename = f"{route_short_name}_{SIGNUP_NAME}.xlsx"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            workbook.save(output_path)
            logging.info("Workbook saved: %s", output_path)
        except Exception as err:  # pylint: disable=broad-except
            logging.error("Error saving workbook '%s': %s", output_filename, err)

def main():
    """Main function to load GTFS data, filter, generate patterns, and export to Excel."""
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logging.info("Created output directory: %s", OUTPUT_DIR)
        except Exception as err:  # pylint: disable=broad-except
            logging.error(
                "Unable to create output directory '%s': %s",
                OUTPUT_DIR,
                err
            )
            return

    try:
        gtfs_data = load_gtfs_files(INPUT_DIR)
        stops_df = gtfs_data['stops']
        trips_df = gtfs_data['trips']
        stop_times_df = gtfs_data['stop_times']
        routes_df = gtfs_data['routes']
    except Exception as err:  # pylint: disable=broad-except
        logging.error(err)
        return

    try:
        filtered_trips_df = filter_trips_by_route(trips_df, routes_df)
    except Exception as err:  # pylint: disable=broad-except
        logging.error(err)
        return

    if filtered_trips_df.empty:
        logging.error("No trips to process after filtering. Exiting.")
        return

    try:
        patterns_dict = generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df)
        pattern_records = assign_pattern_ids(patterns_dict)
    except Exception as err:  # pylint: disable=broad-except
        logging.error("Error generating patterns: %s", err)
        return

    if not pattern_records:
        logging.warning("No unique patterns found. Exiting.")
        return

    try:
        export_patterns_to_excel(pattern_records, routes_df)
    except Exception as err:  # pylint: disable=broad-except
        logging.error("Error exporting patterns to Excel: %s", err)

if __name__ == "__main__":
    main()
