"""
Script to extract unique stop patterns from GTFS data and export them using openpyxl.
For each route, an Excel workbook is created. Within that workbook, each unique stop pattern
appears as a separate worksheet. In each worksheet:
  - The first column ("Route") contains the route name.
  - The second column ("Direction") contains the route direction (from direction_id).
  - The third column ("Distance") contains the total distance (sum of all per-stop distances, in miles).
  - The remaining columns contain the stops, with headers formatted as "StopName (StopID)"
    and the corresponding distance values (in miles).

The script allows you to choose the unit in which shape_dist_traveled values are provided
("feet" or "meters") and, if enabled, converts the distances into miles.

Optional filtering by route_short_name is supported.
An optional signup name can also be provided so that it will be appended to the output file names.
"""

import os
import pandas as pd
import numpy as np
import logging
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

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def is_number(s):
    """Return True if s can be converted to a float."""
    try:
        float(s)
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
        raise FileNotFoundError(f"One or more GTFS files not found in {input_dir}: {fnf_error}")
    except pd.errors.ParserError as parse_error:
        raise Exception(f"Error parsing one of the GTFS files: {parse_error}")
    except Exception as e:
        raise Exception(f"Unexpected error loading GTFS files: {e}")

    logging.info("Successfully loaded GTFS files.")
    return {'stops': stops, 'trips': trips, 'stop_times': stop_times, 'routes': routes}

def filter_trips_by_route(trips_df, routes_df):
    """
    Filter trips based on FILTER_IN_ROUTE_SHORT_NAMES and FILTER_OUT_ROUTE_SHORT_NAMES.
    Returns a filtered DataFrame of trips.
    """
    try:
        trips_routes = pd.merge(trips_df, routes_df[['route_id', 'route_short_name']],
                                on='route_id', how='left')
    except Exception as e:
        raise Exception(f"Error merging trips and routes: {e}")

    if FILTER_IN_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[trips_routes['route_short_name'].isin(FILTER_IN_ROUTE_SHORT_NAMES)]
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        trips_routes = trips_routes[~trips_routes['route_short_name'].isin(FILTER_OUT_ROUTE_SHORT_NAMES)]

    if trips_routes.empty:
        logging.warning("No trips remaining after filtering based on route_short_name.")

    return trips_routes

def generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df):
    """
    For each trip, merge with stop_times and stops; sort by stop_sequence;
    and build a pattern as an ordered tuple of stops.

    Each stop is represented as a tuple: (stop_name, stop_id, distance)
    where distance is computed as the difference in shape_dist_traveled
    from the previous stop (first stop gets "-").

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
            on='trip_id', how='inner'
        )
    except Exception as e:
        raise Exception(f"Error merging stop_times with filtered trips: {e}")

    if 'shape_dist_traveled' not in trips_stop_times.columns:
        trips_stop_times['shape_dist_traveled'] = np.nan

    try:
        trips_stop_times = pd.merge(
            trips_stop_times,
            stops_df[['stop_id', 'stop_name']],
            on='stop_id', how='left'
        )
    except Exception as e:
        raise Exception(f"Error merging stop_times with stops: {e}")

    trips_stop_times.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)

    trip_patterns = []
    for trip_id, group in trips_stop_times.groupby('trip_id'):
        group = group.sort_values('stop_sequence')
        stops_list = []
        prev_dist = None
        for _, row in group.iterrows():
            stop_name = row.get('stop_name', 'Unknown')
            stop_id = row.get('stop_id', 'Unknown')
            current_dist = row.get('shape_dist_traveled', np.nan)
            if prev_dist is None:
                distance = "-"
            else:
                if pd.notnull(current_dist) and pd.notnull(prev_dist):
                    try:
                        diff = float(current_dist) - float(prev_dist)
                        if CONVERT_TO_MILES:
                            if INPUT_DISTANCE_UNIT.lower() == "feet":
                                conv_factor = 5280.0
                            elif INPUT_DISTANCE_UNIT.lower() == "meters":
                                conv_factor = 1609.34
                            else:
                                logging.warning(f"Unknown input distance unit '{INPUT_DISTANCE_UNIT}'. No conversion applied.")
                                conv_factor = 1.0
                            diff = diff / conv_factor
                        distance = f"{diff:.2f}"
                    except (ValueError, TypeError) as e:
                        logging.error(f"Error calculating distance difference for trip {trip_id}: {e}")
                        distance = ""
                else:
                    distance = ""
            stops_list.append((stop_name, stop_id, distance))
            if pd.notnull(current_dist):
                prev_dist = current_dist
            else:
                prev_dist = None
        if group.empty:
            continue
        first_row = group.iloc[0]
        trip_patterns.append({
            'trip_id': trip_id,
            'route_id': first_row.get('route_id', 'Unknown'),
            'direction_id': first_row.get('direction_id', 'Unknown'),
            'pattern': tuple(stops_list)
        })

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

    logging.info(f"Generated {len(patterns_dict)} unique patterns.")
    return patterns_dict

def assign_pattern_ids(patterns_dict):
    """
    Given a dictionary of patterns (keyed by (route_id, direction_id, pattern)),
    assign a sequential pattern_id for each (route_id, direction_id) group.

    Returns a list of records, each with:
      route_id, direction_id, pattern_id, trip_count, pattern
    """
    patterns_by_route_dir = {}
    for key, rec in patterns_dict.items():
        route_id = rec['route_id']
        direction_id = rec['direction_id']
        patterns_by_route_dir.setdefault((route_id, direction_id), []).append(rec)

    pattern_records = []
    for (route_id, direction_id), recs in patterns_by_route_dir.items():
        recs = sorted(recs, key=lambda r: r['pattern'])
        for i, rec in enumerate(recs, start=1):
            rec['pattern_id'] = i
            pattern_records.append({
                'route_id': route_id,
                'direction_id': direction_id,
                'pattern_id': i,
                'trip_count': rec['trip_count'],
                'pattern': rec['pattern']
            })

    logging.info("Assigned pattern IDs.")
    return pattern_records

def export_patterns_to_excel(pattern_records, routes_df):
    """
    Export pattern records to Excel files, one file per route.
    Each Excel file contains worksheets for each unique stop pattern.
    Uses route_short_name for file names and the "Route" column values.
    """
    # Group pattern records by route_id
    route_groups = {}
    for rec in pattern_records:
        route_groups.setdefault(rec['route_id'], []).append(rec)

    for route_id, records in route_groups.items():
        # Retrieve the route_short_name for the given route_id.
        try:
            route_info = routes_df[routes_df['route_id'] == route_id]
        except Exception as e:
            logging.error(f"Error retrieving route info for route_id {route_id}: {e}")
            route_info = None

        if route_info is not None and not route_info.empty:
            route_short_name = route_info.iloc[0].get('route_short_name', f"Route_{route_id}")
        else:
            route_short_name = f"Route_{route_id}"

        # Create a new workbook for the route
        workbook = Workbook()
        # Remove the default sheet
        default_sheet = workbook.active
        workbook.remove(default_sheet)

        for rec in records:
            direction_id = rec.get('direction_id', 'Unknown')
            pattern_id = rec.get('pattern_id', 'Unknown')
            worksheet_title = f"Dir{direction_id}_Pat{pattern_id}"
            try:
                worksheet = workbook.create_sheet(title=worksheet_title)
            except Exception as e:
                logging.error(f"Error creating worksheet '{worksheet_title}' for route {route_id}: {e}")
                continue

            # Build header row: Route, Direction, Distance, plus one column per stop
            header = ["Route", "Direction", "Distance"]
            stops = rec.get('pattern', [])
            stop_headers = [f"{stop_name} ({stop_id})" for stop_name, stop_id, _ in stops]
            header.extend(stop_headers)

            try:
                worksheet.append(header)
            except Exception as e:
                logging.error(f"Error writing header in worksheet '{worksheet_title}' for route {route_id}: {e}")
                continue

            # Calculate the total distance (sum the numeric stop distances)
            total_distance = 0.0
            for stop in stops:
                dist = stop[2]
                if dist not in ("-", "", None) and is_number(dist):
                    try:
                        total_distance += float(dist)
                    except Exception as e:
                        logging.error(f"Error converting stop distance '{dist}' in route {route_id}: {e}")
            total_distance_str = f"{total_distance:.2f}"

            # Use route_short_name for the Route column.
            row = [route_short_name, direction_id, total_distance_str] + [stop[2] for stop in stops]

            try:
                worksheet.append(row)
            except Exception as e:
                logging.error(f"Error writing data row in worksheet '{worksheet_title}' for route {route_id}: {e}")

            # Adjust column widths for better readability
            for i, _ in enumerate(header, start=1):
                col_letter = get_column_letter(i)
                worksheet.column_dimensions[col_letter].width = 20

        output_filename = f"{route_short_name}_{SIGNUP_NAME}.xlsx"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            workbook.save(output_path)
            logging.info(f"Workbook saved: {output_path}")
        except Exception as e:
            logging.error(f"Error saving workbook '{output_filename}': {e}")

# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logging.info(f"Created output directory: {OUTPUT_DIR}")
        except Exception as e:
            logging.error(f"Unable to create output directory '{OUTPUT_DIR}': {e}")
            return

    try:
        gtfs_data = load_gtfs_files(INPUT_DIR)
        stops_df = gtfs_data['stops']
        trips_df = gtfs_data['trips']
        stop_times_df = gtfs_data['stop_times']
        routes_df = gtfs_data['routes']
    except Exception as e:
        logging.error(e)
        return

    try:
        filtered_trips_df = filter_trips_by_route(trips_df, routes_df)
    except Exception as e:
        logging.error(e)
        return

    if filtered_trips_df.empty:
        logging.error("No trips to process after filtering. Exiting.")
        return

    try:
        patterns_dict = generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df)
        pattern_records = assign_pattern_ids(patterns_dict)
    except Exception as e:
        logging.error(f"Error generating patterns: {e}")
        return

    if not pattern_records:
        logging.warning("No unique patterns found. Exiting.")
        return

    try:
        export_patterns_to_excel(pattern_records, routes_df)
    except Exception as e:
        logging.error(f"Error exporting patterns to Excel: {e}")

if __name__ == "__main__":
    main()
