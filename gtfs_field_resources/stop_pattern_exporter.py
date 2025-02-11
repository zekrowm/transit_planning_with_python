#!/usr/bin/env python
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
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ============================================================
# CONFIGURATION
# ============================================================
# Input directory where GTFS files are stored.
INPUT_DIR = r'\\S40SHAREPGC01\DOTWorking\zkrohm\data_archive\connector_gtfs\connector_gtfs_2025_01_13'

# Output directory where the Excel files will be saved.
OUTPUT_DIR = r'\\S40SHAREPGC01\DOTWorking\zkrohm\data_requests\stop_pattern_export_test'

# GTFS filenames
TRIPS_FILE = 'trips.txt'
STOP_TIMES_FILE = 'stop_times.txt'
STOPS_FILE = 'stops.txt'
ROUTES_FILE = 'routes.txt'

# Optional filtering based on route_short_name.
# If FILTER_IN_ROUTE_SHORT_NAMES is not empty, only routes with a route_short_name in this list will be processed.
# If FILTER_OUT_ROUTE_SHORT_NAMES is not empty, routes with a route_short_name in this list will be excluded.
FILTER_IN_ROUTE_SHORT_NAMES = []     # e.g., ['10', '20']
FILTER_OUT_ROUTE_SHORT_NAMES = []    # e.g., ['30']

# Optional signup name to append to output filenames.
SIGNUP_NAME = ""  # e.g., "January2025Signup"

# ============================================================
# DISTANCE CONVERSION CONFIGURATION
# ============================================================
# Specify the unit of the shape_dist_traveled values in the GTFS data.
# Options: "feet" or "meters"
INPUT_DISTANCE_UNIT = "feet"  # Change to "meters" if your GTFS uses meters.

# Set to True to convert distances to miles.
CONVERT_TO_MILES = True

# ============================================================
# FUNCTION DEFINITIONS
# ============================================================

def load_gtfs_files(input_dir):
    """
    Load the necessary GTFS files from the input directory.
    
    Returns a dictionary with keys: stops, trips, stop_times, and routes.
    """
    try:
        stops = pd.read_csv(os.path.join(input_dir, STOPS_FILE))
        trips = pd.read_csv(os.path.join(input_dir, TRIPS_FILE))
        stop_times = pd.read_csv(os.path.join(input_dir, STOP_TIMES_FILE))
        routes = pd.read_csv(os.path.join(input_dir, ROUTES_FILE))
    except Exception as e:
        raise Exception(f"Error loading GTFS files: {e}")
    
    return {'stops': stops, 'trips': trips, 'stop_times': stop_times, 'routes': routes}

def generate_unique_patterns(filtered_trips_df, stop_times_df, stops_df):
    """
    For each trip, merge with stop_times and stops; sort by stop_sequence;
    and build a pattern as an ordered tuple of stops.
    
    Each stop is represented as a tuple: (stop_name, stop_id, distance)
    where distance is computed as the difference in shape_dist_traveled
    from the previous stop (first stop gets "-").
    
    If CONVERT_TO_MILES is True, the numeric differences are converted into miles.
    
    Returns:
        A dictionary keyed by (route_id, direction_id, pattern) containing:
            - route_id
            - direction_id
            - pattern (tuple of (stop_name, stop_id, distance))
            - trip_count (number of trips with that pattern)
    """
    trips_stop_times = pd.merge(
        stop_times_df, 
        filtered_trips_df[['trip_id', 'route_id', 'direction_id']], 
        on='trip_id', how='inner'
    )
    
    if 'shape_dist_traveled' not in trips_stop_times.columns:
        trips_stop_times['shape_dist_traveled'] = np.nan
    
    trips_stop_times = pd.merge(
        trips_stop_times, 
        stops_df[['stop_id', 'stop_name']], 
        on='stop_id', how='left'
    )
    
    trips_stop_times.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)
    
    trip_patterns = []
    for trip_id, group in trips_stop_times.groupby('trip_id'):
        group = group.sort_values('stop_sequence')
        stops_list = []
        prev_dist = None
        for _, row in group.iterrows():
            stop_name = row['stop_name']
            stop_id = row['stop_id']
            current_dist = row['shape_dist_traveled']
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
                                conv_factor = 1.0
                            diff = diff / conv_factor
                        distance = f"{diff:.2f}"
                    except Exception:
                        distance = ""
                else:
                    distance = ""
            stops_list.append( (stop_name, stop_id, distance) )
            if pd.notnull(current_dist):
                prev_dist = current_dist
            else:
                prev_dist = None
        first_row = group.iloc[0]
        trip_patterns.append({
            'trip_id': trip_id,
            'route_id': first_row['route_id'],
            'direction_id': first_row['direction_id'],
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
            rec['pattern_id']
