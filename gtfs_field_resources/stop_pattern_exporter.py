"""
This module extracts unique stop patterns from GTFS data and exports them to Excel workbooks.
It provides detailed summaries of stop patterns by route, direction, and service ID, including
the earliest start times and pattern frequencies.

Key features:
- Route-based and calendar-based (service_id) filtering.
- Distance unit conversions (meters or feet to miles).
- Export of timepoint-only stop patterns, with optional validation of distances.
- Computation and inclusion of earliest departure times for each pattern.
- Improved Excel outputs, organized by route and direction, featuring a master-trip structure to
  clearly visualize stop patterns across different trips.
"""
import logging
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================
# CONFIGURATION
# ===============================
INPUT_DIR = r'\\Path\\To\\Your\\GTFS'
OUTPUT_DIR = r'\\Path\\To\\Output'
TRIPS_FILE = 'trips.txt'
STOP_TIMES_FILE = 'stop_times.txt'
STOPS_FILE = 'stops.txt'
ROUTES_FILE = 'routes.txt'

FILTER_IN_ROUTE_SHORT_NAMES = []
FILTER_OUT_ROUTE_SHORT_NAMES = []
FILTER_IN_CALENDAR_IDS = []

SIGNUP_NAME = "January2025Signup"

INPUT_DISTANCE_UNIT = "meters"
CONVERT_TO_MILES = True
EXPORT_TIMEPOINTS_ONLY = True
VALIDATE_TIMEPOINT_DISTANCE = True

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def convert_dist_to_miles(distance, input_unit):
    """
    Convert 'distance' to miles if configured. Otherwise return as is.
    """
    if not CONVERT_TO_MILES or pd.isna(distance):
        return distance
    if input_unit.lower() == "feet":
        conv = 5280.0
    elif input_unit.lower() == "meters":
        conv = 1609.34
    else:
        logging.warning("Unknown distance unit '%s'. No conversion done.", input_unit)
        conv = 1.0
    return distance / conv

def parse_time_to_minutes(timestr):
    """
    Parse HH:MM:SS (GTFS style) to float minutes. Return None if invalid.
    Allows hours >= 24 for service passing midnight.
    """
    if not isinstance(timestr, str):
        return None
    parts = timestr.strip().split(':')
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2])
        return hh*60 + mm + ss/60.0
    except:
        return None

def minutes_to_hhmm(m):
    """Convert float minutes to 'HH:MM' (24-hour)."""
    if m is None or pd.isna(m):
        return ""
    total_m = int(m)
    hh = total_m // 60
    mm = total_m % 60
    return f"{hh:02d}:{mm:02d}"

# ------------------------------------------------------------
# Loading GTFS
# ------------------------------------------------------------
def load_gtfs_files(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    stops = pd.read_csv(os.path.join(input_dir, STOPS_FILE))
    trips = pd.read_csv(os.path.join(input_dir, TRIPS_FILE))
    stop_times = pd.read_csv(os.path.join(input_dir, STOP_TIMES_FILE))
    routes = pd.read_csv(os.path.join(input_dir, ROUTES_FILE))

    logging.info("Loaded GTFS files successfully.")
    return {
        'stops': stops,
        'trips': trips,
        'stop_times': stop_times,
        'routes': routes
    }

def filter_trips(trips_df, routes_df, cal_ids):
    # Merge to get route_short_name
    merged = pd.merge(
        trips_df, 
        routes_df[['route_id','route_short_name']], 
        on='route_id', how='left'
    )
    # Filter in
    if FILTER_IN_ROUTE_SHORT_NAMES:
        merged = merged[merged['route_short_name'].isin(FILTER_IN_ROUTE_SHORT_NAMES)]
    # Filter out
    if FILTER_OUT_ROUTE_SHORT_NAMES:
        merged = merged[~merged['route_short_name'].isin(FILTER_OUT_ROUTE_SHORT_NAMES)]
    # Filter calendar
    if cal_ids and 'service_id' in merged.columns:
        merged = merged[merged['service_id'].isin(cal_ids)]
    elif cal_ids:
        logging.warning("No service_id in data for filtering.")
    return merged

# ------------------------------------------------------------
# Build Patterns (same as you had, or something similar)
# ------------------------------------------------------------
def generate_unique_patterns(trips_df, stop_times_df, stops_df):
    """
    Creates unique patterns keyed by (route_id, direction_id, service_id, patternOfStops).
    Each pattern is a tuple of stops: [ (stop_id, distanceFromPrevious), ...]
    The first entry has distance='-'. Also does optional timepoint filtering, etc.
    """
    # Merge to get route/direction
    tmp = pd.merge(
        stop_times_df,
        trips_df[['trip_id','route_id','direction_id','service_id']],
        on='trip_id', how='inner'
    )
    if 'shape_dist_traveled' not in tmp.columns:
        tmp['shape_dist_traveled'] = np.nan

    # Merge stop names so we can see them, but let's keep stop_id for matching
    tmp = pd.merge(tmp, stops_df[['stop_id','stop_name']], on='stop_id', how='left')
    tmp.sort_values(['trip_id','stop_sequence'], inplace=True)

    # If we want timepoint-only
    if EXPORT_TIMEPOINTS_ONLY and 'timepoint' in tmp.columns:
        tmp = tmp[tmp['timepoint'] == 1]

    # For distance validation
    trip_dist = {}
    if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
        # compute original full-trip dist
        for tid, grp in tmp.groupby('trip_id'):
            grp = grp.dropna(subset=['shape_dist_traveled'])
            if grp.empty:
                trip_dist[tid] = None
            else:
                dist_val = grp.iloc[-1]['shape_dist_traveled'] - grp.iloc[0]['shape_dist_traveled']
                trip_dist[tid] = convert_dist_to_miles(dist_val, INPUT_DISTANCE_UNIT)

    # Build patterns
    patterns_list = []
    for tid, grp in tmp.groupby('trip_id'):
        grp = grp.sort_values('stop_sequence')
        stops_for_this_trip = []
        prevDistVal = None
        for _, row in grp.iterrows():
            sid = row['stop_id']
            shapeVal = row['shape_dist_traveled']
            if prevDistVal is None:
                dStr = "-"
            else:
                if pd.notnull(shapeVal) and pd.notnull(prevDistVal):
                    diff = convert_dist_to_miles(shapeVal - prevDistVal, INPUT_DISTANCE_UNIT)
                    dStr = f"{diff:.2f}" if diff else ""
                else:
                    dStr = ""
            stops_for_this_trip.append( (sid, dStr) )
            prevDistVal = shapeVal if pd.notnull(shapeVal) else None

        # Validate if desired
        if VALIDATE_TIMEPOINT_DISTANCE and EXPORT_TIMEPOINTS_ONLY:
            sumSeg = 0.0
            for (_, ds) in stops_for_this_trip:
                if ds not in ("-", "", None) and is_number(ds):
                    sumSeg += float(ds)
            fullTrip = trip_dist.get(tid, None)
            if fullTrip is not None and abs(sumSeg - fullTrip) > 0.02:
                logging.warning(
                    "Trip %s sum of segments=%.2f vs. full=%.2f mismatch >0.02",
                    tid, sumSeg, fullTrip
                )

        # Store record
        firstRow = grp.iloc[0]
        rid = firstRow['route_id']
        did = firstRow['direction_id']
        sid = firstRow['service_id']
        patterns_list.append({
            'trip_id': tid,
            'route_id': rid,
            'direction_id': did,
            'service_id': sid,
            'pattern_stops': tuple(stops_for_this_trip)  # immutable
        })

    # Accumulate unique patterns
    patterns_dict = {}
    for rec in patterns_list:
        key = (rec['route_id'], rec['direction_id'], rec['service_id'], rec['pattern_stops'])
        if key not in patterns_dict:
            patterns_dict[key] = {
                'route_id': rec['route_id'],
                'direction_id': rec['direction_id'],
                'service_id': rec['service_id'],
                'pattern_stops': rec['pattern_stops'],
                'trip_count': 0,
                'trip_ids': []
            }
        patterns_dict[key]['trip_count'] += 1
        patterns_dict[key]['trip_ids'].append(rec['trip_id'])

    logging.info("Found %d unique patterns.", len(patterns_dict))
    return patterns_dict

def assign_pattern_ids(patterns_dict):
    """
    For each route/service/direction, assign pattern_id in ascending order.
    """
    from collections import defaultdict
    group_map = defaultdict(list)
    for v in patterns_dict.values():
        route_id = v['route_id']
        dir_id = v['direction_id']
        srv_id = v['service_id']
        group_map[(route_id, srv_id, dir_id)].append(v)

    out = []
    for (rid, sid, did), recs in group_map.items():
        # sort stable by pattern_stops
        recs = sorted(recs, key=lambda x: x['pattern_stops'])
        for i, pattern_rec in enumerate(recs, 1):
            pattern_rec['pattern_id'] = i
            out.append({
                'route_id': pattern_rec['route_id'],
                'direction_id': pattern_rec['direction_id'],
                'service_id': pattern_rec['service_id'],
                'pattern_stops': pattern_rec['pattern_stops'],
                'trip_count': pattern_rec['trip_count'],
                'trip_ids': pattern_rec['trip_ids'],
                'pattern_id': i
            })
    logging.info("Assigned pattern IDs to pattern records.")
    return out

# ------------------------------------------------------------
# EARLIEST START TIME
# ------------------------------------------------------------
def compute_earliest_start_times(pattern_records, stop_times_df):
    """
    For each pattern, find the earliest arrival/departure time for the *first* stop of
    each trip in that pattern.  Then store the min as earliest_time_minutes & earliest_time_str.
    """
    if 'arrival_time' not in stop_times_df.columns:
        for r in pattern_records:
            r['earliest_time_minutes'] = None
            r['earliest_time_str'] = ""
        return

    st_by_trip = stop_times_df.groupby('trip_id')
    for rec in pattern_records:
        trip_ids = rec['trip_ids']
        earliest_val = None
        for tid in trip_ids:
            if tid not in st_by_trip.groups:
                continue
            g2 = st_by_trip.get_group(tid).sort_values('stop_sequence')
            if g2.empty:
                continue
            arr = g2.iloc[0].get('arrival_time','')
            dep = g2.iloc[0].get('departure_time','')
            arrm = parse_time_to_minutes(str(arr)) if arr else None
            depm = parse_time_to_minutes(str(dep)) if dep else None
            candidates = []
            if arrm is not None:
                candidates.append(arrm)
            if depm is not None:
                candidates.append(depm)
            if not candidates:
                continue
            thisMin = min(candidates)
            if earliest_val is None or thisMin < earliest_val:
                earliest_val = thisMin
        rec['earliest_time_minutes'] = earliest_val
        rec['earliest_time_str'] = minutes_to_hhmm(earliest_val) if earliest_val else ""

# ------------------------------------------------------------
# MASTER TRIP for each (Route/Direction)
# ------------------------------------------------------------
def find_master_trip_stops(route_id, direction_id, relevant_trips, stop_times_df, stops_df):
    """
    Among all trips in 'relevant_trips' (which is already filtered to route+dir),
    find the one with the largest number of stops, and read them in order.

    Return a list: [ (stop_id, stop_name), (stop_id, stop_name), ... ] 
    for that 'master' trip.
    """
    if relevant_trips.empty:
        return []

    # find the trip with the maximum # of stops/timepoints
    st_sub = stop_times_df[stop_times_df['trip_id'].isin(relevant_trips['trip_id'])]
    if 'timepoint' in st_sub.columns and EXPORT_TIMEPOINTS_ONLY:
        st_sub = st_sub[st_sub['timepoint'] == 1]

    # find trip_id with largest # of stops
    sizes = st_sub.groupby('trip_id').size()
    if sizes.empty:
        return []
    best_trip_id = sizes.idxmax()

    # get that trip's stops in order
    best_grp = st_sub[st_sub['trip_id']==best_trip_id].sort_values('stop_sequence')
    # merge to get name
    best_grp = pd.merge(best_grp, stops_df[['stop_id','stop_name']], on='stop_id', how='left')
    out_list = []
    for _, row in best_grp.iterrows():
        sid = row['stop_id']
        sname = row.get('stop_name','Unknown')
        out_list.append( (sid, sname) )
    return out_list

def forward_match_pattern_to_master(pattern_stops, master_stops):
    """
    We have:
      pattern_stops = [(stop_id1, dist_str1), (stop_id2, dist_str2), ...]
      master_stops =  [(sidA, nameA), (sidB, nameB), ...]

    We'll produce a list of length == len(master_stops) with the distance or ''.
    The *first* matched pattern stop => '-'.

    We'll do a forward-only match:
      i = 0 (master), j = 0 (pattern)
      while i < len(master_stops) and j < len(pattern_stops):
        if master_stops[i].stop_id == pattern_stops[j].stop_id:
          => place pattern_stops[j].dist in the result for column i
             i++, j++
        else:
          i++

    Because each pattern has the "first stop" labeled as '-', that means 
    the *very first time* we match j=0 => we place '-'. Then for j=1 => we place dist of pattern_stops[1], etc.
    """
    result = [""] * len(master_stops)
    i = 0
    j = 0
    while i < len(master_stops) and j < len(pattern_stops):
        master_sid = master_stops[i][0]
        pat_sid = pattern_stops[j][0]
        dist_str = pattern_stops[j][1]  # distance from previous stop
        if master_sid == pat_sid:
            result[i] = dist_str  # place the pattern distance
            i += 1
            j += 1
        else:
            i += 1
    return result

# ------------------------------------------------------------
# EXCEL EXPORT
# ------------------------------------------------------------
def create_workbook():
    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)
    return wb

def fill_worksheet_for_direction(
    wb, sheet_title, route_short_name, direction_id, service_id,
    pattern_records_dir, master_stops
):
    """
    Create one Excel sheet for route/direction, each unique pattern is one row.
    Columns are defined by 'master_stops' => one column per stop.

    Steps:
     1) We'll place [Route, Direction, Calendar, Pattern ID, TripCount, EarliestTime]
        then one column per "master stop" labeled with its 'stop_name'.
     2) For each pattern row, we do forward_match_pattern_to_master(...) 
        to fill the distance in the right columns.
     3) The first matched pattern stop is always '-', so the user sees that 
        as the distance to the first stop is none.
    """
    try:
        ws = wb.create_sheet(title=sheet_title)
    except ValueError:
        ws = wb.create_sheet(title=f"{sheet_title[:25]}_X")

    if not master_stops:
        ws.append(["No master stops found for direction."])
        return

    # Build header
    header = [
        "Route",
        "Direction",
        "Calendar (service_id)",
        "Pattern ID",
        "Trip Count",
        "Earliest Start Time"
    ]
    # add one column per master stop
    for (_, sname) in master_stops:
        header.append(sname)
    ws.append(header)

    # sort pattern records by earliest_time_minutes
    pattern_records_dir = sorted(
        pattern_records_dir,
        key=lambda r: (r.get('earliest_time_minutes') is None, r.get('earliest_time_minutes', 9999999))
    )

    # fill rows
    for rec in pattern_records_dir:
        pat_id = rec['pattern_id']
        tc = rec['trip_count']
        e_str = rec.get('earliest_time_str', "")
        pattern_stops = rec['pattern_stops']  # e.g. [ (stop_id, dist_str), ... ]
        # forward match
        row_distances = forward_match_pattern_to_master(pattern_stops, master_stops)

        row = [
            route_short_name,
            direction_id,
            service_id,
            pat_id,
            tc,
            e_str
        ]
        row.extend(row_distances)
        ws.append(row)

    # Column widths
    from openpyxl.utils import get_column_letter
    for col_index, _ in enumerate(header, 1):
        col_letter = get_column_letter(col_index)
        ws.column_dimensions[col_letter].width = 30

def export_patterns_to_excel(pattern_records, routes_df, stop_times_df, stops_df):
    """
    For each route/service, group patterns by direction. For each direction:
      1) find the "master trip" stops from the actual GTFS.
      2) fill one sheet with all patterns in that direction, using forward match.
    """
    from collections import defaultdict
    # group pattern_records by (route_id, service_id)
    group_map = defaultdict(list)
    for pr in pattern_records:
        rid = pr['route_id']
        sid = pr['service_id']
        group_map[(rid, sid)].append(pr)

    for (rid, sid), group_list in group_map.items():
        # find route_short_name
        route_info = routes_df[routes_df['route_id']==rid]
        if not route_info.empty:
            short_name = route_info.iloc[0].get('route_short_name', f"Route_{rid}")
        else:
            short_name = f"Route_{rid}"

        # Create workbook
        wb = create_workbook()

        # group by direction
        dir_map = defaultdict(list)
        for r2 in group_list:
            dir_id = r2['direction_id']
            dir_map[dir_id].append(r2)

        # We need to gather the actual trips that belong to *this route & service_id & direction*
        # so we can find the "master trip" from GTFS
        # But let's read the 'trip_ids' from pattern_records, or we can do a simpler approach:
        #    find the matching trips from the original "filter_trips" results. 
        # We'll do the simpler approach: each pattern has 'trip_ids', but we might want them all.
        # We'll do a union of trip_ids in each direction.
        # Then from those trips, we pick the largest (timepoint) trip to define columns.

        # But we need the filtered trips that match route_id & service_id => then direction
        # We'll do that outside, so let's pass the entire trips df? 
        # For simplicity, let's gather all trip_ids from pattern_records.

        # We'll store them by direction
        for d_id, recs_dir in dir_map.items():
            all_trip_ids = set()
            for r3 in recs_dir:
                all_trip_ids.update(r3['trip_ids'])
            # Now we have all relevant trip_ids => find master trip
            # We need a DF of those trips in that route/direction
            # Actually we only need the subset of trips that match route_id, service_id, direction_id
            # The user might have multiple service_ids? We have only one service_id here => sid
            # We'll do that below.

            # Build a small DataFrame "relevant_trips_dir" with trip_id in all_trip_ids
            # plus route_id=rid, service_id=sid, direction_id=d_id
            # But we never loaded that "filtered_trips_df" as a global. Let's do it quickly:
            # We'll rely on stop_times_df? Actually we want the "trips" file for route, service, direction.
            # We'll do an inner join approach:
            # But we might not *strictly* need route=, service= etc. if the trip_id set is correct.

            # For the master trip logic, we'll just need the largest # of stops among these trip_ids in stop_times_df
            # so let's do that approach. We'll do "stop_times_df[stop_times_df.trip_id.isin(all_trip_ids)]"

            st_sub = stop_times_df[stop_times_df['trip_id'].isin(all_trip_ids)]
            if 'timepoint' in st_sub.columns and EXPORT_TIMEPOINTS_ONLY:
                st_sub = st_sub[st_sub['timepoint'] == 1]

            # pick the trip_id with the largest # of stops
            sizes = st_sub.groupby('trip_id').size()
            if sizes.empty:
                master_stops = []
            else:
                best_tid = sizes.idxmax()
                # read that trip in order
                mg = st_sub[st_sub['trip_id']==best_tid].sort_values('stop_sequence')
                mg = pd.merge(mg, stops_df[['stop_id','stop_name']], on='stop_id', how='left')
                master_stops = []
                for _, rowz in mg.iterrows():
                    master_stops.append( (rowz['stop_id'], rowz.get('stop_name','Unknown')) )

            # Now we have master_stops. 
            # Fill the sheet
            sheet_title = f"Dir{d_id}"
            fill_worksheet_for_direction(
                wb, sheet_title, short_name, d_id, sid, recs_dir, master_stops
            )

        # Save
        fn = f"{short_name}_{sid}_{SIGNUP_NAME}.xlsx"
        fp = os.path.join(OUTPUT_DIR, fn)
        try:
            wb.save(fp)
            logging.info("Saved workbook: %s", fp)
        except Exception as e:
            logging.error("Could not save workbook '%s': %s", fn, e)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # 1) Load GTFS
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        gtfs_data = load_gtfs_files(INPUT_DIR)
    except Exception as e:
        logging.error("Failed to load GTFS: %s", e)
        return

    stops_df = gtfs_data['stops']
    trips_df = gtfs_data['trips']
    stop_times_df = gtfs_data['stop_times']
    routes_df = gtfs_data['routes']

    # 2) Filter
    filtered_trips = filter_trips(
        trips_df, routes_df,
        cal_ids=FILTER_IN_CALENDAR_IDS
    )
    if filtered_trips.empty:
        logging.error("No trips after filtering. Exiting.")
        return

    # 3) Generate unique patterns
    patterns_dict = generate_unique_patterns(filtered_trips, stop_times_df, stops_df)
    if not patterns_dict:
        logging.warning("No patterns found. Exiting.")
        return

    pattern_records = assign_pattern_ids(patterns_dict)
    if not pattern_records:
        logging.warning("No pattern records. Exiting.")
        return

    # 4) Earliest Start Times
    compute_earliest_start_times(pattern_records, stop_times_df)

    # 5) Export
    export_patterns_to_excel(pattern_records, routes_df, stop_times_df, stops_df)


if __name__ == "__main__":
    main()
