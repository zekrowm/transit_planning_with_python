"""
GTFS Schedule Processor
=======================

This script processes General Transit Feed Specification (GTFS) data to generate
Excel reports for specified transit routes and schedules. It reads GTFS files
such as trips, stop times, routes, stops, and calendar, and produces organized
Excel sheets with schedule times that emulate public printed schedules.

Configuration:
--------------
- **Input Paths**: Specify the directories containing GTFS data files.
- **Output Path**: Define where the Excel reports will be saved.
- **Route Selection**: Choose specific route short names or use 'all'.
- **Time Format**: Select 12-hour or 24-hour time formats for the output.

Features:
---------
- Filters to timepoints (major stops) or includes all stops if timepoint data is absent.
- Validates schedule times for chronological consistency.
- Exports each route's direction and schedule type as separate Excel sheets.
- Uses topological sorting for non-loop routes; special fallback for loops
  (preserving repeated appearances of a stop, plus ensuring first and last stops).
"""

import os
import re
import sys
from collections import deque

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

BASE_INPUT_PATH = r"C:\Path\To\Your\System\GTFS_Data"
BASE_OUTPUT_PATH = r"C:\Path\To\Your\Output_Folder"
if not os.path.exists(BASE_OUTPUT_PATH):
    os.makedirs(BASE_OUTPUT_PATH)

route_short_names_input = ['101', '202']  # 'all' or a list of short names
TIME_FORMAT_OPTION = '24'  # '12' or '24'
MISSING_TIME = "---"
MAX_COLUMN_WIDTH = 30

trips_file = os.path.join(BASE_INPUT_PATH, "trips.txt")
stop_times_file = os.path.join(BASE_INPUT_PATH, "stop_times.txt")
routes_file = os.path.join(BASE_INPUT_PATH, "routes.txt")
stops_file = os.path.join(BASE_INPUT_PATH, "stops.txt")
calendar_file = os.path.join(BASE_INPUT_PATH, "calendar.txt")

# Define module-level variables so they are recognized by linters
TRIPS = None
STOP_TIMES = None
ROUTES = None
STOPS = None
CALENDAR = None
TIMEPOINTS = None

# ==============================
# END OF CONFIGURATION SECTION
# ==============================


def time_to_minutes(time_str):
    """
    Converts a time string to total minutes since midnight.
    Supports 'HH:MM' and 'HH:MM AM/PM' formats, including hours >= 24 in 24-hr mode.
    Returns None if the format is invalid or time_str == MISSING_TIME.
    """
    if time_str == MISSING_TIME:
        return None
    try:
        match = re.match(
            r'^(\d{1,2}):(\d{2})(?:\s*(AM|PM))?$',
            time_str,
            re.IGNORECASE
        )
        if not match:
            return None
        hour_str, minute_str, period = match.groups()
        hour = int(hour_str)
        minute = int(minute_str)
        if period:
            period = period.upper()
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
        return hour * 60 + minute
    except (ValueError, TypeError, re.error):
        # Catches invalid int conversion or regex error
        return None


def remove_empty_schedule_columns(input_df):
    """
    Drops any columns in input_df that are entirely '---' (MISSING_TIME).
    """
    schedule_cols = [col for col in input_df.columns if col.endswith("Schedule")]
    all_blank_cols = [col for col in schedule_cols if (input_df[col] == MISSING_TIME).all()]
    input_df.drop(columns=all_blank_cols, inplace=True)
    return input_df


def check_schedule_order(input_df, ordered_stop_names, route_short_name, schedule_type, dir_id):
    """
    Checks times in the DataFrame to ensure they increase across rows (within a trip)
    and down columns (across trips).
    """
    violations = False

    # Row-wise check (each trip should have increasing times)
    for _, row in input_df.iterrows():
        last_time = None
        for stop in ordered_stop_names:
            col_name = f"{stop} Schedule"
            if col_name not in row:
                continue
            time_str = row[col_name]
            current_time = time_to_minutes(time_str)
            if current_time is None:
                continue
            if last_time is not None and current_time < last_time:
                print(
                    f"⚠️ Time order violation in Route '{route_short_name}', "
                    f"Schedule '{schedule_type}', Direction '{dir_id}', "
                    f"Trip '{row['Trip Headsign']}': '{stop}' time {time_str} "
                    f"is earlier than the previous stop's time."
                )
                violations = True
                break
            last_time = current_time

    # Column-wise check (each stop across multiple trips)
    for stop in ordered_stop_names:
        col_name = f"{stop} Schedule"
        if col_name not in input_df.columns:
            continue
        last_time = None
        for _, row in input_df.iterrows():
            time_str = row[col_name]
            current_time = time_to_minutes(time_str)
            if current_time is None:
                continue
            if last_time is not None and current_time < last_time:
                print(
                    f"⚠️ Time order violation in Route '{route_short_name}', "
                    f"Schedule '{schedule_type}', Direction '{dir_id}', "
                    f"Stop '{stop}': time {time_str} is earlier than the previous trip."
                )
                violations = True
                break
            last_time = current_time

    if not violations:
        print("✅ Schedule order check passed.")


def adjust_time(time_str, time_format='24'):
    """
    Adjusts time strings to the desired format:
      - '12' => 12-hour with AM/PM
      - '24' => 24-hour
    Returns MISSING_TIME if input is MISSING_TIME, or None if invalid.
    """
    if time_str.strip() == MISSING_TIME:
        return MISSING_TIME

    parts = time_str.strip().split(":")
    if len(parts) >= 2:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            if time_format == '12':
                # If hours >= 24, we can't fully convert, so keep it as is
                if hours >= 24:
                    print(
                        f"Warning: Cannot fully convert time '{time_str}' to 12-hour format. Keeping as is."
                    )
                    return time_str
                period = 'AM' if hours < 12 else 'PM'
                adjusted_hour = hours % 12
                if adjusted_hour == 0:
                    adjusted_hour = 12
                return f"{adjusted_hour}:{minutes:02d} {period}"
            # Otherwise 24-hour format
            return f"{hours:02d}:{minutes:02d}"
        except ValueError:
            print(f"Warning: Invalid time format encountered: '{time_str}'")
            return None
    print(f"Warning: Invalid time format encountered: '{time_str}'")
    return None


def fallback_preserve_repeats(all_stops_df, raw_stop_times):
    """
    Fallback logic that:
      1) Forces the first and last stops of each trip into the DataFrame
         (in case they were not marked as timepoints).
      2) Preserves multiple appearances of the same stop_id by
         storing (stop_id, stop_sequence) pairs.
      3) Sorts by stop_sequence and renames repeated stops as "(2)", "(3)", etc.
    """
    global STOPS

    # 1) Force-include first & last stops for each trip
    trip_ids = all_stops_df['trip_id'].unique()
    forced_rows = []
    for tid in trip_ids:
        trip_data = raw_stop_times[raw_stop_times['trip_id'] == tid].copy()
        if trip_data.empty:
            continue
        trip_data['stop_sequence'] = pd.to_numeric(trip_data['stop_sequence'], errors='coerce')
        trip_data.dropna(subset=['stop_sequence'], inplace=True)
        if trip_data.empty:
            continue

        # Identify the earliest and latest sequence
        min_seq_row = trip_data.loc[trip_data['stop_sequence'].idxmin()]
        max_seq_row = trip_data.loc[trip_data['stop_sequence'].idxmax()]
        forced_rows.append(min_seq_row)
        if max_seq_row['stop_sequence'] != min_seq_row['stop_sequence']:
            forced_rows.append(max_seq_row)

    if forced_rows:
        forced_df = pd.DataFrame(forced_rows)
        forced_df = pd.merge(
            forced_df,
            STOPS[['stop_id', 'stop_name']],
            on='stop_id',
            how='left'
        )
        # Union them with all_stops_df
        all_stops_df = pd.concat([all_stops_df, forced_df], ignore_index=True)

    # 2) Now gather (stop_id, stop_sequence, stop_name) pairs
    loop_entries = all_stops_df[['stop_id', 'stop_sequence', 'stop_name']].dropna(subset=['stop_id']).copy()
    loop_entries['stop_sequence'] = pd.to_numeric(loop_entries['stop_sequence'], errors='coerce')
    loop_entries.drop_duplicates(subset=['stop_id', 'stop_sequence'], inplace=True)

    # Sort by stop_sequence
    loop_entries.sort_values('stop_sequence', inplace=True)

    # 3) If the same stop_id appears multiple times, label them "StopName", "StopName (2)", etc.
    counter_map = {}
    new_names = []
    for _, row in loop_entries.iterrows():
        sid = row['stop_id']
        sname = row['stop_name']
        count = counter_map.get(sid, 0) + 1
        counter_map[sid] = count
        if count == 1:
            new_names.append(sname)
        else:
            new_names.append(f"{sname} ({count})")

    loop_entries['stop_name'] = new_names
    loop_entries.reset_index(drop=True, inplace=True)

    return loop_entries


def get_ordered_stops(dir_id, relevant_trips_dir):
    """
    For a given direction_id, returns (ordered_stop_names, DataFrame_of_stops).
    1) Subset TIMEPOINTS to relevant trips.
    2) Detect loop route if any trip visits the same stop_id more than once.
    3) If not a loop, do a topological sort of all stops across trips.
    4) If a loop or cycle, fallback to 'fallback_preserve_repeats'.
    """
    global TIMEPOINTS, STOPS, STOP_TIMES

    relevant_dir = relevant_trips_dir[relevant_trips_dir['direction_id'] == dir_id]
    if relevant_dir.empty:
        print(f"Warning: No trips found for direction_id '{dir_id}'.")
        return [], pd.DataFrame()

    # Subset timepoints for only those trips
    all_stops = TIMEPOINTS[TIMEPOINTS['trip_id'].isin(relevant_dir['trip_id'])].copy()
    if all_stops.empty:
        print(f"Warning: No stop times found (timepoints) for direction_id '{dir_id}'.")
        return [], pd.DataFrame()

    # Convert sequences to numeric
    all_stops['stop_sequence'] = pd.to_numeric(all_stops['stop_sequence'], errors='coerce')

    # Detect loop route
    loop_route_detected = False
    for trip_id, group in all_stops.groupby('trip_id'):
        if group['stop_id'].duplicated().any():
            loop_route_detected = True
            break

    # Merge so we have 'stop_name'
    all_stops_merged = pd.merge(
        all_stops,
        STOPS[['stop_id', 'stop_name']],
        on='stop_id',
        how='left'
    )

    # If loop => fallback
    if loop_route_detected:
        print(f"Detected loop route in direction_id '{dir_id}'. Using fallback to preserve repeats.")
        fallback_df = fallback_preserve_repeats(all_stops_merged, STOP_TIMES)
        return fallback_df['stop_name'].tolist(), fallback_df

    # Otherwise, attempt topological sort
    unique_stops_list = all_stops['stop_id'].unique().tolist()
    adjacency = {s: set() for s in unique_stops_list}
    in_degree = {s: 0 for s in unique_stops_list}

    # Build directed edges from consecutive stops in each trip
    for trip_id, group in all_stops.groupby('trip_id'):
        g_sorted = group.sort_values('stop_sequence')
        stops_in_order = g_sorted['stop_id'].tolist()
        for i in range(len(stops_in_order) - 1):
            cur_s = stops_in_order[i]
            nxt_s = stops_in_order[i+1]
            if nxt_s not in adjacency[cur_s]:
                adjacency[cur_s].add(nxt_s)
                in_degree[nxt_s] += 1

    # Kahn's Algorithm
    queue = deque([s for s in unique_stops_list if in_degree[s] == 0])
    topo_result = []

    while queue:
        node = queue.popleft()
        topo_result.append(node)
        for nbr in adjacency[node]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    # If not all included => cycle => fallback
    if len(topo_result) < len(unique_stops_list):
        print(f"Cycle detected in direction_id '{dir_id}'. Using fallback to preserve repeats.")
        fallback_df = fallback_preserve_repeats(all_stops_merged, STOP_TIMES)
        return fallback_df['stop_name'].tolist(), fallback_df

    # Build final DataFrame
    stop_info = STOPS.set_index('stop_id')['stop_name'].to_dict()
    final_df = pd.DataFrame({
        'stop_id': topo_result,
        'stop_sequence': range(len(topo_result)),
        'stop_name': [stop_info.get(s, f"Unknown stop {s}") for s in topo_result]
    })

    return final_df['stop_name'].tolist(), final_df


def map_service_id_to_schedule(service_row_local):
    """
    Maps a service_id row to a schedule type based on the days it serves.
    """
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    served_days = [day for day in days if service_row_local.get(day, '0') == '1']

    weekday = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday'}
    weekday_except_friday = {'monday', 'tuesday', 'wednesday', 'thursday'}
    saturday = {'saturday'}
    sunday = {'sunday'}
    weekend = {'saturday', 'sunday'}
    daily = set(days)

    if not served_days:
        return 'Holiday'

    served_set = set(served_days)
    if served_set == weekday:
        return 'Weekday'
    if served_set == weekday_except_friday:
        return 'Weekday_except_Friday'
    if served_set == saturday:
        return 'Saturday'
    if served_set == sunday:
        return 'Sunday'
    if served_set == weekend:
        return 'Weekend'
    if served_set == {'friday', 'saturday'}:
        return 'Friday-Saturday'
    if served_set == daily:
        return 'Daily'
    return 'Special'


def process_single_trip(trip_id, group, ordered_stop_ids, stop_index_map,
                        route_short, sched_type, dir_id, time_fmt):
    """
    Process a single trip, returning one row of data plus a 'max_sort_time' for sorting.
    """
    global TRIPS, ROUTES

    trip_info = TRIPS[TRIPS['trip_id'] == trip_id].iloc[0]
    route_name_val = ROUTES[ROUTES['route_id'] == trip_info['route_id']]['route_short_name'].values[0]
    trip_headsign = trip_info.get('trip_headsign', '')
    row_data = [route_name_val, trip_info['direction_id'], trip_headsign]

    # Prepare columns (one for each stop)
    schedule_times = [MISSING_TIME] * len(ordered_stop_ids)
    valid_24h_times = []

    for _, stop in group.iterrows():
        departure_raw = stop['departure_time'].strip()
        time_str_display = adjust_time(departure_raw, time_fmt)
        time_str_24 = adjust_time(departure_raw, '24')

        if time_str_display is None or time_str_24 is None:
            print(
                f"Warning: Invalid time '{stop['departure_time']}' in trip '{trip_id}' "
                f"at stop '{stop['stop_id']}'"
            )
            continue

        s_id = stop['stop_id']
        if s_id in stop_index_map:
            idx = stop_index_map[s_id]
            schedule_times[idx] = time_str_display
        valid_24h_times.append(time_str_24)

    # Use the max (latest) departure time for final sorting of rows
    if valid_24h_times:
        try:
            timedeltas = [pd.to_timedelta(f"{t}:00") for t in valid_24h_times]
            max_sort_time = max(timedeltas)
        except (ValueError, TypeError) as e:
            max_sort_time = pd.to_timedelta('00:00')
            print(
                f"Warning: Could not parse times for trip '{trip_id}'. "
                f"Defaulting sort_time=00:00. Error: {e}"
            )
    else:
        max_sort_time = pd.to_timedelta('9999:00:00')

    row_data.extend(schedule_times)
    row_data.append(max_sort_time)

    # Quick check of within-trip sequences
    times_in_minutes = [time_to_minutes(t) for t in schedule_times if t != MISSING_TIME]
    for i in range(1, len(times_in_minutes)):
        if times_in_minutes[i] < times_in_minutes[i - 1]:
            print(
                f"⚠️ Non-sequential departure times in trip_id '{trip_id}', "
                f"Route '{route_short}', Schedule '{sched_type}', Direction '{dir_id}'."
            )
            break

    return row_data


def process_trips_for_direction(params):
    """
    Processes trips for a specific direction_id => returns a DataFrame for that direction.
    """
    global TIMEPOINTS

    trips_dir = params["trips_dir"]
    stop_names_ordered = params["stop_names_ordered"]
    stops_unique = params["stops_unique"]
    time_fmt = params["time_fmt"]
    route_short = params["route_short"]
    sched_type = params["sched_type"]
    dir_id = params["dir_id"]

    if trips_dir.empty:
        print(f"No trips for direction {dir_id}. Skipping.")
        return pd.DataFrame()

    # Build a map from stop_id => column index
    ordered_stop_ids = stops_unique['stop_id'].tolist()
    stop_index_map = {s_id: i for i, s_id in enumerate(ordered_stop_ids)}

    # Collect rows of data
    output_data = []

    # Subset TIMEPOINTS to just these trips
    group_mask = TIMEPOINTS['trip_id'].isin(trips_dir['trip_id'])
    timepoints_dir = TIMEPOINTS[group_mask].copy()

    # For each trip, produce one row
    for trip_id, group in timepoints_dir.groupby('trip_id'):
        row_data = process_single_trip(
            trip_id,
            group,
            ordered_stop_ids,
            stop_index_map,
            route_short,
            sched_type,
            dir_id,
            time_fmt
        )
        output_data.append(row_data)

    # Build columns: first 3 = meta, next are stops, final is 'sort_time'
    col_names = (
        ['Route Name', 'Direction ID', 'Trip Headsign']
        + [f"{sn} Schedule" for sn in stop_names_ordered]
        + ['sort_time']
    )
    output_df = pd.DataFrame(output_data, columns=col_names)

    # Sort by 'sort_time' so it’s roughly chronological
    output_df.sort_values(by='sort_time', inplace=True)
    output_df.drop(columns=['sort_time'], inplace=True, errors='ignore')

    # Optional: check ordering across the table
    check_schedule_order(output_df, stop_names_ordered, route_short, sched_type, dir_id)

    # Remove any fully empty columns
    remove_empty_schedule_columns(output_df)

    return output_df


def export_to_excel_multiple_sheets(df_dict, out_file):
    """
    Exports multiple DataFrames to an Excel file with each DataFrame on its own sheet.
    """
    if not df_dict:
        print(f"No data to export to {out_file}.")
        return

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        for sheet_name, input_df in df_dict.items():
            if input_df.empty:
                print(f"No data for sheet '{sheet_name}'. Skipping...")
                continue

            input_df.to_excel(writer, index=False, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]

            for col_num, _ in enumerate(input_df.columns, 1):
                col_letter = get_column_letter(col_num)
                header_cell = worksheet[f'{col_letter}1']
                header_cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
                for row_num in range(2, worksheet.max_row + 1):
                    cell = worksheet[f'{col_letter}{row_num}']
                    cell.alignment = Alignment(horizontal='left')

                # Calculate column width
                column_cells = worksheet[col_letter]
                try:
                    max_length = max(
                        len(str(cell.value)) for cell in column_cells if cell.value is not None
                    )
                except (ValueError, TypeError):
                    max_length = 10
                adjusted_width = min(max_length + 2, MAX_COLUMN_WIDTH)
                worksheet.column_dimensions[col_letter].width = adjusted_width

    print(f"Data exported to {out_file}")


def load_gtfs_files():
    """
    Loads all GTFS data into global dataframes.
    """
    global TRIPS, STOP_TIMES, ROUTES, STOPS, CALENDAR
    try:
        TRIPS = pd.read_csv(trips_file, dtype=str)
        STOP_TIMES = pd.read_csv(stop_times_file, dtype=str)
        ROUTES = pd.read_csv(routes_file, dtype=str)
        STOPS = pd.read_csv(stops_file, dtype=str)
        CALENDAR = pd.read_csv(calendar_file, dtype=str)
        print("Successfully loaded all GTFS files.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error reading GTFS files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading GTFS files: {e}")
        sys.exit(1)


def prepare_timepoints():
    """
    Subset STOP_TIMES to timepoint=1 if available, else use all.
    This attempts to limit to major stops only, unless 'timepoint' is missing.
    """
    global STOP_TIMES, TIMEPOINTS

    STOP_TIMES['stop_sequence'] = pd.to_numeric(STOP_TIMES['stop_sequence'], errors='coerce')
    if STOP_TIMES['stop_sequence'].isnull().any():
        print("Warning: Some 'stop_sequence' values could not be converted to numeric.")

    if 'timepoint' in STOP_TIMES.columns:
        TIMEPOINTS = STOP_TIMES[STOP_TIMES['timepoint'] == '1'].copy()
        print("Filtered STOP_TIMES to rows with timepoint=1.")
    else:
        print("Warning: 'timepoint' column not found. Using all stops as timepoints.")
        TIMEPOINTS = STOP_TIMES.copy()


def determine_route_short_names():
    """
    Returns a list of route short names based on route_short_names_input.
    """
    if isinstance(route_short_names_input, str):
        if route_short_names_input.lower() == 'all':
            return ROUTES['route_short_name'].dropna().unique().tolist()
        return [name.strip() for name in route_short_names_input.split(',')]
    elif isinstance(route_short_names_input, list):
        # If the list contains 'all', pick all
        if any(name.lower() == 'all' for name in route_short_names_input):
            return ROUTES['route_short_name'].dropna().unique().tolist()
        return route_short_names_input
    else:
        print("Error: 'route_short_names_input' must be 'all', or comma-separated string, or a list.")
        sys.exit(1)


def build_service_id_schedule_map():
    """
    Creates a dict: service_id -> schedule_type from CALENDAR,
    plus a set of all schedule types found.
    """
    service_id_schedule_map = {}
    schedule_types_set = set()
    for _, service_row_local in CALENDAR.iterrows():
        sid_val = service_row_local['service_id']
        stype_var = map_service_id_to_schedule(service_row_local)
        service_id_schedule_map[sid_val] = stype_var
        schedule_types_set.add(stype_var)
    return service_id_schedule_map, schedule_types_set


def main():
    # 1. Load GTFS Files
    load_gtfs_files()

    # 2. Prepare Timepoints
    prepare_timepoints()

    # 3. Map service IDs to schedule types
    service_id_schedule_map, schedule_types_set = build_service_id_schedule_map()
    print(f"Identified schedule types: {schedule_types_set}")

    # 4. Determine routes
    route_short_names = determine_route_short_names()
    print(f"Selected routes: {route_short_names}")

    # 5. Process each route
    for route_short_name in route_short_names:
        print(f"\nProcessing route '{route_short_name}'...")

        route_ids = ROUTES[ROUTES['route_short_name'] == route_short_name]['route_id']
        if route_ids.empty:
            print(f"Error: Route '{route_short_name}' not found in routes.txt.")
            continue

        # For each schedule type
        for schedule_type in schedule_types_set:
            print(f"  Processing schedule type '{schedule_type}'...")
            relevant_service_ids = [
                sid for sid, stype in service_id_schedule_map.items()
                if stype == schedule_type
            ]
            if not relevant_service_ids:
                print(f"    No services for schedule type '{schedule_type}'.")
                continue

            relevant_trips = TRIPS[
                (TRIPS['route_id'].isin(route_ids)) &
                (TRIPS['service_id'].isin(relevant_service_ids))
            ]
            if relevant_trips.empty:
                print(
                    f"    No trips found for route '{route_short_name}' "
                    f"with schedule '{schedule_type}'."
                )
                continue

            # Build direction-based outputs
            direction_ids_local = relevant_trips['direction_id'].unique()
            df_sheets = {}

            for dir_id in direction_ids_local:
                print(f"    Processing direction_id '{dir_id}'...")
                trips_direction = relevant_trips[relevant_trips['direction_id'] == dir_id]

                # Get the final stop ordering
                stop_names_ordered, stops_unique = get_ordered_stops(dir_id, relevant_trips)
                if not stop_names_ordered:
                    print(f"      No stops found for direction '{dir_id}'. Skipping.")
                    continue

                params_dict = {
                    "trips_dir": trips_direction,
                    "stop_names_ordered": stop_names_ordered,
                    "stops_unique": stops_unique,
                    "time_fmt": TIME_FORMAT_OPTION,
                    "route_short": route_short_name,
                    "sched_type": schedule_type,
                    "dir_id": dir_id
                }

                output_df = process_trips_for_direction(params_dict)
                if output_df.empty:
                    print(f"      No data to export for direction_id '{dir_id}'.")
                    continue

                sheet_name = f"Direction_{dir_id}"
                df_sheets[sheet_name] = output_df

            # If we have data for any direction in this schedule type, export
            if df_sheets:
                schedule_type_safe = schedule_type.replace(' ', '_').replace('-', '_').replace('/', '_')
                out_file = os.path.join(
                    BASE_OUTPUT_PATH,
                    f"route_{route_short_name}_schedule_{schedule_type_safe}.xlsx"
                )
                export_to_excel_multiple_sheets(df_sheets, out_file)
            else:
                print(f"    No data to export for route '{route_short_name}' with schedule '{schedule_type}'.")


if __name__ == "__main__":
    main()
