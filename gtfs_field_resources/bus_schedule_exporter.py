"""
GTFS Schedule Processor
=======================

This script processes General Transit Feed Specification (GTFS) data to generate
Excel reports for specified transit routes and schedules (e.g. route 99 weekday
schedule). It reads GTFS files such as trips, stop times, routes, stops, and
calendar, and produces organized Excel sheets with schedule times that emulate
public printed schedules.

Configuration:
--------------
- **Input Paths**: Specify the directories containing GTFS data files.
- **Output Path**: Define the directory where the Excel reports will be saved.
- **Route Selection**: Choose specific route short names to process or set to 'all'
  to include all routes.
- **Time Format**: Select between 12-hour or 24-hour time formats for the output.

Features:
---------
- Validates the chronological order of schedule times within trips and across stops.
- Supports customization of maximum column widths in the Excel output.
- Handles various schedule types based on service days, including weekdays, weekends,
  and special schedules.
- Provides informative warnings and confirmations to assist in data verification.
"""

import os
import re
import sys

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

route_short_names_input = ['101', '102']  # 'all' or list
TIME_FORMAT_OPTION = '12'  # or '24'
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
    Supports 'HH:MM' and 'HH:MM AM/PM' formats.
    Allows hours >=24 for 24-hour format.
    Returns None if the format is invalid.
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
    Drops any columns in input_df that are entirely '---'.
    """
    schedule_cols = [col for col in input_df.columns if col.endswith("Schedule")]
    all_blank_cols = [col for col in schedule_cols if (input_df[col] == MISSING_TIME).all()]
    input_df.drop(columns=all_blank_cols, inplace=True)
    return input_df


def check_schedule_order(input_df, ordered_stop_names, route_short_name, schedule_type, dir_id):
    """
    Checks times in the DataFrame to ensure they increase across rows and down columns.
    """
    violations = False

    # Row-wise check
    for _, row in input_df.iterrows():
        last_time = None
        for stop in ordered_stop_names:
            time_str = row.get(f"{stop} Schedule", MISSING_TIME)
            current_time = time_to_minutes(time_str)
            if current_time is None:
                continue
            if last_time is not None and current_time < last_time:
                print(
                    f"⚠️ Time order violation in Route '{route_short_name}', "
                    f"Schedule '{schedule_type}', Direction '{dir_id}', "
                    f"Trip '{row['Trip Headsign']}': '{stop}' time {time_str} "
                    f"is earlier than previous stop."
                )
                violations = True
                break
            last_time = current_time

    # Column-wise check
    for stop in ordered_stop_names:
        last_time = None
        for _, row in input_df.iterrows():
            time_str = row.get(f"{stop} Schedule", MISSING_TIME)
            current_time = time_to_minutes(time_str)
            if current_time is None:
                continue
            if last_time is not None and current_time < last_time:
                print(
                    f"⚠️ Time order violation in Route '{route_short_name}', "
                    f"Schedule '{schedule_type}', Direction '{dir_id}', "
                    f"Stop '{stop}': time {time_str} is earlier than previous trip."
                )
                violations = True
                break
            last_time = current_time

    if not violations:
        print("✅ Schedule order check passed.")


def adjust_time(time_str, time_format='24'):
    """
    Adjusts time strings to the desired format.
    Returns None if the format is invalid, or the original string if hours >=24 in 12-hour mode.
    """
    parts = time_str.strip().split(":")
    if len(parts) >= 2:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            if time_format == '12':
                if hours >= 24:
                    print(
                        f"Warning: Cannot convert time '{time_str}' to 12-hour format. Keeping as is."
                    )
                    return time_str
                period = 'AM' if hours < 12 else 'PM'
                adjusted_hour = hours % 12
                if adjusted_hour == 0:
                    adjusted_hour = 12
                return f"{adjusted_hour}:{minutes:02} {period}"
            return f"{hours:02}:{minutes:02}"
        except ValueError:
            print(f"Warning: Invalid time format encountered: '{time_str}'")
            return None
    print(f"Warning: Invalid time format encountered: '{time_str}'")
    return None


def get_ordered_stops(dir_id, relevant_trips_dir):
    """
    Retrieves and orders the stops for the given dir_id from relevant_trips_dir.
    """
    global TIMEPOINTS, STOPS

    relevant_dir = relevant_trips_dir[relevant_trips_dir['direction_id'] == dir_id]
    if relevant_dir.empty:
        print(f"Warning: No trips found for direction_id '{dir_id}'.")
        return [], []

    all_stops = TIMEPOINTS[TIMEPOINTS['trip_id'].isin(relevant_dir['trip_id'])]
    all_stops = all_stops.sort_values(['trip_id', 'stop_sequence'])
    if all_stops.empty:
        print(f"Warning: No stop times found for direction_id '{dir_id}'.")
        return [], []

    unique_stops = (
        all_stops[['stop_id', 'stop_sequence']]
        .drop_duplicates()
        .sort_values('stop_sequence')
    )
    stop_names = STOPS.set_index('stop_id')['stop_name']
    ordered_stop_names = [
        f"{stop_names.get(s_id, f'Unknown Stop ID {s_id}')} ({seq})"
        for s_id, seq in zip(unique_stops['stop_id'], unique_stops['stop_sequence'])
    ]
    return ordered_stop_names, unique_stops


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
    Process a single trip, returning one row of data (list) plus a 'max_sort_time' value.
    """
    global TRIPS, ROUTES, TIMEPOINTS

    trip_info = TRIPS[TRIPS['trip_id'] == trip_id].iloc[0]
    route_name_val = ROUTES[ROUTES['route_id'] == trip_info['route_id']]['route_short_name'].values[0]
    trip_headsign = trip_info.get('trip_headsign', '')
    row_data = [route_name_val, trip_info['direction_id'], trip_headsign]

    schedule_times = [MISSING_TIME] * len(ordered_stop_ids)
    valid_departure_times_24 = []

    for _, stop in group.iterrows():
        departure_str = stop['departure_time'].strip()
        time_str_display = adjust_time(departure_str, time_fmt)
        time_str_24 = adjust_time(departure_str, '24')
        if time_str_display is None or time_str_24 is None:
            print(
                f"Warning: Invalid time '{stop['departure_time']}' in trip_id '{trip_id}' "
                f"at stop_id '{stop['stop_id']}'"
            )
            continue
        seq_val = stop['stop_sequence']
        index_val = stop_index_map[seq_val]
        schedule_times[index_val] = time_str_display
        valid_departure_times_24.append(time_str_24)

    # Sort time is determined by the maximum departure time
    if valid_departure_times_24:
        try:
            departure_timedeltas = [
                pd.to_timedelta(f"{t_24}:00") for t_24 in valid_departure_times_24
            ]
            max_sort_time = max(departure_timedeltas)
        except (ValueError, TypeError) as e:
            max_sort_time = pd.to_timedelta('00:00')
            print(
                f"Warning: Failed to determine maximum departure time for trip_id '{trip_id}'. "
                f"Defaulting to '00:00'. Error: {e}"
            )
    else:
        max_sort_time = pd.to_timedelta('9999:00:00')  # Push trip to bottom

    row_data.extend(schedule_times)
    row_data.append(max_sort_time)

    # Check sequential times
    times_in_seconds = []
    for t_str_24 in valid_departure_times_24:
        total_minutes_val = time_to_minutes(t_str_24)
        if total_minutes_val is not None:
            times_in_seconds.append(total_minutes_val * 60)
        else:
            print(f"Warning: Failed to parse time '{t_str_24}' in trip_id '{trip_id}'.")

    for i in range(1, len(times_in_seconds)):
        if times_in_seconds[i] < times_in_seconds[i - 1]:
            print(
                f"⚠️ Non-sequential departure times in trip_id '{trip_id}' for "
                f"Route '{route_short}', Schedule '{sched_type}', Direction '{dir_id}'. "
                f"Stop {i + 1} is earlier than Stop {i}."
            )
            break

    return row_data


def process_trips_for_direction(params):
    """
    Processes trips for a specific direction_id and returns a DataFrame.
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
        print("Warning: No trips to process for this direction.")
        return pd.DataFrame()

    # Prepare indexing
    ordered_stop_ids = stops_unique['stop_id'].tolist()
    ordered_stop_sequences = stops_unique['stop_sequence'].tolist()
    stop_index_map = {seq: i for i, seq in enumerate(ordered_stop_sequences)}

    output_data = []
    group_mask = TIMEPOINTS['trip_id'].isin(trips_dir['trip_id'])
    for trip_id, group in TIMEPOINTS[group_mask].groupby('trip_id'):
        row_data = process_single_trip(trip_id, group, ordered_stop_ids,
                                       stop_index_map, route_short, sched_type,
                                       dir_id, time_fmt)
        output_data.append(row_data)

    col_names = (
        ['Route Name', 'Direction ID', 'Trip Headsign']
        + [f"{sn} Schedule" for sn in stop_names_ordered]
        + ['sort_time']
    )
    output_df = pd.DataFrame(output_data, columns=col_names)
    output_df.sort_values(by='sort_time', inplace=True)
    output_df.drop(columns=['sort_time'], inplace=True, errors='ignore')

    check_schedule_order(output_df, stop_names_ordered, route_short, sched_type, dir_id)

    # Remove empty columns
    schedule_cols = [c for c in output_df.columns if c.endswith("Schedule")]
    all_blank_cols = [c for c in schedule_cols if (output_df[c] == MISSING_TIME).all()]
    if all_blank_cols:
        output_df.drop(columns=all_blank_cols, inplace=True)
        print(
            f"Dropped empty columns for Route '{route_short}', "
            f"Schedule '{sched_type}', Direction '{dir_id}': {all_blank_cols}"
        )

    return output_df


def export_to_excel_multiple_sheets(df_dict, out_file):
    """
    Exports multiple DataFrames to an Excel file with each DataFrame in a separate sheet.
    Uses engine='openpyxl' to avoid abstract-class-instantiated warnings.
    """
    if not df_dict:
        print(f"No data to export to {out_file}.")
        return

    # Force openpyxl engine
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

                # Safely calculate max width
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
    Based on the STOP_TIMES DataFrame, subset only the rows with timepoints=1
    (if the column exists), else use all rows.
    """
    global STOP_TIMES, TIMEPOINTS

    STOP_TIMES['stop_sequence'] = pd.to_numeric(STOP_TIMES['stop_sequence'], errors='coerce')
    if STOP_TIMES['stop_sequence'].isnull().any():
        print("Warning: Some 'stop_sequence' values could not be converted to numeric.")

    if 'timepoint' in STOP_TIMES.columns:
        TIMEPOINTS = STOP_TIMES[STOP_TIMES['timepoint'] == '1']
        print("Filtered stop_times based on 'timepoint' column.")
    else:
        print("Warning: 'timepoint' column not found. Using all stops as timepoints.")
        TIMEPOINTS = STOP_TIMES.copy()


def determine_route_short_names():
    """
    Returns a list of route short names based on route_short_names_input global setting.
    """
    if isinstance(route_short_names_input, str):
        if route_short_names_input.lower() == 'all':
            return ROUTES['route_short_name'].dropna().unique().tolist()
        # If it's a string but not 'all', assume comma-separated
        return [name.strip() for name in route_short_names_input.split(',')]
    elif isinstance(route_short_names_input, list):
        # If the list contains 'all', select all
        if any(name.lower() == 'all' for name in route_short_names_input):
            return ROUTES['route_short_name'].dropna().unique().tolist()
        return route_short_names_input
    else:
        print(
            "Error: 'route_short_names_input' must be 'all', "
            "a comma-separated string, or a list of short names."
        )
        sys.exit(1)


def build_service_id_schedule_map():
    """
    Builds a dict mapping service_id -> schedule_type from CALENDAR DataFrame.
    Also returns a set of all schedule types.
    """
    service_id_schedule_map = {}
    schedule_types_set = set()
    for _, service_row_local in CALENDAR.iterrows():
        service_id_val = service_row_local['service_id']
        service_type_var = map_service_id_to_schedule(service_row_local)
        service_id_schedule_map[service_id_val] = service_type_var
        schedule_types_set.add(service_type_var)
    return service_id_schedule_map, schedule_types_set


def main():
    # 1. Load GTFS Files
    load_gtfs_files()

    # 2. Prepare Timepoints
    prepare_timepoints()

    # 3. Build schedule map
    service_id_schedule_map, schedule_types_set = build_service_id_schedule_map()
    print(f"Identified schedule types: {schedule_types_set}")

    # 4. Determine which routes to process
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
                sid for sid, stype_ in service_id_schedule_map.items()
                if stype_ == schedule_type
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
                    f"with schedule type '{schedule_type}'."
                )
                continue

            direction_ids_local = relevant_trips['direction_id'].unique()
            df_sheets = {}

            for dir_id in direction_ids_local:
                print(f"    Processing direction_id '{dir_id}'...")
                trips_direction = relevant_trips[relevant_trips['direction_id'] == dir_id]
                stop_names_ordered, stops_unique = get_ordered_stops(dir_id, relevant_trips)
                if not stop_names_ordered:
                    print(f"      No stops found for direction_id '{dir_id}'. Skipping...")
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

                df_sheets[f"Direction_{dir_id}"] = output_df

            # Export to Excel if we have data
            if df_sheets:
                schedule_type_safe = (
                    schedule_type.replace(' ', '_').replace('-', '_').replace('/', '_')
                )
                out_file = os.path.join(
                    BASE_OUTPUT_PATH,
                    f"route_{route_short_name}_schedule_{schedule_type_safe}.xlsx"
                )
                export_to_excel_multiple_sheets(df_sheets, out_file)
            else:
                print(
                    f"    No data to export for route '{route_short_name}' "
                    f"with schedule '{schedule_type}'."
                )


if __name__ == "__main__":
    main()
