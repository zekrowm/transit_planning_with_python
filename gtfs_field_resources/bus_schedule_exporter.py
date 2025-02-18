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

Usage:
------
1. **Configure the Settings**: Update the configuration section with appropriate
   input and output paths, route selections, and time format.
2. **Run the Script**: Execute the script using a Python interpreter.
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

# Input file paths for GTFS files (update these paths accordingly)
BASE_INPUT_PATH = r"C:\Path\To\Your\System\GTFS_Data"  # Replace with your file path
trips_file = os.path.join(BASE_INPUT_PATH, "trips.txt")
stop_times_file = os.path.join(BASE_INPUT_PATH, "stop_times.txt")
routes_file = os.path.join(BASE_INPUT_PATH, "routes.txt")
stops_file = os.path.join(BASE_INPUT_PATH, "stops.txt")
calendar_file = os.path.join(BASE_INPUT_PATH, "calendar.txt")

# Output directory (update this path accordingly)
BASE_OUTPUT_PATH = r"C:\Path\To\Your\Output_Folder"  # Replace with your file path
if not os.path.exists(BASE_OUTPUT_PATH):
    os.makedirs(BASE_OUTPUT_PATH)

# List of route short names to process
# Set to 'all' (string) to include all routes,
# Or provide a list like ['101', '102', '103']
route_short_names_input = ['101', '102']  # Modify as needed

# Time format option: '24' for 24-hour time, '12' for 12-hour time
TIME_FORMAT_OPTION = '12'  # Change to '24' for 24-hour format

# Placeholder values
MISSING_TIME = "---"

# Maximum column width for Excel output (used to wrap long headers)
MAX_COLUMN_WIDTH = 30  # Adjust as needed

# ==============================
# END OF CONFIGURATION SECTION
# ==============================

# ==============================
# UTILITY FUNCTIONS
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
        # Allow hours >=24 by not constraining the hour value
        return hour * 60 + minute
    except Exception as err:  # Catch broad exception, renamed 'e' -> 'err'
        return None

def remove_empty_schedule_columns(input_df):
    """
    Drops any columns in input_df (Schedule columns) that are entirely '---'.
    """
    schedule_cols = [
        col for col in input_df.columns if col.endswith("Schedule")
    ]
    all_blank_cols = [
        col for col in schedule_cols
        if (input_df[col] == '---').all()
    ]
    input_df.drop(columns=all_blank_cols, inplace=True)
    return input_df

def check_schedule_order(
    input_df,
    ordered_stop_names,
    route_short_name,
    schedule_type,
    dir_id
):
    """
    Checks that times in the DataFrame increase across rows and down columns,
    ignoring MISSING_TIME or '---'. Prints warnings if violations are found.
    """
    violations = False

    # Row-wise check: times increase left-to-right in each trip
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
                break  # Warn once per trip
            last_time = current_time

    # Column-wise check: times increase top-to-bottom at each stop
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
                break  # Warn once per stop
            last_time = current_time

    if not violations:
        print("✅ Schedule order check passed.")

def adjust_time(time_str, time_format='24'):
    """
    Adjusts time strings to the desired format.
    If '24', keeps hours as-is without wrapping.
    If '12', converts to 12-hour format with AM/PM, unless hours >=24.
    Returns None if the format is invalid.
    """
    parts = time_str.strip().split(":")
    if len(parts) >= 2:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            if time_format == '12':
                if hours >= 24:
                    print(
                        f"Warning: Cannot convert time '{time_str}' "
                        "to 12-hour format. Keeping as is."
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
    Retrieves and orders the stops for the given dir_id from 'relevant_trips_dir'.
    Returns a tuple of (ordered_stop_names, unique_stops DataFrame).
    """
    relevant_dir = relevant_trips_dir[relevant_trips_dir['direction_id'] == dir_id]
    if relevant_dir.empty:
        print(f"Warning: No trips found for direction_id '{dir_id}'.")
        return [], []

    all_stops = timepoints[timepoints['trip_id'].isin(relevant_dir['trip_id'])]
    all_stops = all_stops.sort_values(['trip_id', 'stop_sequence'])
    if all_stops.empty:
        print(f"Warning: No stop times found for direction_id '{dir_id}'.")
        return [], []

    unique_stops = (
        all_stops[['stop_id', 'stop_sequence']]
        .drop_duplicates()
        .sort_values('stop_sequence')
    )
    stop_names = stops.set_index('stop_id')['stop_name']
    ordered_stop_names = [
        f"{stop_names.get(s_id, f'Unknown Stop ID {s_id}')} ({seq})"
        for s_id, seq in zip(
            unique_stops['stop_id'],
            unique_stops['stop_sequence']
        )
    ]
    return ordered_stop_names, unique_stops

def map_service_id_to_schedule(service_row_local):
    """
    Maps a service_id row to a schedule type based on days served.
    Includes 'Weekday except Friday'.
    """
    days = [
        'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday'
    ]
    served_days = [
        day for day in days if service_row_local.get(day, '0') == '1'
    ]

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

def process_trips_for_direction(params):
    """
    Processes trips for a specific direction_id and returns a DataFrame.
    Each stop occurrence is preserved, ensuring repeated visits to the same stop.
    Sorts trips by the latest departure time in 24-hour format.
    Checks for sequential departure times and prints warnings if inconsistencies.
    """

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

    # Build an index map for quick column lookups
    ordered_stop_ids = stops_unique['stop_id'].tolist()
    ordered_stop_sequences = stops_unique['stop_sequence'].tolist()
    stop_index_map = {
        seq: i for i, seq in enumerate(ordered_stop_sequences)
    }

    output_data = []
    group_mask = timepoints['trip_id'].isin(trips_dir['trip_id'])
    for trip_id, group in timepoints[group_mask].groupby('trip_id'):
        trip_info = trips_dir[trips_dir['trip_id'] == trip_id].iloc[0]
        route_name_val = routes[
            routes['route_id'] == trip_info['route_id']
        ]['route_short_name'].values[0]
        trip_headsign = trip_info.get('trip_headsign', '')
        row_data = [route_name_val, trip_info['direction_id'], trip_headsign]

        # Fill schedule times with placeholders
        schedule_times = [MISSING_TIME] * len(ordered_stop_ids)
        valid_departure_times_24 = []

        for _, stop in group.iterrows():
            departure_str = stop['departure_time'].strip()
            time_str_display = adjust_time(departure_str, time_fmt)
            time_str_24 = adjust_time(departure_str, '24')
            if time_str_display is None or time_str_24 is None:
                print(
                    f"Warning: Invalid time format '{stop['departure_time']}' "
                    f"in trip_id '{trip_id}' at stop_id '{stop['stop_id']}'"
                )
                continue
            seq_val = stop['stop_sequence']
            index_val = stop_index_map[seq_val]
            schedule_times[index_val] = time_str_display
            valid_departure_times_24.append(time_str_24)

        # Determine the sorting time from the maximum departure time
        if valid_departure_times_24:
            try:
                departure_timedeltas = [
                    pd.to_timedelta(f"{t_24}:00") for t_24 in valid_departure_times_24
                ]
                max_sort_time = max(departure_timedeltas)
            except Exception as err:
                max_sort_time = pd.to_timedelta('00:00')
                print(
                    f"Warning: Failed to determine maximum departure time "
                    f"for trip_id '{trip_id}'. Defaulting to '00:00'. Error: {err}"
                )
        else:
            # If no valid times, push trip to bottom
            max_sort_time = pd.to_timedelta('9999:00:00')

        row_data.extend(schedule_times)
        row_data.append(max_sort_time)
        output_data.append(row_data)

        # Check sequential times within the trip
        times_in_seconds = []
        for t_str_24 in valid_departure_times_24:
            total_minutes_val = time_to_minutes(t_str_24)
            if total_minutes_val is not None:
                times_in_seconds.append(total_minutes_val * 60)
            else:
                print(
                    f"Warning: Failed to parse time '{t_str_24}' "
                    f"in trip_id '{trip_id}'."
                )
        for i in range(1, len(times_in_seconds)):
            if times_in_seconds[i] < times_in_seconds[i - 1]:
                print(
                    f"⚠️ Non-sequential departure times in trip_id '{trip_id}' for "
                    f"Route '{route_short}', Schedule '{sched_type}', Direction '{dir_id}'. "
                    f"Stop {i + 1} is earlier than Stop {i}."
                )
                break

    # Build DataFrame
    col_names = (
        ['Route Name', 'Direction ID', 'Trip Headsign']
        + [f"{sn} Schedule" for sn in stop_names_ordered]
        + ['sort_time']
    )
    output_df = pd.DataFrame(output_data, columns=col_names)
    output_df = output_df.sort_values(by='sort_time').drop(columns=['sort_time'])

    # Perform schedule order checks
    check_schedule_order(
        output_df,
        stop_names_ordered,
        route_short,
        sched_type,
        dir_id
    )

    # Drop columns that are entirely placeholders
    schedule_cols = [c for c in output_df.columns if c.endswith("Schedule")]
    all_blank_cols = [
        c for c in schedule_cols if (output_df[c] == MISSING_TIME).all()
    ]
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
    """
    if not df_dict:
        print(f"No data to export to {out_file}.")
        return

    # Using engine='openpyxl' to avoid abstract-class-instantiated warning
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        for sheet_name, input_df in df_dict.items():
            if input_df.empty:
                print(f"No data for sheet '{sheet_name}'. Skipping...")
                continue
            input_df.to_excel(writer, index=False, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]

            # Apply alignment to headers and adjust column widths
            for col_num, _ in enumerate(input_df.columns, 1):
                col_letter = get_column_letter(col_num)
                header_cell = worksheet[f'{col_letter}1']
                header_cell.alignment = Alignment(
                    horizontal='left', 
                    vertical='top', 
                    wrap_text=True
                )
                for row_num in range(2, worksheet.max_row + 1):
                    cell = worksheet[f'{col_letter}{row_num}']
                    cell.alignment = Alignment(horizontal='left')

                # Calculate max width for this column
                column_cells = worksheet[col_letter]
                try:
                    max_length = max(
                        len(str(cell.value))
                        for cell in column_cells if cell.value is not None
                    )
                except Exception as err:
                    max_length = 10
                    print(
                        f"Warning: Failed to calculate max_length. Error: {err}"
                    )
                adjusted_width = min(max_length + 2, MAX_COLUMN_WIDTH)
                worksheet.column_dimensions[col_letter].width = adjusted_width

    print(f"Data exported to {out_file}")

# ==============================
# MAIN SCRIPT LOGIC
# ==============================

try:
    trips = pd.read_csv(trips_file, dtype=str)
    stop_times = pd.read_csv(stop_times_file, dtype=str)
    routes = pd.read_csv(routes_file, dtype=str)
    stops = pd.read_csv(stops_file, dtype=str)
    calendar = pd.read_csv(calendar_file, dtype=str)
    print("Successfully loaded all GTFS files.")
except FileNotFoundError as err:
    print(f"Error: {err}")
    print("Please check your input file paths in the configuration section.")
    sys.exit(1)
except Exception as err:
    print(f"An unexpected error occurred while reading GTFS files: {err}")
    sys.exit(1)

stop_times['stop_sequence'] = pd.to_numeric(
    stop_times['stop_sequence'],
    errors='coerce'
)
if stop_times['stop_sequence'].isnull().any():
    print("Warning: Some 'stop_sequence' values could not be converted to numeric.")

if isinstance(route_short_names_input, str):
    if route_short_names_input.lower() == 'all':
        route_short_names = routes['route_short_name'].dropna().unique().tolist()
        print(f"Selected all routes: {route_short_names}")
    else:
        route_short_names = [
            name.strip() for name in route_short_names_input.split(',')
        ]
        print(f"Selected routes: {route_short_names}")
elif isinstance(route_short_names_input, list):
    if 'all' in [name.lower() for name in route_short_names_input]:
        route_short_names = routes['route_short_name'].dropna().unique().tolist()
        print(f"Selected all routes: {route_short_names}")
    else:
        route_short_names = route_short_names_input
        print(f"Selected routes: {route_short_names}")
else:
    print(
        "Error: 'route_short_names_input' must be either 'all', "
        "a comma-separated string, or a list of route short names."
    )
    sys.exit(1)

if 'timepoint' in stop_times.columns:
    timepoints = stop_times[stop_times['timepoint'] == '1']
    print("Filtered stop_times based on 'timepoint' column.")
else:
    print("Warning: 'timepoint' column not found. Using all stops as timepoints.")
    timepoints = stop_times.copy()

service_id_schedule_map = {}
schedule_types_set = set()
for _, service_row_local in calendar.iterrows():
    service_id_val = service_row_local['service_id']
    service_type_var = map_service_id_to_schedule(service_row_local)
    service_id_schedule_map[service_id_val] = service_type_var
    schedule_types_set.add(service_type_var)

print(f"Identified schedule types: {schedule_types_set}")

for route_short_name in route_short_names:
    print(f"\nProcessing route '{route_short_name}'...")

    route_ids = routes[routes['route_short_name'] == route_short_name]['route_id']
    if route_ids.empty:
        print(f"Error: Route '{route_short_name}' not found in routes.txt.")
        continue

    for schedule_type in schedule_types_set:
        print(f"  Processing schedule type '{schedule_type}'...")
        relevant_service_ids = [
            sid for sid, stype_ in service_id_schedule_map.items()
            if stype_ == schedule_type
        ]
        if not relevant_service_ids:
            print(f"    No services found for schedule type '{schedule_type}'.")
            continue

        relevant_trips = trips[
            (trips['route_id'].isin(route_ids))
            & (trips['service_id'].isin(relevant_service_ids))
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
                print(
                    f"      No stops found for direction_id '{dir_id}'. Skipping..."
                )
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

            sheet_name_var = f"Direction_{dir_id}"
            df_sheets[sheet_name_var] = output_df

        if not df_sheets:
            print(
                f"    No data to export for route '{route_short_name}' "
                f"with schedule '{schedule_type}'."
            )
            continue

        schedule_type_safe = (schedule_type
                              .replace(' ', '_')
                              .replace('-', '_')
                              .replace('/', '_'))
        output_file_path = os.path.join(
            BASE_OUTPUT_PATH,
            f"route_{route_short_name}_schedule_{schedule_type_safe}.xlsx"
        )
        export_to_excel_multiple_sheets(df_sheets, output_file_path)
