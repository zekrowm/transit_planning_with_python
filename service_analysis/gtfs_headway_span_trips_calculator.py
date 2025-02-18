"""
GTFS Headway Span Calculator

This script processes General Transit Feed Specification (GTFS) data to calculate
headways for different routes and schedules. It also checks the GTFS block_id to
determine interlined routes and lists them in an additional column.

Configuration:
    - Define input and output paths.
    - Specify GTFS files to load.
    - Configure time blocks and schedule types.

Usage:
    Ensure all GTFS files are placed in the input directory and run the script.
    The output Excel files will be saved in the specified output directory.
"""

import os
from datetime import timedelta

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# ==============================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ==============================

# Input directory containing GTFS files
GTFS_INPUT_PATH = r'\\your_file_path\here\\'

# Output directory for the Excel file
OUTPUT_PATH = r'\\your_file_path\here\\'

# GTFS files to load
gtfs_files = [
    'routes.txt',
    'trips.txt',
    'stop_times.txt',
    'calendar.txt',
    'calendar_dates.txt'
]

# Output Excel file name (base name)
OUTPUT_EXCEL = "route_schedule_headway_with_modes.xlsx"

# Define time blocks with start and end times in 'HH:MM' format
time_blocks_config = {
    'am': ('04:00', '09:00'),
    'midday': ('09:00', '15:00'),
    'pm': ('15:00', '21:00'),
    'night': ('21:00', '28:00')  # 28:00 is equivalent to 04:00 the next day
}

# Define multiple schedule types and their corresponding days
schedule_types = {
    'Weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'Saturday': ['saturday'],
    'Sunday': ['sunday'],
    # Add more if desired
}

# ==============================
# END OF CONFIGURATION SECTION
# ==============================


def check_input_files(base_path, files):
    """
    Verify that the input directory and all required GTFS files exist.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The input directory {base_path} does not exist.")
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The required GTFS file {file_name} does not exist in {base_path}."
            )


def load_gtfs_data(base_path, files):
    """
    Load GTFS data from specified files into a dictionary of pandas DataFrames.
    """
    data = {}
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        data_name = file_name.replace('.txt', '')
        try:
            data[data_name] = pd.read_csv(file_path)
            print(f"Loaded {file_name} with {len(data[data_name])} records.")
        except Exception as error:
            # Raise from the original error to preserve traceback
            raise Exception(f"Error loading {file_name}: {error}") from error
    return data


def parse_time_blocks(time_blocks_str):
    """
    Convert time block definitions from string format to timedelta objects.
    """
    parsed_blocks = {}
    for block_name, (start_str, end_str) in time_blocks_str.items():
        start_parts = start_str.split(':')
        end_parts = end_str.split(':')
        start_time_delta = timedelta(hours=int(start_parts[0]), minutes=int(start_parts[1]))
        end_time_delta = timedelta(hours=int(end_parts[0]), minutes=int(end_parts[1]))
        parsed_blocks[block_name] = (start_time_delta, end_time_delta)
    return parsed_blocks


def assign_time_block(time_delta, blocks):
    """
    Assign a given time to the appropriate time block.
    """
    for block_name, (start, end) in blocks.items():
        if start <= time_delta < end:
            return block_name
    return 'other'


def format_timedelta(time_delta):
    """
    Format a timedelta object into a string 'HH:MM'.
    """
    if pd.isna(time_delta):
        return None
    total_seconds = int(time_delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"


def find_large_break(trip_times):
    """
    Determine if there is a large break (over 3 hours) between trips during midday.
    """
    late_morning = pd.Timedelta(hours=10)
    early_afternoon = pd.Timedelta(hours=14)
    midday_trips = trip_times[(trip_times >= late_morning) & (trip_times <= early_afternoon)]
    midday_trips = midday_trips.reset_index(drop=True)
    if len(midday_trips) < 2:
        return False
    for i in range(1, len(midday_trips)):
        if (midday_trips[i] - midday_trips[i - 1]) > pd.Timedelta(hours=3):
            return True
    return False


def calculate_trip_times(group):
    """
    Calculate first/last trip times, AM/PM trip times, and the number of trips
    based on trip schedule.
    """
    trip_times = group['departure_time'].sort_values()
    first_trip = trip_times.min()
    last_trip = trip_times.max()
    trips_count = len(trip_times)

    # Use sequential if-statements (pylint R1705)
    if first_trip >= pd.Timedelta(hours=15):
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': None,
            'pm_first_trip_time': format_timedelta(first_trip),
            'trips': trips_count
        })

    if last_trip <= pd.Timedelta(hours=10):
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': format_timedelta(last_trip),
            'pm_first_trip_time': None,
            'trips': trips_count
        })

    if find_large_break(trip_times):
        am_last_trip = trip_times[trip_times < pd.Timedelta(hours=10)].max()
        pm_first_trip = trip_times[trip_times > pd.Timedelta(hours=14)].min()
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': format_timedelta(am_last_trip),
            'pm_first_trip_time': format_timedelta(pm_first_trip),
            'trips': trips_count
        })

    return pd.Series({
        'first_trip_time': format_timedelta(first_trip),
        'last_trip_time': format_timedelta(last_trip),
        'am_last_trip_time': None,
        'pm_first_trip_time': None,
        'trips': trips_count
    })


def calculate_headways(departure_times):
    """
    Calculate the mode of headways (in minutes) between consecutive departure times.
    """
    sorted_times = departure_times.sort_values()
    headways = sorted_times.diff().dropna().apply(lambda x: x.total_seconds() / 60)
    if headways.empty:
        return None
    return headways.mode()[0]


def process_headways(merged_data):
    """
    Process merged data to calculate headways for each route, direction, and time block.
    """
    headways = (
        merged_data
        .groupby(['route_short_name', 'route_long_name', 'direction_id', 'time_block'])['departure_time']
        .apply(calculate_headways)
        .reset_index()
    )

    headway_dict = {
        'weekday_am_headway': {},
        'weekday_midday_headway': {},
        'weekday_pm_headway': {},
        'weekday_night_headway': {}
    }

    for _, row in headways.iterrows():
        route_key = (row['route_short_name'], row['route_long_name'], row['direction_id'])
        block = row['time_block']
        hw_value = row['departure_time']

        if block == 'am':
            headway_dict['weekday_am_headway'][route_key] = hw_value
        elif block == 'midday':
            headway_dict['weekday_midday_headway'][route_key] = hw_value
        elif block == 'pm':
            headway_dict['weekday_pm_headway'][route_key] = hw_value
        elif block == 'night':
            headway_dict['weekday_night_headway'][route_key] = hw_value

    return headway_dict


def merge_headways(trip_times_df, headway_dict):
    """
    Merge calculated headways into the trip times DataFrame.
    """
    trip_times_df['weekday_am_headway'] = trip_times_df.apply(
        lambda row: headway_dict['weekday_am_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times_df['weekday_midday_headway'] = trip_times_df.apply(
        lambda row: headway_dict['weekday_midday_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times_df['weekday_pm_headway'] = trip_times_df.apply(
        lambda row: headway_dict['weekday_pm_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times_df['weekday_night_headway'] = trip_times_df.apply(
        lambda row: headway_dict['weekday_night_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    return trip_times_df


def save_to_excel(final_data, output_dir, output_file):
    """
    Save the final DataFrame to an Excel file with formatted columns.
    """
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Route_Schedule_Headway"

    headers = final_data.columns.tolist()
    worksheet.append(headers)

    for row in final_data.itertuples(index=False, name=None):
        worksheet.append(row)

    for col in worksheet.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col) + 2
        col_letter = get_column_letter(col[0].column)
        worksheet.column_dimensions[col_letter].width = max_length
        for cell in col:
            cell.alignment = Alignment(horizontal='center')

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_file)
    workbook.save(output_file_path)
    print(f"Final data successfully saved to {output_file_path}")


def process_schedule_type(schedule_type, days, data):
    """
    Process a single schedule type (e.g., Weekday, Saturday, Sunday).
    """
    calendar_df = data['calendar']
    trips_df = data['trips']
    routes_df = data['routes']
    stop_times_df = data['stop_times']

    print(f"Processing schedule: {schedule_type}")

    # 1. Create a mask for services that run on all the specified days
    mask = pd.Series([True] * len(calendar_df))
    for day in days:
        mask &= (calendar_df[day] == 1)
    relevant_service_ids = calendar_df[mask]['service_id']

    if relevant_service_ids.empty:
        print(f"No services found for {schedule_type}. Skipping.\n")
        return

    # 2. Filter trips
    trips_filtered = trips_df[trips_df['service_id'].isin(relevant_service_ids)]
    if trips_filtered.empty:
        print(f"No trips found for {schedule_type}. Skipping.\n")
        return

    # 3. Merge routes and trip info
    trip_info = (
        trips_filtered[['trip_id', 'route_id', 'service_id', 'direction_id', 'block_id']]
        .merge(
            routes_df[['route_id', 'route_short_name', 'route_long_name']],
            on='route_id',
            how='left'
        )
    )
    print(
        "Merged trip information has "
        f"{len(trip_info)} records for {schedule_type}.\n"
    )

    # 4. Merge trip info with stop_times
    merged_data = stop_times_df[['trip_id', 'departure_time', 'stop_sequence']].merge(
        trip_info, on='trip_id'
    )
    print(
        "Merged data has "
        f"{len(merged_data)} records for {schedule_type}.\n"
    )

    # 5. Filter to only starting stops
    merged_data = merged_data[merged_data['stop_sequence'] == 1]
    print(
        f"Filtered starting trips count: {len(merged_data)} for {schedule_type}\n"
    )

    if merged_data.empty:
        print(f"No starting trips for {schedule_type}. Skipping.\n")
        return

    # 6. Convert departure_time to timedelta and drop invalid
    merged_data['departure_time'] = pd.to_timedelta(
        merged_data['departure_time'], errors='coerce'
    )
    merged_data = merged_data.dropna(subset=['departure_time'])
    if merged_data.empty:
        print(f"All departure_times invalid for {schedule_type}. Skipping.\n")
        return

    # 7. Assign time blocks
    time_blocks = parse_time_blocks(time_blocks_config)
    merged_data['time_block'] = merged_data['departure_time'].apply(
        lambda x: assign_time_block(x, time_blocks)
    )
    merged_data = merged_data[merged_data['time_block'] != 'other']
    print(
        f"Trips after filtering 'other' time blocks: {len(merged_data)} "
        f"for {schedule_type}\n"
    )

    if merged_data.empty:
        print(f"No trips left after filtering 'other' time blocks for {schedule_type}. Skipping.\n")
        return

    # -------------------------------------------------------------------------
    #         NEW SECTION: Determine Interlined Routes Using block_id
    # -------------------------------------------------------------------------
    # block_id -> set of route_short_names
    block_to_routes = (
        trip_info
        .groupby('block_id')['route_short_name']
        .apply(lambda routes: set(routes.dropna()))
        .to_dict()
    )

    # route_short_name -> set of other short names it interlines with
    interlined_routes_map = {}
    for block_id, route_set in block_to_routes.items():
        for rt in route_set:
            interlined_routes_map.setdefault(rt, set()).update(route_set - {rt})

    # -------------------------------------------------------------------------
    # 8. Group by route and direction, calculate trip times
    print("Calculating trip times...")
    trip_times = (
        merged_data
        .groupby(['route_short_name', 'route_long_name', 'direction_id'])
        .apply(calculate_trip_times)
        .reset_index()
    )
    print("Trip times calculated.\n")

    # 9. Calculate headways
    print("Calculating headways...")
    headway_dict = process_headways(merged_data)
    print("Headways calculated.\n")

    # 10. Merge headways with trip times
    print("Merging headways with trip times...")
    final_data = merge_headways(trip_times, headway_dict)
    print("Headways merged.\n")

    # 11. Add interlined_routes column
    ## For each row, look up which route_short_name it has,
    ## and pull all routes it is interlined with. Join them in a string.
    final_data['interlined_routes'] = final_data['route_short_name'].apply(
        lambda rt: ", ".join(sorted(interlined_routes_map.get(rt, [])))
    )

    # 12. Save to Excel with schedule_type in filename
    output_file_for_schedule = f"{schedule_type}_{OUTPUT_EXCEL}"
    print(f"Saving data for {schedule_type} to Excel...")
    save_to_excel(final_data, OUTPUT_PATH, output_file_for_schedule)
    print(f"Data for {schedule_type} saved.\n")


def main():
    """
    Main function to execute the GTFS headway span calculation and interlining check.
    """
    try:
        print("Checking input files...")
        check_input_files(GTFS_INPUT_PATH, gtfs_files)
        print("All input files are present.\n")

        print("Loading GTFS data...")
        data = load_gtfs_data(GTFS_INPUT_PATH, gtfs_files)
        print("GTFS data loaded successfully.\n")

        # Process each schedule type separately
        for schedule_type, days in schedule_types.items():
            process_schedule_type(schedule_type, days, data)

        print("All schedule types processed successfully!")

    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
