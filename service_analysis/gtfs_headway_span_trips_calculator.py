"""
GTFS Headway Span Calculator

This script processes General Transit Feed Specification (GTFS) data to calculate
headways for different routes and schedules. It reads GTFS files, assigns trips to
defined time blocks, calculates headway, span, and trip counts, and exports the results
to Excel files.

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

    Args:
        base_path (str): The path to the input directory containing GTFS files.
        files (list): A list of GTFS file names to check for existence.

    Raises:
        FileNotFoundError: If the input directory or any of the required files are missing.
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

    Args:
        base_path (str): The path to the input directory containing GTFS files.
        files (list): A list of GTFS file names to load.

    Returns:
        dict: A dictionary where keys are data names (without .txt) and values are DataFrames.

    Raises:
        Exception: If there is an error loading any of the GTFS files.
    """
    data = {}
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        data_name = file_name.replace('.txt', '')
        try:
            data[data_name] = pd.read_csv(file_path)
            print(f"Loaded {file_name} with {len(data[data_name])} records.")
        except Exception as e:
            raise Exception(f"Error loading {file_name}: {e}")
    return data

def parse_time_blocks(time_blocks_str):
    """
    Convert time block definitions from string format to timedelta objects.

    Args:
        time_blocks_str (dict): A dictionary with time block names as keys and
                                tuples of start and end times as values.

    Returns:
        dict: A dictionary with time block names as keys and tuples of start and end
              times as timedelta objects.
    """
    parsed_blocks = {}
    for block_name, (start_str, end_str) in time_blocks_str.items():
        start_parts = start_str.split(':')
        end_parts = end_str.split(':')
        start_td = timedelta(
            hours=int(start_parts[0]), minutes=int(start_parts[1])
        )
        end_td = timedelta(hours=int(end_parts[0]), minutes=int(end_parts[1]))
        parsed_blocks[block_name] = (start_td, end_td)
    return parsed_blocks

def assign_time_block(time, blocks):
    """
    Assign a given time to the appropriate time block.

    Args:
        time (timedelta): The time to assign.
        blocks (dict): A dictionary with time block names as keys and tuples of
                       start and end times as timedelta objects.

    Returns:
        str: The name of the time block the time falls into, or 'other' if it doesn't fit any.
    """
    for block_name, (start, end) in blocks.items():
        if start <= time < end:
            return block_name
    return 'other'

def format_timedelta(td):
    """
    Format a timedelta object into a string 'HH:MM'.

    Args:
        td (timedelta): The timedelta to format.

    Returns:
        str or None: The formatted time string, or None if td is NaN.
    """
    if pd.isna(td):
        return None
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02}:{minutes:02}"

def find_large_break(trip_times):
    """
    Determine if there is a large break (over 3 hours) between trips during midday.

    Args:
        trip_times (pd.Series): A series of departure times as timedeltas.

    Returns:
        bool: True if a large break is found, False otherwise.
    """
    late_morning = pd.Timedelta(hours=10)
    early_afternoon = pd.Timedelta(hours=14)
    midday_trips = trip_times[
        (trip_times >= late_morning) & (trip_times <= early_afternoon)
    ]
    midday_trips = midday_trips.reset_index(drop=True)
    if len(midday_trips) < 2:
        return False
    for i in range(1, len(midday_trips)):
        if (midday_trips[i] - midday_trips[i - 1]) > pd.Timedelta(hours=3):
            return True
    return False

def calculate_trip_times(group):
    """
    Calculate first and last trip times, AM/PM trip times, and the number of trips
    based on trip schedule.

    Args:
        group (pd.DataFrame): A DataFrame group containing departure times for
        a route and direction.

    Returns:
        pd.Series: A series with calculated trip times and trip count.
    """
    trip_times = group['departure_time'].sort_values()
    first_trip = trip_times.min()
    last_trip = trip_times.max()
    trips_count = len(trip_times)  # Count the number of trips

    if first_trip >= pd.Timedelta(hours=15):
        # PM-only route
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': None,
            'pm_first_trip_time': format_timedelta(first_trip),
            'trips': trips_count  # Add trips count
        })
    elif last_trip <= pd.Timedelta(hours=10):
        # AM-only route
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': format_timedelta(last_trip),
            'pm_first_trip_time': None,
            'trips': trips_count  # Add trips count
        })
    elif find_large_break(trip_times):
        # Normal route with midday break
        am_last_trip = trip_times[trip_times < pd.Timedelta(hours=10)].max()
        pm_first_trip = trip_times[trip_times > pd.Timedelta(hours=14)].min()
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': format_timedelta(am_last_trip),
            'pm_first_trip_time': format_timedelta(pm_first_trip),
            'trips': trips_count  # Add trips count
        })
    else:
        # Normal all-day route
        return pd.Series({
            'first_trip_time': format_timedelta(first_trip),
            'last_trip_time': format_timedelta(last_trip),
            'am_last_trip_time': None,
            'pm_first_trip_time': None,
            'trips': trips_count  # Add trips count
        })

def calculate_headways(departure_times):
    """
    Calculate the mode of headways (in minutes) between consecutive departure times.

    Args:
        departure_times (pd.Series): A series of departure times as timedeltas.

    Returns:
        float or None: The most common headway in minutes, or None if no headways are found.
    """
    sorted_times = departure_times.sort_values()
    headways = sorted_times.diff().dropna().apply(lambda x: x.total_seconds() / 60)
    if headways.empty:
        return None
    return headways.mode()[0]

def process_headways(merged_data):
    """
    Process merged data to calculate headways for each route, direction, and time block.

    Args:
        merged_data (pd.DataFrame): The merged GTFS data containing departure times and other info.

    Returns:
        dict: A dictionary containing headway information categorized by schedule and time block.
    """
    headways = merged_data.groupby(
        ['route_short_name', 'route_long_name', 'direction_id', 'time_block']
    )['departure_time'].apply(calculate_headways).reset_index()
    headway_dict = {
        'weekday_am_headway': {},
        'weekday_midday_headway': {},
        'weekday_pm_headway': {},
        'weekday_night_headway': {}
    }
    for _, row in headways.iterrows():
        route = (row['route_short_name'], row['route_long_name'], row['direction_id'])
        if row['time_block'] == 'am':
            headway_dict['weekday_am_headway'][route] = row['departure_time']
        elif row['time_block'] == 'midday':
            headway_dict['weekday_midday_headway'][route] = row['departure_time']
        elif row['time_block'] == 'pm':
            headway_dict['weekday_pm_headway'][route] = row['departure_time']
        elif row['time_block'] == 'night':
            headway_dict['weekday_night_headway'][route] = row['departure_time']
    return headway_dict

def merge_headways(trip_times, headway_dict):
    """
    Merge calculated headways into the trip times DataFrame.

    Args:
        trip_times (pd.DataFrame): DataFrame containing trip times for each route and direction.
        headway_dict (dict): Dictionary containing headway information.

    Returns:
        pd.DataFrame: Updated trip_times DataFrame with headway columns added.
    """
    trip_times['weekday_am_headway'] = trip_times.apply(
        lambda row: headway_dict['weekday_am_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times['weekday_midday_headway'] = trip_times.apply(
        lambda row: headway_dict['weekday_midday_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times['weekday_pm_headway'] = trip_times.apply(
        lambda row: headway_dict['weekday_pm_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    trip_times['weekday_night_headway'] = trip_times.apply(
        lambda row: headway_dict['weekday_night_headway'].get(
            (row['route_short_name'], row['route_long_name'], row['direction_id']), None
        ),
        axis=1
    )
    return trip_times

def save_to_excel(final_data, OUTPUT_PATH, output_file):
    """
    Save the final DataFrame to an Excel file with formatted columns.

    Args:
        final_data (pd.DataFrame): The DataFrame containing all final data to save.
        OUTPUT_PATH (str): The directory path where the Excel file will be saved.
        output_file (str): The name of the Excel file.

    Returns:
        None
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Route_Schedule_Headway"

    headers = final_data.columns.tolist()
    ws.append(headers)

    for row in final_data.itertuples(index=False, name=None):
        ws.append(row)

    for col in ws.columns:
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0 for cell in col
        ) + 2
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max_length
        for cell in col:
            cell.alignment = Alignment(horizontal='center')

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_PATH, output_file)
    wb.save(output_file_path)
    print(f"Final data successfully saved to {output_file_path}")

def main():
    """
    Main function to execute the GTFS headway span calculation process.

    Steps:
        1. Check for the existence of input files.
        2. Load GTFS data into DataFrames.
        3. Parse time block configurations.
        4. Iterate over each schedule type to process trips.
            a. Filter services based on schedule type.
            b. Merge trip and route information.
            c. Assign trips to time blocks.
            d. Calculate trip times and headways.
            e. Save the results to Excel.
        5. Handle any exceptions that occur during processing.

    Returns:
        None
    """
    try:
        print("Checking input files...")
        check_input_files(GTFS_INPUT_PATH, gtfs_files)
        print("All input files are present.\n")

        print("Loading GTFS data...")
        data = load_gtfs_data(GTFS_INPUT_PATH, gtfs_files)
        print("GTFS data loaded successfully.\n")

        print("Parsing time block definitions...")
        time_blocks = parse_time_blocks(time_blocks_config)
        print("Time block definitions parsed.\n")

        # We now loop over each schedule type defined in schedule_types
        calendar = data['calendar']
        trips = data['trips']
        routes = data['routes']
        stop_times = data['stop_times']

        for schedule_type, days in schedule_types.items():
            print(f"Processing schedule: {schedule_type}")

            # Create a mask for services that run on all the specified days
            mask = pd.Series([True]*len(calendar))
            for day in days:
                mask &= (calendar[day] == 1)
            relevant_service_ids = calendar[mask]['service_id']

            if relevant_service_ids.empty:
                print(f"No services found for {schedule_type}. Skipping.\n")
                continue

            # Filter trips
            trips_filtered = trips[trips['service_id'].isin(relevant_service_ids)]
            if trips_filtered.empty:
                print(f"No trips found for {schedule_type}. Skipping.\n")
                continue

            # Merge routes and trip info
            trip_info = trips_filtered[
                ['trip_id', 'route_id', 'service_id', 'direction_id']
            ].merge(
                routes[['route_id', 'route_short_name', 'route_long_name']],
                on='route_id'
            )
            print(
                f"Merged trip information has {len(trip_info)} records for {schedule_type}.\n"
            )

            # Merge trip info with stop_times
            merged_data = stop_times[['trip_id', 'departure_time', 'stop_sequence']].merge(
                trip_info, on='trip_id'
            )
            print(
                f"Merged data has {len(merged_data)} records for {schedule_type}.\n"
            )

            # Filter to include only starting stops
            merged_data = merged_data[merged_data['stop_sequence'] == 1]
            print(
                f"Filtered starting trips count: {len(merged_data)} for {schedule_type}\n"
            )

            if merged_data.empty:
                print(
                    f"No starting trips for {schedule_type}. Skipping.\n"
                )
                continue

            # Convert departure_time to timedelta
            merged_data['departure_time'] = pd.to_timedelta(
                merged_data['departure_time'], errors='coerce'
            )
            merged_data = merged_data.dropna(subset=['departure_time'])
            if merged_data.empty:
                print(
                    f"All departure_times invalid for {schedule_type}. Skipping.\n"
                )
                continue

            print("Assigning time blocks...")
            merged_data['time_block'] = merged_data['departure_time'].apply(
                lambda x: assign_time_block(x, time_blocks)
            )
            print("Time blocks assigned.\n")

            # Filter out 'other' time blocks
            merged_data = merged_data[merged_data['time_block'] != 'other']
            print(
                f"Trips after filtering 'other' time blocks: {len(merged_data)} for {schedule_type}\n"
            )

            if merged_data.empty:
                print(
                    f"No trips left after filtering 'other' time blocks for {schedule_type}. Skipping.\n"
                )
                continue

            # Group by route and direction, calculate trip times
            print("Calculating trip times...")
            trip_times = merged_data.groupby(
                ['route_short_name', 'route_long_name', 'direction_id']
            ).apply(calculate_trip_times).reset_index()
            print("Trip times calculated.\n")

            # Calculate headways
            print("Calculating headways...")
            headway_dict = process_headways(merged_data)
            print("Headways calculated.\n")

            # Merge headways with trip times
            print("Merging headways with trip times...")
            final_data = merge_headways(trip_times, headway_dict)
            print("Headways merged.\n")

            # Save to Excel with schedule_type in filename
            output_file_for_schedule = f"{schedule_type}_{OUTPUT_EXCEL}"
            print(f"Saving data for {schedule_type} to Excel...")
            save_to_excel(final_data, OUTPUT_PATH, output_file_for_schedule)
            print(f"Data for {schedule_type} saved.\n")

        print("All schedule types processed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
