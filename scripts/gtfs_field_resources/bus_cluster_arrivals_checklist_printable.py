"""
Generates printable Excel checklists of GTFS bus arrivals, grouped by customizable stop clusters.

This module processes GTFS data to create structured Excel reports for transit operations,
supporting clustering based on either 'stop_id' or 'stop_code'. The user can configure schedules,
time windows, and stop clusters flexibly. Output includes formatted arrival and departure checklists
for each defined cluster and schedule.

Configuration Options:
- Choose to cluster stops by either 'stop_id' or 'stop_code' (controlled by STOP_IDENTIFIER_FIELD).
- Define multiple schedules with corresponding service days.
- Set specific time windows to filter trips.
- Customize stop clusters to group related stops for reporting.

Output:
- Excel files containing detailed arrival and departure schedules with placeholders
  for manual data collection during operations.
"""

import os

import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
BASE_OUTPUT_PATH = r'\\your_file_path\here\\'

# Input file paths to load GTFS files with specified dtypes
BASE_INPUT_PATH = r'\\your_file_path\here\\'

# Which field to use for clustering filters, 'stop_id' or 'stop_code'
STOP_IDENTIFIER_FIELD = 'stop_code'  # or 'stop_id'

# Define columns to read as strings
DTYPE_DICT = {
    'stop_id': str,
    'trip_id': str,
    'route_id': str,
    'service_id': str,
    # Add other ID fields as needed
}

# List of required GTFS files
GTFS_FILES = ['trips.txt', 'stop_times.txt', 'routes.txt', 'stops.txt', 'calendar.txt']

# Define clusters with stop IDs or stop_codes (depending on STOP_IDENTIFIER_FIELD)
# Format: {'Cluster Name': ['identifier1', 'identifier2', ...]}
CLUSTERS = {
    'Your Cluster 1': ['1', '2', '3'],   # If using 'stop_id', these are stop_ids
    'Your Cluster 2': ['4', '5', '6'],   # If using 'stop_code', these must be stop_codes
    'Your Cluster 3': ['7', '8', '9', '10'],
}

# Define schedule types and corresponding days in the calendar
# Format: {'Schedule Type': ['day1', 'day2', ...]}
SCHEDULE_TYPES = {
    'Weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'Saturday': ['saturday'],
    'Sunday': ['sunday'],
    # 'Friday': ['friday'],  # Uncomment if you have a unique Friday schedule
}

# Time windows for filtering trips
# Format: {'Schedule Type': {'Time Window Name': ('Start Time', 'End Time')}}
# Times should be in 'HH:MM' 24-hour format
TIME_WINDOWS = {
    'Weekday': {
        'morning': ('06:00', '09:59'),
        'afternoon': ('14:00', '17:59'),
        # 'evening': ('18:00', '21:59'),  # Add as needed
    },
    'Saturday': {
        'midday': ('10:00', '13:59'),
        # Add more time windows for Saturday if needed
    },
    # 'Sunday': {  # Uncomment and customize for Sunday if needed
    #     'morning': ('08:00', '11:59'),
    #     'afternoon': ('12:00', '15:59'),
    # },
}

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def validate_input_directory(base_input_path, gtfs_files):
    """
    Validate that the GTFS input directory and required files exist.
    """
    if not os.path.exists(base_input_path):
        raise FileNotFoundError(f"The input directory {base_input_path} does not exist.")

    for file_name in gtfs_files:
        file_path = os.path.join(base_input_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The required GTFS file {file_name} does not exist in {base_input_path}."
            )


def create_output_directory(base_output_path):
    """
    Create the output directory if it doesn't already exist.
    """
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)


def load_gtfs_data(base_input_path, dtype_dict):
    """
    Load all required GTFS files into Pandas DataFrames and return them.
    """
    trips = pd.read_csv(os.path.join(base_input_path, 'trips.txt'), dtype=dtype_dict)
    stop_times = pd.read_csv(os.path.join(base_input_path, 'stop_times.txt'), dtype=dtype_dict)
    routes = pd.read_csv(os.path.join(base_input_path, 'routes.txt'), dtype=dtype_dict)
    stops = pd.read_csv(os.path.join(base_input_path, 'stops.txt'), dtype=dtype_dict)
    calendar = pd.read_csv(os.path.join(base_input_path, 'calendar.txt'), dtype=dtype_dict)
    return trips, stop_times, routes, stops, calendar


def apply_stop_identifier_mode(stops_df, stop_identifier_field):
    """
    If user chooses 'stop_code' as STOP_IDENTIFIER_FIELD, rename the stops_df column
    'stop_code' to 'stop_id' so that downstream code can remain unchanged.
    """
    if stop_identifier_field not in ['stop_id', 'stop_code']:
        raise ValueError("STOP_IDENTIFIER_FIELD must be 'stop_id' or 'stop_code'.")

    if stop_identifier_field == 'stop_code':
        if 'stop_code' not in stops_df.columns:
            raise ValueError("No 'stop_code' column found in stops data.")
        # Overwrite stops['stop_id'] with the values from stop_code
        stops_df['stop_id'] = stops_df['stop_code']


def fix_time_format(time_str):
    """
    Convert the given time to HH:MM format, ignoring seconds if present.
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    if hours >= 24:
        hours -= 24
    return f"{hours:02}:{minutes:02}"


def export_to_excel(df, output_file):
    """
    Export the given DataFrame to an Excel file with basic formatting.
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Align all headers to the left
        for cell in worksheet[1]:
            cell.alignment = Alignment(horizontal='left')

        # Adjust the width of all columns
        for idx, col in enumerate(df.columns, 1):  # 1-based indexing for Excel columns
            column_letter = get_column_letter(idx)
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2  # Extra space
            worksheet.column_dimensions[column_letter].width = max_length


def process_cluster_data(
    cluster_data, stops_df, cluster_name, schedule_name, base_output_path, time_windows=None
):
    """
    Perform transformations on the cluster data (fix times, add placeholders) and export
    both the full data set and any time-window-specific subsets to Excel files.
    """
    # Fix arrival and departure times
    cluster_data['arrival_time'] = cluster_data['arrival_time'].apply(fix_time_format)
    cluster_data['departure_time'] = cluster_data['departure_time'].apply(fix_time_format)

    # Convert columns to string in HH:MM format
    cluster_data['arrival_time'] = cluster_data['arrival_time'].astype(str)
    cluster_data['departure_time'] = cluster_data['departure_time'].astype(str)

    # Sort by arrival_time using a temporary datetime conversion
    cluster_data['arrival_sort'] = pd.to_datetime(cluster_data['arrival_time'], format='%H:%M')
    cluster_data = cluster_data.sort_values(by='arrival_sort').drop(columns='arrival_sort')

    # Insert placeholder columns
    cluster_data.insert(
        cluster_data.columns.get_loc('arrival_time') + 1,
        'act_arrival',
        '________'
    )
    cluster_data.insert(
        cluster_data.columns.get_loc('departure_time') + 1,
        'act_departure',
        '________'
    )
    cluster_data.insert(
        cluster_data.columns.get_loc('block_id') + 1,
        'act_block',
        '________'
    )

    # Modify placeholders for first/last stop in each trip
    cluster_data.loc[cluster_data['sequence_long'] == 'start', 'act_arrival'] = '__XXXX__'
    cluster_data.loc[cluster_data['sequence_long'] == 'last', 'act_departure'] = '__XXXX__'

    # Add bus_number and comments columns
    cluster_data['bus_number'] = '________'
    cluster_data['comments'] = '________________'

    # Merge with stop names
    cluster_data = pd.merge(
        cluster_data,
        stops_df[['stop_id', 'stop_name']],
        on='stop_id',
        how='left'
    )

    # Reorder columns
    first_columns = [
        'route_short_name', 'trip_headsign', 'stop_sequence', 'sequence_long',
        'stop_id', 'stop_name', 'arrival_time', 'act_arrival',
        'departure_time', 'act_departure', 'block_id', 'act_block',
        'bus_number', 'comments'
    ]
    other_columns = [col for col in cluster_data.columns if col not in first_columns]
    cluster_data = cluster_data[first_columns + other_columns]

    # Drop unnecessary columns
    cluster_data = cluster_data.drop(
        columns=[
            'shape_dist_traveled', 'shape_id', 'route_id', 'service_id',
            'trip_id', 'timepoint', 'direction_id', 'stop_headsign',
            'pickup_type', 'drop_off_type', 'wheelchair_accessible',
            'bikes_allowed', 'trip_short_name', 'stop_code'
        ],
        errors='ignore'
    )

    # Export full cluster data
    output_file_name = f'{cluster_name}_{schedule_name}_data.xlsx'
    output_file = os.path.join(base_output_path, output_file_name)
    export_to_excel(cluster_data, output_file)

    print(f"Processed and exported data for {cluster_name} on {schedule_name} schedule.")

    # Process time windows if applicable
    if time_windows and schedule_name in time_windows:
        for time_window_name, time_range in time_windows[schedule_name].items():
            start_time_str, end_time_str = time_range

            # Parse the start and end times
            start_dt = pd.to_datetime(start_time_str, format='%H:%M').time()
            end_dt = pd.to_datetime(end_time_str, format='%H:%M').time()

            # Convert arrival_time strings (HH:MM) to datetime.time for filtering
            arrival_times = pd.to_datetime(cluster_data['arrival_time'], format='%H:%M').dt.time
            filtered_data = cluster_data[(arrival_times >= start_dt) & (arrival_times <= end_dt)]

            if filtered_data.empty:
                print(
                    f"No data found for {cluster_name} on {schedule_name} schedule in "
                    f"{time_window_name} time window. Skipping."
                )
                continue

            # Export filtered data to Excel
            output_file_name = f'{cluster_name}_{schedule_name}_{time_window_name}_data.xlsx'
            output_file = os.path.join(base_output_path, output_file_name)
            export_to_excel(filtered_data, output_file)

            print(
                f"Processed and exported data for {cluster_name} on {schedule_name} schedule "
                f"in {time_window_name} time window."
            )


def generate_gtfs_checklists():
    """
    Generates GTFS checklists in Excel format by schedule and cluster.
    Allows for filtering by either stop_id or stop_code, based on STOP_IDENTIFIER_FIELD.
    """
    # 1) Validate input directory
    validate_input_directory(BASE_INPUT_PATH, GTFS_FILES)

    # 2) Create output directory
    create_output_directory(BASE_OUTPUT_PATH)

    # 3) Load GTFS data
    trips, stop_times, routes, stops, calendar = load_gtfs_data(BASE_INPUT_PATH, DTYPE_DICT)

    # 4) Potentially replace stop_id with stop_code
    apply_stop_identifier_mode(stops, STOP_IDENTIFIER_FIELD)

    # Ensure stop_id is a string in stops
    stops['stop_id'] = stops['stop_id'].astype(str)

    # 5) Process each schedule type
    for schedule_name, days in SCHEDULE_TYPES.items():
        print(f"Processing schedule: {schedule_name}")

        # Filter calendar by days
        service_mask = calendar[days].astype(bool).all(axis=1)
        relevant_service_ids = calendar.loc[service_mask, 'service_id']

        # Filter trips by relevant service IDs
        trips_filtered = trips[trips['service_id'].isin(relevant_service_ids)]

        if trips_filtered.empty:
            print(f"No trips found for {schedule_name} schedule. Skipping.")
            continue

        # Merge with stop_times and routes
        merged_data = pd.merge(stop_times, trips_filtered, on='trip_id')
        merged_data = pd.merge(
            merged_data,
            routes[['route_id', 'route_short_name']],
            on='route_id'
        )

        # Ensure stop_id is string in merged_data
        merged_data['stop_id'] = merged_data['stop_id'].astype(str)

        # Create sequence_long column
        merged_data['sequence_long'] = 'middle'
        merged_data.loc[merged_data['stop_sequence'] == 1, 'sequence_long'] = 'start'
        max_sequence = merged_data.groupby('trip_id')['stop_sequence'].transform('max')
        merged_data.loc[merged_data['stop_sequence'] == max_sequence, 'sequence_long'] = 'last'

        # 6) Process each cluster
        for cluster_name, cluster_stop_ids in CLUSTERS.items():
            print(f"Processing cluster: {cluster_name} for {schedule_name} schedule")

            # Ensure cluster_stop_ids are strings
            cluster_stop_ids = [str(sid) for sid in cluster_stop_ids]

            # Filter merged data by cluster's stops
            cluster_data = merged_data[merged_data['stop_id'].isin(cluster_stop_ids)]

            if cluster_data.empty:
                print(f"No data found for {cluster_name} on {schedule_name} schedule. Skipping.")
                continue

            # Transform and export data
            process_cluster_data(
                cluster_data,
                stops,
                cluster_name,
                schedule_name,
                BASE_OUTPUT_PATH,
                time_windows=TIME_WINDOWS
            )

    print("All clusters and schedules have been processed and exported.")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main entry point of the script.
    """
    generate_gtfs_checklists()


if __name__ == '__main__':
    main()
