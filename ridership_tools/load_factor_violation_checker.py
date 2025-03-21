"""
Script to process bus data and determine whether maximum load
standards are being observed or violated.
"""

import pandas as pd
from openpyxl.utils import get_column_letter

# =======================
# Configuration Section
# =======================
INPUT_FILE = r"\\File\Path\To\Your\STATISTICS_BY_ROUTE_AND_TRIP.XLSX"
OUTPUT_FILE = INPUT_FILE.replace(".XLSX", "_processed.xlsx")
BUS_CAPACITY = 39

# Routes in this list use a load factor limit of 1.25.
# Any route not in LOWER_LIMIT_ROUTES will default to 1.25.
HIGHER_LIMIT_ROUTES = [
    '101', '102', '103', '104'
]

# Routes in this list use a load factor limit of 1.0.
# Many agencies have a lower load factor for express routes.
LOWER_LIMIT_ROUTES = [
    '105', '106'
]

LOWER_LOAD_FACTOR_LIMIT = 1.0
HIGHER_LOAD_FACTOR_LIMIT = 1.25

# =======================
# Helper Functions
# =======================

def load_data(input_file: str) -> pd.DataFrame:
    """Load the Excel file and select the relevant columns."""
    data_frame = pd.read_excel(input_file)
    selected_columns = [
        'SERIAL_NUMBER', 'ROUTE_NAME', 'DIRECTION_NAME', 'TRIP_START_TIME',
        'BLOCK', 'MAX_LOAD', 'MAX_LOAD_P', 'ALL_RECORDS_MAX_LOAD'
    ]
    return data_frame[selected_columns]


def assign_service_period(ts):
    """ts is assumed to be a datetime or time object."""
    hour = ts.hour
    if 4 <= hour < 6:
        return 'AM Early'
    elif 6 <= hour < 9:
        return 'AM Peak'
    elif 9 <= hour < 15:
        return 'Midday'
    elif 15 <= hour < 18:
        return 'PM Peak'
    elif 18 <= hour < 21:
        return 'PM Late'
    elif 21 <= hour < 24:
        return 'PM Nite'
    else:
        return 'Other'


def get_route_load_limit(route_name: str) -> float:
    """
    Get the appropriate load factor limit based on the route.

    If the route is in LOWER_LIMIT_ROUTES, returns the lower limit;
    otherwise, returns the higher limit.
    """
    if route_name in LOWER_LIMIT_ROUTES:
        return LOWER_LOAD_FACTOR_LIMIT
    return HIGHER_LOAD_FACTOR_LIMIT


def check_load_factor_violation(row: pd.Series) -> str:
    """Determine if the row exceeds the route's load factor limit."""
    limit = get_route_load_limit(row['ROUTE_NAME'])
    return 'TRUE' if row['LOAD_FACTOR'] > limit else 'FALSE'


def process_data(data_frame: pd.DataFrame, bus_capacity: int) -> pd.DataFrame:
    """
    Process the DataFrame to calculate load factor and determine service periods
    and violations.
    """
    # Assign service period
    data_frame['SERVICE_PERIOD'] = data_frame['TRIP_START_TIME'].apply(assign_service_period)

    # Calculate LOAD_FACTOR using the BUS_CAPACITY
    data_frame['LOAD_FACTOR'] = data_frame['MAX_LOAD'] / bus_capacity

    # Create LOAD_FACTOR_VIOLATION column
    data_frame['LOAD_FACTOR_VIOLATION'] = data_frame.apply(check_load_factor_violation, axis=1)

    # Sort by 'LOAD_FACTOR' in descending order
    return data_frame.sort_values(by='LOAD_FACTOR', ascending=False)


def export_to_excel(data_frame: pd.DataFrame, output_file: str) -> None:
    """Export the DataFrame to an Excel file with adjusted column widths."""
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:  # pylint: disable=abstract-class-instantiated
        data_frame.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']

        # Adjust column widths based on the maximum length of the content in each column
        for idx, col in enumerate(data_frame.columns, 1):
            series = data_frame[col].astype(str)
            max_length = max(series.map(len).max(), len(str(col)))
            adjusted_width = max_length + 2  # Add extra space for clarity
            column_letter = get_column_letter(idx)
            worksheet.column_dimensions[column_letter].width = adjusted_width


def print_high_load_trips(data_frame: pd.DataFrame) -> None:
    """Print trips where 'MAX_LOAD' is over 30."""
    high_load_trips = data_frame[data_frame['MAX_LOAD'] > 30]
    if not high_load_trips.empty:
        print("Trips with MAX_LOAD over 30:")
        print(high_load_trips)


# =======================
# Main Routine
# =======================
def main():
    """Main routine to load, process, and export bus load data."""
    # Load and process data
    data_frame = load_data(INPUT_FILE)
    processed_data = process_data(data_frame, BUS_CAPACITY)

    # Export processed data to Excel
    export_to_excel(processed_data, OUTPUT_FILE)

    # Print high load trips
    print_high_load_trips(processed_data)

    print(f"Processed file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
