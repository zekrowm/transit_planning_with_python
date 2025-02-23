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
    df = pd.read_excel(input_file)
    selected_columns = [
        'SERIAL_NUMBER', 'ROUTE_NAME', 'DIRECTION_NAME', 'TRIP_START_TIME',
        'BLOCK', 'MAX_LOAD', 'MAX_LOAD_P', 'ALL_RECORDS_MAX_LOAD'
    ]
    return df[selected_columns]

def assign_service_period(trip_start_time: str) -> str:
    """Assign a service period based on TRIP_START_TIME."""
    if '04:00' <= trip_start_time < '06:00':
        return 'AM Early'
    elif '06:00' <= trip_start_time < '09:00':
        return 'AM Peak'
    elif '09:00' <= trip_start_time < '15:00':
        return 'Midday'
    elif '15:00' <= trip_start_time < '18:00':
        return 'PM Peak'
    elif '18:00' <= trip_start_time < '21:00':
        return 'PM Late'
    elif '21:00' <= trip_start_time < '24:00':
        return 'PM Nite'
    else:
        return 'Other'

def get_route_load_limit(route_name: str) -> float:
    """
    Get the appropriate load factor limit based on the route.
    If the route is in LOWER_LIMIT_ROUTES, returns the lower limit; otherwise, returns the higher limit.
    """
    if route_name in LOWER_LIMIT_ROUTES:
        return LOWER_LOAD_FACTOR_LIMIT
    return HIGHER_LOAD_FACTOR_LIMIT

def check_load_factor_violation(row: pd.Series) -> str:
    """Determine if the row exceeds the route's load factor limit."""
    limit = get_route_load_limit(row['ROUTE_NAME'])
    return 'TRUE' if row['LOAD_FACTOR'] > limit else 'FALSE'

def process_data(df: pd.DataFrame, bus_capacity: int) -> pd.DataFrame:
    """Process the DataFrame to calculate load factor and determine service periods and violations."""
    # Assign service period
    df['SERVICE_PERIOD'] = df['TRIP_START_TIME'].apply(assign_service_period)

    # Calculate LOAD_FACTOR using the BUS_CAPACITY
    df['LOAD_FACTOR'] = df['MAX_LOAD'] / bus_capacity

    # Create LOAD_FACTOR_VIOLATION column
    df['LOAD_FACTOR_VIOLATION'] = df.apply(check_load_factor_violation, axis=1)

    # Sort by 'LOAD_FACTOR' in descending order
    return df.sort_values(by='LOAD_FACTOR', ascending=False)

def export_to_excel(df: pd.DataFrame, output_file: str) -> None:
    """Export the DataFrame to an Excel file with adjusted column widths."""
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']

        # Adjust column widths based on the maximum length of the content in each column
        for idx, col in enumerate(df.columns, 1):
            series = df[col].astype(str)
            max_length = max(series.map(len).max(), len(str(col)))
            adjusted_width = max_length + 2  # Add extra space for clarity
            column_letter = get_column_letter(idx)
            worksheet.column_dimensions[column_letter].width = adjusted_width

def print_high_load_trips(df: pd.DataFrame) -> None:
    """Print trips where 'MAX_LOAD' is over 30."""
    high_load_trips = df[df['MAX_LOAD'] > 30]
    if not high_load_trips.empty:
        print("Trips with MAX_LOAD over 30:")
        print(high_load_trips)

# =======================
# Main Routine
# =======================
def main():
    """Main routine to load, process, and export bus load data."""
    # Load and process data
    df = load_data(INPUT_FILE)
    processed_df = process_data(df, BUS_CAPACITY)

    # Export processed data to Excel
    export_to_excel(processed_df, OUTPUT_FILE)

    # Print high load trips
    print_high_load_trips(processed_df)

    print(f"Processed file saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
