"""
Ridership by Route and Stop Processor

This script processes ridership data from an input Excel file by filtering specific routes and stop IDs,
aggregating the data for defined time periods, and exporting the results to a new Excel file with multiple
formatted sheets.
"""

import os
import sys

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

# ==========================
# Configuration Section
# ==========================

# Input and Output File Paths
INPUT_FILE_PATH = r'\\Your\File\Path\to\RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX'
OUTPUT_FILE_SUFFIX = '_processed'
OUTPUT_FILE_EXTENSION = '.xlsx'

# Routes and Stop IDs for Filtering
ROUTES = ["101", "202", "303"]  # Replace with your route names
STOP_IDS = [
    1001, 1002, 1003, 1004
]  # Replace with your stops of interest

# Required Columns in the Input Excel File
REQUIRED_COLUMNS = ['TIME_PERIOD', 'ROUTE_NAME', 'STOP', 'STOP_ID', 'BOARD_ALL', 'ALIGHT_ALL']

# Columns to Retain in the Output Sheets
COLUMNS_TO_RETAIN = ['ROUTE_NAME', 'STOP', 'STOP_ID', 'BOARD_ALL', 'ALIGHT_ALL']

# Time Periods for Aggregation
TIME_PERIODS = ['AM Early', 'AM PEAK', 'MIDDAY', 'PM PEAK', 'PM LATE', 'PM NITE', 'OTHER']  # Replace with your time periods of interest

# ==========================
# End of Configuration
# ==========================


def aggregate_by_stop(data_subset):
    """
    Aggregate ridership data by STOP and STOP_ID.

    Parameters:
        data_subset (pd.DataFrame): Subset of the DataFrame to aggregate.

    Returns:
        pd.DataFrame: Aggregated DataFrame with totals and routes.
    """
    aggregated = data_subset.groupby(['STOP', 'STOP_ID'], as_index=False).agg({
        'BOARD_ALL': 'sum',
        'ALIGHT_ALL': 'sum',
        # Collect unique ROUTE_NAMEs into a comma-separated string
        'ROUTE_NAME': lambda x: ', '.join(sorted(x.unique()))
    })
    # Rename columns to indicate they are totals
    aggregated.rename(columns={
        'BOARD_ALL': 'BOARD_ALL_TOTAL',
        'ALIGHT_ALL': 'ALIGHT_ALL_TOTAL',
        'ROUTE_NAME': 'ROUTES'
    }, inplace=True)
    return aggregated


def read_excel_file(input_file):
    """
    Read the Excel file into a pandas DataFrame.

    Parameters:
        input_file (str): Path to the input Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the Excel data.
    """
    try:
        return pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
    except pd.errors.ExcelFileError as error:
        print(f"Error reading the Excel file: {error}")
        sys.exit(1)


def verify_required_columns(data_frame, required_columns):
    """
    Verify that all required columns are present in the DataFrame.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to check.
        required_columns (list): List of required column names.

    Raises:
        SystemExit: If any required columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in data_frame.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing in the data: {missing_columns}")
        sys.exit(1)


def filter_data(data_frame, routes, stop_ids):
    """
    Filter the DataFrame based on specified routes and stop IDs.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to filter.
        routes (list): List of route names to include.
        stop_ids (list): List of stop IDs to include.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return data_frame[
        data_frame['ROUTE_NAME'].isin(routes) &
        data_frame['STOP_ID'].isin(stop_ids)
    ]


def write_to_excel(output_file, filtered_data, aggregated_peaks, all_time_aggregated):
    """
    Write the filtered and aggregated data to an Excel file with multiple sheets.

    Parameters:
        output_file (str): Path to the output Excel file.
        filtered_data (pd.DataFrame): Filtered DataFrame to write to 'Original' sheet.
        aggregated_peaks (dict): Dictionary of aggregated DataFrames for each time period.
        all_time_aggregated (pd.DataFrame): Aggregated DataFrame for all time periods.
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write the original filtered data to a sheet named 'Original'
            filtered_data.to_excel(writer, sheet_name='Original', index=False)

            # Write the aggregated data for each time period
            for period, df_agg in aggregated_peaks.items():
                df_agg.to_excel(writer, sheet_name=period, index=False)

            # Write the All Time Periods aggregated data to a new sheet
            all_time_aggregated.to_excel(writer, sheet_name='All Time Periods', index=False)

            writer.save()

        adjust_excel_formatting(output_file)
        print(f"Success: The processed file has been saved as '{output_file}'.")
    except Exception as error:
        print(f"Error writing the processed Excel file: {error}")
        sys.exit(1)


def adjust_excel_formatting(output_file):
    """
    Adjust column widths and format headers in the Excel file.

    Parameters:
        output_file (str): Path to the Excel file to format.
    """
    try:
        workbook = load_workbook(output_file)

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]

            # Bold the header row
            for cell in sheet[1]:
                cell.font = Font(bold=True)

            for column_cells in sheet.columns:
                # Get the maximum length of the content in the column
                max_length = 0
                column = column_cells[0].column_letter  # Get the column name (e.g., 'A', 'B', ...)
                for cell in column_cells:
                    try:
                        cell_value = str(cell.value)
                        if cell_value is None:
                            cell_length = 0
                        else:
                            cell_length = len(cell_value)
                        if cell_length > max_length:
                            max_length = cell_length
                    except:
                        pass
                # Set the column width with a little extra space
                adjusted_width = max_length + 2
                sheet.column_dimensions[column].width = adjusted_width

        # Save the workbook after adjusting column widths and formatting
        workbook.save(output_file)
    except Exception as error:
        print(f"Error adjusting Excel formatting: {error}")
        sys.exit(1)


def main():
    """
    Main function to process ridership data.

    Processes ridership data by filtering specific routes and stop IDs, aggregating the data
    for defined time periods, and exporting the results to a new Excel file with multiple
    formatted sheets.
    """
    input_file = INPUT_FILE_PATH
    base, ext = os.path.splitext(input_file)
    ext = ext.lower()

    # Ensure the output has the correct extension
    if ext != OUTPUT_FILE_EXTENSION:
        print(f"Warning: The input file has an unexpected extension '{ext}'. "
              f"The output file will use '{OUTPUT_FILE_EXTENSION}' extension.")
        ext = OUTPUT_FILE_EXTENSION

    output_file = f"{base}{OUTPUT_FILE_SUFFIX}{ext}"

    # Read and verify the Excel data
    ridership_df = read_excel_file(input_file)
    verify_required_columns(ridership_df, REQUIRED_COLUMNS)

    # Filter the DataFrame based on ROUTES and STOP_IDS
    filtered_data = filter_data(ridership_df, ROUTES, STOP_IDS)

    # Standardize the 'TIME_PERIOD' values by stripping whitespace and converting to uppercase
    filtered_data['TIME_PERIOD'] = filtered_data['TIME_PERIOD'].astype(str).str.strip().str.upper()

    # Filter data for each specified time period
    peak_data_dict = {}
    for period in TIME_PERIODS:
        peak_data_dict[period] = filtered_data[filtered_data['TIME_PERIOD'] == period][COLUMNS_TO_RETAIN]

    # Aggregate data for All Time Periods
    all_time_aggregated = aggregate_by_stop(filtered_data)

    # Aggregate data for each specified time period
    aggregated_peaks = {}
    for period, data_subset in peak_data_dict.items():
        aggregated_peaks[period] = aggregate_by_stop(data_subset)

    # Round ridership columns to 1 decimal place
    ridership_columns = ['BOARD_ALL_TOTAL', 'ALIGHT_ALL_TOTAL']
    for df_agg in [all_time_aggregated] + list(aggregated_peaks.values()):
        for col in ridership_columns:
            if col in df_agg.columns:
                df_agg[col] = df_agg[col].round(1)

    # Also round the original dataframe's ridership columns if needed
    for col in ['BOARD_ALL', 'ALIGHT_ALL']:
        if col in filtered_data.columns:
            filtered_data[col] = filtered_data[col].round(1)

    # Write the data to a new Excel file with multiple sheets and adjust column widths
    write_to_excel(output_file, filtered_data, aggregated_peaks, all_time_aggregated)


if __name__ == "__main__":
    main()
