"""
Process NTD ridership data from one or more monthly Excel files.

This script automates the extraction, combination, filtering, and saving of 
ridership data from National Transit Database (NTD) Excel files. It's designed 
to handle multiple monthly reports, combine them into a single DataFrame, 
filter the data based on specified routes, and save the processed data to a 
new Excel file.

Configuration variables at the top of the script allow users to easily 
specify file paths, sheet names, months to process, routes of interest, and 
output settings.

Key Features:
- Loads data from multiple monthly Excel files.
- Combines data from specified months into a single DataFrame.
- Filters data based on a list of routes of interest (or processes all routes).
- Provides detailed debug output, including file loading confirmations, 
  unique route names, and row counts before and after filtering.
- Handles file not found and data loading errors gracefully, skipping 
  problematic months with warnings.
- Saves the processed data to a specified output file.
"""

import os

import pandas as pd

# -------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------

# File names by month
FILE_NAMES = {
    'October': 'NTD RIDERSHIP BY ROUTE_OCTOBER_2024.XLSX',
    'November':'NTD RIDERSHIP BY ROUTE_NOVEMBER_2024.XLSX'
}

# Sheet names by month
SHEET_NAMES = {
    'October': 'Temporary_Query_N',
    'November':'Temporary_Query_N'
}

# Ordered list of months to process
ORDERED_MONTHS = ['October', 'November']

# Base folder path (used for both input files and output directory)
BASE_FOLDER = r'\\Folder\Path\To\Your\Project'

# Default routes of interest. If an empty list is provided, all routes will be used.
DEFAULT_ROUTES = ['101', '202', '303']

# Output configuration
OUTPUT_DIR = BASE_FOLDER
OUTPUT_FILENAME = 'NTD_ridership_processed.xlsx'


# -------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------

def load_data(excel_path, sheet_name):
    """
    Load data from a given Excel file and sheet.
    Prints debug statements to confirm file existence and display columns.
    """
    print(f"Attempting to load data from: {excel_path} (sheet: {sheet_name})")

    # Check if the file exists
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"ERROR: File not found: {excel_path}")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"  Loaded {len(df)} rows. Columns found: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"Error reading {excel_path} (sheet: {sheet_name}): {e}")

    return df


def combine_data(months, base_folder, file_names, sheet_names):
    """
    Load data for each month and combine into a single DataFrame.
    Skips months if there's an error but warns the user.
    """
    data_frames = []
    for month in months:
        file_path = os.path.join(base_folder, file_names[month])
        print(f"\nProcessing month: {month}")
        try:
            df = load_data(file_path, sheet_names[month])
        except Exception as e:
            print(f"  Skipping month {month} due to error: {e}")
            continue
        data_frames.append(df)

    if not data_frames:
        raise RuntimeError("No data was loaded. Please check file paths and sheet names.")

    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"\nCombined data has {len(combined_df)} rows total.")
    return combined_df


def debug_route_names(df, route_col='ROUTE_NAME'):
    """
    Prints out unique route names for debugging. 
    This helps confirm that the routes in the file match your filter list.
    """
    unique_routes = df[route_col].unique()
    print(f"\nUnique route names found in '{route_col}' column: {list(unique_routes)}")


def filter_routes(df, routes_of_interest=None, route_col='ROUTE_NAME'):
    """
    Filter the DataFrame to include only the specified routes based on `route_col`.
    If `routes_of_interest` is None or an empty list, returns all routes.
    Prints debug output to show the number of rows before and after filtering.
    """
    if route_col not in df.columns:
        raise KeyError(
            f"The column '{route_col}' was not found in the DataFrame columns: {list(df.columns)}"
        )

    # Optionally, ensure consistent string values (trim whitespace, etc.)
    df[route_col] = df[route_col].astype(str).str.strip()

    # Debug: show unique route names
    debug_route_names(df, route_col=route_col)

    initial_count = len(df)
    if not routes_of_interest:
        print("No route filtering applied; using all routes.")
        return df

    filtered_df = df[df[route_col].isin(routes_of_interest)]
    final_count = len(filtered_df)
    print(f"\nFiltering routes: {routes_of_interest}")
    print(f"  Rows before filtering: {initial_count}")
    print(f"  Rows after filtering:  {final_count}")
    return filtered_df


def save_data(df, output_folder, output_filename):
    """
    Save the DataFrame to an Excel file in the specified output folder.
    """
    output_path = os.path.join(output_folder, output_filename)
    try:
        df.to_excel(output_path, index=False)
        print(f"\nFiltered data saved successfully to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save filtered data to {output_path}: {e}")


# -------------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------------

def main():
    # 1. Combine data from all months
    combined_df = combine_data(
        ORDERED_MONTHS,
        BASE_FOLDER,
        FILE_NAMES,
        SHEET_NAMES
    )

    # 2. Set routes of interest (use DEFAULT_ROUTES; use [] or None for all routes)
    routes_of_interest = DEFAULT_ROUTES

    # 3. Filter the combined DataFrame by the routes of interest
    filtered_df = filter_routes(combined_df, routes_of_interest=routes_of_interest, route_col='ROUTE_NAME')

    # 4. Save the filtered data to an Excel file
    save_data(filtered_df, OUTPUT_DIR, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
