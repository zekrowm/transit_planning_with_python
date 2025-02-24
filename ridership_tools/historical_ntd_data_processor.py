"""
Ridership Data Processing & Analysis

This script consolidates transit ridership data from multiple Excel files, 
exports a cleaned CSV, and optionally generates ridership trend plots. 
It also identifies routes with declining ridership using a rolling 12-month 
average and regression analysis.

Configuration:
- Set `ENABLE_PLOTTING = True` to generate plots, or `False` to skip.
- Choose between `USE_DYNAMIC_SCALE` (adaptive y-axis) or static scaling.
- Modify `BASE_INPUT_DIR`, `OUTPUT_DIR`, and `PLOTS_OUTPUT_FOLDER` as needed.
- Exclude specific routes using `ROUTES_TO_EXCLUDE`.

Usage:
Run with:
    python ridership_analysis.py

Dependencies: pandas, matplotlib, numpy, openpyxl
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Configuration Section
###############################################################################

# === Part 1: Consolidation Config ===
BASE_INPUT_DIR = r"C:\Your\Folder\Path\NTD_files"
OUTPUT_DIR = r"C:\Your\Folder\Path\Output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of routes to exclude from the final consolidated file
ROUTES_TO_EXCLUDE = [
    '101', '202', '303'
]
# This may include discontinued or non-standard routes
# Leave empty if you want no exclusions (e.g. [])

FILE_SHEET_MAPPING = {
    'September-23': {
        'file_name': 'NTD RIDERSHIP BY ROUTE SEP 2023.xlsx',
        'sheet_name': 'System Sep2023'
    },
    'October-23': {
        'file_name': 'OCTOBER 2023  NTD RIDERSHIP_BY_ROUTE.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'November-23': {
        'file_name': 'November 2023 NTD RIDERSHIP BY ROUTE.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'December-23': {
        'file_name': 'NTD RIDERSHIP BY ROUTE DECEMBER 2023.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'January-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE JANUARY 2024 FINAL 22824.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'February-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE FEBRUARY 2024.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'March-24': {
        'file_name': 'MARCH 2024 NTD RIDERSHIP BY ROUTE AND LOCATION.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'April-24': {
        'file_name': 'APRIL 2024 NTD RIDERSHIP BY ROUTE (002).xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'May-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE MAY 2024.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'June-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE JUNE 2024.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'July-24': {
        'file_name': 'JULY 2024 NTD RIDERSHIP BY ROUTE.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'August-24': {
        'file_name': 'AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'September-24': {
        'file_name': 'SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.xlsx',
        'sheet_name': 'Sep.2024 Finals'
    },
    'October-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'November-24': {
        'file_name': 'NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx',
        'sheet_name': 'Temporary_Query_N'
    },
    'December-24': {
        'file_name': 'NTD RIDERSHIP BY MONTH_DECEMBER 2024.xlsx',
        'sheet_name': 'Dec. 2024'
    }
}

ROUTE_COLUMN_NAME = 'ROUTE_NAME'
RIDERSHIP_COLUMN_NAME = 'MTH_BOARD'

# === Part 2: Plotting & Analysis Config ===
INPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'Consolidated_Ridership_Data.csv')

# Set this to False if you do NOT want to create plots.
ENABLE_PLOTTING = True

# Which routes to plot/analyze? If empty, includes all
ROUTES_OF_INTEREST = []

# Where to save the plots
PLOTS_OUTPUT_FOLDER = r"C:\Your\Folder\Path\Plots"
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)

# Plot appearance settings
PLOT_TITLE_PREFIX = "Ridership Over Time for Route"
FIG_SIZE = (10, 6)
MARKER_STYLE = 'o'
LINE_STYLE = '-'
OUTPUT_FORMAT = 'jpeg'  # e.g., 'jpeg' or 'png'

# Choose between dynamic or static y-axis scale
USE_DYNAMIC_SCALE = True  # Set to False for static y-axis scale
STATIC_Y_MIN = 0
STATIC_Y_MAX = 100000

###############################################################################
# Part 1: Consolidation Functions
###############################################################################


def extract_route_ridership(df, route_column='ROUTE_NAME',
                            ridership_column='MTH_BOARD'):
    """
    Extract the route and ridership data from the DataFrame. Cleans the
    route names by removing spaces, converting to uppercase, and ensuring
    numeric columns are properly handled.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame to extract data from.
    route_column : str, optional
        The name of the column containing route identifiers.
    ridership_column : str, optional
        The name of the column containing ridership values.

    Returns
    -------
    pd.DataFrame
        A DataFrame with just the relevant route and ridership columns.
    """
    missing_columns = []
    for col in [route_column, ridership_column]:
        if col not in df.columns:
            missing_columns.append(col)
    if missing_columns:
        raise ValueError(
            f"Missing columns in the data: {', '.join(missing_columns)}"
        )

    df = df[[route_column, ridership_column]].copy()
    df.dropna(subset=[route_column, ridership_column], inplace=True)

    # Clean the route name
    df[route_column] = (
        df[route_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(' ', '', regex=False)
    )
    # Remove any trailing ".0" if present
    df[route_column] = df[route_column].apply(
        lambda x: re.sub(r'\.0$', '', x)
    )

    # Convert ridership to numeric, replace non-convertible with 0
    df[ridership_column] = pd.to_numeric(df[ridership_column],
                                         errors='coerce').fillna(0)

    return df


def consolidate_ridership_data():
    """
    Consolidate ridership data from multiple Excel files and sheets into a
    single CSV file. The consolidated CSV file will be saved to
    `Consolidated_Ridership_Data.csv` in the specified OUTPUT_DIR.

    Returns
    -------
    str
        Path to the generated CSV file of consolidated data, or None if
        no data was processed.
    """
    consolidated_df = pd.DataFrame()
    excluded_routes_list = []
    total_excluded_routes = 0

    for month, info in FILE_SHEET_MAPPING.items():
        file_path = os.path.join(BASE_INPUT_DIR, info['file_name'])

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping {month}.")
            continue

        try:
            df = pd.read_excel(file_path, sheet_name=info['sheet_name'])
            print(f"Processing {month}: {info['file_name']} - Sheet: "
                  f"{info['sheet_name']}")
        except Exception as exc:
            print(f"Error reading {file_path} - Sheet: {info['sheet_name']}: "
                  f"{exc}")
            continue

        try:
            df_extracted = extract_route_ridership(
                df,
                route_column=ROUTE_COLUMN_NAME,
                ridership_column=RIDERSHIP_COLUMN_NAME
            )
            print(f"Extracted {df_extracted.shape[0]} records for {month}.")
        except ValueError as ve:
            print(f"Data extraction error for {month}: {ve}")
            continue

        # Exclude specified routes if any
        if ROUTES_TO_EXCLUDE:
            df_excluded = df_extracted[
                df_extracted[ROUTE_COLUMN_NAME].isin(ROUTES_TO_EXCLUDE)
            ].copy()
            if not df_excluded.empty:
                df_excluded['Month'] = month
                excluded_routes_list.append(df_excluded)

                excluded_count = df_excluded.shape[0]
                total_excluded_routes += excluded_count
                print(f"Excluded {excluded_count} routes for {month}.")
            else:
                excluded_count = 0
                print(f"No routes excluded for {month}.")

            # Perform the exclusion
            df_extracted = df_extracted[
                ~df_extracted[ROUTE_COLUMN_NAME].isin(ROUTES_TO_EXCLUDE)
            ]

        if df_extracted.empty:
            print(f"No data left after excluding routes for {month}. Skipping.")
            continue

        # Aggregate ridership by route
        df_grouped = df_extracted.groupby(ROUTE_COLUMN_NAME)[
            RIDERSHIP_COLUMN_NAME
        ].sum().reset_index()

        # Convert the month text like 'July-25' to 'Jul-25' if possible
        try:
            month_datetime = pd.to_datetime(month, format='%B-%y')
            month_abbr = month_datetime.strftime('%b-%y')
        except ValueError:
            month_abbr = month
            print(f"Warning: Could not parse month '{month}'. Using original.")

        df_grouped = df_grouped.rename(columns={
            RIDERSHIP_COLUMN_NAME: month_abbr
        })

        if consolidated_df.empty:
            consolidated_df = df_grouped
            print(f"Initialized consolidated DataFrame with "
                  f"{df_grouped.shape[0]} routes.")
        else:
            consolidated_df = pd.merge(
                consolidated_df, df_grouped,
                on=ROUTE_COLUMN_NAME,
                how='outer'
            )
            print(f"Merged data for {month}. Consolidated DataFrame now has "
                  f"{consolidated_df.shape[0]} routes.")

    if consolidated_df.empty:
        print("No data was processed. Please check the input files.")
        return None

    # Fill NaN with 0
    consolidated_df.fillna(0, inplace=True)

    # Sort by route name (attempt numeric sort if possible)
    try:
        consolidated_df['ROUTE_SORT'] = consolidated_df[
            ROUTE_COLUMN_NAME
        ].astype(int)
    except ValueError:
        consolidated_df['ROUTE_SORT'] = consolidated_df[ROUTE_COLUMN_NAME]

    consolidated_df.sort_values('ROUTE_SORT', inplace=True)
    consolidated_df.drop('ROUTE_SORT', axis=1, inplace=True)

    # Reorder columns: route first, then sorted months
    month_cols = [
        col for col in consolidated_df.columns
        if re.match(r'^[A-Za-z]{3}-\d{2}$', col)
    ]
    # Sort month columns by date
    month_cols_sorted = sorted(
        month_cols, key=lambda x: pd.to_datetime(x, format='%b-%y')
    )
    # Final reordering
    consolidated_df = consolidated_df[[ROUTE_COLUMN_NAME] + month_cols_sorted]

    # Export
    output_file = os.path.join(OUTPUT_DIR, 'Consolidated_Ridership_Data.csv')
    consolidated_df.to_csv(output_file, index=False)

    print(f"\nConsolidated data exported to {output_file}")

    # Export excluded routes if any
    if ROUTES_TO_EXCLUDE and excluded_routes_list:
        excluded_routes_df = pd.concat(excluded_routes_list, ignore_index=True)
        excluded_routes_df = excluded_routes_df[
            ['Month', ROUTE_COLUMN_NAME, RIDERSHIP_COLUMN_NAME]
        ]
        excluded_file = os.path.join(OUTPUT_DIR, 'Excluded_Routes.xlsx')
        try:
            excluded_routes_df.to_excel(excluded_file, index=False)
            print(f"Excluded routes exported to {excluded_file}")
        except Exception as exc:
            print(f"Error exporting excluded routes: {exc}")
    else:
        print("No excluded routes or none specified.")

    if ROUTES_TO_EXCLUDE:
        print(f"Total routes excluded across all months: {total_excluded_routes}")

    return output_file


###############################################################################
# Part 2: Plotting & Analysis Functions
###############################################################################


def load_data(csv_path):
    """
    Load the CSV data into a pandas DataFrame, normalizing column names.

    Parameters
    ----------
    csv_path : str
        File path to the consolidated ridership data CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names.
    """
    try:
        df = pd.read_csv(csv_path)
        # Normalize column names
        df.columns = df.columns.str.strip().str.upper()
        print(f"Successfully loaded data from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty.")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error loading data: {exc}")
        sys.exit(1)


def filter_routes(df, route_column, routes_of_interest):
    """
    Filter the DataFrame for the specified routes. Returns all
    routes if the list is empty.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    route_column : str
        The name of the route column.
    routes_of_interest : list of str
        Routes to include; if empty, includes all.

    Returns
    -------
    pd.DataFrame
        Filtered (or unfiltered) DataFrame.
    """
    if not routes_of_interest:
        print("No specific routes specified. Plotting/analyzing all routes.")
        return df.copy()

    df_filtered = df[df[route_column].astype(str).isin(routes_of_interest)]
    if df_filtered.empty:
        print(f"No data found for routes: {routes_of_interest}")
        sys.exit(1)
    else:
        print(f"Filtered data has {df_filtered.shape[0]} rows for routes: "
              f"{routes_of_interest}")
    return df_filtered


def parse_month_columns(df, route_column):
    """
    Identify and sort month columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing ridership data.
    route_column : str
        Name of the route column.

    Returns
    -------
    list
        Chronologically sorted month column names.
    """
    month_cols = [c for c in df.columns if c != route_column]
    try:
        month_dates = pd.to_datetime(month_cols, format='%b-%y', errors='coerce')
        if month_dates.isnull().any():
            fail_cols = [col for col, d in zip(month_cols, month_dates)
                         if pd.isnull(d)]
            raise ValueError(f"Unable to parse columns: {fail_cols}")
    except Exception as exc:
        print(f"Error parsing month columns: {exc}")
        sys.exit(1)

    # Pair up the parsed dates with the column names
    month_pairs = list(zip(month_dates, month_cols))
    # Filter out anything that didn't parse
    month_pairs = [mp for mp in month_pairs if not pd.isnull(mp[0])]
    # Sort by the datetime
    sorted_months = [col for _, col in sorted(month_pairs, key=lambda x: x[0])]
    return sorted_months


def plot_ridership(df, route_column, sorted_months, output_folder):
    """
    Create and save ridership plots for each route in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame with ridership data.
    route_column : str
        Name of the route column.
    sorted_months : list
        Chronologically sorted month columns.
    output_folder : str
        Directory where plots are saved.
    """
    for _, row in df.iterrows():
        route = row[route_column]
        ridership = row[sorted_months].values.astype(float)

        if len(ridership) == 0:
            print(f"No ridership data for Route {route}. Skipping.")
            continue

        plt.figure(figsize=FIG_SIZE)
        plt.plot(sorted_months, ridership,
                 marker=MARKER_STYLE,
                 linestyle=LINE_STYLE,
                 label=f'Route {route}')
        plt.title(f"{PLOT_TITLE_PREFIX} {route}")
        plt.xlabel("Month")
        plt.ylabel("Ridership")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)

        if USE_DYNAMIC_SCALE:
            # Dynamic scale based on route's ridership
            if ridership.max() > 0:
                plt.ylim(0, ridership.max() * 1.1)
            else:
                plt.ylim(0, 1)
        else:
            # Static (fixed) scale
            plt.ylim(STATIC_Y_MIN, STATIC_Y_MAX)

        plt.tight_layout()
        out_file = os.path.join(output_folder, f"Route_{route}.{OUTPUT_FORMAT}")
        try:
            plt.savefig(out_file, format=OUTPUT_FORMAT)
            print(f"Saved plot for Route {route} to {out_file}")
        except Exception as exc:
            print(f"Error saving plot for Route {route}: {exc}")
        finally:
            plt.close()


def identify_negative_trends(df, route_column, sorted_months,
                             window=12, slope_threshold=0):
    """
    Identify routes with negative ridership trends using a rolling average
    and a linear regression on that rolling average.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of ridership data for each route.
    route_column : str
        Name of the route column.
    sorted_months : list
        Chronologically sorted month columns.
    window : int, optional
        Window size for rolling average (default 12).
    slope_threshold : float, optional
        If slope < slope_threshold, consider negative trend (default 0).

    Returns
    -------
    list
        Routes with negative ridership trends.
    """
    negative_routes = []
    trend_details = []

    for _, row in df.iterrows():
        route = row[route_column]
        ridership = row[sorted_months].astype(float).values

        ts_df = pd.DataFrame({
            'Month': pd.to_datetime(sorted_months, format='%b-%y'),
            'Ridership': ridership
        }).sort_values('Month')
        ts_df.set_index('Month', inplace=True)

        # Rolling average
        ts_df['Rolling_Avg'] = ts_df['Ridership'].rolling(window=window).mean()
        rolling_avg = ts_df['Rolling_Avg'].dropna()

        if len(rolling_avg) < 2:
            # Not enough data to judge slope
            continue

        initial_avg = rolling_avg.iloc[0]
        final_avg = rolling_avg.iloc[-1]
        avg_diff = final_avg - initial_avg

        # Linear regression on the rolling average
        x_vals = np.arange(len(rolling_avg))
        y_vals = rolling_avg.values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)

        if slope < slope_threshold:
            negative_routes.append(route)
            trend_details.append({
                'Route': route,
                'Initial_Rolling_Avg': initial_avg,
                'Final_Rolling_Avg': final_avg,
                'Average_Diff': avg_diff,
                'Slope': slope
            })

    if negative_routes:
        print("\nRoutes with Negative Ridership Trends:")
        for r in negative_routes:
            print(f" - Route {r}")

        trend_df = pd.DataFrame(trend_details)
        print("\nDetailed Trend Information:")
        print(trend_df)

        # Save to CSV
        trend_path = os.path.join(
            PLOTS_OUTPUT_FOLDER, "negative_ridership_trends.csv"
        )
        try:
            trend_df.to_csv(trend_path, index=False)
            print(f"Trend details saved to {trend_path}")
        except Exception as exc:
            print(f"Error saving trend details: {exc}")
    else:
        print("\nNo routes with negative ridership trends.")

    return negative_routes


###############################################################################
# Main
###############################################################################


def main():
    """
    Run the entire pipeline:
     1) Consolidate the data into a CSV.
     2) Load the CSV, filter for routes, parse months.
     3) (Optional) Plot ridership.
     4) Identify negative trends (rolling 12-month).
    """
    # === Step 1: Consolidate data ===
    csv_path = consolidate_ridership_data()
    if csv_path is None:
        print("No data to plot or analyze. Exiting.")
        sys.exit(0)

    # === Step 2: Load and filter data ===
    df = load_data(csv_path)

    # The route column will be 'ROUTE_NAME' after normalization => 'ROUTE_NAME'
    route_col = 'ROUTE_NAME'
    if route_col not in df.columns:
        print(f"Error: '{route_col}' not found in consolidated data.")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    df_filtered = filter_routes(df, route_col, ROUTES_OF_INTEREST)

    # === Step 3: Identify and sort month columns ===
    sorted_months = parse_month_columns(df_filtered, route_col)
    print("Sorted month columns:", sorted_months)

    # === Step 4 (Optional): Plot ridership ===
    if ENABLE_PLOTTING:
        plot_ridership(df_filtered, route_col, sorted_months, PLOTS_OUTPUT_FOLDER)
    else:
        print("Plotting is disabled. Skipping plot generation.")

    # === Step 5: Identify negative trends ===
    identify_negative_trends(df_filtered, route_col, sorted_months)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
