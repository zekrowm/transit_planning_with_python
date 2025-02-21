"""
This module reads an Excel file containing transit ridership data by stop, filters
by an optional list of STOP_IDs, and then for each route generates an aggregated
Excel output. The aggregation can include boardings, alightings, or both, and
calculates the total boardings/alightings as well as percentages for each stop.

Usage:
1. Adjust the configuration variables in the CONFIGURATION SECTION.
2. Run `main()` in a Python environment or import this module in a Jupyter Notebook.
3. For each route, an Excel file is created in the specified output directory.
"""

import os

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIGURATION SECTION
# ------------------------------------------------------------------------------

# Path to the input Excel file
input_file_path = r'\\Your\File\Path\To\STOP_USAGE_(BY_STOP_NAME).XLSX'

# Path to the directory where output files will be saved
output_dir = r'\\Your\Folder\Path\To\Output'

# List of STOP_IDs to filter on. If empty, no filter is applied.
stop_filter_list = [1107, 2816, 6548]  # Example default

# Decide which columns to use in the output (True/False)
use_boardings = True
use_alightings = True

# ------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ------------------------------------------------------------------------------

def load_data(excel_path):
    """
    Reads the Excel file into a pandas DataFrame.
    
    Parameters
    ----------
    excel_path : str
        Path to the Excel (.xlsx) file.
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the contents of the Excel file.
    """
    df = pd.read_excel(excel_path)
    return df

def filter_by_stops(df, stop_ids):
    """
    Filters the DataFrame by a list of stop_ids if the list is not empty.
    If stop_ids is empty, the unfiltered DataFrame is returned.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing ridership data.
    stop_ids : list of int
        The list of stop IDs to filter on.
        
    Returns
    -------
    filtered_df : pandas.DataFrame
        Filtered DataFrame if stop_ids is non-empty; otherwise the original df.
    """
    if stop_ids:
        return df[df['STOP_ID'].isin(stop_ids)]
    else:
        return df

def get_route_names(df):
    """
    Returns a list of unique route names in the given DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing ridership data.
        
    Returns
    -------
    routes : numpy.ndarray
        Array of unique route names.
    """
    return df['ROUTE_NAME'].unique()

def aggregate_route_data(df, route_name, use_boardings, use_alightings):
    """
    For a given route_name, create a DataFrame of stops served by that route,
    and calculate total and percentage of boardings/alightings. If both boardings
    and alightings are selected, also calculate a 'PCT_TOTAL' column which
    measures the sum of boardings+alightings at each stop as a fraction of
    the route total.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The original DataFrame of ridership data.
    route_name : str
        Name of the route to filter on.
    use_boardings : bool
        Whether to include boardings (XBOARDINGS) in the output.
    use_alightings : bool
        Whether to include alightings (XALIGHTINGS) in the output.
        
    Returns
    -------
    grouped : pandas.DataFrame or None
        DataFrame grouped by STOP_ID, STOP_NAME, containing aggregates and
        percentages. Returns None if neither boardings nor alightings is used.
    """
    # Filter the route
    route_df = df[df['ROUTE_NAME'] == route_name].copy()

    # Determine which columns to aggregate
    agg_dict = {}
    if use_boardings:
        agg_dict['XBOARDINGS'] = 'sum'
    if use_alightings:
        agg_dict['XALIGHTINGS'] = 'sum'

    # If neither boardings nor alightings is selected, no meaningful aggregation
    if not agg_dict:
        return None

    # Aggregate
    grouped = route_df.groupby(['STOP_ID', 'STOP_NAME'], as_index=False).agg(agg_dict)

    # Calculate totals
    if use_boardings:
        total_boardings = grouped['XBOARDINGS'].sum()
    else:
        total_boardings = 0

    if use_alightings:
        total_alightings = grouped['XALIGHTINGS'].sum()
    else:
        total_alightings = 0

    # Calculate individual percentages for boardings and alightings
    if use_boardings:
        if total_boardings != 0:
            grouped['PCT_BOARDINGS'] = grouped['XBOARDINGS'] / total_boardings
        else:
            grouped['PCT_BOARDINGS'] = 0.0

    if use_alightings:
        if total_alightings != 0:
            grouped['PCT_ALIGHTINGS'] = grouped['XALIGHTINGS'] / total_alightings
        else:
            grouped['PCT_ALIGHTINGS'] = 0.0

    # If both boardings and alightings are used, calculate a PCT_TOTAL
    if use_boardings and use_alightings:
        grouped['XTOTAL'] = grouped['XBOARDINGS'] + grouped['XALIGHTINGS']
        total_combined = grouped['XTOTAL'].sum()
        if total_combined != 0:
            grouped['PCT_TOTAL'] = grouped['XTOTAL'] / total_combined
        else:
            grouped['PCT_TOTAL'] = 0.0

    return grouped

def save_route_data(route_data, route_name, output_dir):
    """
    Saves the route_data DataFrame to an .xlsx file named after the route.
    
    Parameters
    ----------
    route_data : pandas.DataFrame
        The aggregated ridership data for a specific route.
    route_name : str
        The route name to be used in the output file name.
    output_dir : str
        The directory where the output file should be saved.
    """
    filename = f"{route_name}.xlsx"
    filepath = os.path.join(output_dir, filename)
    route_data.to_excel(filepath, index=False)

def main():
    """
    Main function to:
    1. Load data from Excel.
    2. Optionally filter for specific stops.
    3. Identify route names to process.
    4. Aggregate boardings/alightings for each route.
    5. Save each route's results to an individual Excel file.
    """
    # 1. Read the Excel file
    df = load_data(input_file_path)

    # 2. Optionally filter by stop IDs (just to identify the routes serving them)
    if stop_filter_list:
        filtered_df_for_routes = filter_by_stops(df, stop_filter_list)
        route_names = get_route_names(filtered_df_for_routes)
    else:
        route_names = get_route_names(df)

    # 3. For each route, gather data about all stops that route serves
    for route_name in route_names:
        route_data = aggregate_route_data(df, route_name, use_boardings, use_alightings)
        if route_data is not None and not route_data.empty:
            save_route_data(route_data, route_name, output_dir)
            print(f"Saved data for route '{route_name}' to {output_dir}")
        else:
            print(f"No boardings/alightings selected or no data for route '{route_name}'")

# ------------------------------------------------------------------------------
# RUN THE SCRIPT (if desired)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
