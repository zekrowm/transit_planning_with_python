"""
This module processes Capital Bikeshare data by combining spatial data from shapefiles with trip data from CSV files.
It optionally clips bikeshare locations to a geographic boundary, reconciles station names between datasets,
aggregates monthly trip activity per station, and calculates daily averages by weekday, Saturday, and Sunday.

Data sources:
- https://opendata.dc.gov/datasets/DCGIS::capital-bikeshare-locations/explore?location=38.813802%2C-77.103538%2C9.67
- https://capitalbikeshare.com/system-data
"""

import difflib
import glob
import os
import calendar

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
BIKESHARE_SHP_PATH = r"C:\Your\File\Path\To\Bikeshare_Locations.shp"
# Set BOUNDARY_SHP_PATH to an empty string or None if you want to use all stops.
BOUNDARY_SHP_PATH = r"C:\Your\File\Path\To\Study_Area_Boundary.shp"
CSV_FOLDER = r"C:\Your\Folder\Path\To\Bikeshare_Trip_Data"
OUTPUT_CSV = r"C:\Your\File\Path\To\Output\total_monthly_trip_activity_by_station.csv"
OUTPUT_XLSX = r"C:\Your\File\Path\To\Output\trip_activity_averages_by_station.xlsx"

# New configuration: Users can set the fuzzy matching threshold here.
FUZZY_THRESHOLD = 0.8

# -----------------------------
# Functions
# -----------------------------
def load_shapefiles(bikeshare_shp_path, boundary_shp_path=None):
    """Load the bikeshare shapefile and optionally the boundary shapefile."""
    print("Loading bikeshare shapefile...")
    bikeshare_gdf = gpd.read_file(bikeshare_shp_path)
    
    boundary_gdf = None
    if boundary_shp_path and os.path.exists(boundary_shp_path):
        print("Loading boundary shapefile...")
        boundary_gdf = gpd.read_file(boundary_shp_path)
    else:
        print("No valid boundary shapefile provided. Using all stops.")
    return bikeshare_gdf, boundary_gdf

def load_and_concatenate_csv(csv_folder):
    """
    Find and load all CSV files in the specified folder,
    then concatenate them into a single DataFrame.
    """
    print(f"Loading and concatenating CSV files from folder: {csv_folder}")
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder}")
    csv_dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(csv_dfs, ignore_index=True)
    return combined_df

def clip_shapefile(bikeshare_gdf, boundary_gdf):
    """Reproject if needed and clip the bikeshare GeoDataFrame to the boundary."""
    if bikeshare_gdf.crs != boundary_gdf.crs:
        bikeshare_gdf = bikeshare_gdf.to_crs(boundary_gdf.crs)
    clipped_gdf = gpd.clip(bikeshare_gdf, boundary_gdf)
    return clipped_gdf

def plot_shapefile(boundary_gdf, clipped_gdf):
    """Plot the boundary and clipped bikeshare locations."""
    print("Plotting clipped shapefile...")
    fig, ax = plt.subplots(figsize=(10, 10))
    boundary_gdf.plot(ax=ax, color='lightgray', edgecolor='black')
    clipped_gdf.plot(ax=ax, color='red')
    ax.set_title('Clipped Bikeshare Locations within Boundary')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

def find_close_matches(station, valid_station_names, threshold):
    """Return the best fuzzy match for a station name from the valid_station_names list."""
    matches = difflib.get_close_matches(station, valid_station_names, n=1, cutoff=threshold)
    return matches[0] if matches else None

def rename_stations(df):
    """Rename specific station names based on a predefined mapping dictionary."""
    # Fixed corrections known to be helpful.
    rename_dict = {
        'John McCormack Dr & Michigan Ave NE': 'John McCormack Rd & Michigan Ave NE',
        'Radford St & Osage St': 'Radford & Osage St',
        'Anacostia Roller Skating Pavillion': 'Anacostia Roller Skating Pavilion',
        'Wilson Blvd. & N. Vermont St.': 'Wilson Blvd & N Vermont St',
        '10th & Florida Ave NW': '10th St & Florida Ave NW',
        '14th & Rhode Island Ave NW': '14th St & Rhode Island Ave NW',
        'Kenilworth Terrace & Hayes St. NE': 'Kenilworth Terr & Hayes St. NE',
        '11th & V st NW': '11th & V St NW',
        'Vaden Dr & Royal Victoria Dr/Providence Community Center': 'Vaden Dr & Royal Victoria Dr/Jim Scott Cmty Ctr',
        'Westbranch Dr & Jones Branch Dr': 'Westbranch & Jones Branch Dr',
        '21st NW & E St NW': '21st & E St NW'
    }
    df['start_station_name'] = df['start_station_name'].replace(rename_dict)
    df['end_station_name'] = df['end_station_name'].replace(rename_dict)
    return df

def interactive_update_station_names(unmatched_start, unmatched_end, valid_station_names, threshold):
    """
    For station names not matching any valid stop in the study area,
    interactively prompt the user with a fuzzy match suggestion.
    
    If the suggested name is different from the original, the user can confirm
    the update. Otherwise, a message is shown advising manual correction.
    Returns a mapping dictionary of updates.
    """
    update_mapping = {}
    # Combine unmatched names from both start and end columns.
    combined_unmatched = set(unmatched_start).union(set(unmatched_end))
    
    print("\nInteractive review of fuzzy match suggestions:")
    print("(For each suggestion, if the proposed correction is different, type 'y' to update it.)\n")
    
    for station in combined_unmatched:
        suggestion = find_close_matches(station, valid_station_names, threshold)
        if suggestion:
            if suggestion == station:
                print(f"Station '{station}' appears identical to its fuzzy match suggestion.")
                print("  -> If you believe this is a typo, please update it manually in your mapping.\n")
            else:
                print(f"Suggestion: '{station}' -> '{suggestion}'")
                choice = input("Do you want to update this station name? (y/n): ")
                if choice.strip().lower() == 'y':
                    update_mapping[station] = suggestion
                print()  # Blank line for readability.
        else:
            print(f"No fuzzy match found for station: '{station}'.")
    return update_mapping

def filter_and_aggregate_trips(combined_df, clipped_gdf):
    """
    Filter trips that include a valid station (i.e. in the clipped shapefile)
    and then aggregate monthly trip counts.
    """
    # Clean station names by stripping extra spaces.
    combined_df['start_station_name'] = combined_df['start_station_name'].str.strip()
    combined_df['end_station_name'] = combined_df['end_station_name'].str.strip()

    # Use stations within the boundary (or all stops) as valid.
    valid_stations = clipped_gdf['NAME'].str.strip().unique()

    # Filter trips where at least one station is in the valid list.
    filtered_df = combined_df[
        (combined_df['start_station_name'].notna()) &
        (combined_df['end_station_name'].notna()) &
        ((combined_df['start_station_name'].isin(valid_stations)) |
         (combined_df['end_station_name'].isin(valid_stations)))
    ]

    print(f"Total trips after filtering: {len(filtered_df)}")

    # Convert time columns to datetime and drop rows with invalid times.
    filtered_df['started_at'] = pd.to_datetime(filtered_df['started_at'], errors='coerce')
    filtered_df['ended_at'] = pd.to_datetime(filtered_df['ended_at'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['started_at', 'ended_at'])

    # Extract day of week and month.
    filtered_df['day_of_week'] = filtered_df['started_at'].dt.day_name()
    filtered_df['month'] = filtered_df['started_at'].dt.to_period('M').astype(str)

    # Aggregate start and end trip counts by station and month.
    start_trip_counts = filtered_df[filtered_df['start_station_name'].isin(valid_stations)]\
        .groupby(['start_station_name', 'month']).size().unstack(fill_value=0)
    end_trip_counts = filtered_df[filtered_df['end_station_name'].isin(valid_stations)]\
        .groupby(['end_station_name', 'month']).size().unstack(fill_value=0)

    # Combine counts from both start and end trips.
    total_trip_counts = start_trip_counts.add(end_trip_counts, fill_value=0)
    total_trip_counts['total_activity'] = total_trip_counts.sum(axis=1)

    return total_trip_counts, filtered_df

def get_month_day_counts(year, month):
    """
    Given a year and month, return the number of weekdays (Mon-Fri),
    Saturdays, and Sundays in that month.
    """
    cal = calendar.monthcalendar(year, month)
    weekday_count = sum(1 for week in cal for i in range(5) if week[i] != 0)
    saturday_count = sum(1 for week in cal if week[5] != 0)
    sunday_count = sum(1 for week in cal if week[6] != 0)
    return weekday_count, saturday_count, sunday_count

def compute_daily_averages(filtered_df, valid_station_names):
    """
    Compute, for each station and month, the average trips per day for weekdays,
    Saturday, and Sunday. Both start and end trips are combined. The averages are
    computed by dividing the total trips on a given day type by the number of those days in the month.
    """
    # Compute day-of-week for start and end trips.
    filtered_df['start_day'] = filtered_df['started_at'].dt.day_name()
    filtered_df['end_day'] = filtered_df['ended_at'].dt.day_name()
    
    # Create DataFrames for start and end trips.
    start_df = filtered_df[filtered_df['start_station_name'].isin(valid_station_names)][['start_station_name', 'month', 'start_day']] \
        .rename(columns={'start_station_name': 'station', 'start_day': 'day'})
    end_df = filtered_df[filtered_df['end_station_name'].isin(valid_station_names)][['end_station_name', 'month', 'end_day']] \
        .rename(columns={'end_station_name': 'station', 'end_day': 'day'})
    
    # Combine start and end trip data.
    combined = pd.concat([start_df, end_df], ignore_index=True)
    
    # Group by station, month, and day-of-week.
    grouped = combined.groupby(['station', 'month', 'day']).size().unstack(fill_value=0)
    
    averages_list = []
    for (station, month), row in grouped.iterrows():
        try:
            year, mon = map(int, month.split('-'))
        except Exception as e:
            print(f"Error parsing month '{month}': {e}")
            continue
        weekdays_num, saturdays_num, sundays_num = get_month_day_counts(year, mon)
        # Sum trips for Monday through Friday.
        weekday_trip_count = sum(row.get(day, 0) for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        saturday_trip_count = row.get("Saturday", 0)
        sunday_trip_count = row.get("Sunday", 0)
        weekday_avg = weekday_trip_count / weekdays_num if weekdays_num > 0 else 0
        saturday_avg = saturday_trip_count / saturdays_num if saturdays_num > 0 else 0
        sunday_avg = sunday_trip_count / sundays_num if sundays_num > 0 else 0
        averages_list.append({
            'station': station,
            'month': month,
            'weekday_average': weekday_avg,
            'saturday_average': saturday_avg,
            'sunday_average': sunday_avg
        })
    averages_df = pd.DataFrame(averages_list)
    return averages_df

# -----------------------------
# Main function
# -----------------------------
def main():
    # Use file paths from the configuration.
    bikeshare_shp_path = BIKESHARE_SHP_PATH
    boundary_shp_path = BOUNDARY_SHP_PATH  # Optional; set to empty string or None to use all stops.
    csv_folder = CSV_FOLDER
    output_csv = OUTPUT_CSV
    output_xlsx = OUTPUT_XLSX

    # Load shapefiles.
    bikeshare_gdf, boundary_gdf = load_shapefiles(bikeshare_shp_path, boundary_shp_path)
    
    # Load and concatenate CSV trip data.
    combined_df = load_and_concatenate_csv(csv_folder)
    
    # Clip the bikeshare locations to the boundary if available; otherwise, use all stops.
    if boundary_gdf is not None:
        clipped_gdf = clip_shapefile(bikeshare_gdf, boundary_gdf)
        # Optional: Plot the clipped locations.
        plot_shapefile(boundary_gdf, clipped_gdf)
    else:
        clipped_gdf = bikeshare_gdf
    
    # Extract the valid station names from the study area.
    valid_station_names = clipped_gdf['NAME'].str.strip().unique()
    print(f"\nTotal unique station names within the study area: {len(valid_station_names)}")
    
    # --- Fixed corrections ---
    print("\nApplying fixed station name corrections...")
    combined_df = rename_stations(combined_df)
    
    # Recompute unmatched station names from CSV based on stations in the study area.
    unmatched_start = combined_df.loc[
        ~combined_df['start_station_name'].isin(valid_station_names), 'start_station_name'
    ].dropna().unique()
    unmatched_end = combined_df.loc[
        ~combined_df['end_station_name'].isin(valid_station_names), 'end_station_name'
    ].dropna().unique()
    
    print(f"\nUnique start station names not matching study area stops: {len(unmatched_start)}")
    print(unmatched_start)
    print(f"\nUnique end station names not matching study area stops: {len(unmatched_end)}")
    print(unmatched_end)
    
    # --- Interactive fuzzy match review ---
    update_mapping = interactive_update_station_names(unmatched_start, unmatched_end, valid_station_names, FUZZY_THRESHOLD)
    
    if update_mapping:
        print("Applying interactive station name updates:")
        print(update_mapping)
        combined_df['start_station_name'] = combined_df['start_station_name'].replace(update_mapping)
        combined_df['end_station_name'] = combined_df['end_station_name'].replace(update_mapping)
    else:
        print("No interactive updates applied.")
    
    # Filter trips and aggregate the monthly activity.
    total_trip_counts, filtered_df = filter_and_aggregate_trips(combined_df, clipped_gdf)
    print("\nAggregated trip counts by station and month:")
    print(total_trip_counts)
    
    # Export aggregated data to CSV.
    try:
        total_trip_counts.to_csv(output_csv)
        print(f"\nTotal monthly trip activity exported successfully to {output_csv}.")
    except Exception as e:
        print(f"Error exporting CSV: {e}")
    
    # --- Compute and export daily averages ---
    print("\nComputing daily averages (weekday, Saturday, Sunday) for each station by month...")
    averages_df = compute_daily_averages(filtered_df, valid_station_names)
    print("\nDaily averages:")
    print(averages_df)
    
    try:
        averages_df.to_excel(output_xlsx, index=False)
        print(f"\nDaily averages exported successfully to {output_xlsx}.")
    except Exception as e:
        print(f"Error exporting Excel: {e}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
