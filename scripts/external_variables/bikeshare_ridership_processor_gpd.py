"""
Processes Capital Bikeshare trip and station data to generate usage reports.

Generates monthly trip totals and average daily activity by station, with optional
clipping to a study area and interactive name reconciliation. Designed for adapting
to similar bikeshare datasets with minor modifications.

Inputs:
    - Capital Bikeshare station shapefile (BIKESHARE_SHP_PATH)
    - Optional boundary shapefile (BOUNDARY_SHP_PATH)
    - Monthly trip data CSVs (CSV_FOLDER)
    - Fuzzy match threshold for station reconciliation (FUZZY_THRESHOLD)

Outputs:
    - CSV: Monthly total trips per station (OUTPUT_CSV)
    - Excel: Daily averages per station/month (OUTPUT_XLSX)
    - Optional: Station map if a boundary is provided
"""

import calendar
import difflib
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BIKESHARE_SHP_PATH = r"C:\Your\File\Path\To\Bikeshare_Locations.shp"
# Set BOUNDARY_SHP_PATH to an empty string or None if you want to use all stops.
BOUNDARY_SHP_PATH = r"C:\Your\File\Path\To\Study_Area_Boundary.shp"
CSV_FOLDER = r"C:\Your\Folder\Path\To\Bikeshare_Trip_Data"
OUTPUT_CSV = r"C:\File\Path\To\Output\total_monthly_trip_activity_by_station.csv"
OUTPUT_XLSX = r"C:\File\Path\To\Output\trip_activity_averages_by_station.xlsx"

# New configuration: Users can set the fuzzy matching threshold here.
FUZZY_THRESHOLD = 0.8

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_shapefiles(bikeshare_shp_path, boundary_shp_path=None):
    """
    Load the bikeshare shapefile and optionally the boundary shapefile.
    """
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
    Find and load all CSV files in the specified folder, then
    concatenate them into a single DataFrame.
    """
    print(f"Loading and concatenating CSV files from folder: {csv_folder}")
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder}")
    csv_dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_data_frame = pd.concat(csv_dfs, ignore_index=True)
    return combined_data_frame


def clip_shapefile(bikeshare_gdf, boundary_gdf):
    """
    Reproject if needed and clip the bikeshare GeoDataFrame to the boundary.
    """
    if bikeshare_gdf.crs != boundary_gdf.crs:
        bikeshare_gdf = bikeshare_gdf.to_crs(boundary_gdf.crs)
    clipped_gdf = gpd.clip(bikeshare_gdf, boundary_gdf)
    return clipped_gdf


def plot_shapefile(boundary_gdf, clipped_gdf):
    """
    Plot the boundary and clipped bikeshare locations.
    """
    print("Plotting clipped shapefile...")
    _, axis_obj = plt.subplots(figsize=(10, 10))
    boundary_gdf.plot(ax=axis_obj, color="lightgray", edgecolor="black")
    clipped_gdf.plot(ax=axis_obj, color="red")
    axis_obj.set_title("Clipped Bikeshare Locations within Boundary")
    axis_obj.set_xlabel("Longitude")
    axis_obj.set_ylabel("Latitude")
    plt.show()


def find_close_matches(station_name, valid_station_names, threshold):
    """
    Return the best fuzzy match for a station name from the valid_station_names list.
    """
    matches = difflib.get_close_matches(
        station_name, valid_station_names, n=1, cutoff=threshold
    )
    return matches[0] if matches else None


def rename_stations(data_frame):
    """
    Rename specific station names based on a predefined mapping dictionary.
    """
    rename_dict = {
        "John McCormack Dr & Michigan Ave NE": "John McCormack Rd & Michigan Ave NE",
        "Radford St & Osage St": "Radford & Osage St",
        "Anacostia Roller Skating Pavillion": "Anacostia Roller Skating Pavilion",
        "Wilson Blvd. & N. Vermont St.": "Wilson Blvd & N Vermont St",
        "10th & Florida Ave NW": "10th St & Florida Ave NW",
        "14th & Rhode Island Ave NW": "14th St & Rhode Island Ave NW",
        "Kenilworth Terrace & Hayes St. NE": "Kenilworth Terr & Hayes St. NE",
        "11th & V st NW": "11th & V St NW",
        "Vaden Dr & Royal Victoria Dr/Providence Community Center": "Vaden Dr & Royal Victoria Dr/Jim Scott Cmty Ctr",
        "Westbranch Dr & Jones Branch Dr": "Westbranch & Jones Branch Dr",
        "21st NW & E St NW": "21st & E St NW",
    }

    data_frame["start_station_name"] = data_frame["start_station_name"].replace(
        rename_dict
    )
    data_frame["end_station_name"] = data_frame["end_station_name"].replace(rename_dict)
    return data_frame


def interactive_update_station_names(
    unmatched_start, unmatched_end, valid_station_names, threshold
):
    """
    For station names not matching any valid stop in the study area,
    interactively prompt the user with a fuzzy match suggestion.

    If the suggested name is different from the original, the user can confirm
    the update. Otherwise, a message is shown advising manual correction.
    Returns a mapping dictionary of updates.
    """
    update_mapping = {}
    combined_unmatched = set(unmatched_start).union(set(unmatched_end))

    print("\nInteractive review of fuzzy match suggestions:")
    print("(Type 'y' at the prompt if the proposed correction is acceptable.)\n")

    for station in combined_unmatched:
        suggestion = find_close_matches(station, valid_station_names, threshold)
        if suggestion:
            if suggestion == station:
                print(
                    f"Station '{station}' appears identical to its fuzzy match suggestion."
                )
                print(
                    "  -> If this is incorrect, please update manually in a mapping.\n"
                )
            else:
                print(f"Suggestion: '{station}' -> '{suggestion}'")
                try:
                    choice = input("Do you want to update this station name? (y/n): ")
                except (EOFError, KeyboardInterrupt) as exc:
                    print(f"Input interrupted: {exc}")
                    continue
                if choice.strip().lower() == "y":
                    update_mapping[station] = suggestion
                print()
        else:
            print(f"No fuzzy match found for station: '{station}'.")
    return update_mapping


def filter_valid_stations(combined_data_frame, clipped_gdf):
    """
    Filter trips that include at least one valid station in clipped_gdf,
    and return the filtered DataFrame plus the list of valid station names.
    """
    combined_data_frame["start_station_name"] = combined_data_frame[
        "start_station_name"
    ].str.strip()
    combined_data_frame["end_station_name"] = combined_data_frame[
        "end_station_name"
    ].str.strip()
    valid_stations = clipped_gdf["NAME"].str.strip().unique()

    # Filter trips where at least one station is in the valid list.
    filtered_data_frame = combined_data_frame[
        (combined_data_frame["start_station_name"].notna())
        & (combined_data_frame["end_station_name"].notna())
        & (
            (combined_data_frame["start_station_name"].isin(valid_stations))
            | (combined_data_frame["end_station_name"].isin(valid_stations))
        )
    ]

    print(f"Total trips after filtering: {len(filtered_data_frame)}")

    # Convert time columns to datetime and drop rows with invalid times.
    filtered_data_frame["started_at"] = pd.to_datetime(
        filtered_data_frame["started_at"], errors="coerce"
    )
    filtered_data_frame["ended_at"] = pd.to_datetime(
        filtered_data_frame["ended_at"], errors="coerce"
    )
    filtered_data_frame = filtered_data_frame.dropna(subset=["started_at", "ended_at"])

    # Extract day of week and month for future analysis.
    filtered_data_frame["day_of_week"] = filtered_data_frame["started_at"].dt.day_name()
    filtered_data_frame["month"] = (
        filtered_data_frame["started_at"].dt.to_period("M").astype(str)
    )

    return filtered_data_frame, valid_stations


def aggregate_monthly_trips(filtered_data_frame, valid_stations):
    """
    Given a filtered DataFrame, group trips by station-month and return
    monthly totals for each station plus a 'total_activity' column.
    """
    start_trip_counts = (
        filtered_data_frame[
            filtered_data_frame["start_station_name"].isin(valid_stations)
        ]
        .groupby(["start_station_name", "month"])
        .size()
        .unstack(fill_value=0)
    )
    end_trip_counts = (
        filtered_data_frame[
            filtered_data_frame["end_station_name"].isin(valid_stations)
        ]
        .groupby(["end_station_name", "month"])
        .size()
        .unstack(fill_value=0)
    )

    total_trip_counts = start_trip_counts.add(end_trip_counts, fill_value=0)
    total_trip_counts["total_activity"] = total_trip_counts.sum(axis=1)
    return total_trip_counts


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


def _create_average_record(station, month_str, row_data):
    """
    Helper function to compute average weekday/Saturday/Sunday trips for a single
    station-month row. Returns a dict or None if parsing fails.
    """
    try:
        year, mon = map(int, month_str.split("-"))
    except ValueError as exc:
        print(f"Error parsing month '{month_str}': {exc}")
        return None

    weekdays_num, saturdays_num, sundays_num = get_month_day_counts(year, mon)
    weekday_trip_count = sum(
        row_data.get(day, 0)
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    saturday_trip_count = row_data.get("Saturday", 0)
    sunday_trip_count = row_data.get("Sunday", 0)

    weekday_avg = weekday_trip_count / weekdays_num if weekdays_num > 0 else 0
    saturday_avg = saturday_trip_count / saturdays_num if saturdays_num > 0 else 0
    sunday_avg = sunday_trip_count / sundays_num if sundays_num > 0 else 0

    return {
        "station": station,
        "month": month_str,
        "weekday_average": weekday_avg,
        "saturday_average": saturday_avg,
        "sunday_average": sunday_avg,
    }


def compute_daily_averages(filtered_data_frame, valid_station_names):
    """
    Compute, for each station and month, the average trips per day
    for weekdays, Saturday, and Sunday.
    """
    filtered_data_frame["start_day"] = filtered_data_frame["started_at"].dt.day_name()
    filtered_data_frame["end_day"] = filtered_data_frame["ended_at"].dt.day_name()

    start_data_frame = filtered_data_frame[
        filtered_data_frame["start_station_name"].isin(valid_station_names)
    ][["start_station_name", "month", "start_day"]].rename(
        columns={"start_station_name": "station", "start_day": "day"}
    )

    end_data_frame = filtered_data_frame[
        filtered_data_frame["end_station_name"].isin(valid_station_names)
    ][["end_station_name", "month", "end_day"]].rename(
        columns={"end_station_name": "station", "end_day": "day"}
    )

    combined = pd.concat([start_data_frame, end_data_frame], ignore_index=True)
    grouped = combined.groupby(["station", "month", "day"]).size().unstack(fill_value=0)

    average_records = []
    for (station, month_str), row_data in grouped.iterrows():
        record = _create_average_record(station, month_str, row_data)
        if record is not None:
            average_records.append(record)

    averages_data_frame = pd.DataFrame(average_records)
    return averages_data_frame


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Process Capital Bikeshare data to generate trip activity reports.

    This function:
      - Loads bikeshare station data (and optionally a study area boundary)
        from shapefiles.
      - Loads and concatenates trip data from CSV files.
      - Applies fixed and interactive corrections to station names.
      - Filters trips to valid stations and aggregates monthly counts.
      - Computes daily average trip counts (weekday, Saturday, Sunday).
      - Exports the aggregated results to CSV and Excel files.

    Configuration (file paths, fuzzy matching threshold, etc.) is handled at the
    module level.
    """
    bikeshare_shp_path = BIKESHARE_SHP_PATH
    boundary_shp_path = BOUNDARY_SHP_PATH
    csv_folder = CSV_FOLDER
    output_csv = OUTPUT_CSV
    output_xlsx = OUTPUT_XLSX

    # Load shapefiles.
    bikeshare_gdf, boundary_gdf = load_shapefiles(bikeshare_shp_path, boundary_shp_path)

    # Load and concatenate CSV trip data.
    combined_data_frame = load_and_concatenate_csv(csv_folder)

    # Clip the bikeshare locations to the boundary if available; otherwise, use all stops.
    if boundary_gdf is not None:
        clipped_gdf = clip_shapefile(bikeshare_gdf, boundary_gdf)
        # Optional: Plot the clipped locations.
        plot_shapefile(boundary_gdf, clipped_gdf)
    else:
        clipped_gdf = bikeshare_gdf

    valid_station_names = clipped_gdf["NAME"].str.strip().unique()
    print(
        f"\nTotal unique station names within the study area: {len(valid_station_names)}"
    )

    print("\nApplying fixed station name corrections...")
    combined_data_frame = rename_stations(combined_data_frame)

    # Identify unmatched station names.
    unmatched_start = (
        combined_data_frame.loc[
            ~combined_data_frame["start_station_name"].isin(valid_station_names),
            "start_station_name",
        ]
        .dropna()
        .unique()
    )
    unmatched_end = (
        combined_data_frame.loc[
            ~combined_data_frame["end_station_name"].isin(valid_station_names),
            "end_station_name",
        ]
        .dropna()
        .unique()
    )

    print(
        f"\nUnique start station names not matching study area stops: {len(unmatched_start)}"
    )
    print(unmatched_start)
    print(
        f"\nUnique end station names not matching study area stops: {len(unmatched_end)}"
    )
    print(unmatched_end)

    # Interactive fuzzy match
    update_mapping = interactive_update_station_names(
        unmatched_start, unmatched_end, valid_station_names, FUZZY_THRESHOLD
    )

    if update_mapping:
        print("Applying interactive station name updates:")
        print(update_mapping)
        combined_data_frame["start_station_name"] = combined_data_frame[
            "start_station_name"
        ].replace(update_mapping)
        combined_data_frame["end_station_name"] = combined_data_frame[
            "end_station_name"
        ].replace(update_mapping)
    else:
        print("No interactive updates applied.")

    # Filter trips to valid stations and aggregate.
    filtered_data_frame, valid_stations = filter_valid_stations(
        combined_data_frame, clipped_gdf
    )
    total_trip_counts = aggregate_monthly_trips(filtered_data_frame, valid_stations)

    print("\nAggregated trip counts by station and month:")
    print(total_trip_counts)

    try:
        total_trip_counts.to_csv(output_csv)
        print(f"\nTotal monthly trip activity exported successfully to {output_csv}.")
    except OSError as exc:
        print(f"Error exporting CSV: {exc}")

    print(
        "\nComputing daily averages (weekday, Saturday, Sunday) for each station by month..."
    )
    averages_data_frame = compute_daily_averages(filtered_data_frame, valid_stations)

    print("\nDaily averages:")
    print(averages_data_frame)

    try:
        averages_data_frame.to_excel(output_xlsx, index=False)
        print(f"\nDaily averages exported successfully to {output_xlsx}.")
    except OSError as exc:
        print(f"Error exporting Excel: {exc}")


if __name__ == "__main__":
    main()
