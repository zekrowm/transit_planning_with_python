"""
noaa_weather_processing.py

A module for loading, processing, and analyzing NOAA weather data.

Data can be obtained here:
https://www.ncei.noaa.gov/cdo-web/search?

This script performs the following tasks:
- Loads NOAA weather data from a specified CSV file.
- Renames DataFrame columns to more descriptive and PEP-8 compliant names.
- Processes date columns to extract additional temporal information.
- Classifies days based on defined poor weather criteria.
- Creates daily and monthly summaries of the weather data.
- Adds human-readable day names to summaries.
- Saves the processed data and summaries to the designated output folder.
- Displays the processed data for verification.

Configuration settings such as file paths, column mappings, and weather criteria are centralized
in the configuration section for easy modification and scalability.
"""

import pandas as pd
from pathlib import Path
import os

# ===========================
# Configuration Section
# ===========================

# Path to the NOAA weather data CSV file
FILE_PATH = Path(
    "Path/To/Your/noaa_weather_data.csv"
)

# Path to the output folder where processed data and summaries will be saved
OUTPUT_FOLDER = Path(
    "Path/To/Your/output_folder"
)

# Mapping of original column names to new, PEP-8 compliant column names
COLUMN_MAPPING = {
    'STATION': 'weather_station',
    'NAME': 'station_name',
    'DATE': 'date',
    'AWND': 'average_wind_speed',
    'PGTM': 'peak_gust_time',
    'PRCP': 'precipitation',
    'SNOW': 'snowfall',
    'SNWD': 'snow_depth',
    'TAVG': 'average_temperature',
    'TMAX': 'maximum_temperature',
    'TMIN': 'minimum_temperature',
    'WDF2': 'direction_of_fastest_2_minute_wind',
    'WDF5': 'direction_of_fastest_5_minute_wind',
    'WT01': 'fog_ice_fog_freezing_fog',
    'WT02': 'heavy_fog_heavy_freezing_fog',
    'WT03': 'thunder',
    'WT04': 'ice_pellets_sleet_snow_pellets_small_hail',
    'WT05': 'hail_may_include_small_hail',
    'WT06': 'glaze_or_rime',
    'WT07': 'dust_volcanic_ash_blowing_dust_blowing_sand',
    'WT08': 'smoke_or_haze',
    'WT09': 'blowing_drifting_snow',
    'WT11': 'high_or_damaging_winds',
    'WT13': 'mist',
    'WT14': 'drizzle',
    'WT15': 'freezing_drizzle',
    'WT16': 'rain_may_include_freezing_rain_drizzle',
    'WT17': 'freezing_rain',
    'WT18': 'snow_snow_pellets_snow_grains_ice_crystals',
    'WT21': 'ground_fog',
    'WT22': 'ice_fog_or_freezing_fog',
    'WESD': 'water_equivalent_of_snow_on_ground',
    'WSF2': 'fastest_2_minute_wind_speed',
    'WSF5': 'fastest_5_second_wind_speed',
    'FMTM': 'time_of_fastest_mile_or_1_minute_wind',
}

# Poor weather criteria thresholds
POOR_WEATHER_CRITERIA = {
    'average_wind_speed': 15.0,         # in mph (example threshold)
    'maximum_temperature': 95.0,        # in Fahrenheit (example threshold)
    'minimum_temperature': 20.0,        # in Fahrenheit (example threshold)
    'snowfall': 2.0                      # in inches (example threshold)
}

# ===========================
# Function Definitions
# ===========================


def load_weather_data(file_path: Path) -> pd.DataFrame:
    """
    Load NOAA weather data from a CSV file.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the weather data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: The file {file_path} could not be parsed.")
        raise


def rename_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Rename columns to more descriptive and PEP-8 compliant names.

    Args:
        df (pd.DataFrame): The original weather DataFrame.
        column_mapping (dict): A dictionary mapping original column names to new names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df.rename(columns=column_mapping, inplace=True)
    print("Columns renamed successfully.")
    return df


def process_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'date' column to datetime and extract year, month, day, and day of week.

    Args:
        df (pd.DataFrame): The weather DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with additional date-related columns.
    """
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
        df['day_name'] = df['date'].dt.day_name()  # Human-readable day name
        print("Date columns processed successfully.")
        return df
    except KeyError:
        print("Error: 'date' column not found in DataFrame.")
        raise
    except pd.errors.OutOfBoundsDatetime:
        print("Error: Dates in 'date' column are out of bounds.")
        raise


def classify_poor_weather(df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """
    Classify each day as a poor weather day based on defined criteria.

    Args:
        df (pd.DataFrame): The processed weather DataFrame.
        criteria (dict): A dictionary with thresholds for poor weather classification.

    Returns:
        pd.DataFrame: DataFrame with an additional 'poor_weather' column.
    """
    conditions = (
        (df['average_wind_speed'] > criteria['average_wind_speed']) |
        (df['maximum_temperature'] > criteria['maximum_temperature']) |
        (df['minimum_temperature'] < criteria['minimum_temperature']) |
        (df['snowfall'] > criteria['snowfall'])
    )
    df['poor_weather'] = conditions
    print("Poor weather days classified successfully.")
    return df


def create_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a daily summary of the weather data.

    Args:
        df (pd.DataFrame): The processed weather DataFrame.

    Returns:
        pd.DataFrame: Daily summary DataFrame with aggregated metrics and day names.
    """
    try:
        # Group by 'date' and aggregate relevant metrics
        daily_summary = df.groupby('date').agg({
            'average_wind_speed': 'mean',
            'precipitation': 'sum',
            'snowfall': 'sum',
            'snow_depth': 'mean',
            'average_temperature': 'mean',
            'maximum_temperature': 'max',
            'minimum_temperature': 'min',
            'poor_weather': 'max'  # If any condition met, mark as True
        }).reset_index()

        # Add 'day_name' by merging with original DataFrame
        day_names = df[['date', 'day_name']].drop_duplicates()
        daily_summary = daily_summary.merge(day_names, on='date', how='left')

        print("Daily summary created successfully.")
        return daily_summary
    except KeyError as e:
        print(f"Error: Missing expected column {e} in DataFrame.")
        raise


def create_monthly_poor_weather_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly summary of poor weather days categorized by day type.

    Args:
        df (pd.DataFrame): The processed weather DataFrame with 'poor_weather' and 'day_of_week' columns.

    Returns:
        pd.DataFrame: Monthly summary DataFrame with counts of poor weather days per day type.
    """
    try:
        # Define day types
        def categorize_day(day_num):
            if day_num < 5:
                return 'Weekday'
            elif day_num == 5:
                return 'Saturday'
            else:
                return 'Sunday'

        df['day_type'] = df['day_of_week'].apply(categorize_day)

        # Filter poor weather days
        poor_weather_df = df[df['poor_weather']]

        # Group by year_month and day_type and count poor weather days
        monthly_summary = poor_weather_df.groupby(['year_month', 'day_type']).size().reset_index(name='poor_weather_days')

        # Pivot the table to have day types as columns
        monthly_summary_pivot = monthly_summary.pivot(index='year_month', columns='day_type', values='poor_weather_days').fillna(0).reset_index()

        # Ensure all day types are present
        for day_type in ['Weekday', 'Saturday', 'Sunday']:
            if day_type not in monthly_summary_pivot.columns:
                monthly_summary_pivot[day_type] = 0

        # Convert year_month to string for easier handling
        monthly_summary_pivot['year_month'] = monthly_summary_pivot['year_month'].astype(str)

        print("Monthly poor weather summary created successfully.")
        return monthly_summary_pivot
    except KeyError as e:
        print(f"Error: Missing expected column {e} in DataFrame.")
        raise


def save_dataframe(df: pd.DataFrame, output_path: Path, filename: str):
    """
    Save a DataFrame to a CSV file in the specified output path.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (Path): The directory where the file will be saved.
        filename (str): The name of the output CSV file.
    """
    try:
        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        file_full_path = output_path / filename
        df.to_csv(file_full_path, index=False)
        print(f"Data saved to {file_full_path}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")
        raise


def main():
    """
    Main function to load, process, summarize, and save NOAA weather data.
    """
    # Load the data
    weather_df = load_weather_data(FILE_PATH)

    # Rename the columns
    weather_df = rename_columns(weather_df, COLUMN_MAPPING)

    # Process date columns
    weather_df = process_date_columns(weather_df)

    # Classify poor weather days
    weather_df = classify_poor_weather(weather_df, POOR_WEATHER_CRITERIA)

    # Create daily summary
    daily_summary_df = create_daily_summary(weather_df)

    # Create monthly poor weather summary
    monthly_poor_weather_summary_df = create_monthly_poor_weather_summary(weather_df)

    # Save processed data
    save_dataframe(weather_df, OUTPUT_FOLDER, 'processed_weather_data.csv')

    # Save daily summary
    save_dataframe(daily_summary_df, OUTPUT_FOLDER, 'daily_summary.csv')

    # Save monthly poor weather summary
    save_dataframe(monthly_poor_weather_summary_df, OUTPUT_FOLDER, 'monthly_poor_weather_summary.csv')

    # Display the first few rows of the processed DataFrame
    print("\nProcessed Weather Data:")
    print(weather_df.head())

    # Display the first few rows of the daily summary
    print("\nDaily Summary:")
    print(daily_summary_df.head())

    # Display the first few rows of the monthly poor weather summary
    print("\nMonthly Poor Weather Summary:")
    print(monthly_poor_weather_summary_df.head())


if __name__ == "__main__":
    main()
