"""
Script Name:
        historical_otp_data_processor.py

Purpose:
        Processes transit agency on-time performance (OTP) (and running time)
        data. It calculates key performance metrics, identifies operational
        deviations from schedules, and optionally generates OTP trend plots
        for individual routes and directions.

Inputs:
        1. Primary CSV data file ('INPUT_FILE') containing detailed trip-level
           data including scheduled/actual runtimes, start times, and OTP counts.
        2. (If plotting enabled) Processed CSV data file ('OTP_FILE_PATH') for
           OTP plotting, typically derived from the primary input.
        3. Configuration constants defined in the script for file paths,
           filters (e.g., ROUTE_FILTER), thresholds (e.g.,
           DEVIATION_THRESHOLD_SECONDS), and plot appearance.

Outputs:
        1. Excel files detailing processed running time data, including
           calculated deviations and problem flags, saved to 'OUTPUT_DIR'.
           One file per route and one consolidated file.
        2. (If plotting enabled via RUN_PLOTTING=True) JPEG image files of
           OTP trends over time, one for each route and direction, saved
           to 'OTP_OUTPUT_DIR'.
        3. Status messages and error logs printed to the console.
        
Dependencies:
        pandas, matplotlib, os, sys, datetime
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------
#                             CONFIGURATION SECTION
# ------------------------------------------------------------------------------

# =========================
#  1) Running Time Data
# =========================
INPUT_FILE = r"\\Path\To\Your\Runtime and OTP Trip Level Data.csv"
OUTPUT_DIR = r"\\Path\To\Your\Processed\Data\Output"

# Route Filter Configuration
# Set to a list of routes to include (e.g., ['101', '202']) or an empty list [] for all routes.
ROUTE_FILTER = []
# ROUTE_FILTER = ['101', '202']  # Example usage

# Threshold for Deviation in Seconds (5 minutes)
DEVIATION_THRESHOLD_SECONDS = 5 * 60  # 300 seconds

# =========================
#  2) OTP Data & Plotting
# =========================
RUN_PLOTTING = True  # Set to False to skip OTP plotting entirely

OTP_FILE_PATH = r"\\Path\To\Your\Output\Runtime and OTP Trip Level Data_processed.csv"
OTP_OUTPUT_DIR = r"\\Path\To\Your\Plot\Output"

# Date Range for Full Range of Months (for OTP plots)
FULL_RANGE_START = "2024-01-01"
FULL_RANGE_END = "2024-12-31"
DATE_FORMAT = "%Y-%b-%d"  # Format for converting to datetime

# Y-axis Limits Adjustment (for OTP plots)
Y_MIN_CUTOFF = 50  # Ensure the minimum y-axis starts at 50%
Y_MARGIN = 5  # Margin added/subtracted to y-axis limits

# Percentage Thresholds for Horizontal Lines
PERCENTAGE_LEVELS = [95, 85, 75]
LINE_COLOR = "r"
LINE_STYLE = "--"
LINE_WIDTH = 0.7
TEXT_FONT_SIZE = 9

# Plot Appearance
FIG_SIZE = (12, 6)
MARKER_STYLE = "o"
LINE_STYLE_PLOT = "-"
LABEL_ROTATION = 45

# Columns Configuration for OTP Data
ROUTE_COLUMN = "Route"
DIRECTION_COLUMN = "Direction"
MONTH_COLUMN = "Month"
YEAR_COLUMN = "Year"
ON_TIME_COLUMN = "Sum # On Time"
EARLY_COLUMN = "Sum # Early"
LATE_COLUMN = "Sum # Late"

# ------------------------------------------------------------------------------
#                         HELPER FUNCTIONS (RUN-TIME DATA)
# ------------------------------------------------------------------------------


def time_str_to_seconds(time_str):
    """
    Convert a time string in 'H:MM:SS' or 'HH:MM:SS' format to total seconds.

    If the format is 'HH:MM', assumes zero seconds.
    """
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes = parts
            seconds = "0"
        else:
            return None
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except Exception as exc:
        print(f"Error parsing time string '{time_str}': {exc}")
        return None


def parse_start_time(time_str):
    """
    Parse 'Scheduled Start Time (HH:MM)' into seconds since midnight.
    """
    try:
        dt = datetime.strptime(time_str.strip(), "%H:%M")
        return dt.hour * 3600 + dt.minute * 60
    except Exception as exc:
        print(f"Error parsing start time '{time_str}': {exc}")
        return None


def parse_start_delta(delta_str):
    """
    Parse 'Average Start Delta' (could have formats like '-H:MM:SS', etc.) into total seconds.
    """
    try:
        delta_str = delta_str.strip()
        negative = False
        if delta_str.startswith("-"):
            negative = True
            delta_str = delta_str[1:]
        parts = delta_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes = parts
            seconds = "0"
        else:
            return None
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        return -total_seconds if negative else total_seconds
    except Exception as exc:
        print(f"Error parsing start delta '{delta_str}': {exc}")
        return None


def process_runtime_data():
    """
    Process running time data from a configured CSV file, filter by route (optional),
    compute deviations, and export results to per-route Excel files and one
    processed Excel file.
    """
    # Configuration references
    input_file = INPUT_FILE
    output_dir = OUTPUT_DIR
    route_filter = ROUTE_FILTER
    threshold = DEVIATION_THRESHOLD_SECONDS

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as exc:
            print(f"Error creating output directory '{output_dir}': {exc}")
            return

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as exc:
        print(f"Error reading CSV file '{input_file}': {exc}")
        return

    # Display initial data sample
    print("Initial Data Sample:")
    print(df.head())

    # Apply Route Filter if specified
    if route_filter:
        original_count = len(df)
        df = df[df["Route"].astype(str).isin([str(route) for route in route_filter])]
        filtered_count = len(df)
        print(f"Route Filter Applied: {route_filter}")
        print(f"Rows before filter: {original_count}, after filter: {filtered_count}")
    else:
        print("No Route Filter Applied: Including all routes")

    # Parse 'Scheduled Start Time (HH:MM)' to seconds
    df["Scheduled_Start_Seconds"] = df["Scheduled Start Time (HH:MM)"].apply(parse_start_time)

    # Parse 'Average Scheduled Running Time' and 'Average Actual Running Time' to seconds
    df["Avg_Scheduled_Running_Time_sec"] = df["Average Scheduled Running Time"].apply(
        time_str_to_seconds
    )
    df["Avg_Actual_Running_Time_sec"] = df["Average Actual Running Time"].apply(time_str_to_seconds)

    # Parse 'Average Start Delta' to seconds
    df["Avg_Start_Delta_sec"] = df["Average Start Delta"].apply(parse_start_delta)

    # Recalculate deviations
    df["Recalc_Running_Time_Deviation_sec"] = (
        df["Avg_Actual_Running_Time_sec"] - df["Avg_Scheduled_Running_Time_sec"]
    )

    # Convert deviations from seconds to minutes
    df["Running Time Deviation (minutes)"] = df["Recalc_Running_Time_Deviation_sec"] / 60
    df["Start Time Deviation (minutes)"] = df["Avg_Start_Delta_sec"] / 60

    # Determine runtime problems (>5 mins off scheduled runtime)
    df["Runtime_Problem"] = df["Recalc_Running_Time_Deviation_sec"].abs() > threshold

    # Determine start time problems (>5 mins off)
    df["Start_Time_Problem"] = df["Avg_Start_Delta_sec"].abs() > threshold

    # Create a new 'Total_Trips'
    df["Total_Trips"] = df["Sum # Early"] + df["Sum # Late"] + df["Sum # On Time"]

    # Compute Pct_Early, Pct_Late, Pct_On_Time
    df["Pct_Early"] = df.apply(
        lambda row: (
            100.0 * row["Sum # Early"] / row["Total_Trips"] if row["Total_Trips"] != 0 else 0
        ),
        axis=1,
    )
    df["Pct_Late"] = df.apply(
        lambda row: (
            100.0 * row["Sum # Late"] / row["Total_Trips"] if row["Total_Trips"] != 0 else 0
        ),
        axis=1,
    )
    df["Pct_On_Time"] = df.apply(
        lambda row: (
            100.0 * row["Sum # On Time"] / row["Total_Trips"] if row["Total_Trips"] != 0 else 0
        ),
        axis=1,
    )

    # Round to 1 decimal place and convert to proportion
    df["Pct_Early"] = (df["Pct_Early"].round(1)) / 100
    df["Pct_Late"] = (df["Pct_Late"].round(1)) / 100
    df["Pct_On_Time"] = (df["Pct_On_Time"].round(1)) / 100

    # Loop through each route and save an Excel file
    routes = df["Route"].unique()
    for route in routes:
        route_data = df[df["Route"] == route]
        excel_filename = f"runtime_problems_route_{route}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)

        output_columns = [
            "Route",
            "Direction",
            "Scheduled Start Time (HH:MM)",
            "Count Trip",
            "Sum # Early",
            "Sum # Late",
            "Sum # On Time",
            "Total_Trips",
            "Pct_Early",
            "Pct_Late",
            "Pct_On_Time",
            "Average Scheduled Running Time",
            "Average Actual Running Time",
            "Recalc_Running_Time_Deviation_sec",
            "Running Time Deviation (minutes)",
            "Average Start Delta",
            "Avg_Start_Delta_sec",
            "Start Time Deviation (minutes)",
            "Runtime_Problem",
            "Start_Time_Problem",
        ]

        try:
            route_data.to_excel(excel_path, columns=output_columns, index=False)
            print(f"Entries for Route {route} (all trips) saved to '{excel_path}'")
        except Exception as exc:
            print(f"Error saving Excel file '{excel_path}': {exc}")

    # Save the complete processed data to a single Excel file
    processed_excel_filename = "processed_runtime_data.xlsx"
    processed_excel_path = os.path.join(output_dir, processed_excel_filename)
    try:
        df.to_excel(processed_excel_path, index=False)
        print(f"Processed data with deviations saved to '{processed_excel_path}'")
    except Exception as exc:
        print(f"Error saving processed Excel file '{processed_excel_path}': {exc}")


# ------------------------------------------------------------------------------
#                         HELPER FUNCTIONS (OTP DATA)
# ------------------------------------------------------------------------------


def load_csv(file_path):
    """
    Load CSV data into a pandas DataFrame with error handling.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file loaded successfully: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found:\n{file_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        sys.exit(1)
    except Exception as exc:
        print(f"An unexpected error occurred while loading the CSV: {exc}")
        sys.exit(1)


def define_full_date_range(start, end):
    """
    Define a full range of months between start and end dates.
    """
    return pd.date_range(start=start, end=end, freq="MS")


def clean_route_column(df, route_col):
    """
    Clean the 'Route' column by extracting text before '-' and stripping spaces.
    """
    df[route_col] = df[route_col].astype(str).str.split("-").str[0].str.strip()
    return df


def ensure_numeric_columns(df, columns):
    """
    Ensure specified columns are numeric integers, handling non-numeric entries.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    return df


def calculate_sum_and_percentages(df, on_time, early, late):
    """
    Calculate 'Sum All Trips' and percentage columns.
    """
    df["Sum All Trips"] = df[on_time] + df[early] + df[late]

    # Avoid division by zero by replacing 0 with NaN
    df["Percent On Time"] = (df[on_time] / df["Sum All Trips"].replace(0, pd.NA)) * 100
    df["Percent Early"] = (df[early] / df["Sum All Trips"].replace(0, pd.NA)) * 100
    df["Percent Late"] = (df[late] / df["Sum All Trips"].replace(0, pd.NA)) * 100

    # Fill NaN values with 0
    df["Percent On Time"] = df["Percent On Time"].fillna(0)
    df["Percent Early"] = df["Percent Early"].fillna(0)
    df["Percent Late"] = df["Percent Late"].fillna(0)

    return df


def create_date_column(df, year_col, month_col, date_format):
    """
    Create a 'Date' column from 'Year' and 'Month' columns.
    """
    df["YY-MM"] = df[year_col].astype(str) + "-" + df[month_col].str[:3].str.capitalize()

    try:
        # Force day = '01' for each month
        df["Date"] = pd.to_datetime(df["YY-MM"] + "-01", format=date_format)
        print("Date column created successfully.")
    except Exception as exc:
        print(f"Error converting 'YY-MM' to datetime: {exc}")
        sys.exit(1)

    return df


def sort_dataframe_by_date(df, date_col):
    """
    Sort the DataFrame by the 'Date' column.
    """
    return df.sort_values(by=date_col)


def determine_y_axis_limits(df, y_min_cutoff, y_margin):
    """
    Determine global y-axis limits based on the 'Percent On Time' column.
    """
    global_min_otp = df["Percent On Time"].min()
    global_max_otp = df["Percent On Time"].max()

    global_y_min = max(global_min_otp - y_margin, y_min_cutoff)
    global_y_max = global_max_otp + y_margin
    return global_y_min, global_y_max


def create_output_directory(output_dir):
    """
    Create the output directory if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")


def calculate_variability(df, route_col, direction_col):
    """
    Calculate the variability (range) of 'Percent On Time' for each Route/Direction.
    """
    range_df = df.groupby([route_col, direction_col])["Percent On Time"].agg(["max", "min"])
    range_df["Range"] = range_df["max"] - range_df["min"]
    range_df = range_df.sort_values(by="Range", ascending=False)
    return range_df


def sanitize_filename(text):
    """
    Sanitize text to create a safe filename by replacing or removing
    problematic characters.
    """
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_on_time_percentage(group, route, direction, config, y_limits, full_range):
    """
    Plot the 'Percent On Time' over time for a specific Route and Direction.
    """
    plt.figure(figsize=config["FIG_SIZE"])

    # Plot the data
    plt.plot(
        group["Date"],
        group["Percent On Time"],
        marker=config["MARKER_STYLE"],
        linestyle=config["LINE_STYLE_PLOT"],
        label=f"{route} - {direction}",
    )

    # Set y-axis limits
    plt.ylim(y_limits)

    # Set x-axis limits
    plt.xlim(full_range[0], full_range[-1])

    # Add x-axis labels
    plt.xticks(
        ticks=full_range,
        labels=[date.strftime("%b %Y") for date in full_range],
        rotation=config["LABEL_ROTATION"],
    )

    # Add horizontal lines and labels
    for level in config["PERCENTAGE_LEVELS"]:
        plt.axhline(
            y=level,
            color=config["LINE_COLOR"],
            linestyle=config["LINE_STYLE"],
            linewidth=config["LINE_WIDTH"],
        )
        plt.text(
            full_range[-1],
            level,
            f"{level}%",
            color=config["LINE_COLOR"],
            fontsize=config["TEXT_FONT_SIZE"],
            verticalalignment="center",
            horizontalalignment="left",
        )

    # Set labels and title
    plt.xlabel("Date")
    plt.ylabel("Percent On Time")
    plt.title(f"On Time Percentage for Route {route} - Direction {direction}")
    plt.tight_layout()

    # Save the plot
    safe_route = sanitize_filename(route)
    safe_direction = sanitize_filename(direction)
    plot_filename = f"{safe_route}_{safe_direction}_on_time_percentage.jpeg"
    plot_path = os.path.join(config["OUTPUT_DIR"], plot_filename)

    try:
        plt.savefig(plot_path, format="jpeg")
        print(f"Plot saved: {plot_path}")
    except Exception as exc:
        print(f"Error saving plot for Route {route} - Direction {direction}: {exc}")
    finally:
        plt.close()


def process_otp_data():
    """
    Process the OTP data and generate on-time-percentage plots (one per Route/Direction).
    """
    # Load the CSV file
    otp_df = load_csv(OTP_FILE_PATH)

    # Define the full range of months
    full_range = define_full_date_range(FULL_RANGE_START, FULL_RANGE_END)

    # Clean the "Route" column
    otp_df = clean_route_column(otp_df, ROUTE_COLUMN)

    # Ensure columns used in calculations are numeric
    otp_df = ensure_numeric_columns(otp_df, [ON_TIME_COLUMN, EARLY_COLUMN, LATE_COLUMN])

    # Create "Sum All Trips" and percentage columns
    otp_df = calculate_sum_and_percentages(otp_df, ON_TIME_COLUMN, EARLY_COLUMN, LATE_COLUMN)

    # Create "Date" column using existing "Year" and "Month" columns
    otp_df = create_date_column(otp_df, YEAR_COLUMN, MONTH_COLUMN, DATE_FORMAT)

    # Sort DataFrame by the new "Date" column
    otp_df = sort_dataframe_by_date(otp_df, "Date")

    # Determine global y-axis limits
    y_limits = determine_y_axis_limits(otp_df, Y_MIN_CUTOFF, Y_MARGIN)

    # Create output directory for plots
    create_output_directory(OTP_OUTPUT_DIR)

    # Calculate variability in OTP
    variability_df = calculate_variability(otp_df, ROUTE_COLUMN, DIRECTION_COLUMN)
    print("\nRoutes with the highest variability in OTP:")
    print(variability_df)

    # Configuration dictionary for plotting
    plot_config = {
        "FIG_SIZE": FIG_SIZE,
        "MARKER_STYLE": MARKER_STYLE,
        "LINE_STYLE_PLOT": LINE_STYLE_PLOT,
        "LABEL_ROTATION": LABEL_ROTATION,
        "PERCENTAGE_LEVELS": PERCENTAGE_LEVELS,
        "LINE_COLOR": LINE_COLOR,
        "LINE_STYLE": LINE_STYLE,
        "LINE_WIDTH": LINE_WIDTH,
        "TEXT_FONT_SIZE": TEXT_FONT_SIZE,
        "OUTPUT_DIR": OTP_OUTPUT_DIR,
    }

    # Plot On-Time Percentage over Time for each Route and Direction
    for (route, direction), group in otp_df.groupby([ROUTE_COLUMN, DIRECTION_COLUMN]):
        plot_on_time_percentage(group, route, direction, plot_config, y_limits, full_range)

    print("\nOTP processing and plotting complete. Plots saved to:", OTP_OUTPUT_DIR)


# ------------------------------------------------------------------------------
#                                   MAIN
# ------------------------------------------------------------------------------


def main():
    """
    Main pipeline function. Processes Running Time Data, then
    optionally processes (and plots) OTP Data based on configuration.
    """
    print("=== Processing Running Time Data ===")
    process_runtime_data()

    if RUN_PLOTTING:
        print("\n=== Processing OTP Data and Generating Plots ===")
        process_otp_data()
    else:
        print("\nPlotting is disabled. Skipping OTP processing.")


if __name__ == "__main__":
    main()
