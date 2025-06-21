"""Process historical on-time-performance (OTP) and running-time data.

The module provides a CLI-driven pipeline that

* parses trip-level CSV exports,
* flags trips whose actual running time deviates from schedule,
* writes per-route Excel files summarising issues, and
* (optionally) produces monthly JPEG trend plots of OTP by route and
  direction.

Typical usage is via ArcPro or standalone Python notebook.

Configuration is set via the constants in the *CONFIGURATION* section.

Outputs
-------
└── <OUTPUT_DIR>/
    ├── processed_runtime_data.xlsx
    ├── runtime_problems_route_<ROUTE>.xlsx
    └── <plot_output_dir>/  # if RUN_PLOTTING = True
        └── <route>_<dir>_on_time_percentage.jpeg
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# -------------------------
#  1) Running Time Data
# -------------------------
INPUT_FILE = r"\\Path\To\Your\Runtime and OTP Trip Level Data.csv"
OUTPUT_DIR = r"\\Path\To\Your\Processed\Data\Output"

# Route Filter Configuration
# Set to a list of routes to include (e.g., ['101', '202']) or an empty list [] for all routes.
ROUTE_FILTER = []
# ROUTE_FILTER = ['101', '202']  # Example usage

# Threshold for Deviation in Seconds (5 minutes)
DEVIATION_THRESHOLD_SECONDS = 5 * 60  # 300 seconds

# -------------------------
#  2) OTP Data & Plotting
# -------------------------
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

# =============================================================================
# FUNCTIONS
# =============================================================================


def time_str_to_seconds(time_str: str) -> int | None:
    """Convert an H:MM(:SS) time string to seconds.

    Args:
        time_str: Time value in ``H:MM:SS``, ``HH:MM:SS`` or ``HH:MM``
            format. Seconds are assumed to be ``00`` if omitted.

    Returns:
        Total number of seconds represented by *time_str*, or
        ``None`` if the value cannot be parsed.

    Examples:
        >>> time_str_to_seconds("1:23:45")
        5025
        >>> time_str_to_seconds("09:15")
        33300
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


def parse_start_time(time_str: str) -> int | None:
    """Convert a scheduled ``HH:MM`` start time to seconds since midnight.

    Args:
        time_str: Scheduled start time in 24-hour ``HH:MM`` format.

    Returns:
        Seconds since midnight, or ``None`` when parsing fails.
    """
    try:
        dt = datetime.strptime(time_str.strip(), "%H:%M")
        return dt.hour * 3600 + dt.minute * 60
    except Exception as exc:
        print(f"Error parsing start time '{time_str}': {exc}")
        return None


def parse_start_delta(delta_str: str) -> int | None:
    """Convert a signed start-delta string to seconds.

    Args:
        delta_str: Offset such as ``-0:02:30``, ``+00:03``, or ``4:15``.

    Returns:
        Signed number of seconds (negative for early departures), or
        ``None`` if the format is invalid.
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


def process_runtime_data() -> None:
    """Parse runtime CSV, flag deviations, and write Excel output.

    Steps performed:

    1. Load *INPUT_FILE*.
    2. Optionally filter to routes in *ROUTE_FILTER*.
    3. Convert schedule, actual, and delta fields to seconds.
    4. Flag trips whose start time or running time deviates by more
       than *DEVIATION_THRESHOLD_SECONDS*.
    5. Compute percentage early/late/on-time statistics.
    6. Export a per-route Excel file and a consolidated workbook.

    Returns:
        None.  Results are written to disk.
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
    df["Scheduled_Start_Seconds"] = df["Scheduled Start Time (HH:MM)"].apply(
        parse_start_time
    )

    # Parse 'Average Scheduled Running Time' and 'Average Actual Running Time' to seconds
    df["Avg_Scheduled_Running_Time_sec"] = df["Average Scheduled Running Time"].apply(
        time_str_to_seconds
    )
    df["Avg_Actual_Running_Time_sec"] = df["Average Actual Running Time"].apply(
        time_str_to_seconds
    )

    # Parse 'Average Start Delta' to seconds
    df["Avg_Start_Delta_sec"] = df["Average Start Delta"].apply(parse_start_delta)

    # Recalculate deviations
    df["Recalc_Running_Time_Deviation_sec"] = (
        df["Avg_Actual_Running_Time_sec"] - df["Avg_Scheduled_Running_Time_sec"]
    )

    # Convert deviations from seconds to minutes
    df["Running Time Deviation (minutes)"] = (
        df["Recalc_Running_Time_Deviation_sec"] / 60
    )
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
            100.0 * row["Sum # Early"] / row["Total_Trips"]
            if row["Total_Trips"] != 0
            else 0
        ),
        axis=1,
    )
    df["Pct_Late"] = df.apply(
        lambda row: (
            100.0 * row["Sum # Late"] / row["Total_Trips"]
            if row["Total_Trips"] != 0
            else 0
        ),
        axis=1,
    )
    df["Pct_On_Time"] = df.apply(
        lambda row: (
            100.0 * row["Sum # On Time"] / row["Total_Trips"]
            if row["Total_Trips"] != 0
            else 0
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


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file with robust error handling.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        DataFrame containing the file contents.

    Raises:
        SystemExit: If the file is missing, empty, or unreadable.
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


def define_full_date_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return a monthly DatetimeIndex spanning *start* through *end*.

    Args:
        start: First date (inclusive).
        end:   Last date (inclusive).

    Returns:
        A ``DatetimeIndex`` with frequency ``'MS'`` (month-start).
    """
    return pd.date_range(start=start, end=end, freq="MS")


def clean_route_column(df: pd.DataFrame, route_col: str) -> pd.DataFrame:
    """Standardise the *Route* column in-place.

    Removes text following the first “-” and strips whitespace, so
    “10A-Otis St” becomes “10A”.

    Args:
        df: DataFrame containing the route column.
        route_col: Name of the column to clean.

    Returns:
        The mutated DataFrame (returned for convenience).
    """
    df[route_col] = df[route_col].astype(str).str.split("-").str[0].str.strip()
    return df


def ensure_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Force selected columns to integer dtype, coercing errors to 0.

    Args:
        df: Input DataFrame.
        columns: Columns to coerce.

    Returns:
        The mutated DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    return df


def calculate_sum_and_percentages(
    df: pd.DataFrame,
    on_time: str,
    early: str,
    late: str,
) -> pd.DataFrame:
    """Add “Sum All Trips” and percentage OTP columns.

    Args:
        df: DataFrame containing OTP counts.
        on_time: Column with on-time counts.
        early: Column with early counts.
        late: Column with late counts.

    Returns:
        DataFrame with added *Sum All Trips*, *Percent On Time*,
        *Percent Early*, and *Percent Late* columns.
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


def create_date_column(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    date_format: str,
) -> pd.DataFrame:
    """Combine *Year* and *Month* into a datetime column.

    Args:
        df: DataFrame to modify.
        year_col: Column containing four-digit year values.
        month_col: Column containing month names or abbreviations.
        date_format: ``strftime`` pattern used for parsing.

    Returns:
        The DataFrame with new ``'Date'`` and ``'YY-MM'`` columns.

    Raises:
        SystemExit: If the combined strings cannot be parsed.
    """
    df["YY-MM"] = (
        df[year_col].astype(str) + "-" + df[month_col].str[:3].str.capitalize()
    )

    try:
        # Force day = '01' for each month
        df["Date"] = pd.to_datetime(df["YY-MM"] + "-01", format=date_format)
        print("Date column created successfully.")
    except Exception as exc:
        print(f"Error converting 'YY-MM' to datetime: {exc}")
        sys.exit(1)

    return df


def sort_dataframe_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return *df* sorted by the specified datetime column."""
    return df.sort_values(by=date_col)


def determine_y_axis_limits(
    df: pd.DataFrame,
    y_min_cutoff: int,
    y_margin: int,
) -> tuple[float, float]:
    """Compute global y-axis limits for OTP plots.

    Args:
        df: DataFrame containing ``'Percent On Time'``.
        y_min_cutoff: Lower bound below which the axis should not start.
        y_margin: Padding to add above and below the observed range.

    Returns:
        Tuple ``(y_min, y_max)``.
    """
    global_min_otp = df["Percent On Time"].min()
    global_max_otp = df["Percent On Time"].max()

    global_y_min = max(global_min_otp - y_margin, y_min_cutoff)
    global_y_max = global_max_otp + y_margin
    return global_y_min, global_y_max


def create_output_directory(output_dir: str) -> None:
    """Ensure *output_dir* exists, creating parents as needed.

    Args:
        output_dir: Directory path.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")


def calculate_variability(
    df: pd.DataFrame,
    route_col: str,
    direction_col: str,
) -> pd.DataFrame:
    """Calculate OTP variability (range) by route and direction.

    Args:
        df: DataFrame with percentage OTP data.
        route_col: Column identifying the route.
        direction_col: Column identifying the direction.

    Returns:
        DataFrame sorted by descending range, with columns
        ``['max', 'min', 'Range']``.
    """
    range_df = df.groupby([route_col, direction_col])["Percent On Time"].agg(
        ["max", "min"]
    )
    range_df["Range"] = range_df["max"] - range_df["min"]
    range_df = range_df.sort_values(by="Range", ascending=False)
    return range_df


def sanitize_filename(text: str) -> str:
    """Return *text* made safe for use as a filename.
    
    Replaces slashes and spaces with underscores.

    Args:
        text: Raw filename candidate.

    Returns:
        Sanitised filename.
    """
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_on_time_percentage(
    group: pd.DataFrame,
    route: str,
    direction: str,
    config: dict[str, object],
    y_limits: tuple[float, float],
    full_range: pd.DatetimeIndex,
) -> None:
    """Create and save a JPEG OTP trend plot for a route-direction pair.

    Args:
        group: Sub-DataFrame for one route and direction.
        route: Route identifier.
        direction: Direction name (e.g., “Eastbound”).
        config: Dictionary of plotting constants.
        y_limits: Global y-axis limits produced by
            :func:`determine_y_axis_limits`.
        full_range: Continuous monthly index for the x-axis.

    Returns:
        None.  The figure is written to *config['OUTPUT_DIR']*.
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


def process_otp_data() -> None:
    """End-to-end OTP processing and plotting pipeline.

    Loads OTP CSV, cleans and augments the data set, produces plots for
    every Route/Direction combination, and prints variability rankings.

    Returns:
        None.  Plots and console output are side-effects.
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
    otp_df = calculate_sum_and_percentages(
        otp_df, ON_TIME_COLUMN, EARLY_COLUMN, LATE_COLUMN
    )

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
        plot_on_time_percentage(
            group, route, direction, plot_config, y_limits, full_range
        )

    print("\nOTP processing and plotting complete. Plots saved to:", OTP_OUTPUT_DIR)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Execute the full pipeline based on configuration flags.

    Returns:
        None.
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
