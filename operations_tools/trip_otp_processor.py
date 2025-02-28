"""
Python script for filtering and aggregating running time and OTP data,
with configurable OTP aggregation methods, export options, branch filtering,
a recalculated total trips column, and percent calculations in both the aggregate
and individual route-level exports.
"""

import os
import pandas as pd
from typing import List, Optional

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------
INPUT_FILE = (
    r"\\Path\To\Your\CLEVER_Runtime_and_OTP_Trip_Level.csv"
)
OUTPUT_DIR = (
    r"\\Path\To\Your\Output_Folder"
)

# If ROUTES_OF_INTEREST is non-empty, only rows with Branch in this list are kept.
ROUTES_OF_INTEREST: List[str] = []

# If FILTER_OUT_BRANCHES is non-empty, rows with Branch in this list will be excluded.
FILTER_OUT_BRANCHES: List[str] = ['9999A ', '9999B', '9999C ', 'STRGH ', 'STRGR ', 'STRGW ']

# Set to True to aggregate by [Branch, Direction], or False to aggregate by [Branch] only.
AGGREGATE_BY_DIRECTION = True

# OTP Aggregation Options:
# If True, use aggregated counts (sum counts then compute percentages).
# If False (default), average the OTP percentages computed for each trip.
AGGREGATE_OTP_USING_COUNTS = False

# Export Options:
# If True, export the aggregate spreadsheet plus individual Excel files for each route
# (or route/direction). If False, only the aggregate spreadsheet is exported.
EXPORT_INDIVIDUAL_ROUTE_FILES = True

# Column names in the CSV (adjust if necessary)
SCHEDULED_COLUMN = "Average Scheduled Running Time"
ACTUAL_COLUMN = "Average Actual Running Time"
DEVIATION_COLUMN = "Average Running Time Deviation"  # in seconds
START_DELTA_COLUMN = "Average Start Delta"           # in HH:MM:SS
COUNT_TRIP_COLUMN = "Count Trip"
SUM_EARLY_COLUMN = "Sum # Early"
SUM_LATE_COLUMN = "Sum # Late"
SUM_ON_TIME_COLUMN = "Sum # On Time"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_time_string_to_minutes(time_str: str) -> float:
    """
    Convert a time string in the format HH:MM:SS to minutes (float) with one decimal place.
    Returns NaN if the format is invalid.
    Example: "0:31:00" -> 31.0
    """
    time_str = str(time_str).strip()
    if not time_str or ":" not in time_str:
        return float("nan")
    parts = time_str.split(":")
    if len(parts) != 3:
        return float("nan")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    except ValueError:
        return float("nan")
    total_minutes = hours * 60 + minutes + seconds / 60
    return round(total_minutes, 1)


def parse_seconds_to_minutes(value) -> float:
    """
    Convert a value in seconds (int or float) to minutes (float) with one decimal place.
    Returns NaN if the value is invalid.
    Example: -309 -> -5.2
    """
    try:
        seconds = float(value)
        return round(seconds / 60, 1)
    except (ValueError, TypeError):
        return float("nan")


def filter_by_branch(
    df: pd.DataFrame,
    include_branches: Optional[List[str]] = None,
    exclude_branches: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filters the DataFrame based on branch inclusion and exclusion lists.
    - If exclude_branches is provided and non-empty, rows with Branch in that list are removed.
    - If include_branches is provided and non-empty, only rows with Branch in that list are kept.
    """
    df_filtered = df.copy()
    if exclude_branches:
        df_filtered = df_filtered[~df_filtered["Branch"].isin(exclude_branches)]
    if include_branches:
        df_filtered = df_filtered[df_filtered["Branch"].isin(include_branches)]
    return df_filtered


def create_aggregations(df: pd.DataFrame, group_by_direction: bool = True) -> pd.DataFrame:
    """
    Create aggregations that average running times and OTP metrics over all trips
    for each route (Branch) or route/direction.

    OTP metrics include:
        - Calculated Total Trips
        - Early %
        - Late %
        - On-Time %

    The method for calculating OTP percentages is determined by the configuration:
      - If AGGREGATE_OTP_USING_COUNTS is True, counts are summed and percentages computed.
      - Otherwise, OTP percentages are computed per row using Calculated Total Trips and then averaged.

    Returns a DataFrame with the aggregated results.
    """
    group_cols = ["Branch"]
    if group_by_direction:
        group_cols.append("Direction")

    # Aggregation for running time columns (if available)
    agg_dict = {}
    if "Average Scheduled Running Time (min)" in df.columns:
        agg_dict["avg_scheduled_minutes"] = ("Average Scheduled Running Time (min)", "mean")
    if "Average Actual Running Time (min)" in df.columns:
        agg_dict["avg_actual_minutes"] = ("Average Actual Running Time (min)", "mean")
    if "Average Running Time Deviation (min)" in df.columns:
        agg_dict["avg_deviation_minutes"] = ("Average Running Time Deviation (min)", "mean")
    if "Average Start Delta (min)" in df.columns:
        agg_dict["avg_start_delta_minutes"] = ("Average Start Delta (min)", "mean")

    rt_agg = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    # OTP Aggregation (if the necessary columns exist)
    otp_columns = {SUM_EARLY_COLUMN, SUM_LATE_COLUMN, SUM_ON_TIME_COLUMN, "Calculated Total Trips"}
    if otp_columns.issubset(set(df.columns)):
        if AGGREGATE_OTP_USING_COUNTS:
            # Sum counts and then compute percentages using the calculated total trips.
            otp_agg = df.groupby(group_cols, dropna=False).agg(
                total_trips=("Calculated Total Trips", "sum"),
                total_early=(SUM_EARLY_COLUMN, "sum"),
                total_late=(SUM_LATE_COLUMN, "sum"),
                total_on_time=(SUM_ON_TIME_COLUMN, "sum"),
            ).reset_index()
            otp_agg["early_pct"] = otp_agg.apply(
                lambda row: round((row["total_early"] / row["total_trips"] * 100)
                                  if row["total_trips"] else float("nan"), 1), axis=1
            )
            otp_agg["late_pct"] = otp_agg.apply(
                lambda row: round((row["total_late"] / row["total_trips"] * 100)
                                  if row["total_trips"] else float("nan"), 1), axis=1
            )
            otp_agg["on_time_pct"] = otp_agg.apply(
                lambda row: round((row["total_on_time"] / row["total_trips"] * 100)
                                  if row["total_trips"] else float("nan"), 1), axis=1
            )
        else:
            # Compute OTP percentages per row using Calculated Total Trips and then average by group.
            df = df.copy()
            df["early_pct"] = df.apply(
                lambda row: (row[SUM_EARLY_COLUMN] / row["Calculated Total Trips"] * 100)
                if row["Calculated Total Trips"] else float("nan"),
                axis=1,
            )
            df["late_pct"] = df.apply(
                lambda row: (row[SUM_LATE_COLUMN] / row["Calculated Total Trips"] * 100)
                if row["Calculated Total Trips"] else float("nan"),
                axis=1,
            )
            df["on_time_pct"] = df.apply(
                lambda row: (row[SUM_ON_TIME_COLUMN] / row["Calculated Total Trips"] * 100)
                if row["Calculated Total Trips"] else float("nan"),
                axis=1,
            )
            otp_agg = df.groupby(group_cols, dropna=False).agg(
                total_trips=("Calculated Total Trips", "sum"),
                early_pct=("early_pct", "mean"),
                late_pct=("late_pct", "mean"),
                on_time_pct=("on_time_pct", "mean"),
            ).reset_index()
            otp_agg["early_pct"] = otp_agg["early_pct"].round(1)
            otp_agg["late_pct"] = otp_agg["late_pct"].round(1)
            otp_agg["on_time_pct"] = otp_agg["on_time_pct"].round(1)

        # Merge OTP aggregation with running time aggregation
        agg_df = pd.merge(rt_agg, otp_agg, on=group_cols, how="left")
    else:
        agg_df = rt_agg

    # Final rounding for running time columns
    for col in agg_df.columns:
        if col.startswith("avg_"):
            agg_df[col] = agg_df[col].round(1)

    return agg_df


def export_individual_files(agg_df: pd.DataFrame, group_by_direction: bool, output_dir: str) -> None:
    """
    Export individual Excel files for each unique route (or route/direction)
    from the aggregated DataFrame, which includes OTP percentages.
    """
    group_keys = ["Branch", "Direction"] if group_by_direction else ["Branch"]

    groups = agg_df.groupby(group_keys)
    for name, group in groups:
        if isinstance(name, tuple):
            file_name = "_".join(str(x).strip().replace(" ", "_") for x in name)
        else:
            file_name = str(name).strip().replace(" ", "_")
        file_path = os.path.join(output_dir, f"{file_name}_aggregated.xlsx")
        group.to_excel(file_path, index=False)
        print(f"Exported individual aggregated file for group {name} to: {file_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    # 1. Read the CSV
    df = pd.read_csv(INPUT_FILE)

    # 2. Parse the relevant time columns to minutes
    if SCHEDULED_COLUMN in df.columns:
        df["Average Scheduled Running Time (min)"] = df[SCHEDULED_COLUMN].apply(parse_time_string_to_minutes)
    if ACTUAL_COLUMN in df.columns:
        df["Average Actual Running Time (min)"] = df[ACTUAL_COLUMN].apply(parse_time_string_to_minutes)
    if DEVIATION_COLUMN in df.columns:
        df["Average Running Time Deviation (min)"] = df[DEVIATION_COLUMN].apply(parse_seconds_to_minutes)
    if START_DELTA_COLUMN in df.columns:
        df["Average Start Delta (min)"] = df[START_DELTA_COLUMN].apply(parse_time_string_to_minutes)

    # 3. Calculate total trips using the sum of early, late, and on-time counts
    otp_cols = {SUM_EARLY_COLUMN, SUM_LATE_COLUMN, SUM_ON_TIME_COLUMN}
    if otp_cols.issubset(set(df.columns)):
        df["Calculated Total Trips"] = df[SUM_EARLY_COLUMN] + df[SUM_LATE_COLUMN] + df[SUM_ON_TIME_COLUMN]

    # 4. Filter the data by branch if needed
    df_filtered = filter_by_branch(
        df, include_branches=ROUTES_OF_INTEREST, exclude_branches=FILTER_OUT_BRANCHES
    )

    # 5. Create the aggregated DataFrame (includes OTP percentages)
    agg_df = create_aggregations(df_filtered, group_by_direction=AGGREGATE_BY_DIRECTION)

    # 6. Write the aggregated results to an Excel file
    aggregated_output_path = os.path.join(OUTPUT_DIR, "route_level_aggregations.xlsx")
    agg_df.to_excel(aggregated_output_path, index=False)
    print(f"Aggregation results saved to: {aggregated_output_path}")

    # 7. Optionally, export individual aggregated route files (with percent calculations)
    if EXPORT_INDIVIDUAL_ROUTE_FILES:
        export_individual_files(agg_df, AGGREGATE_BY_DIRECTION, OUTPUT_DIR)


if __name__ == "__main__":
    main()
