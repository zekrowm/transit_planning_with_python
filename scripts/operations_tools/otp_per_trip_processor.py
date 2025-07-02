"""Aggregate and analyze trip-level on-time-performance (OTP) data for transit routes.

The script ingests a CSV exported from CLEVER (or a similar AVL system),
applies optional route filtering, converts duration columns to minutes,
computes OTP percentages, and writes:

- A system-wide Excel workbook of aggregate metrics.
- (Optionally) one workbook per route or route–direction pair.

Typical workflow
1. Adjust the constants in the *CONFIGURATION* section.
2. Execute the script in ArcPro’s Python console or a Jupyter notebook.
3. Review the exported workbooks in ``OUTPUT_DIR``.

Note:
    The heavy lifting happens in :func:`create_aggregations`; if you need
    a different aggregation grain (e.g. by service period), modify the
    ``group_cols`` list inside that function.
"""

import os
from typing import List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"\\Path\To\Your\CLEVER_Runtime_and_OTP_Trip_Level.csv"
OUTPUT_DIR = r"\\Path\To\Your\Output_Folder"

# If ROUTES_OF_INTEREST is non-empty, only rows with Branch in this list are kept.
ROUTES_OF_INTEREST: List[str] = []

# If FILTER_OUT_BRANCHES is non-empty, rows with Branch in this list will be excluded.
FILTER_OUT_BRANCHES: List[str] = [
    "9999A ",
    "9999B",
    "9999C ",
    "STRGH ",
    "STRGR ",
    "STRGW ",
]

# Set to True to aggregate by [Branch, Direction], or False to aggregate by [Branch] only.
AGGREGATE_BY_DIRECTION = True

# OTP performance standard (percentage)
# Any group with on_time_pct < this value will be flagged
OTP_STANDARD_PCT = 85  # Adjust if needed

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
START_DELTA_COLUMN = "Average Start Delta"  # in HH:MM:SS
COUNT_TRIP_COLUMN = "Count Trip"
SUM_EARLY_COLUMN = "Sum # Early"
SUM_LATE_COLUMN = "Sum # Late"
SUM_ON_TIME_COLUMN = "Sum # On Time"

# =============================================================================
# FUNCTIONS
# =============================================================================


def parse_time_string_to_minutes(time_str: str) -> float:
    """Convert an ``HH:MM:SS`` time string to minutes.

    The computation preserves one decimal place of precision. Values that
    cannot be parsed (e.g. malformed strings, missing data) return
    ``nan`` to allow downstream arithmetic without raising exceptions.

    Args:
        time_str: A time string such as ``"0:31:00"`` or ``"02:05:30"``.
                  Surrounding whitespace is ignored.

    Returns:
        Total minutes represented by the input, e.g. ``31.0`` or
        ``125.5``. Non-conformant inputs yield ``float('nan')``.
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
    """Convert a raw seconds value to minutes.

    Args:
        value: Seconds as ``int``, ``float``, or numeric **str**.

    Returns:
        The equivalent minutes (rounded to one decimal) or
        ``float('nan')`` if ``value`` is not numeric.
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
    """Subset records by the ``Branch`` column.

    Args:
        df: The original trip-level DataFrame.
        include_branches: If provided, retain *only* rows whose
            ``Branch`` appears in this list.
        exclude_branches: If provided, drop rows whose ``Branch`` appears
            in this list **before** applying *include_branches*.

    Returns:
        A new DataFrame reflecting the requested filters. The original
        ``df`` is left unmodified.

    Notes:
        Inclusion and exclusion are case-sensitive and depend on
        *exact* string matches. Trim or standardise branch values
        upstream if necessary.
    """
    df_filtered = df.copy()
    if exclude_branches:
        df_filtered = df_filtered[~df_filtered["Branch"].isin(exclude_branches)]
    if include_branches:
        df_filtered = df_filtered[df_filtered["Branch"].isin(include_branches)]
    return df_filtered


def create_aggregations(
    df: pd.DataFrame, group_by_direction: bool = True
) -> pd.DataFrame:
    """Aggregate running-time and OTP metrics.

    Args:
        df: Pre-processed (and, if desired, filtered) trip-level data.
        group_by_direction: Whether to keep *Direction* separate.

    Returns:
        A tidy, one-row-per-group DataFrame ready for export.

    Raises:
        KeyError: If required columns (e.g. OTP counts) are missing.
    """
    # -----------------------------------------------------------------
    # Which columns define a “group”?
    # -----------------------------------------------------------------
    group_cols: list[str] = ["Branch"]
    if group_by_direction:
        group_cols.append("Direction")

    # -----------------------------------------------------------------
    # Running-time metrics
    # -----------------------------------------------------------------
    agg_dict: dict[str, tuple[str, str]] = {}
    if "Average Scheduled Running Time (min)" in df.columns:
        agg_dict["avg_scheduled_minutes"] = (
            "Average Scheduled Running Time (min)",
            "mean",
        )
    if "Average Actual Running Time (min)" in df.columns:
        agg_dict["avg_actual_minutes"] = ("Average Actual Running Time (min)", "mean")
    if "Average Running Time Deviation (min)" in df.columns:
        agg_dict["avg_deviation_minutes"] = (
            "Average Running Time Deviation (min)",
            "mean",
        )
    if "Average Start Delta (min)" in df.columns:
        agg_dict["avg_start_delta_minutes"] = ("Average Start Delta (min)", "mean")

    rt_agg = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    # -----------------------------------------------------------------
    # OTP metrics
    # -----------------------------------------------------------------
    otp_cols = {
        SUM_EARLY_COLUMN,
        SUM_LATE_COLUMN,
        SUM_ON_TIME_COLUMN,
        "Calculated Total Trips",
    }
    if otp_cols.issubset(df.columns):
        if AGGREGATE_OTP_USING_COUNTS:
            # Sum counts first, then compute percentages
            otp_agg = (
                df.groupby(group_cols, dropna=False)
                .agg(
                    total_trips=("Calculated Total Trips", "sum"),
                    total_early=(SUM_EARLY_COLUMN, "sum"),
                    total_late=(SUM_LATE_COLUMN, "sum"),
                    total_on_time=(SUM_ON_TIME_COLUMN, "sum"),
                )
                .reset_index()
            )
            otp_agg["early_pct"] = (
                otp_agg["total_early"] / otp_agg["total_trips"] * 100
            ).round(1)
            otp_agg["late_pct"] = (
                otp_agg["total_late"] / otp_agg["total_trips"] * 100
            ).round(1)
            otp_agg["on_time_pct"] = (
                otp_agg["total_on_time"] / otp_agg["total_trips"] * 100
            ).round(1)
        else:
            # Compute percentages row-by-row, then average
            df_pct = df.copy()
            df_pct["early_pct"] = (
                df_pct[SUM_EARLY_COLUMN] / df_pct["Calculated Total Trips"] * 100
            )
            df_pct["late_pct"] = (
                df_pct[SUM_LATE_COLUMN] / df_pct["Calculated Total Trips"] * 100
            )
            df_pct["on_time_pct"] = (
                df_pct[SUM_ON_TIME_COLUMN] / df_pct["Calculated Total Trips"] * 100
            )
            otp_agg = (
                df_pct.groupby(group_cols, dropna=False)
                .agg(
                    total_trips=("Calculated Total Trips", "sum"),
                    early_pct=("early_pct", "mean"),
                    late_pct=("late_pct", "mean"),
                    on_time_pct=("on_time_pct", "mean"),
                )
                .reset_index()
            ).round({"early_pct": 1, "late_pct": 1, "on_time_pct": 1})

        # Merge running-time and OTP metrics
        agg_df = pd.merge(rt_agg, otp_agg, on=group_cols, how="left")
    else:
        agg_df = rt_agg

    # -----------------------------------------------------------------
    # Final rounding for running-time columns
    # -----------------------------------------------------------------
    for col in agg_df.filter(like="avg_").columns:
        agg_df[col] = agg_df[col].round(1)

    # -----------------------------------------------------------------
    # Flag groups below the OTP performance standard
    # -----------------------------------------------------------------
    if "on_time_pct" in agg_df.columns:
        agg_df["otp_below_standard"] = agg_df["on_time_pct"] < OTP_STANDARD_PCT

    return agg_df


def export_individual_files(
    agg_df: pd.DataFrame, group_by_direction: bool, output_dir: str
) -> None:
    """Write one Excel workbook per group.

    Args:
        agg_df: The DataFrame returned by :func:`create_aggregations`.
        group_by_direction: Mirrors the argument of the same name in
            :func:`create_aggregations`; ensures filenames align with
            aggregation grain.
        output_dir: Destination folder (must already exist).

    Returns:
        None. Files are written to disk as a side effect.
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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the full OTP analysis pipeline.

    Workflow:

    1. Read the raw CLEVER CSV (``INPUT_FILE``).
    2. Convert duration fields to minutes.
    3. Derive ``Calculated Total Trips`` from OTP counts.
    4. Filter branches per *CONFIGURATION* lists.
    5. Aggregate metrics with :func:`create_aggregations`.
    6. Export the system-wide workbook plus (optionally) per-route files.

    Returns:
        None. The primary output is a set of Excel workbooks.

    Raises:
        FileNotFoundError: If ``INPUT_FILE`` cannot be read.
        OSError
    """
    # 1. Read the CSV
    df = pd.read_csv(INPUT_FILE)

    # 2. Parse the relevant time columns to minutes
    if SCHEDULED_COLUMN in df.columns:
        df["Average Scheduled Running Time (min)"] = df[SCHEDULED_COLUMN].apply(
            parse_time_string_to_minutes
        )
    if ACTUAL_COLUMN in df.columns:
        df["Average Actual Running Time (min)"] = df[ACTUAL_COLUMN].apply(
            parse_time_string_to_minutes
        )
    if DEVIATION_COLUMN in df.columns:
        df["Average Running Time Deviation (min)"] = df[DEVIATION_COLUMN].apply(
            parse_seconds_to_minutes
        )
    if START_DELTA_COLUMN in df.columns:
        df["Average Start Delta (min)"] = df[START_DELTA_COLUMN].apply(
            parse_time_string_to_minutes
        )

    # 3. Calculate total trips using the sum of early, late, and on-time counts
    otp_cols = {SUM_EARLY_COLUMN, SUM_LATE_COLUMN, SUM_ON_TIME_COLUMN}
    if otp_cols.issubset(set(df.columns)):
        df["Calculated Total Trips"] = (
            df[SUM_EARLY_COLUMN] + df[SUM_LATE_COLUMN] + df[SUM_ON_TIME_COLUMN]
        )

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
