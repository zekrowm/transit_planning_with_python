"""Concatenate bike share trip data and export summaries by station.

This script processes raw Capital Bikeshare trip data stored as monthly CSV files
and generates station-level usage summaries. It performs the following steps:

1. Loads and concatenates all CSV files in a given folder.
2. Optionally filters trips by a specified start and end month (inclusive).
3. Computes monthly trip totals per station (start + end).
4. Computes overall trip totals per station across the entire filtered period.
5. Exports:
   - A combined trip dataset (`OUTPUT_CSV`)
   - A monthly station usage summary (`SUMMARY_CSV`)
   - A period-wide station summary (`PERIOD_SUMMARY_CSV`)

Outputs:
    - Combined trip file (OUTPUT_CSV)
    - Monthly summary per station (SUMMARY_CSV)
    - Total activity per station for the filtered period (PERIOD_SUMMARY_CSV)

This script is designed to support exploratory analysis, reporting, and data
preparation for downstream visualization or modeling.
"""

import glob
import os
from typing import List

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Folder containing the input CSV files.
CSV_FOLDER: str = r"/path/to/your/csv_folder"

# Path for the output, combined CSV.
OUTPUT_CSV: str = r"/path/to/your/output/combined.csv"

# Path for the output, station monthly summary CSV.
SUMMARY_CSV: str = r"/path/to/your/output/station_monthly_summary.csv"

# Path for the output, overall period summary CSV.
PERIOD_SUMMARY_CSV: str = r"/path/to/your/output/period_summary.csv"

# Optional time period filter (inclusive) in 'YYYY-MM' format.
# Leave blank ("") to disable filtering.
FILTER_START: str = ""  # e.g. "2024-01"
FILTER_END: str = ""  # e.g. "2024-03"


# =============================================================================
# FUNCTIONS
# =============================================================================


def find_csv_files(folder: str) -> List[str]:
    """Find all CSV files in the specified folder.

    Args:
        folder (str): Path to the folder containing CSV files.

    Returns:
        List[str]: A list of full file paths for every '*.csv' in `folder`.
    """
    pattern = os.path.join(folder, "*.csv")
    return glob.glob(pattern)


def load_and_concatenate_csv(files: List[str]) -> pd.DataFrame:
    """Load multiple CSV files and concatenate them into one DataFrame.

    Args:
        files (List[str]): List of CSV file paths to load.

    Returns:
        pandas.DataFrame: A single DataFrame containing all rows from `files`.
    """
    data_frames = [pd.read_csv(fp) for fp in files]
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


# =============================================================================
# CONCATENATION SECTION
# =============================================================================


def concatenate_csvs(csv_folder: str) -> pd.DataFrame:
    """Locate CSVs in a folder and merge them into one DataFrame.

    Args:
        csv_folder (str): Directory path where CSV files are stored.

    Returns:
        pandas.DataFrame: The concatenated DataFrame of all CSVs.
    """
    print(f"Searching for CSV files in: {csv_folder}")
    csv_files = find_csv_files(csv_folder)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{csv_folder}'")
    print(f"Found {len(csv_files)} file(s); loading and concatenating now...")
    combined_df = load_and_concatenate_csv(csv_files)
    print(f"Concatenation complete: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    return combined_df


# =============================================================================
# SUMMARY CALCULATIONS
# =============================================================================


def aggregate_monthly_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly trip totals per station (start + end).

    Args:
        df (pandas.DataFrame): Trip DataFrame with 'started_at',
            'start_station_name', and 'end_station_name'.

    Returns:
        pandas.DataFrame: Indexed by station name, with one column per month
        and a 'total_activity' column summing starts and ends.
    """
    # Ensure datetime
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"])

    # Extract month period for grouping
    df["month"] = df["started_at"].dt.to_period("M").astype(str)

    # Count trips by origin station
    start_counts = (
        df[df["start_station_name"].notna()]
        .groupby(["start_station_name", "month"])
        .size()
        .unstack(fill_value=0)
    )

    # Count trips by destination station
    end_counts = (
        df[df["end_station_name"].notna()]
        .groupby(["end_station_name", "month"])
        .size()
        .unstack(fill_value=0)
    )
    end_counts.index.rename("start_station_name", inplace=True)

    # Sum origin + destination counts
    total_counts = start_counts.add(end_counts, fill_value=0)
    total_counts["total_activity"] = total_counts.sum(axis=1)

    total_counts.index.name = "station"
    return total_counts


def aggregate_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total trip activity per station over the entire filtered period.

    Args:
        df (pandas.DataFrame): Trip DataFrame with 'start_station_name'
            and 'end_station_name'.

    Returns:
        pandas.DataFrame: Indexed by station, with a single
        'total_activity' column summing starts and ends.
    """
    # Count all start events
    start_counts = df["start_station_name"].value_counts().rename("start_trips")

    # Count all end events
    end_counts = df["end_station_name"].value_counts().rename("end_trips")

    # Combine and sum
    total = start_counts.add(end_counts, fill_value=0).rename("total_activity")

    period_df = pd.concat([start_counts, end_counts, total], axis=1).fillna(0)
    period_df.index.name = "station"
    return period_df


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run concatenation, apply optional filtering, and export all reports."""
    # 1) Concatenate all CSVs
    combined_df = concatenate_csvs(CSV_FOLDER)

    # 2) Export the raw combined data
    print(f"Writing combined CSV to: {OUTPUT_CSV}")
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print("Combined CSV export complete.\n")

    # 3) Prepare data for summaries
    df_for_summary = combined_df.copy()
    df_for_summary["started_at"] = pd.to_datetime(df_for_summary["started_at"], errors="coerce")

    # Apply time‐period filtering if configured
    if FILTER_START:
        start_period = pd.Period(FILTER_START, "M")
        df_for_summary = df_for_summary[
            df_for_summary["started_at"].dt.to_period("M") >= start_period
        ]
        print(f"Filtered out trips before {FILTER_START}.")

    if FILTER_END:
        end_period = pd.Period(FILTER_END, "M")
        df_for_summary = df_for_summary[
            df_for_summary["started_at"].dt.to_period("M") <= end_period
        ]
        print(f"Filtered out trips after {FILTER_END}.")

    print(f"Trips remaining for summary: {len(df_for_summary)}\n")

    # 4) Monthly summary (only months in period)
    print("Computing monthly trip totals per station...")
    monthly_summary = aggregate_monthly_trips(df_for_summary)
    print(f"Writing station monthly summary CSV to: {SUMMARY_CSV}")
    monthly_summary.to_csv(SUMMARY_CSV)
    print("Monthly summary export complete.\n")

    # 5) Period summary (overall totals)
    print("Computing overall trip summary for filtered period...")
    period_summary = aggregate_period_summary(df_for_summary)
    print(f"Writing period summary CSV to: {PERIOD_SUMMARY_CSV}")
    period_summary.to_csv(PERIOD_SUMMARY_CSV)
    print("Period summary export complete.")


if __name__ == "__main__":
    main()
