"""
Script Name:
    runtime_per_trip_processor.py

Purpose:
    Analyse scheduled vs. actual running times for bus routes,
    flagging trips and route–direction pairs whose deviations
    exceed user-defined percent or minute thresholds, and
    exporting both trip-level and summary CSVs.

    The script now auto-detects multiple column variants (HH:MM:SS,
    already-in-minutes, etc.) so it can run on the *same* file that
    the OTP trip processor uses—no manual renaming required.

Inputs:
    1. A trip-level CSV defined by INPUT_FILE
       (e.g. r"\\Path\\To\\Runtime Trip Level - 05-05-2025.csv")
    2. Configuration constants in the CONFIGURATION section below.

Outputs:
    1. trip_level_with_flags.csv   – trip rows + new metrics & flags
    2. route_direction_summary.csv – aggregated summary with flags

Dependencies:
    pandas
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"\\Path\To\Runtime Trip Level - MM-DD-YYYY.csv"
OUTPUT_DIR = r"\\Path\To\Your\Output_Folder"

PCT_THRESHOLD: float = 10.0    # flag if |% deviation| > this
MIN_THRESHOLD: float = 10.0    # flag if |deviation|   > this (minutes)

WEIGHTED_AVERAGE: bool = False  # False = simple mean (default)

TRIP_CSV_NAME = "trip_level_with_flags.csv"
SUMMARY_CSV_NAME = "route_direction_summary.csv"

# =============================================================================
# COLUMN NORMALISATION
# =============================================================================

COLUMN_ALIASES: dict[str, list[str]] = {
    # logical name → preferred + fall-backs in incoming files
    "route":                 ["Route", "Branch"],
    "direction":             ["Direction"],
    "scheduled_time_raw":    ["Average Scheduled Running Time"],
    "scheduled_time_min":    ["Average Scheduled Running Time (min)"],
    "actual_time_raw":       ["Average Actual Running Time"],
    "actual_time_min":       ["Average Actual Running Time (min)"],
    "deviation_time_min":    ["Average Running Time Deviation (min)"],
    "count_trip":            ["Count Trip"],
}

def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    """Return the first column name that exists in *df* (or None)."""
    for n in names:
        if n in df.columns:
            return n
    return None


# =============================================================================
# FUNCTIONS
# =============================================================================

def parse_time_to_minutes(value) -> float:
    """
    Convert HH:MM:SS (or H:MM) strings to a float number of minutes.
    If *value* is already numeric, return it unchanged (rounded to 1 dp).
    """
    # value already numeric?
    if isinstance(value, (int, float)) and not pd.isna(value):
        return round(float(value), 1)

    if pd.isna(value):
        return float("nan")

    txt = str(value).strip()
    if ":" not in txt:
        return float("nan")

    parts = txt.split(":")
    try:
        hours, minutes, seconds = (parts + ["0"])[:3]
        total_sec = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        return round(total_sec / 60, 1)
    except ValueError:
        return float("nan")


def flag_row(dev_min: float, sched_min: float) -> bool:
    """True if deviation exceeds either configured threshold."""
    if pd.isna(dev_min) or pd.isna(sched_min) or sched_min == 0:
        return False
    pct_diff = abs(dev_min) / sched_min * 100
    return abs(dev_min) > MIN_THRESHOLD or pct_diff > PCT_THRESHOLD


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean with safe divide-by-zero handling."""
    w_sum = weights.sum()
    return (values * weights).sum() / w_sum if w_sum else float("nan")


def prepare_trip_level(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns and compute deviation / flags, agnostic to source file."""
    # ---------------------------------------------------------------------
    # Route name
    # ---------------------------------------------------------------------
    df["route_name"] = (
        df["route"]
        .astype(str)
        .str.split("-", n=1)
        .str[0]
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    # ---------------------------------------------------------------------
    # Minutes conversion
    # ---------------------------------------------------------------------
    def to_minutes(v):
        return parse_time_to_minutes(v)

    # Scheduled
    if "scheduled_time_min" in df.columns:
        df["scheduled_minutes"] = df["scheduled_time_min"].apply(to_minutes)
    else:
        df["scheduled_minutes"] = df["scheduled_time_raw"].apply(to_minutes)

    # Actual
    if "actual_time_min" in df.columns:
        df["actual_minutes"] = df["actual_time_min"].apply(to_minutes)
    else:
        df["actual_minutes"] = df["actual_time_raw"].apply(to_minutes)

    # Deviation
    if "deviation_time_min" in df.columns:
        df["deviation_minutes"] = df["deviation_time_min"].apply(to_minutes)
    else:
        df["deviation_minutes"] = df["actual_minutes"] - df["scheduled_minutes"]

    # Percent deviation
    df["percent_deviation"] = (
        df["deviation_minutes"] / df["scheduled_minutes"] * 100
    ).round(1)

    # Flag
    df["flag_runtime_issue"] = df.apply(
        lambda r: flag_row(r["deviation_minutes"], r["scheduled_minutes"]), axis=1
    )

    return df


def build_summary(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to Route × Direction and flag summary rows."""
    group_cols = ["route_name", "direction"]

    if WEIGHTED_AVERAGE:
        summary = (
            trips.groupby(group_cols, dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "count_trips": g["count_trip"].sum(),
                        "scheduled_avg": weighted_mean(
                            g["scheduled_minutes"], g["count_trip"]
                        ),
                        "actual_avg": weighted_mean(
                            g["actual_minutes"], g["count_trip"]
                        ),
                        "deviation_avg": weighted_mean(
                            g["deviation_minutes"], g["count_trip"]
                        ),
                    }
                )
            )
            .reset_index()
        )
    else:
        summary = (
            trips.groupby(group_cols, dropna=False)
            .agg(
                count_trips=("count_trip", "sum"),
                scheduled_avg=("scheduled_minutes", "mean"),
                actual_avg=("actual_minutes", "mean"),
                deviation_avg=("deviation_minutes", "mean"),
            )
            .reset_index()
        )

    summary[["scheduled_avg", "actual_avg", "deviation_avg"]] = summary[
        ["scheduled_avg", "actual_avg", "deviation_avg"]
    ].round(1)
    summary["percent_deviation"] = (
        summary["deviation_avg"] / summary["scheduled_avg"] * 100
    ).round(1)

    summary["flag_runtime_issue"] = summary.apply(
        lambda r: flag_row(r["deviation_avg"], r["scheduled_avg"]), axis=1
    )

    return summary


def run_pipeline() -> Tuple[str, str]:
    """Load, normalise, process, save.  Returns (trip_path, summary_path)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Read CSV and normalise column names
    # ---------------------------------------------------------------------
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw.columns = df_raw.columns.str.strip()  # defensive: trim spaces

    rename_map = {
        old: logical
        for logical, alts in COLUMN_ALIASES.items()
        if (old := first_existing(df_raw, alts))
    }
    trips_raw = df_raw.rename(columns=rename_map)

    # ---------------------------------------------------------------------
    # 2. Prepare and summarise
    # ---------------------------------------------------------------------
    trips = prepare_trip_level(trips_raw)
    summary = build_summary(trips)

    # ---------------------------------------------------------------------
    # 3. Write outputs
    # ---------------------------------------------------------------------
    trip_path = os.path.join(OUTPUT_DIR, TRIP_CSV_NAME)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV_NAME)
    trips.to_csv(trip_path, index=False)
    summary.to_csv(summary_path, index=False)

    return trip_path, summary_path


# =============================================================================
# ENTRY-POINT
# =============================================================================
if __name__ == "__main__":
    trip_csv, summary_csv = run_pipeline()

    print("✔ Trip-level output :", trip_csv)
    print("✔ Summary output    :", summary_csv)
    print(
        f"Flagged when |deviation| > {MIN_THRESHOLD} min "
        f"OR |% deviation| > {PCT_THRESHOLD}%."
    )
    print("Aggregation uses", "weighted averages." if WEIGHTED_AVERAGE else "simple averages.")
