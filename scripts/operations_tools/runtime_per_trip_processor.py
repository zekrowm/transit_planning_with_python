"""
Script Name:
    runtime_per_trip_processor.py

Purpose:
    Analyse scheduled vs. actual running times for bus routes,
    flagging trips and route–direction pairs whose deviations
    exceed user‑defined percent or minute thresholds, and
    exporting both trip‑level and summary CSVs.

Inputs:
    1. A trip‑level CSV defined by INPUT_FILE
       (e.g. r"\\Path\\To\\Runtime Trip Level - 05-05-2025.csv")
    2. Configuration constants in the CONFIGURATION section below
       (thresholds, weighted vs. simple averaging, output folder).

Outputs:
    1. trip_level_with_flags.csv   – trip rows + new metrics & flags
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

# expected column names
SCHEDULED_COL = "Average Scheduled Running Time"
ACTUAL_COL = "Average Actual Running Time"
COUNT_TRIP_COL = "Count Trip"

# =============================================================================
# FUNCTIONS
# =============================================================================

def parse_time_to_minutes(time_str: str) -> float:
    """Convert 'H:MM:SS' (or 'HH:MM') to minutes, NaN on failure."""
    if pd.isna(time_str):
        return float("nan")
    txt = str(time_str).strip()
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
    """Weighted mean with safe divide‑by‑zero handling."""
    w_sum = weights.sum()
    return (values * weights).sum() / w_sum if w_sum else float("nan")


def prepare_trip_level(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean columns and per‑trip flag."""
    df["route_name"] = (
        df["Route"]
        .astype(str)
        .str.split("-", n=1)
        .str[0]
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    df["scheduled_minutes"] = df[SCHEDULED_COL].apply(parse_time_to_minutes)
    df["actual_minutes"] = df[ACTUAL_COL].apply(parse_time_to_minutes)

    df["deviation_minutes"] = df["actual_minutes"] - df["scheduled_minutes"]
    df["percent_deviation"] = (
        df["deviation_minutes"] / df["scheduled_minutes"] * 100
    ).round(1)

    df["flag_runtime_issue"] = df.apply(
        lambda r: flag_row(r["deviation_minutes"], r["scheduled_minutes"]), axis=1
    )
    return df


def build_summary(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to Route × Direction and flag summary rows."""
    group_cols = ["route_name", "Direction"]

    if WEIGHTED_AVERAGE:
        summary = (
            trips.groupby(group_cols, dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "count_trips": g[COUNT_TRIP_COL].sum(),
                        "scheduled_avg": weighted_mean(g["scheduled_minutes"], g[COUNT_TRIP_COL]),
                        "actual_avg": weighted_mean(g["actual_minutes"], g[COUNT_TRIP_COL]),
                        "deviation_avg": weighted_mean(g["deviation_minutes"], g[COUNT_TRIP_COL]),
                    }
                )
            )
            .reset_index()
        )
    else:
        summary = (
            trips.groupby(group_cols, dropna=False)
            .agg(
                count_trips=(COUNT_TRIP_COL, "sum"),
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
    """Load, process, save.  Returns (trip_path, summary_path)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trips_raw = pd.read_csv(INPUT_FILE)
    trips = prepare_trip_level(trips_raw)
    summary = build_summary(trips)

    trip_path = os.path.join(OUTPUT_DIR, TRIP_CSV_NAME)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV_NAME)
    trips.to_csv(trip_path, index=False)
    summary.to_csv(summary_path, index=False)

    return trip_path, summary_path


if __name__ == "__main__":
    trip_csv, summary_csv = run_pipeline()

    print("✔ Trip‑level output :", trip_csv)
    print("✔ Summary output    :", summary_csv)
    print(
        f"Flagged when |deviation| > {MIN_THRESHOLD} min "
        f"OR |% deviation| > {PCT_THRESHOLD}%."
    )
    print("Aggregation uses", "weighted averages." if WEIGHTED_AVERAGE else "simple averages.")
