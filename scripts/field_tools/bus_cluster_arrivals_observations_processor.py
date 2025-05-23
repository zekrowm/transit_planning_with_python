"""
Script Name:
        bus_cluster_arrivals_checklist_processor.py

Purpose:
        Processes observed transit arrival/departure data from input
        files based on field data to analyze service punctuality.
        Calculates schedule adherence against configurable on-time
        tolerances and outputs summary reports.

Inputs:
        1. Directory path (`OBSERVED_DATA_PATH`) containing observed
           transit data as .xlsx or .csv files.
        2. Configuration constants in the script (e.g., on-time
           tolerances `EARLY_TOLERANCE_MIN`, `LATE_TOLERANCE_MIN`;
           input/output paths; `PLACEHOLDER_PATTERN`).

Outputs:
        1. Excel file (default: `arrival_performance_summary.xlsx`) with
           punctuality summaries (overall, by route, by route/direction).
        2. Corresponding CSV summary files (default prefix:
           `arrival_performance_summary`).
        3. Diagnostic CSV files (`observed_data_valid_events.csv`,
           `observed_data_invalid_events.csv`) detailing processed events.
        4. Console status messages during execution.

Dependencies:
        pandas, openpyxl, re, pathlib
"""
# =============================================================================
# CONFIGURATION
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import re

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

OBSERVED_DATA_PATH = (
    r"\\Path\To\Your\Field_Data_Folder"
)
ANALYSIS_RESULTS_PATH = (
    r"\\Path\To\Your\Output_Folder"
)

EARLY_TOLERANCE_MIN = -1   # minutes early that STILL counts on-time
LATE_TOLERANCE_MIN = 6     # minutes late  that STILL counts on-time

PLACEHOLDER_PATTERN = r"[_X]{4,}"          # e.g. “____”, “__XXXX__”
OUTPUT_EXCEL_NAME   = "arrival_performance_summary.xlsx"
OUTPUT_CSV_PREFIX   = "arrival_performance_summary"

TIME_EXTRACT_RE = re.compile(r'(\d{1,2})\s*[:]?(\d{2})')

# =============================================================================
# HELPERS
# =============================================================================
def list_observed_files(base_path: str) -> List[Path]:
    """Return list of .xlsx/.csv files in *base_path* (raise if none)."""
    path = Path(base_path)
    if not path.exists():
        raise FileNotFoundError(f"Observed-data folder not found: {base_path}")

    files = [p for p in path.iterdir() if p.suffix.lower() in {".xlsx", ".csv"}]
    if not files:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {base_path}")
    return files


def is_placeholder(val: str | float | int | None) -> bool:
    """True if *val* is NaN, empty, or matches the placeholder pattern."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    return not bool(re.search(r"\d", s)) or re.fullmatch(PLACEHOLDER_PATTERN, s)


def time_str_to_minutes(time_str: str | float | int | None) -> Optional[int]:
    """
    Convert a variety of messy time strings to minutes past midnight.

    Returns
    -------
    int | None
        Minutes after 00:00, or *None* if no valid HH:MM pattern is found.
    """
    if is_placeholder(time_str):                      # still screens out blanks/“____”
        return None

    match = TIME_EXTRACT_RE.search(str(time_str))     # look for the first HHMM group
    if not match:
        return None

    hh, mm = map(int, match.groups())                 # safe – both groups are digits
    if 0 <= hh < 24 and 0 <= mm < 60:                 # sanity-check
        return hh * 60 + mm
    return None


def compute_diff(actual: pd.Series, scheduled: pd.Series) -> pd.Series:
    """Scheduled–actual difference (minutes)."""
    actual_min    = actual.map(time_str_to_minutes)
    scheduled_min = scheduled.map(time_str_to_minutes)
    return pd.to_numeric(actual_min) - pd.to_numeric(scheduled_min)


# -----------------------------------------------------------------------------
# PUNCTUALITY CLASSIFICATION & FLAGS
# -----------------------------------------------------------------------------
def classify_punctuality(diff: float | int | None) -> str | None:
    """Return 'early', 'on_time', or 'late' based on *diff* (minutes)."""
    if pd.isna(diff):
        return None
    if diff < EARLY_TOLERANCE_MIN:
        return "early"
    if diff > LATE_TOLERANCE_MIN:
        return "late"
    return "on_time"


def flag_on_time(diff_series: pd.Series) -> pd.Series:
    """Return 'Y'/'N' on-time flag (kept for backward compatibility)."""
    return diff_series.apply(
        lambda d: "Y"
        if pd.notna(d) and EARLY_TOLERANCE_MIN <= d <= LATE_TOLERANCE_MIN
        else "N"
    )


def load_single_file(path: Path) -> pd.DataFrame:
    """Load a single .xlsx or .csv file as dataframe of str columns."""
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df["source_file"] = path.name
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, drop SAMPLE rows, keep original column names."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Exclude template rows (route == SAMPLE)
    df = df[df["route_short_name"].str.upper() != "SAMPLE"]

    # Strip leading/trailing spaces in key text columns
    for col in ("route_short_name", "trip_headsign"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 1-row = 1-event **long-format** transformer
# -----------------------------------------------------------------------------
EVENT_MAP: Dict[str, tuple[str, str]] = {
    "arrival": ("arrival_time", "act_arrival"),
    "departure": ("departure_time", "act_departure"),
}


def longify_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode arrivals & departures into one-event-per-row *long* format.

    Rows whose *act_time* is a placeholder / lacks digits are dropped.
    Adds ``diff_min``, ``on_time`` flag, and ``punctuality`` category.
    """
    parts: list[pd.DataFrame] = []
    for evt, (sched_col, act_col) in EVENT_MAP.items():
        part = (
            df[["route_short_name", "trip_headsign", sched_col, act_col]]
            .copy()
            .rename(columns={sched_col: "sched_time", act_col: "act_time"})
            .assign(event_type=evt)
        )
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)

    # Keep only rows whose *act_time* is a real timestamp
    mask_valid_act = long_df["act_time"].apply(lambda v: not is_placeholder(v))
    long_df = long_df[mask_valid_act].reset_index(drop=True)

    # Compute diff, flags, and category
    long_df["diff_min"]    = compute_diff(long_df["act_time"], long_df["sched_time"])
    long_df["on_time"]     = flag_on_time(long_df["diff_min"])
    long_df["punctuality"] = long_df["diff_min"].apply(classify_punctuality)

    return long_df


# -----------------------------------------------------------------------------
# PUNCTUALITY SUMMARY
# -----------------------------------------------------------------------------
def summarise_punctuality(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a summary with *early_pct*, *on_time_pct*, *late_pct* (0–100).

    Percentages are rounded to one decimal place and always sum to ~100.
    """
    if group_cols:
        grp_df = df
    else:  # overall summary needs a dummy key
        grp_df = df.assign(__overall__="all")
        group_cols = ["__overall__"]

    pct_table = (
        grp_df.groupby(group_cols)["punctuality"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("pct")
        .round(1)
        .unstack(fill_value=0)
        .reindex(columns=["early", "on_time", "late"], fill_value=0)
        .reset_index()
        .rename(
            columns={
                "early": "early_pct",
                "on_time": "on_time_pct",
                "late": "late_pct",
            }
        )
    )

    if "__overall__" in pct_table.columns:
        pct_table = pct_table.drop(columns="__overall__")

    return pct_table


# -----------------------------------------------------------------------------
# Valid vs. invalid splitter (for diagnostics)
# -----------------------------------------------------------------------------
def split_valid_invalid(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (valid, invalid) where validity = numeric diff."""
    valid_mask = events_df["diff_min"].notna()
    return events_df[valid_mask].copy(), events_df[~valid_mask].copy()


# =============================================================================
# EXPORT HELPERS
# =============================================================================
def ensure_output_folder(path: str) -> None:
    """Create *path* (plus parents) if it does not yet exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def export_extra_dataframes(
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    out_folder: str,
) -> None:
    """Write valid/invalid diagnostic CSVs to *out_folder*."""
    ensure_output_folder(out_folder)
    valid_df.to_csv(Path(out_folder) / "observed_data_valid_events.csv", index=False)
    invalid_df.to_csv(Path(out_folder) / "observed_data_invalid_events.csv", index=False)


def export_results(
    overall: pd.DataFrame,
    by_route: pd.DataFrame,
    by_route_dir: pd.DataFrame,
    out_folder: str,
) -> None:
    """Write Excel + CSV outputs to *out_folder*."""
    ensure_output_folder(out_folder)
    excel_path = Path(out_folder) / OUTPUT_EXCEL_NAME
    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    def add_sheet(df: pd.DataFrame, title: str) -> None:
        ws = wb.create_sheet(title=title)
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)

        # Autosize columns
        for col in ws.columns:
            max_len = (
                max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                + 2
            )
            ws.column_dimensions[col[0].column_letter].width = max_len

    add_sheet(overall, "Overall")
    add_sheet(by_route, "By_Route")
    add_sheet(by_route_dir, "By_Route_Direction")
    wb.save(excel_path)

    overall.to_csv(Path(out_folder) / f"{OUTPUT_CSV_PREFIX}_overall.csv", index=False)
    by_route.to_csv(Path(out_folder) / f"{OUTPUT_CSV_PREFIX}_by_route.csv", index=False)
    by_route_dir.to_csv(
        Path(out_folder) / f"{OUTPUT_CSV_PREFIX}_by_route_dir.csv",
        index=False,
    )
    print(f"✔ Results written to {excel_path}")
    print("  (+ parallel CSVs in the same folder)")


# =============================================================================
# MAIN ROUTINE
# =============================================================================
def main() -> None:
    """Orchestrate ETL → analysis → export."""
    print("▸ Listing observed-data files …")
    observed_files = list_observed_files(OBSERVED_DATA_PATH)
    print(f"  {len(observed_files)} files found.")

    # load → clean → concatenate (still WIDE at this point)
    cleaned_frames: List[pd.DataFrame] = []
    for path in observed_files:
        print(f"  – loading {path.name}")
        wide_raw = load_single_file(path)
        wide_clean = clean_dataframe(wide_raw)
        cleaned_frames.append(wide_clean)

    wide_all = pd.concat(cleaned_frames, ignore_index=True)
    print(f"✔ Combined wide DataFrame shape: {wide_all.shape}")

    # explode to LONG, drop placeholder events, compute diffs
    events_long = longify_events(wide_all)
    print(
        "✔ Long-format events shape (after dropping placeholders): "
        f"{events_long.shape}"
    )

    # split for diagnostics
    valid_events, invalid_events = split_valid_invalid(events_long)
    export_extra_dataframes(valid_events, invalid_events, ANALYSIS_RESULTS_PATH)

    # summaries
    print("▸ Calculating punctuality summaries …")
    overall_summary = summarise_punctuality(valid_events)  # 1 row
    by_route_summary = summarise_punctuality(valid_events, ["route_short_name"])
    by_route_dir_summary = summarise_punctuality(
        valid_events,
        ["route_short_name", "trip_headsign"],
    )

    export_results(
        overall_summary,
        by_route_summary,
        by_route_dir_summary,
        ANALYSIS_RESULTS_PATH,
    )
    print("✓ All done.")


if __name__ == "__main__":
    main()
