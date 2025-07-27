"""Assess schedule adherence from field-observed bus arrivals and departures.

The module extracts and cleans raw observations, converts them to a
long “one event = one row” format, classifies punctuality using
configurable early/late tolerances, and exports:

- A nicely formatted Excel workbook with overall, by-route, and
  by-route-direction punctuality summaries.
- Parallel CSVs for each summary table.
- Diagnostic CSVs listing every valid and invalid event record.

Typical usage: ArcGIS Pro, Jupyter notebook, or command line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, cast

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.worksheet import Worksheet

# =============================================================================
# CONFIGURATION
# =============================================================================

OBSERVED_DATA_PATH = r"\\Path\To\Your\Field_Data_Folder"
ANALYSIS_RESULTS_PATH = r"\\Path\To\Your\Output_Folder"

EARLY_TOLERANCE_MIN = -1  # minutes early that STILL counts on-time
LATE_TOLERANCE_MIN = 6  # minutes late  that STILL counts on-time

PLACEHOLDER_PATTERN = r"[_X]{4,}"  # e.g. “____”, “__XXXX__”
OUTPUT_EXCEL_NAME = "arrival_performance_summary.xlsx"
OUTPUT_CSV_PREFIX = "arrival_performance_summary"

TIME_EXTRACT_RE = re.compile(r"(\d{1,2})\s*[:]?(\d{2})")

CORE_EVENT_COLS: list[str] = [
    "route_short_name",
    "trip_headsign",
    "stop_id",  #  ← new
    "stop_name",  #  ← new
]

# =============================================================================
# FUNCTIONS
# =============================================================================


def list_observed_files(base_path: str) -> List[Path]:
    """Return every ``.xlsx`` or ``.csv`` file in *base_path*.

    Args:
        base_path: Directory that *should* contain the field-observed
            arrival/departure files.

    Returns:
        A list of :class:`pathlib.Path` objects, each pointing to an
        ``.xlsx`` or ``.csv`` file.

    Raises:
        FileNotFoundError: If *base_path* does not exist **or** if it
        contains no matching files.
    """
    path = Path(base_path)
    if not path.exists():
        raise FileNotFoundError(f"Observed-data folder not found: {base_path}")

    files = [p for p in path.iterdir() if p.suffix.lower() in {".xlsx", ".csv"}]
    if not files:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {base_path}")
    return files


def is_placeholder(val: str | float | int | None) -> bool:
    """Check whether *val* is blank, NaN, or a recognised placeholder.

    Args:
        val: Any scalar value that may have come from the raw spreadsheet.

    Returns:
        ``True`` if *val* is empty, contains no digits, **or** matches
        :data:`PLACEHOLDER_PATTERN`; otherwise ``False``.
    """
    if pd.isna(val):
        return True
    s = str(val).strip()
    return (not bool(re.search(r"\d", s))) or bool(re.fullmatch(PLACEHOLDER_PATTERN, s))


def time_str_to_minutes(time_str: str | float | int | None) -> Optional[int]:
    """Convert a messy HH:MM value to minutes past midnight.

    The helper tolerates common data-entry issues such as embedded spaces,
    missing colons, or times entered as numbers (e.g. ``530`` → 5 : 30).

    Args:
        time_str: Raw string or numeric representation of a time.

    Returns:
        The number of minutes after 00 : 00, or ``None`` if no valid
        time can be parsed.
    """
    if is_placeholder(time_str):  # still screens out blanks/“____”
        return None

    match = TIME_EXTRACT_RE.search(str(time_str))  # look for the first HHMM group
    if not match:
        return None

    hh, mm = map(int, match.groups())  # safe – both groups are digits
    if 0 <= hh < 24 and 0 <= mm < 60:  # sanity-check
        return hh * 60 + mm
    return None


def compute_diff(actual: pd.Series, scheduled: pd.Series) -> pd.Series:
    """Return *scheduled – actual* (minutes).

    Args:
        actual: Series of observed arrival or departure values.
        scheduled: Series of scheduled values.

    Returns:
        :class:`pandas.Series` of numeric minute differences. Cells that
        fail to parse become *NaN*.
    """
    actual_min = actual.map(time_str_to_minutes)
    scheduled_min = scheduled.map(time_str_to_minutes)
    return pd.to_numeric(actual_min) - pd.to_numeric(scheduled_min)


# -----------------------------------------------------------------------------
# PUNCTUALITY CLASSIFICATION & FLAGS
# -----------------------------------------------------------------------------


def classify_punctuality(diff: float | int | None) -> str | None:
    """Label an event as ``'early'``, ``'on_time'`` or ``'late'``.

    Args:
        diff: Minute difference produced by :func:`compute_diff`.

    Returns:
        A string classification, or ``None`` when *diff* is *NaN*.
    """
    if pd.isna(diff):
        return None
    if diff < EARLY_TOLERANCE_MIN:
        return "early"
    if diff > LATE_TOLERANCE_MIN:
        return "late"
    return "on_time"


def flag_on_time(diff_series: pd.Series) -> pd.Series:
    """Return a legacy ``'Y'``/``'N'`` on-time flag.

    Args:
        diff_series: Output from :func:`compute_diff`.

    Returns:
        A Series whose dtype is ``string[pyarrow]`` and whose values are
        ``'Y'`` if the event is within tolerance, otherwise ``'N'``.
    """
    return diff_series.apply(
        lambda d: ("Y" if pd.notna(d) and EARLY_TOLERANCE_MIN <= d <= LATE_TOLERANCE_MIN else "N")
    )


def load_single_file(path: Path) -> pd.DataFrame:
    """Load one ``.xlsx`` or ``.csv`` file as a string-typed DataFrame.

    Args:
        path: Absolute or relative path to the input file.

    Returns:
        Raw contents with an added ``source_file`` column.

    Raises:
        ValueError: If *path* has an unsupported extension.
    """
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df["source_file"] = path.name
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip noise and drop template rows.

    Operations performed:

    * Trim column names.
    * Remove rows whose ``route_short_name`` is ``'SAMPLE'`` (case-insensitive).
    * Trim whitespace in ``route_short_name`` and ``trip_headsign``.

    Args:
        df: Raw field-observed data.

    Returns:
        A cleaned copy of *df*.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Exclude template rows (route == SAMPLE)
    df = df[df["route_short_name"].str.upper() != "SAMPLE"]

    # Strip leading/trailing spaces in key text columns
    for col in ("route_short_name", "trip_headsign"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df.reset_index(drop=True)


def _get_invalid_reason(
    act_time: str | float | int | None,
    sched_time: str | float | int | None,
) -> str:
    """Explain *why* an event row is invalid.

    An empty string means the row is valid; otherwise a semicolon-separated
    list enumerates all detected problems.

    Args:
        act_time: Observed arrival/departure value.
        sched_time: Scheduled arrival/departure value.

    Returns:
        ``""`` (valid) **or** a descriptive string (invalid).
    """
    reasons: list[str] = []

    # Actual time checks -----------------------------------------------------
    if is_placeholder(act_time):
        reasons.append("blank/placeholder act_time")
    elif time_str_to_minutes(act_time) is None:
        reasons.append("bad act_time format")

    # Scheduled time checks --------------------------------------------------
    if is_placeholder(sched_time):
        reasons.append("blank/placeholder sched_time")
    elif time_str_to_minutes(sched_time) is None:
        reasons.append("bad sched_time format")

    return "; ".join(reasons)


# -----------------------------------------------------------------------------
# 1-row = 1-event **long-format** transformer
# -----------------------------------------------------------------------------
EVENT_MAP: Dict[str, tuple[str, str]] = {
    "arrival": ("arrival_time", "act_arrival"),
    "departure": ("departure_time", "act_departure"),
}


def longify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Explode wide arrival/departure columns into long format.

    The result contains every event in the original data: **nothing is
    dropped**. Invalid events carry an ``invalid_reason`` description and
    *NaN* for ``diff_min``. Valid events receive:

    * ``diff_min`` – numeric difference (minutes).
    * ``on_time``   – ``'Y'`` or ``'N'``.
    * ``punctuality`` – ``'early'``, ``'on_time'``, or ``'late'``.

    Args:
        df: Wide-format DataFrame whose columns follow *EVENT_MAP*.

    Returns:
        Long-format DataFrame with one row per event.
    """
    parts: list[pd.DataFrame] = []
    for evt, (sched_col, act_col) in EVENT_MAP.items():
        cols_needed = CORE_EVENT_COLS + [sched_col, act_col]

        part = (
            df[cols_needed]
            .copy()
            .rename(
                columns={
                    sched_col: "sched_time",
                    act_col: "act_time",
                }
            )
            .assign(event_type=evt)
        )
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)

    # Work out (in-)validity before any maths -------------------------------
    long_df["invalid_reason"] = long_df.apply(
        lambda r: _get_invalid_reason(r["act_time"], r["sched_time"]),
        axis=1,
    )
    valid_mask = long_df["invalid_reason"] == ""

    # Compute diffs only where both times parsed OK --------------------------
    long_df.loc[valid_mask, "diff_min"] = compute_diff(
        long_df.loc[valid_mask, "act_time"],
        long_df.loc[valid_mask, "sched_time"],
    )
    long_df.loc[~valid_mask, "diff_min"] = pd.NA

    # Flags / categorisation --------------------------------------------------
    long_df["on_time"] = flag_on_time(long_df["diff_min"])
    long_df["punctuality"] = long_df["diff_min"].apply(classify_punctuality)

    return long_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# PUNCTUALITY SUMMARY
# -----------------------------------------------------------------------------


def summarise_punctuality(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return early/on-time/late percentages rounded to one decimal.

    Args:
        df: Long-format events table, *already filtered to valid rows*.
        group_cols: Column names to group by. ``None`` → overall summary.

    Returns:
        A tidy DataFrame with three columns: ``early_pct``, ``on_time_pct``,
        and ``late_pct``.
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
    """Partition *events_df* into valid and invalid subsets.

    Validity is defined strictly as “``diff_min`` is not NA”.

    Args:
        events_df: Output from :func:`longify_events`.

    Returns:
        Two DataFrames: ``(valid_events, invalid_events)``.
    """
    valid_mask = events_df["diff_min"].notna()
    return events_df[valid_mask].copy(), events_df[~valid_mask].copy()


# -----------------------------------------------------------------------------
# EXPORT HELPERS
# -----------------------------------------------------------------------------


def ensure_output_folder(path: str) -> None:
    """Create *path* and its parents if they do not already exist.

    Args:
        path: Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def export_extra_dataframes(
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    out_folder: str,
) -> None:
    """Write the diagnostic CSVs to *out_folder*.

    Args:
        valid_df: All events deemed valid.
        invalid_df: All events deemed invalid.
        out_folder: Destination directory.
    """
    ensure_output_folder(out_folder)
    valid_df.to_csv(Path(out_folder) / "observed_data_valid_events.csv", index=False)
    invalid_df.to_csv(Path(out_folder) / "observed_data_invalid_events.csv", index=False)


def export_results(
    overall: pd.DataFrame,
    by_route: pd.DataFrame,
    by_route_dir: pd.DataFrame,
    out_folder: str,
) -> None:
    """Export the punctuality summaries to Excel **and** CSV.

    Args:
        overall: Output from :func:`summarise_punctuality` with
            *group_cols* = ``None``.
        by_route: Summary grouped by ``route_short_name``.
        by_route_dir: Summary grouped by
            ``['route_short_name', 'trip_headsign']``.
        out_folder: Destination directory.

    Side Effects:
        Writes one Excel workbook and three CSV files.
    """
    ensure_output_folder(out_folder)
    excel_path = Path(out_folder) / OUTPUT_EXCEL_NAME
    wb = Workbook()
    default_ws = wb.active
    if default_ws is not None:  # appease static checker
        wb.remove(cast("Worksheet", default_ws))

    def add_sheet(df: pd.DataFrame, title: str) -> None:
        ws = wb.create_sheet(title=title)
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)

        # Autosize columns
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col) + 2
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
# MAIN
# =============================================================================


def main() -> None:
    """Command-line entry point."""
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
    print(f"✔ Long-format events shape (after dropping placeholders): {events_long.shape}")

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
