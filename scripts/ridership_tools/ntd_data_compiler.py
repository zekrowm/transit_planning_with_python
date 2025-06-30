"""Compiles monthly NTD ridership reports into a unified dataset with validation and diagnostics.

Reads multiple Excel files containing National Transit Database (NTD) ridership data,
applies numeric cleaning and row filtering rules, and writes a consolidated output
with optional audit of discarded rows.

Typical use: Batch processing of agency-level monthly NTD data for analysis or reporting.

Inputs:
    - List of Excel file paths (with optional sheet names)
    - Configurable drop rules for null rows
    - Period column name for validation (optional)

Outputs:
    - Consolidated file (.csv or .xlsx) with cleaned data
    - Optional audit file of discarded rows (same format as output)
"""

from __future__ import annotations  # postpone evaluation of type-hints

import os
from typing import Any, Callable, List, Mapping, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

FILES_TO_PROCESS: List[Tuple[str, Optional[str]]] = [
    (r"\\Your\File\Path\JULY 2024 NTD RIDERSHIP.XLSX", "Temporary_Query_N"),
    (r"\\Your\File\Path\AUGUST 2024 NTD RIDERSHIP.XLSX", "Temporary_Query_N",),
    (r"\\Your\File\Path\SEPTEMBER 2024 NTD RIDERSHIP.XLSX", "Sep.2024 Finals"),
]

OUTPUT_FILE_PATH: str = r"\\Path\to\Your\Output_Folder\Compiled_NTD_Data.csv"  # or .xlsx

# -----------------------------------------------------------------------------

# Optional row-exclusion rules
DROPNA_SUBSET_ALL_NAN: Optional[List[str]] = None  # e.g. ["ROUTE_NAME", "MTH_BOARD"]
DROPNA_SUBSET_ANY_NAN: Optional[List[str]] = None  # e.g. ["ROUTE_NAME", "MTH_BOARD"]

EXISTING_PERIOD_COLUMN_NAME: str = "MTH_YR"  # ← update

# -----------------------------------------------------------------------------

SUMMARY_ROW_PATTERNS: tuple[str, ...] = (
    "TOTAL",  # “TOTAL”, “Grand Total”, etc.
    "SUMMARY",  # “SUMMARY”
    "ALL ROUTES",  # “ALL ROUTES”
    "SYSTEM",  # “SYSTEM TOTAL”, “SYSTEM”, …
)

# -----------------------------------------------------------------------------
# QUALITY-CHECK FLAGGER
# -----------------------------------------------------------------------------

ROLLING_WINDOW_MONTHS: int = 12
NEG_THRESHOLD: float = -0.10              # –10 %
LOG_FILE_PATH: str = os.path.join(
    os.path.dirname(OUTPUT_FILE_PATH) or ".",
    "Data_Quality_Log.txt",
)
ROLLING_AVG_EXPORT: str = os.path.join(
    os.path.dirname(OUTPUT_FILE_PATH) or ".",
    "Rolling_Weekday_Averages.csv",
)

COL_ROUTE = "ROUTE_NAME"
COL_PERIOD = EXISTING_PERIOD_COLUMN_NAME or "PERIOD"
COL_DAYTYPE = "SERVICE_PERIOD"
COL_BOARD = "MTH_BOARD"
COL_DAYS = "DAYS"

# =============================================================================
# FUNCTIONS
# =============================================================================

def robust_numeric_converter(value: Any) -> float | None:
    """Convert strings such as "1,234" to 1234.0 (float).

    Returns None on blanks / NA or on conversion failure (with a warning).

    Args:
        value: The value to attempt to convert to a float.

    Returns:
        The converted float value, or None if conversion fails or the input is
        blank/NA.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        print(f"Warning: Could not convert '{value}' to float. Returning None.")
        return None


# -----------------------------------------------------------------------------
# Mapping of column names → converters passed to pandas.read_excel
# -----------------------------------------------------------------------------

COMMON_CONVERTERS: Mapping[int | str, Callable[[Any], object]] = {
    "MTH_BOARD": robust_numeric_converter,
    "MTH_REV_HOURS": robust_numeric_converter,
    "MTH_PASS_MILES": robust_numeric_converter,
    "ASCH_TRIPS": robust_numeric_converter,
    "ACTUAL_TRIPS": robust_numeric_converter,
    "DAYS": robust_numeric_converter,
    "REV_MILES": robust_numeric_converter,
}

# -----------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------


def read_and_prepare_ntd_file(
    file_path: str, sheet_name: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Reads an NTD Excel file, applies cleansing rules, and separates rows.

    Returns a tuple of two DataFrames: `(kept_rows, discarded_rows)`. Both
    DataFrames will have identical columns. Either element may be empty; both
    are None if the file cannot be read.

    Args:
        file_path: The full path to the Excel file to process.
        sheet_name: The name of the sheet to read. If None, the first available
            sheet will be used.

    Returns:
        A tuple containing:
            - A pandas.DataFrame of rows that passed the cleansing rules.
            - A pandas.DataFrame of rows that were discarded by the cleansing rules.
        Returns (None, None) if the file does not exist or an error occurs during
        reading.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}. Skipping.")
        return None, None

    print(
        f"Processing file: {os.path.basename(file_path)}, Sheet: {sheet_name or 'first available'}"
    )

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, converters=COMMON_CONVERTERS)

        # ------------------ filtering logic ------------------
        keep_mask = pd.Series(True, index=df.index)

        # (1) user-configured NaN checks
        if DROPNA_SUBSET_ALL_NAN:
            keep_mask &= ~df[DROPNA_SUBSET_ALL_NAN].isna().all(axis=1)
        if DROPNA_SUBSET_ANY_NAN:
            keep_mask &= ~df[DROPNA_SUBSET_ANY_NAN].isna().any(axis=1)

        # (2) NEW SUMMARY-ROW FILTER
        summary_mask = _is_summary_row(df)
        keep_mask &= ~summary_mask  # keep everything except summary rows

        kept_df = df[keep_mask].copy()
        dropped_df = df[~keep_mask].copy()

        # diagnostics
        if EXISTING_PERIOD_COLUMN_NAME and EXISTING_PERIOD_COLUMN_NAME not in df.columns:
            print(f"  Warning: period column '{EXISTING_PERIOD_COLUMN_NAME}' not found.")

        print(f"  Rows kept: {len(kept_df):>6} | Rows discarded: {len(dropped_df):>6}")

        return kept_df, dropped_df

    except Exception as exc:  # broad catch to keep batch processing alive
        print(f"ERROR: failed to read {os.path.basename(file_path)}: {exc}")
        return None, None


def compile_ntd_data() -> Optional[pd.DataFrame]:
    """Compile monthly NTD files into a single DataFrame.

    The function iterates over the list defined in ``FILES_TO_PROCESS``,
    applies row-exclusion rules, concatenates the retained rows, and writes
    both the compiled dataset and any discarded rows to disk.

    Returns:
        Optional[pd.DataFrame]: The compiled NTD data if any rows were kept;
        otherwise ``None``.
    """
    if not FILES_TO_PROCESS:
        print("ERROR: 'FILES_TO_PROCESS' list is empty.")
        return None

    print(f"\nFound {len(FILES_TO_PROCESS)} file(s) listed for processing.")
    kept_frames: List[pd.DataFrame] = []
    dropped_frames: List[pd.DataFrame] = []

    for file_path, sheet_name in FILES_TO_PROCESS:
        kept, dropped = read_and_prepare_ntd_file(file_path, sheet_name)
        if kept is not None and not kept.empty:
            kept_frames.append(kept)
        if dropped is not None and not dropped.empty:
            dropped_frames.append(dropped)

    if not kept_frames:
        print("\nNo data collected. Compilation aborted.")
        return None

    # ------------------ concatenate and save ------------------
    try:
        compiled = pd.concat(kept_frames, ignore_index=True, sort=False)
        print(f"\nConcatenation complete. Total rows kept: {len(compiled)}")

        # Ensure target directory exists, create if needed
        out_dir = os.path.dirname(OUTPUT_FILE_PATH)
        out_path = OUTPUT_FILE_PATH
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
                print(f"Created output directory: {out_dir}")
            except OSError as ose:
                print(f"ERROR: could not create directory {out_dir}: {ose}")
                out_path = os.path.basename(OUTPUT_FILE_PATH)
                print(f"Saving to current working directory as {out_path}")

        def write_dataframe(df: pd.DataFrame, path: str) -> None:
            """Write DataFrame to XLSX or CSV based on file extension.

            Args:
                df: The DataFrame to write.
                path: The full path to the output file.
            """
            if path.lower().endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)

        # Save kept rows
        write_dataframe(compiled, out_path)
        print(f"Compiled NTD data written to: {os.path.abspath(out_path)}")

        # Save discarded rows (audit), if any
        if dropped_frames:
            discarded = pd.concat(dropped_frames, ignore_index=True, sort=False)
            base, ext = os.path.splitext(out_path)
            discard_path = f"{base}_discarded{ext or '.csv'}"
            write_dataframe(discarded, discard_path)
            print(f"Discarded rows written to: {os.path.abspath(discard_path)}")
        else:
            print("No rows were discarded; no audit file created.")

        return compiled

    except Exception as exc:
        print(f"ERROR during concatenation / saving: {exc}")
        return None


def _is_summary_row(frame: pd.DataFrame) -> pd.Series:
    """Identifies ridership summary or total rows within a DataFrame.

    A row is flagged as a summary if **either** of the following is true:
      1. Any text identifier column (`ROUTE_NAME`, `SCHEDULE_NAME`,
         `SERVICE_PERIOD`) contains a keyword defined in `SUMMARY_ROW_PATTERNS`.
      2. All identifier columns (`ROUTE_NAME`, `SCHEDULE_NAME`) are blank
         *and* at least one numeric field contains data (typical unlabeled
         “Totals” row).

    Args:
        frame: The pandas.DataFrame to check for summary rows.

    Returns:
        A pandas.Series of boolean values, where True indicates a summary row.
    """
    # --- 1. keyword match --------------------------------------------------
    text_cols = ("ROUTE_NAME", "SCHEDULE_NAME", "SERVICE_PERIOD")
    contains_keyword = (
        frame.reindex(columns=text_cols, fill_value="")
        .apply(
            lambda col: col.astype(str, copy=False)
            .str.upper()
            .str.contains("|".join(SUMMARY_ROW_PATTERNS), regex=True, na=False)
        )
        .any(axis=1)
    )

    # --- 2. unlabeled totals ----------------------------------------------
    id_cols = ["ROUTE_NAME", "SCHEDULE_NAME"]
    num_cols = [c for c in frame.columns if frame[c].dtype.kind in "fi"]  # floats/ints
    blank_ids = frame[id_cols].isna().all(axis=1)
    has_numbers = frame[num_cols].notna().any(axis=1)

    unlabeled_total = blank_ids & has_numbers

    return contains_keyword | unlabeled_total


# -----------------------------------------------------------------------------
# QUALITY-CHECK FLAGGER
# -----------------------------------------------------------------------------

def _periodify(series: pd.Series) -> pd.Series:
    """Parse many month formats → pandas.Period ('M')."""
    known = ("%b-%y", "%B-%y", "%Y-%m", "%Y%m", "%m/%Y")
    def _p(s: str):
        for f in known:
            try:
                return datetime.strptime(s, f)
            except ValueError:
                continue
        return pd.to_datetime(s, errors="coerce")
    return series.astype(str, copy=False).map(_p).dt.to_period("M")


def _fmt(p: "pd.Period") -> str:   # noqa: F821
    return p.to_timestamp().strftime("%b-%y") if pd.notna(p) else "?"


# ─────────────────────────────────────────────────────────────────────────────
def _build_rolling_weekday_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame: route × period_end × 12-mo weekday average."""
    wk = df[df[COL_DAYTYPE].astype(str).str.upper().str.startswith("WEEKDAY")].copy()
    if wk.empty:
        return pd.DataFrame(columns=[COL_ROUTE, "PERIOD_END", "WKDY_AVG"])

    wk[COL_BOARD] = pd.to_numeric(wk[COL_BOARD], errors="coerce")
    wk[COL_DAYS] = pd.to_numeric(wk[COL_DAYS], errors="coerce")
    wk["PERIOD_M"] = _periodify(wk[COL_PERIOD])

    # aggregate duplicates
    agg = (
        wk.groupby([COL_ROUTE, "PERIOD_M"], as_index=False)[[COL_BOARD, COL_DAYS]]
        .sum(min_count=1)
        .dropna(subset=["PERIOD_M"])
    )

    results = []
    latest = agg["PERIOD_M"].max()
    earliest = agg["PERIOD_M"].min()

    end = latest
    while end - (ROLLING_WINDOW_MONTHS - 1) >= earliest:
        window = pd.period_range(end - (ROLLING_WINDOW_MONTHS - 1), end, freq="M")
        slice_ = agg[agg["PERIOD_M"].isin(window)].copy()

        if slice_.empty:
            end -= 1
            continue

        roll = (
            slice_.groupby(COL_ROUTE, as_index=False)[[COL_BOARD, COL_DAYS]]
            .sum(min_count=1)
            .assign(
                PERIOD_END=end,
                WKDY_AVG=lambda x: np.where(
                    x[COL_DAYS] > 0, x[COL_BOARD] / x[COL_DAYS], np.nan
                ),
            )[[COL_ROUTE, "PERIOD_END", "WKDY_AVG"]]
        )
        results.append(roll)
        end -= 1

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _flag_negative_trends(roll_tbl: pd.DataFrame) -> list[str]:
    """Compare consecutive windows and flag ≥10 % drops."""
    msgs: list[str] = []
    if roll_tbl.empty:
        return msgs

    roll_tbl = roll_tbl.sort_values([COL_ROUTE, "PERIOD_END"])

    for route, grp in roll_tbl.groupby(COL_ROUTE):
        avgs = grp.set_index("PERIOD_END")["WKDY_AVG"].dropna()
        if avgs.size < 2:
            continue
        pct = (avgs.iloc[-1] - avgs.iloc[-2]) / avgs.iloc[-2]
        if pct <= NEG_THRESHOLD:
            msgs.append(
                f"[WKDY AVG TREND ↓] Route {route}: "
                f"{ROLLING_WINDOW_MONTHS}-mo avg {_fmt(avgs.index[-2])}–{_fmt(avgs.index[-1])} "
                f"dropped {pct:.1%} "
                f"({avgs.iloc[-2]:,.0f} → {avgs.iloc[-1]:,.0f})"
            )
    return msgs


def _routes_with_missing_months(df: pd.DataFrame) -> list[str]:
    msgs: list[str] = []
    if COL_PERIOD not in df.columns:
        return msgs

    df = df[[COL_ROUTE, COL_PERIOD]].dropna()
    df["PERIOD_M"] = _periodify(df[COL_PERIOD])
    for route, grp in df.groupby(COL_ROUTE):
        months = grp["PERIOD_M"].dropna().unique()
        if months.size == 0:
            continue
        full = pd.period_range(months.min(), months.max(), freq="M")
        miss = full.difference(months)
        if not miss.empty:
            msgs.append(
                f"[MISSING DATA] Route {route}: "
                f"missing {miss.size} month(s) between "
                f"{_fmt(months.min())} and {_fmt(months.max())} "
                f"→ {', '.join(_fmt(m) for m in miss)}"
            )
    return msgs


def write_quality_log(compiled_df: pd.DataFrame) -> None:
    """Create a one-row-per-route, date-ordered wide table + a plain-text log."""
    # 1. build long rolling table (route, period_end, avg)
    roll_long = _build_rolling_weekday_table(compiled_df)

    # 2. reshape → wide (routes as rows, months as columns), date-sorted
    if not roll_long.empty:
        roll_wide = (
            roll_long.pivot(index=COL_ROUTE, columns="PERIOD_END", values="WKDY_AVG")
            .sort_index(axis=1)                                  # chronological L→R
        )
        # make header friendly "Apr-25", "Mar-25", …
        roll_wide.columns = [_fmt(p) for p in roll_wide.columns]
        roll_wide.reset_index().to_csv(ROLLING_AVG_EXPORT, index=False)
        print(f"Rolling-average table written → {ROLLING_AVG_EXPORT}")
    else:
        roll_wide = pd.DataFrame()  # empty sentinel

    # 3. build log
    items = (
        _routes_with_missing_months(compiled_df)
        + _flag_negative_trends(roll_long)
    )
    if not items:
        items = ["No issues found."]

    try:
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(items))
        print(f"Quality-check log written → {LOG_FILE_PATH}")
    except OSError as err:
        print(f"WARNING: could not write quality log: {err}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Command-line entry point for the NTD compilation workflow.

    Steps
    -----
    1. Validate critical configuration values.
    2. Compile all monthly NTD workbooks listed in ``FILES_TO_PROCESS`` into a
       single cleaned DataFrame (see ``compile_ntd_data``).
    3. **Audit** the compiled data and write a plain-text quality log
       (missing-month gaps, negative weekday-average rolling trends) via
       ``write_quality_log``.
    4. Report a concise summary of the compiled data (row × column count).

    The script aborts early if required configuration items are missing or
    if the compilation returns no data.
    """
    print("--- Starting NTD Data Compilation Script ---")

    # ────────────────────────────── configuration checks ────────────────────
    critical_ok = True
    if not FILES_TO_PROCESS:
        print("CRITICAL ERROR: 'FILES_TO_PROCESS' is empty.")
        critical_ok = False
    if not OUTPUT_FILE_PATH:
        print("CRITICAL ERROR: 'OUTPUT_FILE_PATH' not set.")
        critical_ok = False
    if not EXISTING_PERIOD_COLUMN_NAME:
        print(
            "WARNING: 'EXISTING_PERIOD_COLUMN_NAME' not set; period validation will be skipped."
        )

    # ─────────────────────────────── main pipeline ──────────────────────────
    if critical_ok:
        result_df = compile_ntd_data()

        if result_df is not None:
            # ---------- quality-flagger ------------------------------------
            write_quality_log(result_df)

            # ---------- summary -------------------------------------------
            print(
                f"\nCompilation summary: "
                f"{result_df.shape[0]} rows × {result_df.shape[1]} columns."
            )
        else:
            print("\nCompilation failed or returned no data.")
    else:
        print("\nScript terminated due to configuration errors.")

    print("\n--- Script Finished ---")


# Standard Python entry point
if __name__ == "__main__":  # pragma: no cover
    main()
