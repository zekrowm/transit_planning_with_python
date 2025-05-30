"""
Script Name:
    ntd_data_compiler.py

Purpose:
    Compiles monthly National Transit Database (NTD) ridership Excel reports into
    a single consolidated dataset, applying robust numeric conversions, filtering
    invalid rows, and providing diagnostics for data quality control.

Inputs:
    - Multiple Excel files (.xlsx), each containing ridership data.
    - Optional specific worksheet names within Excel files.

Outputs:
    - A consolidated CSV or Excel file containing all valid, processed rows.
    - (Optional) A separate audit file containing rows discarded during processing.

Dependencies:
    os, typing, pandas
"""

from __future__ import annotations  # postpone evaluation of type-hints

import os
from typing import List, Optional, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
FILES_TO_PROCESS: List[Tuple[str, Optional[str]]] = [
    (r"\\Your\File\Path\JULY 2024 NTD RIDERSHIP BY ROUTE.XLSX", "Temporary_Query_N"),
    (
        r"\\Your\File\Path\AUGUST 2024 NTD RIDERSHIP REPORT BY ROUTE.XLSX",
        "Temporary_Query_N",
    ),
    (r"\\Your\File\Path\SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.XLSX", "Sep.2024 Finals"),
]

OUTPUT_FILE_PATH: str = (
    r"\\Path\to\Your\Output_Folder\Compiled_NTD_Data.csv"  # or .xlsx
)

# Optional row-exclusion rules
DROPNA_SUBSET_ALL_NAN: Optional[List[str]] = None  # e.g. ["ROUTE_NAME", "MTH_BOARD"]
DROPNA_SUBSET_ANY_NAN: Optional[List[str]] = None  # e.g. ["ROUTE_NAME", "MTH_BOARD"]

EXISTING_PERIOD_COLUMN_NAME: str = "NameOfYourExistingMonthYearColumn"  # ← update


# =============================================================================
# FUNCTIONS
# =============================================================================
def robust_numeric_converter(value):
    """
    Convert strings such as "1,234" → 1234.0 (float).
    Returns None on blanks / NA or on conversion failure (with a warning).
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
# Mapping of column names → converters that will be passed to pandas.read_excel
# -----------------------------------------------------------------------------
COMMON_CONVERTERS = {
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
    """
    Return (kept_rows, discarded_rows) as DataFrames with identical columns.
    Either element may be empty; both are None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}. Skipping.")
        return None, None

    print(
        f"Processing file: {os.path.basename(file_path)}, "
        f"Sheet: {sheet_name or 'first available'}"
    )

    try:
        df = pd.read_excel(
            file_path, sheet_name=sheet_name, converters=COMMON_CONVERTERS
        )

        # ------------------ filtering logic ------------------
        keep_mask = pd.Series(True, index=df.index)

        if DROPNA_SUBSET_ALL_NAN:
            keep_mask &= ~df[DROPNA_SUBSET_ALL_NAN].isna().all(axis=1)

        if DROPNA_SUBSET_ANY_NAN:
            keep_mask &= ~df[DROPNA_SUBSET_ANY_NAN].isna().any(axis=1)

        kept_df = df[keep_mask].copy()
        dropped_df = df[~keep_mask].copy()

        # diagnostics
        if (
            EXISTING_PERIOD_COLUMN_NAME
            and EXISTING_PERIOD_COLUMN_NAME not in df.columns
        ):
            print(
                f"  Warning: period column "
                f"'{EXISTING_PERIOD_COLUMN_NAME}' not found."
            )

        print(
            f"  Rows kept: {len(kept_df):>6} | " f"Rows discarded: {len(dropped_df):>6}"
        )

        return kept_df, dropped_df

    except Exception as exc:  # broad catch to keep batch processing alive
        print(f"ERROR: failed to read {os.path.basename(file_path)}: {exc}")
        return None, None


def compile_ntd_data() -> Optional[pd.DataFrame]:
    """
    Iterate over FILES_TO_PROCESS, apply cleansing rules, write outputs,
    and return the concatenated DataFrame (or None on failure).
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
            """Write DataFrame to XLSX or CSV based on file extension."""
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


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Entry point for CLI execution."""
    print("--- Starting NTD Data Compilation Script ---")

    critical_ok = True
    if not FILES_TO_PROCESS:
        print("CRITICAL ERROR: 'FILES_TO_PROCESS' is empty.")
        critical_ok = False
    if not OUTPUT_FILE_PATH:
        print("CRITICAL ERROR: 'OUTPUT_FILE_PATH' not set.")
        critical_ok = False
    if not EXISTING_PERIOD_COLUMN_NAME:
        print(
            "WARNING: 'EXISTING_PERIOD_COLUMN_NAME' not set; "
            "period validation will be skipped."
        )

    if critical_ok:
        result_df = compile_ntd_data()
        if result_df is not None:
            print(
                f"\nCompilation summary: {result_df.shape[0]} rows × "
                f"{result_df.shape[1]} columns."
            )
        else:
            print("\nCompilation failed or returned no data.")
    else:
        print("\nScript terminated due to configuration errors.")

    print("\n--- Script Finished ---")


# Standard Python entry point
if __name__ == "__main__":  # pragma: no cover
    main()
