"""
Processes stop-level ridership data from an Excel file.

Reads an input Excel file (RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX),
filters by route or stop ID, aggregates boardings and alightings by stop and time
period, and saves the results to a new Excel file. Aggregated data can
optionally be rounded or categorized into bins. It is useful for fulfilling
stop-based data requests.

Designed for use in ArcGIS Pro or Jupyter notebooks, typically as part of a
manual or scripted data request workflow.
"""

import os
import sys

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE_PATH = r"\\Path\To\Your\RIDERSHIP_BY_ROUTE_AND_STOP_(ALL_TIME_PERIODS).XLSX"
OUTPUT_FILE_SUFFIX = "_processed"
OUTPUT_FILE_EXTENSION = ".xlsx"

# ROUTES = keep-only list   |  ROUTES_EXCLUDE = toss-out list
ROUTES = []  # keep these (leave empty → keep all)
ROUTES_EXCLUDE = []  # drop these (leave empty → drop none)

# Optional STOP_IDS filter list
STOP_IDS = [1001, 2002, 3003]  # keep these (leave empty → keep all)

# Optional STOP_IDS aggregation list
# If empty, the script will skip time-period breakdown for these lists.
TIME_PERIODS = [
    "AM EARLY",
    "AM PEAK",
    "MIDDAY",
    "PM PEAK",
    "PM LATE",
    "PM NITE",
]  # e.g. [] means skip time-period breakdown

# If True, ridership columns in the "Original" data are rounded to 1 decimal place.
# Also, if AGGREGATE_BIN_RANGES = False, aggregated totals get rounded instead of binned.
APPLY_ROUNDING = True

# If True, aggregated totals (BOARD_ALL_TOTAL, ALIGHT_ALL_TOTAL) are converted to
# "0-4.9", "5-24.9", or "25 or more". If False, they remain numeric and
# get rounded to 1 decimal place only if APPLY_ROUNDING = True.
AGGREGATE_BIN_RANGES = False

REQUIRED_COLUMNS = [
    "TIME_PERIOD",
    "ROUTE_NAME",
    "STOP",
    "STOP_ID",
    "BOARD_ALL",
    "ALIGHT_ALL",
]
COLUMNS_TO_RETAIN = ["ROUTE_NAME", "STOP", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL"]

# =============================================================================
# FUNCTIONS
# =============================================================================


def bin_ridership_value(value):
    """
    Categorize a ridership value into a range (e.g., "0-4.9", "5-24.9", "25 or more").
    """
    if value < 5:
        return "0-4.9"
    if value < 25:
        return "5-24.9"
    return (
        "25 or more"  # Removed unnecessary `elif` and final `else` per no-else-return.
    )


def aggregate_by_stop(data_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ridership by stop, totaling boardings and alightings, and listing
    unique routes.

    Fixes implemented
    -----------------
    1. **FutureWarning** – We keep only the columns we will aggregate before the
       group-by, so pandas never has to “drop invalid columns”.
    2. **TypeError** – ROUTE_NAME is converted to `str`, guaranteeing that
       ','.join() always receives strings.
    """
    # Keep only the columns required for this aggregation step
    cols_needed = ["STOP", "STOP_ID", "BOARD_ALL", "ALIGHT_ALL", "ROUTE_NAME"]
    subset = data_subset[cols_needed].copy()

    # Force ROUTE_NAME to string so join() can’t choke on numeric dtypes
    subset["ROUTE_NAME"] = subset["ROUTE_NAME"].astype(str).str.strip()

    aggregated = (
        subset.groupby(["STOP", "STOP_ID"], as_index=False)
        .agg(
            {
                "BOARD_ALL": "sum",
                "ALIGHT_ALL": "sum",
                "ROUTE_NAME": lambda x: ", ".join(sorted(x.unique())),
            }
        )
        .rename(
            columns={
                "BOARD_ALL": "BOARD_ALL_TOTAL",
                "ALIGHT_ALL": "ALIGHT_ALL_TOTAL",
                "ROUTE_NAME": "ROUTES",
            }
        )
    )
    return aggregated


def read_excel_file(input_file):
    """
    Load an Excel file into a DataFrame. Exits if the file is missing or unreadable.
    """
    try:
        return pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
    except ValueError as error:  # Replaced pd.errors.ExcelFileError with ValueError
        print(f"Error reading the Excel file: {error}")
        sys.exit(1)


def verify_required_columns(data_frame, required_columns):
    """
    Check if all required columns exist in the DataFrame. Exits if any are missing.
    """
    missing_columns = [col for col in required_columns if col not in data_frame.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        sys.exit(1)


def filter_data(data_frame, routes=None, stop_ids=None, routes_exclude=None):
    """
    Apply three *optional* filters in this order:
        1. keep-only ROUTES         (inclusive)
        2. drop-only ROUTES_EXCLUDE (exclusive)
        3. keep-only STOP_IDS
    Any of the three may be an empty list / None to skip that step.
    """
    df = data_frame.copy()

    # 1. inclusive route filter -------------------------------------------
    if routes:
        df = df[df["ROUTE_NAME"].isin(routes)]

    # 2. exclusive route filter -------------------------------------------
    if routes_exclude:
        df = df[~df["ROUTE_NAME"].isin(routes_exclude)]

    # 3. stop-id filter ----------------------------------------------------
    if stop_ids:
        df = df[df["STOP_ID"].isin(stop_ids)]

    return df


def write_to_excel(output_file, filtered_data, aggregated_peaks, all_time_aggregated):
    """
    Save processed ridership data to an Excel file with multiple sheets.
    """
    try:
        # Remove 'engine="openpyxl"' to avoid abstract-class-instantiated warnings
        with pd.ExcelWriter(output_file) as writer:
            filtered_data.to_excel(writer, sheet_name="Original", index=False)

            # Write each time period's aggregated data
            for period, df_agg in aggregated_peaks.items():
                df_agg.to_excel(writer, sheet_name=period, index=False)

            # Always write the all-time aggregated data
            all_time_aggregated.to_excel(
                writer, sheet_name="All Time Periods", index=False
            )

            writer.save()

        adjust_excel_formatting(output_file)
        print(f"Success: The processed file has been saved as '{output_file}'.")
    except (OSError, PermissionError) as error:
        print(f"Error writing the processed Excel file: {error}")
        sys.exit(1)


def adjust_excel_formatting(output_file):
    """
    Format an Excel file by bolding headers and adjusting column widths.
    """
    try:
        workbook = load_workbook(output_file)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            # Bold the header row
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            # Auto-adjust column widths
            for column_cells in sheet.columns:
                max_length = 0
                col_letter = column_cells[0].column_letter
                for cell in column_cells:
                    cell_val = str(cell.value) if cell.value is not None else ""
                    max_length = max(max_length, len(cell_val))
                sheet.column_dimensions[col_letter].width = max_length + 2
        workbook.save(output_file)
    except (OSError, PermissionError) as error:
        print(f"Error adjusting Excel formatting: {error}")
        sys.exit(1)


def process_aggregations(filtered_data: pd.DataFrame):
    """
    Handle rounding/bins for original and aggregated data. Returns the
    final versions of the filtered data, the aggregated peaks, and the
    all-time aggregated DataFrame.

    This wrapper now forces ROUTE_NAME to str *once* so every downstream use
    (original sheet as well as any future aggregations) is safe.
    """
    # ------------------------------------------------------------------
    # 0) Make ROUTE_NAME consistently string *before* any aggregation
    # ------------------------------------------------------------------
    filtered_data["ROUTE_NAME"] = filtered_data["ROUTE_NAME"].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 1) Build time-period subsets if TIME_PERIODS is non-empty
    # ------------------------------------------------------------------
    peak_data_dict = {}
    if TIME_PERIODS:
        for period in TIME_PERIODS:
            period_upper = period.upper()
            subset = filtered_data[filtered_data["TIME_PERIOD"] == period_upper]
            peak_data_dict[period] = subset[COLUMNS_TO_RETAIN]

    # ------------------------------------------------------------------
    # 2) Aggregate data (all-time and by time period)
    # ------------------------------------------------------------------
    all_time_aggregated = aggregate_by_stop(filtered_data)

    aggregated_peaks = {}
    if TIME_PERIODS:
        for period, data_subset in peak_data_dict.items():
            aggregated_peaks[period] = aggregate_by_stop(data_subset)

    # ------------------------------------------------------------------
    # 3) Round the original ridership columns if requested
    # ------------------------------------------------------------------
    if APPLY_ROUNDING:
        filtered_data[["BOARD_ALL", "ALIGHT_ALL"]] = filtered_data[
            ["BOARD_ALL", "ALIGHT_ALL"]
        ].round(1)

    # ------------------------------------------------------------------
    # 4) Format aggregated columns (rounding or binning)
    # ------------------------------------------------------------------
    all_dfs = [all_time_aggregated] + list(aggregated_peaks.values())
    for df_agg in all_dfs:
        if AGGREGATE_BIN_RANGES:
            # Convert numeric aggregated totals into bins
            for col in ["BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"]:
                df_agg[col] = df_agg[col].apply(bin_ridership_value)
        elif APPLY_ROUNDING:
            # Otherwise, if rounding is desired, do decimal rounding
            df_agg[["BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"]] = df_agg[
                ["BOARD_ALL_TOTAL", "ALIGHT_ALL_TOTAL"]
            ].round(1)

    return filtered_data, aggregated_peaks, all_time_aggregated


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Process ridership data: read, filter, aggregate, apply formatting, and save to Excel.
    """
    input_file = INPUT_FILE_PATH
    base, ext = os.path.splitext(input_file)
    ext = ext.lower()
    if ext != OUTPUT_FILE_EXTENSION:
        print(
            f"Warning: The input file has extension '{ext}'. "
            f"Using '{OUTPUT_FILE_EXTENSION}' for output."
        )
        ext = OUTPUT_FILE_EXTENSION
    output_file = f"{base}{OUTPUT_FILE_SUFFIX}{ext}"

    # Read and verify the Excel data
    ridership_df = read_excel_file(input_file)
    verify_required_columns(ridership_df, REQUIRED_COLUMNS)

    # Apply optional filters
    filtered_data = filter_data(
        ridership_df,
        routes=ROUTES,
        stop_ids=STOP_IDS,
        routes_exclude=ROUTES_EXCLUDE,  # <-- NEW ARG
    )

    # Standardize 'TIME_PERIOD' values
    filtered_data["TIME_PERIOD"] = (
        filtered_data["TIME_PERIOD"].astype(str).str.strip().str.upper()
    )

    # Process and retrieve final aggregated data
    final_filtered, aggregated_peaks, all_time_aggregated = process_aggregations(
        filtered_data
    )

    # Write the data to Excel
    write_to_excel(output_file, final_filtered, aggregated_peaks, all_time_aggregated)


if __name__ == "__main__":
    main()
