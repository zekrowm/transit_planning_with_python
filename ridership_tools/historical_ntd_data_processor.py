"""
This module consolidates ridership data across multiple Excel workbooks,
specifically handling separate worksheets for different day types (Weekday,
Saturday, Sunday). For each month, the script extracts route-level totals,
day counts, and averages for each day type, then sums them to produce a
monthly total.

Key Features:
1. Automated Day-Type Detection: Searches workbook sheets for names
   matching "Weekday(s)", "Saturday(s)", "Sunday(s)" (in any case).
2. Consolidated Outputs: Each month generates columns such as
   Jan-24_WeekdayTotal, Jan-24_WeekdayDays, Jan-24_WeekdayAverage, and
   Jan-24_MonthlyTotal.
3. Route Exclusions: Configurable list of routes to exclude from final
   outputs; excluded routes are logged separately for reference.
4. Plotting Enhancements: Each day-type variable (WeekdayTotal,
   SaturdayAverage, etc.) is plotted on its own chart, allowing clearer
   visualization of each metric over time.
5. Easy Configuration: Centralized settings control the input directory,
   output paths, route exclusions, plotting behavior, and more.

The script first consolidates data into a single CSV, then optionally
generates per-route line charts for each day-type variable across all months.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Part 1: Consolidation Config ---

BASE_INPUT_DIR = r"\\S40SHAREPGC01\DOTWorking\zkrohm\_data_archive\ntd_ridership"
OUTPUT_DIR = r"\\S40SHAREPGC01\DOTWorking\zkrohm\analysis_requests\route_622_722_investigation\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Each key is a "Month String", each value is the Excel filename for that month
FILE_SHEET_MAPPING = {
    'September-23':  'NTD RIDERSHIP BY ROUTE SEP 2023.xlsx',
    'October-23':    'OCTOBER 2023  NTD RIDERSHIP_BY_ROUTE.xlsx',
    'November-23':   'November 2023 NTD RIDERSHIP BY ROUTE.xlsx',
    'December-23':   'NTD RIDERSHIP BY ROUTE DECEMBER 2023.xlsx',
    'January-24':    'NTD RIDERSHIP BY ROUTE JANUARY 2024 FINAL 22824.xlsx',
    'February-24':   'NTD RIDERSHIP BY ROUTE FEBRUARY 2024.xlsx',
    'March-24':      'MARCH 2024 NTD RIDERSHIP BY ROUTE AND LOCATION.xlsx',
    'April-24':      'APRIL 2024 NTD RIDERSHIP BY ROUTE (002).xlsx',
    'May-24':        'NTD RIDERSHIP BY ROUTE MAY 2024.xlsx',
    'June-24':       'NTD RIDERSHIP BY ROUTE JUNE 2024.xlsx',
    'July-24':       'JULY 2024 NTD RIDERSHIP BY ROUTE.xlsx',
    'August-24':     'AUGUST 2024  NTD RIDERSHIP REPORT BY ROUTE.xlsx',
    'September-24':  'SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.xlsx',
    'October-24':    'NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.xlsx',
    'November-24':   'NTD RIDERSHIP BY ROUTE-NOVEMBER 2024.xlsx',
    'December-24':   'NTD RIDERSHIP BY MONTH_DECEMBER 2024.xlsx',
    'January-25': 'NTD RIDERSHIP BY MONTH-JANUARY 2025',
    'February-25': 'NTD RIDERSHIP BY MONTH-FEBRUARY 2025'
}

# We want to exclude certain routes from the final data
ROUTES_TO_EXCLUDE = ['101', '202', '303']

# Which columns do we expect in each sheet for routes + ridership + days
ROUTE_COLUMN_NAME = 'ROUTE_NAME'
RIDERSHIP_COLUMN_NAME = 'MTH_BOARD'
DAYS_COLUMN_NAME = 'DAYS'

# Regex patterns to identify each day-type sheet (match singular/plural, any case)
DAYTYPE_PATTERNS = {
    "WEEKDAYS":   re.compile(r'(?i)^weekday(s)?$'),
    "SATURDAYS":  re.compile(r'(?i)^saturday(s)?$'),
    "SUNDAYS":    re.compile(r'(?i)^sunday(s)?$')
}


# --- Part 2: Plotting & Analysis Config ---

ENABLE_PLOTTING = True

# Which routes to plot? If empty, will plot all. If you only want 622, 722, etc.:
ROUTES_OF_INTEREST = []

# Folder to store plot images
PLOTS_OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Plots")
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)

# Plot appearance
FIG_SIZE = (10, 6)
MARKER_STYLE = 'o'
LINE_STYLE = '-'
LINE_WIDTH = 2.0

# If you only want to plot certain lines (day types) in each chart, set them here:
# e.g., ["WeekdayTotal", "SaturdayTotal", "SundayTotal", "MonthlyTotal"]
DAYTYPES_TO_PLOT = ["WeekdayTotal", "SaturdayTotal", "SundayTotal", "MonthlyTotal",
                    "WeekdayAverage", "SaturdayAverage", "SundayAverage"]

# -----------------------------------------------------------------------------
# Part 1: Consolidation Function
# -----------------------------------------------------------------------------


def extract_route_ridership(df, route_col, ridership_col, days_col):
    """
    Extract minimal columns (route, ridership, days), handle missing or invalid values,
    and clean the route names to uppercase without spaces.
    """
    missing = []
    for col in [route_col, ridership_col, days_col]:
        if col not in df.columns:
            missing.append(col)

    # We'll allow DAYS to be missing, just fill with NaN
    if days_col in missing:
        missing.remove(days_col)
        df[days_col] = np.nan

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only these 3 columns
    df = df[[route_col, ridership_col, days_col]].copy()

    # Drop rows missing route or ridership
    df.dropna(subset=[route_col, ridership_col], inplace=True)

    # Clean route
    df[route_col] = (df[route_col]
                     .astype(str)
                     .str.strip()
                     .str.upper()
                     .str.replace(' ', '', regex=False))
    df[route_col] = df[route_col].apply(lambda x: re.sub(r'\.0$', '', x))

    # Convert ridership, days to numeric
    df[ridership_col] = pd.to_numeric(df[ridership_col], errors='coerce').fillna(0)
    df[days_col]      = pd.to_numeric(df[days_col], errors='coerce').fillna(0)

    return df


def consolidate_ridership_data():
    """
    Reads all Excel files from FILE_SHEET_MAPPING, merges day types, outputs one CSV.
    Creates columns for each month+daytype: e.g. 'Jan-24_WeekdayTotal', 'Jan-24_WeekdayDays',
    'Jan-24_WeekdayAverage', etc., plus 'Jan-24_MonthlyTotal'.
    """
    consolidated_df = pd.DataFrame()
    excluded_routes_list = []
    total_excluded_count = 0

    for month_str, file_name in FILE_SHEET_MAPPING.items():
        file_path = os.path.join(BASE_INPUT_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping {month_str}.")
            continue

        # Identify all sheets in the workbook
        try:
            xls = pd.ExcelFile(file_path)
            all_sheets = xls.sheet_names
        except FileNotFoundError as exc:
            print(f"Warning: File not found: {file_path} → {exc}")
            continue
        except PermissionError as exc:
            print(f"Warning: Permission error reading {file_path} → {exc}")
            continue
        except ValueError as exc:
            print(f"Warning: Not a valid Excel file {file_path} → {exc}")
            continue

        daytype_dfs = {}
        # Attempt to find Weekday, Saturday, Sunday sheets
        for canonical_daytype, pattern in DAYTYPE_PATTERNS.items():
            matched = [s for s in all_sheets if pattern.match(str(s).strip())]
            if not matched:
                print(f"Warning: No sheet for {canonical_daytype} in {file_name}.")
                continue

            sheet_name = matched[0]  # If multiple, pick first or do your own logic

            try:
                df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
                df_ext = extract_route_ridership(
                    df_raw,
                    route_col=ROUTE_COLUMN_NAME,
                    ridership_col=RIDERSHIP_COLUMN_NAME,
                    days_col=DAYS_COLUMN_NAME
                )
                daytype_dfs[canonical_daytype] = df_ext
                print(f"Read '{sheet_name}' → {canonical_daytype} for {month_str} ({len(df_ext)} rows).")

            except PermissionError as exc:
                print(f"Warning: Permission error reading sheet '{sheet_name}' in '{file_name}': {exc}")
            except ValueError as exc:
                # Typically raised if the sheet is invalid / format is off
                print(f"Warning: Invalid data/format in sheet '{sheet_name}' of '{file_name}': {exc}")
            # If you truly need a fallback, you can add a final broad except as last resort:
            # except Exception as exc:
            #     print(f"Warning: Unexpected error reading {sheet_name} in {file_name}: {exc}")

        if not daytype_dfs:
            print(f"No day-type sheets found for {month_str}. Skipping.")
            continue

        # Merge them on ROUTE_NAME
        merged = None
        for dt_name, df_temp in daytype_dfs.items():
            # Exclude routes
            routes_ex = df_temp[df_temp[ROUTE_COLUMN_NAME].isin(ROUTES_TO_EXCLUDE)]
            if not routes_ex.empty:
                routes_ex['Month'] = month_str
                excluded_routes_list.append(routes_ex)
                total_excluded_count += len(routes_ex)

            # Keep the rest
            df_temp = df_temp[~df_temp[ROUTE_COLUMN_NAME].isin(ROUTES_TO_EXCLUDE)]
            if df_temp.empty:
                continue

            # Rename columns to something like 'RIDERSHIP_WEEKDAYS', 'DAYS_WEEKDAYS'
            ridership_col_new = f"RIDERSHIP_{dt_name}"
            days_col_new = f"DAYS_{dt_name}"
            df_temp = df_temp.rename(columns={
                RIDERSHIP_COLUMN_NAME: ridership_col_new,
                DAYS_COLUMN_NAME: days_col_new
            })

            if merged is None:
                merged = df_temp
            else:
                merged = pd.merge(
                    merged,
                    df_temp[[ROUTE_COLUMN_NAME, ridership_col_new, days_col_new]],
                    on=ROUTE_COLUMN_NAME,
                    how='outer'
                )

        if merged is None or merged.empty:
            print(f"No data left for {month_str}. Skipping.")
            continue

        merged.fillna(0, inplace=True)

        # For each daytype, create an 'AVERAGE_' column
        for dt in ["WEEKDAYS", "SATURDAYS", "SUNDAYS"]:
            rcol = f"RIDERSHIP_{dt}"
            dcol = f"DAYS_{dt}"
            acol = f"AVERAGE_{dt}"
            merged[acol] = np.where(merged[dcol] > 0, merged[rcol] / merged[dcol], 0)

        # Create a single 'RIDERSHIP_MONTHLY_TOTAL' = sum of the three day types
        merged["RIDERSHIP_MONTHLY_TOTAL"] = (
            merged.get("RIDERSHIP_WEEKDAYS", 0) +
            merged.get("RIDERSHIP_SATURDAYS", 0) +
            merged.get("RIDERSHIP_SUNDAYS", 0)
        )

        # Convert e.g. 'January-24' → 'Jan-24'
        try:
            dt_val = pd.to_datetime(month_str, format='%B-%y')
            month_abbr = dt_val.strftime('%b-%y')
        except ValueError:
            month_abbr = month_str

        # Rename to final columns: 'Jan-24_WeekdayTotal', etc.
        rename_map = {}
        for c in merged.columns:
            if c == ROUTE_COLUMN_NAME:
                continue
            if c == "RIDERSHIP_WEEKDAYS":
                rename_map[c] = f"{month_abbr}_WeekdayTotal"
            elif c == "DAYS_WEEKDAYS":
                rename_map[c] = f"{month_abbr}_WeekdayDays"
            elif c == "AVERAGE_WEEKDAYS":
                rename_map[c] = f"{month_abbr}_WeekdayAverage"

            elif c == "RIDERSHIP_SATURDAYS":
                rename_map[c] = f"{month_abbr}_SaturdayTotal"
            elif c == "DAYS_SATURDAYS":
                rename_map[c] = f"{month_abbr}_SaturdayDays"
            elif c == "AVERAGE_SATURDAYS":
                rename_map[c] = f"{month_abbr}_SaturdayAverage"

            elif c == "RIDERSHIP_SUNDAYS":
                rename_map[c] = f"{month_abbr}_SundayTotal"
            elif c == "DAYS_SUNDAYS":
                rename_map[c] = f"{month_abbr}_SundayDays"
            elif c == "AVERAGE_SUNDAYS":
                rename_map[c] = f"{month_abbr}_SundayAverage"

            elif c == "RIDERSHIP_MONTHLY_TOTAL":
                rename_map[c] = f"{month_abbr}_MonthlyTotal"

        merged.rename(columns=rename_map, inplace=True)

        # Add to final consolidated
        if consolidated_df.empty:
            consolidated_df = merged
        else:
            consolidated_df = pd.merge(
                consolidated_df,
                merged,
                on=ROUTE_COLUMN_NAME,
                how='outer'
            )

    if consolidated_df.empty:
        print("No data was processed at all.")
        return None

    # Save the excluded routes info if any
    if excluded_routes_list:
        excluded_all = pd.concat(excluded_routes_list, ignore_index=True)
        excl_file = os.path.join(OUTPUT_DIR, "Excluded_Routes.xlsx")
        try:
            excluded_all.to_excel(excl_file, index=False)
            print(f"Excluded routes saved to {excl_file} (Count={total_excluded_count}).")
        except Exception as exc:
            # If you only ever expect PermissionError here,
            # you could narrow this down to that if you prefer.
            print(f"Error saving excluded routes: {exc}")

    # Sort by route
    try:
        consolidated_df["ROUTE_SORT"] = consolidated_df[ROUTE_COLUMN_NAME].astype(int)
    except ValueError:
        consolidated_df["ROUTE_SORT"] = consolidated_df[ROUTE_COLUMN_NAME]

    consolidated_df.sort_values("ROUTE_SORT", inplace=True)
    consolidated_df.drop(columns="ROUTE_SORT", inplace=True)

    # Export final CSV
    out_csv = os.path.join(OUTPUT_DIR, "Consolidated_Ridership_Data.csv")
    consolidated_df.to_csv(out_csv, index=False)
    print(f"Consolidated CSV saved to {out_csv}")
    return out_csv


# -----------------------------------------------------------------------------
# Part 2: Load & Filter
# -----------------------------------------------------------------------------


def load_data(csv_path):
    """
    Simple CSV loader with error handling
    """
    try:
        df = pd.read_csv(csv_path)
        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()
        print(f"Loaded data from {csv_path} with {df.shape[0]} rows.")
        return df
    except Exception as exc:
        print(f"Error loading {csv_path}: {exc}")
        sys.exit(1)


def filter_routes(df, route_col="ROUTE_NAME", routes_of_interest=None):
    """
    Return either the entire df if routes_of_interest is empty,
    or only the routes specified.
    """
    if not routes_of_interest:
        return df
    fdf = df[df[route_col].isin(routes_of_interest)].copy()
    print(f"Filtered down to {fdf.shape[0]} rows for routes: {routes_of_interest}")
    return fdf


# -----------------------------------------------------------------------------
# Part 3: Plotting (Revised to separate each variable into its own chart)
# -----------------------------------------------------------------------------


def plot_ridership(df, route_col="ROUTE_NAME"):
    """
    For each route in df, we:
       1) Identify columns that match the pattern of time-series ridership
          e.g. "Jan-24_WeekdayTotal", "Feb-24_WeekdayAverage", etc.
       2) Parse out the month and the type (WeekdayTotal, SaturdayTotal, etc.).
       3) For each distinct variable in DAYTYPES_TO_PLOT (e.g. 'WeekdayTotal'),
          plot it on a *separate* figure for that route.
    """
    pattern = re.compile(r'^([A-Za-z]{3}-\d{2})_(Weekday|Saturday|Sunday|Monthly)(Total|Days|Average)$')

    # Identify the columns that match the day-type pattern
    matched_cols = []
    for col in df.columns:
        if col == route_col:
            continue
        m = pattern.match(col)
        if m:
            base_month = m.group(1)   # e.g. "Jan-24"
            daytype    = m.group(2)   # e.g. "Weekday"
            suffix     = m.group(3)   # e.g. "Total" or "Average" or "Days"
            matched_cols.append((col, base_month, daytype, suffix))

    if not matched_cols:
        print("No recognized columns matching the day-type pattern. Nothing to plot.")
        return

    # Convert base_month to a datetime for sorting, store for convenience
    matched_with_dates = []
    for (col, bm, dt_str, sx) in matched_cols:
        try:
            dt_val = pd.to_datetime(bm, format='%b-%y')
        except ValueError:
            dt_val = None
        # Combine dt_str + sx => e.g. "WeekdayTotal", "MonthlyTotal"
        full_tag = dt_str + sx  # e.g. "WeekdayTotal" or "MonthlyTotal"
        matched_with_dates.append((dt_val, col, bm, full_tag))

    routes = df[route_col].unique()

    for route in routes:
        row = df[df[route_col] == route].squeeze()
        if row.empty:
            continue

        # Build up a dictionary: { "WeekdayTotal": [(dt, "Jan-24", val), ...], ... }
        var_dict = {}
        for (dt_val, col, base_month, full_tag) in matched_with_dates:
            if dt_val is not None and full_tag in DAYTYPES_TO_PLOT:
                if full_tag not in var_dict:
                    var_dict[full_tag] = []
                # numeric value from the row
                val = float(row[col]) if pd.notnull(row[col]) else 0
                var_dict[full_tag].append((dt_val, base_month, val))

        # For each variable in var_dict, create a *separate* figure
        for variable_label, items in var_dict.items():
            # Sort by datetime
            items.sort(key=lambda x: x[0])
            x_vals = [i[1] for i in items]  # e.g. "Jan-24", "Feb-24" ...
            y_vals = [i[2] for i in items]

            # Skip if there's no data
            if len(x_vals) == 0:
                continue

            plt.figure(figsize=FIG_SIZE)
            plt.plot(x_vals, y_vals, marker=MARKER_STYLE, linestyle=LINE_STYLE,
                     linewidth=LINE_WIDTH, label=variable_label)

            plt.title(f"{variable_label} Over Time (Route {route})")
            plt.xlabel("Month")
            plt.ylabel("Ridership")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.tight_layout()

            # Save figure
            safe_var_label = re.sub(r'[^A-Za-z0-9]+', '_', variable_label)
            out_path = os.path.join(
                PLOTS_OUTPUT_FOLDER,
                f"Route_{route}_{safe_var_label}.png"
            )
            try:
                plt.savefig(out_path)
                print(f"Saved plot for route {route}, variable='{variable_label}' → {out_path}")
            except Exception as exc:
                print(f"Error saving plot for route {route}, variable='{variable_label}': {exc}")
            finally:
                plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Main entry point for consolidating ridership data and generating plots.

    Steps:
    1. Consolidates day-type data from multiple Excel files into one CSV.
    2. Loads the resulting CSV into a DataFrame.
    3. Filters the DataFrame for routes of interest (if specified).
    4. Generates per-route line charts for each day-type metric (if plotting is enabled).
    5. Saves final outputs and plots to the configured output directory.
    """
    # 1) Consolidate to CSV
    csv_path = consolidate_ridership_data()
    if not csv_path:
        print("No data to analyze or plot.")
        sys.exit(0)

    # 2) Load data
    df = load_data(csv_path)
    route_col = ROUTE_COLUMN_NAME  # e.g. "ROUTE_NAME"

    # 3) Filter routes
    df_filt = filter_routes(df, route_col=route_col, routes_of_interest=ROUTES_OF_INTEREST)
    if df_filt.empty:
        print(f"No rows found for routes {ROUTES_OF_INTEREST}. Exiting.")
        sys.exit(0)

    # 4) Plot if enabled
    if ENABLE_PLOTTING:
        plot_ridership(df_filt, route_col=route_col)
    else:
        print("Plotting disabled.")

    print("\nDone.")


if __name__ == "__main__":
    main()
