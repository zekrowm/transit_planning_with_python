"""
Script Name:
        load_factor_violation_flagger.py

Purpose:
        Processes bus route data to analyze passenger load factors against
        defined capacity standards, determines compliance, and applies
        configurable route filters and rounding for load factor calculations.

Inputs:
        1. Excel file (INPUT_FILE): Contains raw bus statistics by route and trip.
        2. Configuration constants (defined in the script):
           - BUS_CAPACITY: Standard capacity of a bus.
           - HIGHER_LIMIT_ROUTES: List of routes with a higher load factor limit.
           - LOWER_LIMIT_ROUTES: List of routes with a lower load factor limit.
           - LOWER_LOAD_FACTOR_LIMIT: The load factor limit for routes in LOWER_LIMIT_ROUTES.
           - HIGHER_LOAD_FACTOR_LIMIT: The load factor limit for other routes.
           - FILTER_IN_ROUTES: Optional list of routes to include in processing.
           - FILTER_OUT_ROUTES: Optional list of routes to exclude from processing.
           - DECIMAL_PLACES: Number of decimal places for rounding the load factor.

Outputs:     - Processed Excel file (OUTPUT_FILE): Contains the original data
               enriched with service period, calculated load factor, load factor
               violation status, and route limit type.
             - Console output: Prints trips with MAX_LOAD over 30 and the path
               to the processed output file.

Dependencies:
        pandas, openpyxl
"""

import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"\\File\Path\To\Your\STATISTICS_BY_ROUTE_AND_TRIP.XLSX"
OUTPUT_FILE = INPUT_FILE.replace(".XLSX", "_processed.xlsx")
BUS_CAPACITY = 39

# Routes in this list use a load factor limit of 1.25.
# Any route not in LOWER_LIMIT_ROUTES will default to 1.25.
HIGHER_LIMIT_ROUTES = ["101", "102", "103", "104"]

# Routes in this list use a load factor limit of 1.0.
# Many agencies have a lower load factor for express routes.
LOWER_LIMIT_ROUTES = ["105", "106"]

LOWER_LOAD_FACTOR_LIMIT = 1.0
HIGHER_LOAD_FACTOR_LIMIT = 1.25

# Provide these as lists of route strings.
# Leave them empty if you do not want any filtering.
FILTER_IN_ROUTES = []  # e.g. ['101', '202']
FILTER_OUT_ROUTES = []  # e.g. ['105', '106']

# Specify how many decimals to round the LOAD_FACTOR to
DECIMAL_PLACES = 4  # default is 4 decimals

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

WRITE_VIOLATION_LOG = True
VIOLATION_LOG_FILE = OUTPUT_FILE.replace(".xlsx", "_violations_log.txt")

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(input_file: str) -> pd.DataFrame:
    """
    Loads bus data from an Excel file and returns a DataFrame
    containing selected columns.
    """
    data_frame = pd.read_excel(input_file)
    selected_columns = [
        "SERIAL_NUMBER",
        "ROUTE_NAME",
        "DIRECTION_NAME",
        "TRIP_START_TIME",
        "BLOCK",
        "MAX_LOAD",
        "MAX_LOAD_P",
        "ALL_RECORDS_MAX_LOAD",
    ]
    return data_frame[selected_columns]


def assign_service_period(ts):
    """ts is assumed to be a datetime or time object."""
    hour = ts.hour
    if 4 <= hour < 6:
        return "AM Early"
    elif 6 <= hour < 9:
        return "AM Peak"
    elif 9 <= hour < 15:
        return "Midday"
    elif 15 <= hour < 18:
        return "PM Peak"
    elif 18 <= hour < 21:
        return "PM Late"
    elif 21 <= hour < 24:
        return "PM Nite"
    else:
        return "Other"


def get_route_load_limit(route_name: str) -> float:
    """
    Get the appropriate load factor limit based on the route.

    If the route is in LOWER_LIMIT_ROUTES, returns the lower limit;
    otherwise, returns the higher limit.
    """
    if route_name in LOWER_LIMIT_ROUTES:
        return LOWER_LOAD_FACTOR_LIMIT
    return HIGHER_LOAD_FACTOR_LIMIT


def check_load_factor_violation(row: pd.Series) -> str:
    """Determine if the row exceeds the route's load factor limit."""
    limit = get_route_load_limit(row["ROUTE_NAME"])
    return "TRUE" if row["LOAD_FACTOR"] > limit else "FALSE"


def determine_limit_type(route_name: str) -> str:
    """
    Return 'HIGH' if the route uses the higher limit,
    or 'LOW' if the route uses the lower limit.
    """
    if route_name in LOWER_LIMIT_ROUTES:
        return "LOW"
    return "HIGH"


def process_data(
    data_frame: pd.DataFrame,
    bus_capacity: int,
    filter_in_routes: list,
    filter_out_routes: list,
    decimals: int,
) -> pd.DataFrame:
    """
    Processes bus data to filter routes, calculate load factors,
    determine service periods, identify load factor violations,
    and categorize route limit types. Returns the processed DataFrame.
    """
    # 1) Apply filters
    if filter_in_routes:
        data_frame = data_frame[data_frame["ROUTE_NAME"].isin(filter_in_routes)]
    if filter_out_routes:
        data_frame = data_frame[~data_frame["ROUTE_NAME"].isin(filter_out_routes)]

    # 2) Assign service period and calculate load factor
    data_frame["SERVICE_PERIOD"] = data_frame["TRIP_START_TIME"].apply(
        assign_service_period
    )
    data_frame["LOAD_FACTOR"] = data_frame["MAX_LOAD"] / bus_capacity

    # 5) Round load factor to specified decimals
    data_frame["LOAD_FACTOR"] = data_frame["LOAD_FACTOR"].round(decimals)

    # 3) Mark whether load factor is violated
    data_frame["LOAD_FACTOR_VIOLATION"] = data_frame.apply(
        check_load_factor_violation, axis=1
    )

    # 4) Add column for route limit type
    data_frame["ROUTE_LIMIT_TYPE"] = data_frame["ROUTE_NAME"].apply(
        determine_limit_type
    )

    # Sort by 'LOAD_FACTOR' in descending order
    return data_frame.sort_values(by="LOAD_FACTOR", ascending=False)


def create_route_workbooks(data_frame: pd.DataFrame) -> None:
    """
    For each unique route in the processed data, create an Excel workbook
    named '{route_name}.xlsx' in the same folder as OUTPUT_FILE. Each workbook
    contains one sheet per direction, with trips sorted by TRIP_START_TIME.
    """
    # Determine the directory in which to save per-route files
    output_dir = os.path.dirname(OUTPUT_FILE) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Group by each ROUTE_NAME
    for route_name, route_df in data_frame.groupby("ROUTE_NAME", sort=False):
        wb = Workbook()
        default_sheet = wb.active
        wb.remove(default_sheet)

        # Within each route, group by DIRECTION_NAME
        for direction_name, direction_df in route_df.groupby(
            "DIRECTION_NAME", sort=False
        ):
            # Sort trips by TRIP_START_TIME
            direction_df_sorted = direction_df.sort_values(
                by="TRIP_START_TIME", kind="mergesort"
            ).reset_index(drop=True)

            ws = wb.create_sheet(title=str(direction_name))

            # Write header row (bolded)
            headers = list(direction_df_sorted.columns)
            for col_idx, header in enumerate(headers, start=1):
                cell = ws.cell(row=1, column=col_idx, value=header)
                cell.font = Font(bold=True)

            # Write each trip row
            for row_idx, (_, row) in enumerate(direction_df_sorted.iterrows(), start=2):
                for col_idx, header in enumerate(headers, start=1):
                    val = row[header]
                    if header == "TRIP_START_TIME":
                        # Preserve time formatting if possible
                        if hasattr(val, "strftime"):
                            cell_val = val
                        elif pd.isna(val):
                            cell_val = ""
                        else:
                            cell_val = val
                        cell = ws.cell(row=row_idx, column=col_idx, value=cell_val)
                        cell.number_format = "hh:mm"
                    else:
                        ws.cell(row=row_idx, column=col_idx, value=val)

            # Adjust column widths based on content
            for idx, col in enumerate(headers, start=1):
                content_series = direction_df_sorted[col].astype(str)
                max_length = max(content_series.map(len).max(), len(str(col)))
                adjusted_width = max_length + 2
                column_letter = get_column_letter(idx)
                ws.column_dimensions[column_letter].width = adjusted_width

        # Save the workbook named after the route
        filename = f"{route_name}.xlsx"
        file_path = os.path.join(output_dir, filename)
        wb.save(file_path)
        print(f"Saved workbook: {file_path}")


def export_to_csv(data_frame: pd.DataFrame, csv_file_path: str) -> None:
    """
    Export the entire processed DataFrame to a CSV file.
    """
    data_frame.to_csv(csv_file_path, index=False)
    print(f"Processed file saved to CSV: {csv_file_path}")


def export_to_excel(data_frame: pd.DataFrame, output_file: str) -> None:
    """Export the DataFrame to an Excel file with adjusted column widths."""
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data_frame.to_excel(writer, index=False, sheet_name="Sheet1")
        worksheet = writer.sheets["Sheet1"]

        # Adjust column widths based on the maximum length of the content in each column
        for idx, col in enumerate(data_frame.columns, 1):
            series = data_frame[col].astype(str)
            max_length = max(series.map(len).max(), len(str(col)))
            adjusted_width = max_length + 2  # Add extra space for clarity
            column_letter = get_column_letter(idx)
            worksheet.column_dimensions[column_letter].width = adjusted_width


def print_high_load_trips(data_frame: pd.DataFrame) -> None:
    """Print trips where 'MAX_LOAD' is over 30."""
    high_load_trips = data_frame[data_frame["MAX_LOAD"] > 30]
    if not high_load_trips.empty:
        print("Trips with MAX_LOAD over 30:")
        print(high_load_trips)


def write_violation_log(data_frame: pd.DataFrame, log_file_path: str) -> None:
    """
    Write a plain‐text log of all rows for which LOAD_FACTOR_VIOLATION == "TRUE".
    Each line includes: ROUTE_NAME, DIRECTION_NAME, TRIP_START_TIME, MAX_LOAD,
    LOAD_FACTOR, SERVICE_PERIOD, and ROUTE_LIMIT_TYPE.
    """
    # Filter rows where load‐factor is violated
    violations_df = data_frame[data_frame["LOAD_FACTOR_VIOLATION"] == "TRUE"]

    # Open (or create) the log file and overwrite any existing content
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        if violations_df.empty:
            log_file.write(
                "No load‐factor violations found (all trips within permissible limits).\n"
            )
        else:
            # Write a header
            header = (
                "Trips with load‐factor violations (greater than route‐specific limit):\n\n"
                "ROUTE\tDIRECTION\tSTART_TIME\tMAX_LOAD\tLOAD_FACTOR\t"
                "SERVICE_PERIOD\tROUTE_LIMIT_TYPE\n"
            )
            log_file.write(header)

            # Write one line per violating trip
            for _, row in violations_df.iterrows():
                # Format TRIP_START_TIME (if it’s a time object)
                start_val = row.get("TRIP_START_TIME", None)
                if hasattr(start_val, "strftime"):  # datetime.time or pandas Timestamp
                    start_str = start_val.strftime("%H:%M")
                else:
                    start_str = "" if pd.isna(start_val) else str(start_val)

                line = (
                    f"{row.get('ROUTE_NAME', '')}\t"
                    f"{row.get('DIRECTION_NAME', '')}\t"
                    f"{start_str}\t"
                    f"{row.get('MAX_LOAD', '')}\t"
                    f"{row.get('LOAD_FACTOR', '')}\t"
                    f"{row.get('SERVICE_PERIOD', '')}\t"
                    f"{row.get('ROUTE_LIMIT_TYPE', '')}\n"
                )
                log_file.write(line)
    print(f"Exported load‐factor violation log: {log_file_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main routine to load, process, and export bus load data in three formats."""
    # Load data
    data_frame = load_data(INPUT_FILE)

    # Process data with filtering and limit checks
    processed_data = process_data(
        data_frame,
        BUS_CAPACITY,
        FILTER_IN_ROUTES,
        FILTER_OUT_ROUTES,
        DECIMAL_PLACES,
    )

    # -------------------------------------------------------------------------
    # 1) EXPORT COMBINED CSV (good for programmatic consumption)
    # -------------------------------------------------------------------------
    combined_csv_path = INPUT_FILE.replace(".XLSX", "_processed.csv")
    export_to_csv(processed_data, combined_csv_path)

    # -------------------------------------------------------------------------
    # 2) EXPORT COMBINED EXCEL (good for a quick, single-sheet view)
    # -------------------------------------------------------------------------
    export_to_excel(processed_data, OUTPUT_FILE)
    print(f"Processed file saved to Excel: {OUTPUT_FILE}")

    # -------------------------------------------------------------------------
    # 3) EXPORT PER-ROUTE EXCEL WORKBOOKS (one .xlsx per route, sheets per direction)
    # -------------------------------------------------------------------------
    create_route_workbooks(processed_data)

    # -------------------------------------------------------------------------
    # PRINT HIGH-LOAD TRIPS TO CONSOLE
    # -------------------------------------------------------------------------
    print_high_load_trips(processed_data)

    # -------------------------------------------------------------------------
    # WRITE TEXT LOG OF VIOLATIONS (good for a human-readable, line-by-line summary)
    # -------------------------------------------------------------------------
    if WRITE_VIOLATION_LOG:
        write_violation_log(processed_data, VIOLATION_LOG_FILE)


if __name__ == "__main__":
    main()
