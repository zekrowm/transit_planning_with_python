"""Extract and filter EIA weekly gasoline prices.

This module loads U.S. Energy Information Administration (EIA) weekly gasoline
price data from an Excel workbook, selects a specific price series, filters it
to a configurable date range, and writes the result to a new Excel file.

Inputs
- A published EIA workbook (`.xls` or `.xlsx`) whose columns are a
  two-level MultiIndex.
- Configurable sheet name, header rows, price-series keys, and
  inclusive date range.

Outputs
- An Excel workbook containing only the requested price series and dates.
"""

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"C:\Your\File\Path\To\pswrgvwall.xls"
OUTPUT_FILE = r"C:\Your\Output\File\Path\To\extracted_data.xlsx"
SHEET_NAME = "Data 1"
HEADER_ROWS = [1, 2]  # rows to use as header (0-indexed)

# Define the MultiIndex tuples for columns of interest.
DATE_COLUMN = ("Sourcekey", "Date")
PRICE_COLUMN = (
    "EMM_EPMRU_PTE_R1Y_DPG",
    "Weekly Central Atlantic (PADD 1B) "
    "Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)",
)

# Date filter configuration (inclusive).
DATE_FILTER_START = "2020-01-01"
DATE_FILTER_END = "2024-12-31"

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(input_file: str, sheet_name: str, header_rows: list) -> pd.DataFrame:
    """Load the Excel file using the specified sheet and header rows.

    Args:
        input_file (str): Path to the input Excel file.
        sheet_name (str): Name of the sheet to read.
        header_rows (list): List of row indices to use as header.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    loaded_dataframe = pd.read_excel(input_file, sheet_name=sheet_name, header=header_rows)
    print("Columns available:", loaded_dataframe.columns.tolist())
    return loaded_dataframe


def filter_data(
    input_dataframe: pd.DataFrame,
    date_col: tuple,
    price_col: tuple,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Subset by columns and inclusive date span.

    Args:
        input_dataframe (pd.DataFrame): The original DataFrame.
        date_col (tuple): MultiIndex tuple for the date column.
        price_col (tuple): MultiIndex tuple for the price column.
        start_date (str): Start date (inclusive) in 'YYYY-MM-DD' format.
        end_date (str): End date (inclusive) in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: The filtered DataFrame with renamed columns.
    """
    try:
        filtered_dataframe = input_dataframe[[date_col, price_col]].copy()
    except KeyError as exc:
        raise KeyError(f"One or more specified columns were not found: {exc}") from exc

    # Rename the columns for clarity.
    filtered_dataframe.columns = ["Date", "Weekly Central Atlantic Price"]

    # Convert the 'Date' column to datetime format.
    filtered_dataframe["Date"] = pd.to_datetime(filtered_dataframe["Date"], errors="coerce")

    # Filter rows based on the date range.
    mask = (filtered_dataframe["Date"] >= pd.to_datetime(start_date)) & (
        filtered_dataframe["Date"] <= pd.to_datetime(end_date)
    )
    filtered_dataframe = filtered_dataframe.loc[mask]

    return filtered_dataframe


def export_data(dataframe_to_export: pd.DataFrame, output_file: str) -> None:
    """Export the DataFrame to an Excel file.

    Args:
        dataframe_to_export (pd.DataFrame): The DataFrame to export.
        output_file (str): Path to the output Excel file.
    """
    dataframe_to_export.to_excel(output_file, index=False)
    print(f"Extracted data has been written to {output_file}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Orchestrate the ETL process."""
    # Load data from Excel.
    raw_dataframe = load_data(INPUT_FILE, SHEET_NAME, HEADER_ROWS)

    # Filter data to keep only the columns of interest and apply the date filter.
    filtered_dataframe = filter_data(
        raw_dataframe, DATE_COLUMN, PRICE_COLUMN, DATE_FILTER_START, DATE_FILTER_END
    )

    # Display the first few rows of the filtered DataFrame.
    print(filtered_dataframe.head())

    # Export the filtered data to an Excel file.
    export_data(filtered_dataframe, OUTPUT_FILE)


if __name__ == "__main__":
    main()
