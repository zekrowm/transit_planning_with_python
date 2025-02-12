"""
This script extracts US Energy Information Administration (EIA) price data from
the "Data 1: Regular Conventional" sheet of an Excel file.

Data Source:
    https://www.eia.gov/petroleum/gasdiesel/

The Excel sheet is assumed to have a preliminary row (e.g., a title row) followed by
two rows of headers, so we use these two rows (indices 1 and 2) as the column headers.

Additionally, the data is filtered based on start and end dates.

The extracted data is then exported to the specified output file.
"""

import pandas as pd

# ----------------------------
# Configuration Section
# ----------------------------

INPUT_FILE = r"C:\Your\File\Path\To\pswrgvwall.xls"
OUTPUT_FILE = r"C:\Your\Output\File\Path\To\extracted_data.xlsx"
SHEET_NAME = "Data 1"
HEADER_ROWS = [1, 2]  # rows to use as header (0-indexed)

# Define the MultiIndex tuples for columns of interest.
DATE_COLUMN = ("Sourcekey", "Date")
PRICE_COLUMN = (
    "EMM_EPMRU_PTE_R1Y_DPG",
    "Weekly Central Atlantic (PADD 1B) "
    "Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)"
)

# NOTE:
# If you're interested in retrieving gasoline price data for a different region,
# update the PRICE_COLUMN tuple to match the corresponding column header in Excel.

# Date filter configuration (inclusive).
DATE_FILTER_START = "2020-01-01"
DATE_FILTER_END = "2024-12-31"


def load_data(input_file: str, sheet_name: str, header_rows: list) -> pd.DataFrame:
    """
    Load the Excel file using the specified sheet and header rows.

    Args:
        input_file (str): Path to the input Excel file.
        sheet_name (str): Name of the sheet to read.
        header_rows (list): List of row indices to use as header.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_rows)
    print("Columns available:", df.columns.tolist())
    return df


def filter_data(
    df: pd.DataFrame,
    date_col: tuple,
    price_col: tuple,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Filter the DataFrame to include only the desired columns and rows within the
    date range.

    Args:
        df (pd.DataFrame): The original DataFrame.
        date_col (tuple): MultiIndex tuple for the date column.
        price_col (tuple): MultiIndex tuple for the price column.
        start_date (str): Start date (inclusive) in 'YYYY-MM-DD' format.
        end_date (str): End date (inclusive) in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: The filtered DataFrame with renamed columns.
    """
    try:
        df_filtered = df[[date_col, price_col]].copy()
    except KeyError as exc:
        raise KeyError(f"One or more specified columns were not found: {exc}")

    # Rename the columns for clarity.
    df_filtered.columns = ["Date", "Weekly Central Atlantic Price"]

    # Convert the 'Date' column to datetime format.
    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")

    # Filter rows based on the date range.
    mask = (
        (df_filtered["Date"] >= pd.to_datetime(start_date)) &
        (df_filtered["Date"] <= pd.to_datetime(end_date))
    )
    df_filtered = df_filtered.loc[mask]

    return df_filtered


def export_data(df: pd.DataFrame, output_file: str) -> None:
    """
    Export the DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        output_file (str): Path to the output Excel file.
    """
    df.to_excel(output_file, index=False)
    print(f"Extracted data has been written to {output_file}")


def main() -> None:
    """
    Main function to load, filter, and export the data.
    """
    # Load data from Excel.
    df = load_data(INPUT_FILE, SHEET_NAME, HEADER_ROWS)

    # Filter data to keep only the columns of interest and apply the date filter.
    df_filtered = filter_data(
        df, DATE_COLUMN, PRICE_COLUMN, DATE_FILTER_START, DATE_FILTER_END
    )

    # Display the first few rows of the filtered DataFrame.
    print(df_filtered.head())

    # Export the filtered data to an Excel file.
    export_data(df_filtered, OUTPUT_FILE)


if __name__ == "__main__":
    main()
