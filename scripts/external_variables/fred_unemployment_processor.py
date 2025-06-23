"""Process FRED unemployment-rate CSV data for a user-defined period.

The script performs the following high-level tasks:

1. Load a FRED CSV file that contains an ``observation_date`` column and
   exactly one data-series column (e.g., ``UNRATE``).
2. Filter the time-series to the requested start and end dates.
3. Export the filtered data to Excel for downstream analysis.
4. Visualize the results via two Matplotlib figures:
   - A continuous line chart of the full period.
   - An overlay chart plotting each calendar year on the same axes to enable
     year-over-year comparison.
"""

import os
from typing import Final, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

START_DATE: Final[str] = "2020-01-01"  # Replace with your desired start date
END_DATE: Final[str] = "2024-12-01"  # Replace with your desired end date
CSV_FILE_PATH: Final[str] | os.PathLike[str] = r"C:\Path\To\Your\Downloaded\Unemployment_Data.csv"
OUTPUT_FOLDER: Final[str] | os.PathLike[str] = r"C:\Path\To\Your\Output_Folder"

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_data(csv_file_path: str | os.PathLike[str]) -> Tuple[pd.DataFrame, str]:
    """Load the FRED CSV and detect the data-series column.

    Args:
        csv_file_path: Path to the FRED CSV file.

    Returns:
        Tuple containing:
            - The loaded DataFrame with 'observation_date' parsed as datetime.
            - The name of the detected data-series column.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If 'observation_date' is missing or no data column is found.
    """
    data_frame = pd.read_csv(csv_file_path)
    if "observation_date" not in data_frame.columns:
        raise ValueError("The CSV file must contain an 'observation_date' column.")

    data_frame["observation_date"] = pd.to_datetime(data_frame["observation_date"])

    # Dynamically detect the data series column
    data_columns = [col for col in data_frame.columns if col != "observation_date"]
    if not data_columns:
        raise ValueError("No data series column found in the CSV file.")

    series_column = data_columns[0]
    print(f"Detected data series column: '{series_column}'")
    return data_frame, series_column


def filter_data(
    data_frame: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Filter the DataFrame to include only rows within the given date range.

    Args:
        data_frame: DataFrame containing the full time-series.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).

    Returns:
        A filtered copy of the original DataFrame.

    Raises:
        KeyError: If 'observation_date' column is missing.
        ValueError: If start_date is after end_date.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (data_frame["observation_date"] >= start) & (data_frame["observation_date"] <= end)
    return data_frame.loc[mask].copy()


def export_to_excel(
    data_frame: pd.DataFrame,
    output_folder: str | os.PathLike[str],
    filename: str,
) -> None:
    """Export the filtered DataFrame to an Excel file.

    Args:
        data_frame: The DataFrame to export.
        output_folder: Destination directory.
        filename: Desired name of the Excel file (with .xlsx extension).

    Returns:
        None. Writes a file to disk.

    Raises:
        OSError: If the file cannot be written.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    data_frame.to_excel(output_path, index=False)
    print(f"Filtered data exported to Excel file: '{output_path}'.")


def plot_continuous_line(
    data_frame: pd.DataFrame,
    series_column: str,
    output_folder: str | os.PathLike[str],
    filename: str,
) -> None:
    """Plot a continuous time-series line chart and save it as a JPEG.

    Args:
        data_frame: DataFrame with 'observation_date' and the series column.
        series_column: Name of the column to plot.
        output_folder: Destination directory.
        filename: JPEG filename to save.

    Returns:
        None. Saves a JPEG plot.

    Raises:
        KeyError: If required columns are missing.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.figure(figsize=(10, 6))
    plt.plot(
        data_frame["observation_date"],
        data_frame[series_column],
        marker="o",
        linestyle="-",
    )
    plt.title(f"{series_column} Over Time")
    plt.xlabel("Observation Date")
    plt.ylabel(f"{series_column} (%)")
    plt.grid(True)

    # Format x-axis to display only the year
    axis = plt.gca()
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, format="jpeg")
    plt.close()
    print(f"Continuous line chart saved as '{output_path}'.")


def plot_yearly_comparison(
    data_frame: pd.DataFrame,
    series_column: str,
    output_folder: str | os.PathLike[str],
    filename: str,
) -> None:
    """Plot a year-over-year comparison chart and save it as a JPEG.

    Args:
        data_frame: DataFrame with 'observation_date' and the series column.
        series_column: Name of the column to plot.
        output_folder: Destination directory.
        filename: JPEG filename to save.

    Returns:
        None. Saves a JPEG plot.

    Raises:
        KeyError: If required columns are missing.
    """
    data_frame["Year"] = data_frame["observation_date"].dt.year
    data_frame["Month"] = data_frame["observation_date"].dt.month

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.figure(figsize=(10, 6))

    years = sorted(data_frame["Year"].unique())
    color_map = plt.get_cmap("tab10")

    for i, year in enumerate(years):
        yearly_data = data_frame[data_frame["Year"] == year].sort_values(by="Month")
        plt.plot(
            yearly_data["Month"],
            yearly_data[series_column],
            marker="o",
            linestyle="-",
            color=color_map(i),
            label=str(year),
        )

    plt.title(f"{series_column} by Month (Grouped by Year)")
    plt.xlabel("Month")
    plt.ylabel(f"{series_column} (%)")

    # Map month numbers to 3-letter abbreviations
    month_abbr = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    plt.xticks(range(1, 13), month_abbr)

    plt.legend(title="Year")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="jpeg")
    plt.close()
    print(f"Yearly comparison chart saved as '{output_path}'.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main function to execute the full processing and plotting workflow.

    Loads data, filters by date, exports to Excel, and saves two JPEG charts.
    """
    # Load data from CSV and detect data series column
    data_frame, series_column = load_data(CSV_FILE_PATH)

    # Filter data based on the configuration dates
    filtered_data_frame = filter_data(data_frame, START_DATE, END_DATE)

    # Create a dynamic suffix for output filenames with underscores between year and month
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    dynamic_suffix = f"{start_dt.strftime('%Y_%m')}-{end_dt.strftime('%Y_%m')}"

    # Define dynamic output filenames
    continuous_chart_filename = f"line_graph_{dynamic_suffix}.jpeg"
    yearly_chart_filename = f"yearly_line_graph_{dynamic_suffix}.jpeg"
    excel_filename = f"filtered_data_{dynamic_suffix}.xlsx"

    # Export filtered data to Excel
    export_to_excel(filtered_data_frame, OUTPUT_FOLDER, excel_filename)

    # Generate continuous line chart (with x-axis showing only the year)
    plot_continuous_line(
        filtered_data_frame, series_column, OUTPUT_FOLDER, continuous_chart_filename
    )

    # Generate yearly comparison chart (with 3-letter month abbreviations)
    plot_yearly_comparison(filtered_data_frame, series_column, OUTPUT_FOLDER, yearly_chart_filename)


if __name__ == "__main__":
    main()
