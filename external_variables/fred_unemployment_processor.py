"""
Script to filter employment data from the US Federal Reserve Economic Data (FRED) system to desired dates.
It exports the filtered data to an Excel (.xlsx) file and creates line graphs to show trends.

    1. Download your desired data (e.g., for a County or Metropolitan Statistical Area) in .csv format.
       See:
         https://fred.stlouisfed.org/searchresults/?st=unemployment&t=unemployment%3Bmsa&ob=sr&od=desc
    2. Example, Washington DC Metro Area:
         https://fred.stlouisfed.org/series/WASH911URN
    3. Update the configuration variables below before running the script.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------
# Configuration Section
# -------------------------------
START_DATE = '2020-01-01'      # Replace with your desired start date
END_DATE = '2024-12-01'        # Replace with your desired end date
CSV_FILE_PATH = r'C:\Path\To\Your\Downloaded\Unemployment_Data.csv'
OUTPUT_FOLDER = r'C:\Path\To\Your\Output_Folder'

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV data into a DataFrame and converts the observation_date column to datetime.
    Additionally, it automatically detects the data series column.
    Assumes that aside from 'observation_date', the remaining column(s) are the data series,
    and selects the first one found.
    """
    df = pd.read_csv(csv_path)
    if 'observation_date' not in df.columns:
        raise ValueError("The CSV file must contain an 'observation_date' column.")
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    # Dynamically detect the data series column
    data_columns = [col for col in df.columns if col != 'observation_date']
    if not data_columns:
        raise ValueError("No data series column found in the CSV file.")
    series_column = data_columns[0]
    print(f"Detected data series column: '{series_column}'")
    return df, series_column

def filter_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows between start_date and end_date.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df['observation_date'] >= start) & (df['observation_date'] <= end)
    return df.loc[mask].copy()

def export_to_excel(df: pd.DataFrame, output_folder: str, filename: str):
    """
    Exports the DataFrame to an Excel file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_excel(output_path, index=False)
    print(f"Filtered data exported to Excel file: '{output_path}'.")

def plot_continuous_line(df: pd.DataFrame, series_column: str, output_folder: str, filename: str):
    """
    Creates a continuous line chart of the data series over time and saves it as a JPEG.
    - The y-axis label includes a "%" symbol.
    - The x-axis labels show only the year.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.figure(figsize=(10, 6))
    plt.plot(df['observation_date'], df[series_column], marker='o', linestyle='-')
    plt.title(f'{series_column} Over Time')
    plt.xlabel('Observation Date')
    plt.ylabel(f'{series_column} (%)')
    plt.grid(True)

    # Format x-axis to display only the year
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, format='jpeg')
    plt.close()
    print(f"Continuous line chart saved as '{output_path}'.")

def plot_yearly_comparison(df: pd.DataFrame, series_column: str, output_folder: str, filename: str):
    """
    Creates a chart with each year's data plotted as a separate line (overlaid),
    using months on the x-axis represented as 3-letter abbreviations.
    The y-axis label includes a "%" symbol.
    Saves the chart as a JPEG.
    """
    # Extract Year and Month from observation_date
    df['Year'] = df['observation_date'].dt.year
    df['Month'] = df['observation_date'].dt.month

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.figure(figsize=(10, 6))

    years = sorted(df['Year'].unique())
    cmap = plt.get_cmap('tab10')

    for i, year in enumerate(years):
        yearly_data = df[df['Year'] == year].sort_values(by='Month')
        plt.plot(yearly_data['Month'], yearly_data[series_column],
                 marker='o',
                 linestyle='-',
                 color=cmap(i),
                 label=str(year))

    plt.title(f'{series_column} by Month (Grouped by Year)')
    plt.xlabel('Month')
    plt.ylabel(f'{series_column} (%)')

    # Map month numbers to 3-letter abbreviations
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), month_abbr)

    plt.legend(title='Year')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='jpeg')
    plt.close()
    print(f"Yearly comparison chart saved as '{output_path}'.")

def main():
    # Load data from CSV and detect data series column
    df, series_column = load_data(CSV_FILE_PATH)

    # Filter data based on the configuration dates
    filtered_df = filter_data(df, START_DATE, END_DATE)

    # Create a dynamic suffix for output filenames with underscores between year and month
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    dynamic_suffix = f"{start_dt.strftime('%Y_%m')}-{end_dt.strftime('%Y_%m')}"

    # Define dynamic output filenames
    continuous_chart_filename = f"line_graph_{dynamic_suffix}.jpeg"
    yearly_chart_filename = f"yearly_line_graph_{dynamic_suffix}.jpeg"
    excel_filename = f"filtered_data_{dynamic_suffix}.xlsx"

    # Export filtered data to Excel
    export_to_excel(filtered_df, OUTPUT_FOLDER, excel_filename)

    # Generate continuous line chart (with x-axis showing only the year)
    plot_continuous_line(filtered_df, series_column, OUTPUT_FOLDER, continuous_chart_filename)

    # Generate yearly comparison chart (with 3-letter month abbreviations)
    plot_yearly_comparison(filtered_df, series_column, OUTPUT_FOLDER, yearly_chart_filename)

if __name__ == '__main__':
    main()
