"""
Script Name:
    ntd_data_compiler.py
"""

import os

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- List of files to process ---
# Each item is a tuple: (raw_file_path_string, sheet_name_string)
# sheet_name can be None to read the first sheet.
# --- IMPORTANT: Replace with your actual file paths and sheet names ---
FILES_TO_PROCESS = [
    (r"\\Your\File\Path\JULY 2024 NTD RIDERSHIP BY ROUTE.XLSX", "Temporary_Query_N"),
    (
        r"\\Your\File\Path\AUGUST 2024 NTD RIDERSHIP REPORT BY ROUTE.XLSX",
        "Temporary_Query_N",
    ),
    (r"\\Your\File\Path\SEPTEMBER 2024 NTD RIDERSHIP BY ROUTE.XLSX", "Sep.2024 Finals"),
    # Add more files as (file_path, sheet_name) tuples:
    # (r"\\Your\File\Path\NTD RIDERSHIP BY ROUTE _ OCTOBER _2024.XLSX", "Temporary_Query_N"),
    # ... (add all other files)
]

# --- Output Configuration ---
OUTPUT_FILE_PATH = r"\\Path\to\Your\Output_Folder\Compiled_NTD_Data.csv"  # Or .xlsx

# --- Optional: Data Cleaning and Conversion ---
# Define converters for specific columns. These will be applied to all files.
COMMON_CONVERTERS = {
    # Example using the robust_numeric_converter defined above:
    "MTH_BOARD": robust_numeric_converter,
    "MTH_REV_HOURS": robust_numeric_converter,
    "MTH_PASS_MILES": robust_numeric_converter,
    "ASCH_TRIPS": robust_numeric_converter,
    "ACTUAL_TRIPS": robust_numeric_converter,
    "DAYS": robust_numeric_converter,
    "REV_MILES": robust_numeric_converter,
    # Add other columns that need this specific numeric conversion
}

# Optional: Specify a list of columns. If a row has NaN in ALL of these columns, it will be dropped.
DROPNA_SUBSET_ALL_NAN = None  # e.g., ["ROUTE_NAME", "MTH_BOARD"]

# Optional: Specify a list of columns. If a row has NaN in ANY of these columns, it will be dropped.
DROPNA_SUBSET_ANY_NAN = None  # e.g., ["ROUTE_NAME", "MTH_BOARD"]

# --- Existing Period Column Information ---
# Specify the name of the column in your datasets that ALREADY contains the month/year.
# The script will rely on this column being present.
# Ensure this column name is consistent across all your input files.
EXISTING_PERIOD_COLUMN_NAME = (
    "NameOfYourExistingMonthYearColumn"  # <--- IMPORTANT: UPDATE THIS
)

# =============================================================================
# FUNCTIONS
# =============================================================================


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def robust_numeric_converter(value):
    """
    Safely converts a value to float, handling commas and empty/NA values.
    Preserves 0 or 0.0.
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
# SCRIPT FUNCTIONS
# -----------------------------------------------------------------------------


def read_and_prepare_ntd_file(
    file_path: str, sheet_name: str | None
) -> pd.DataFrame | None:
    """
    Reads a single NTD Excel file, and applies common cleaning/conversions.
    Relies on an existing column for period information as specified by global config.

    Args:
        file_path (str): The full path to the Excel file.
        sheet_name (str | None): The name of the sheet to read. If None, reads the first sheet.

    Returns:
        pd.DataFrame or None: The processed DataFrame, or None if an error occurs or file not found.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}. Skipping.")
        return None

    print(
        f"Processing file: {os.path.basename(file_path)}, Sheet: {sheet_name or 'first available'}"
    )

    try:
        # Access global COMMON_CONVERTERS
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            converters=COMMON_CONVERTERS if COMMON_CONVERTERS else None,
        )

        # Check for the existing period column specified by global EXISTING_PERIOD_COLUMN_NAME
        if EXISTING_PERIOD_COLUMN_NAME:  # Check if the global variable is set
            if EXISTING_PERIOD_COLUMN_NAME not in df.columns:
                print(
                    f"  Warning: Specified existing period column '{EXISTING_PERIOD_COLUMN_NAME}' was NOT found in {os.path.basename(file_path)}."
                )
            else:
                print(
                    f"  Info: Using existing period column '{EXISTING_PERIOD_COLUMN_NAME}' from {os.path.basename(file_path)}."
                )
        else:
            print(
                f"  Warning: 'EXISTING_PERIOD_COLUMN_NAME' not specified in config. No specific period column checked by script."
            )

        # Optional: Drop rows where a subset of columns are ALL NaN (access global DROPNA_SUBSET_ALL_NAN)
        if DROPNA_SUBSET_ALL_NAN:
            original_rows = len(df)
            df.dropna(subset=DROPNA_SUBSET_ALL_NAN, how="all", inplace=True)
            if len(df) < original_rows:
                print(
                    f"  Dropped {original_rows - len(df)} rows based on 'DROPNA_SUBSET_ALL_NAN': {DROPNA_SUBSET_ALL_NAN}"
                )

        # Optional: Drop rows where a subset of columns are ANY NaN (access global DROPNA_SUBSET_ANY_NAN)
        if DROPNA_SUBSET_ANY_NAN:
            original_rows = len(df)
            df.dropna(subset=DROPNA_SUBSET_ANY_NAN, how="any", inplace=True)
            if len(df) < original_rows:
                print(
                    f"  Dropped {original_rows - len(df)} rows based on 'DROPNA_SUBSET_ANY_NAN': {DROPNA_SUBSET_ANY_NAN}"
                )

        print(
            f"  Successfully processed {len(df)} rows from {os.path.basename(file_path)}."
        )
        return df

    except FileNotFoundError:
        print(f"ERROR: File not found during read attempt: {file_path}.")
        return None
    except ValueError as ve:
        print(
            f"ERROR reading sheet '{sheet_name}' from {os.path.basename(file_path)}: {ve}"
        )
        return None
    except Exception as e:
        print(
            f"ERROR: An unexpected error occurred while reading {os.path.basename(file_path)} (Sheet: {sheet_name}): {e}"
        )
        return None


def compile_ntd_data() -> pd.DataFrame | None:
    """
    Compiles all NTD files specified in the global FILES_TO_PROCESS into a single DataFrame.

    Returns:
        pd.DataFrame or None: The concatenated DataFrame, or None if no data was compiled.
    """
    all_dfs = []

    # Access global FILES_TO_PROCESS
    if not FILES_TO_PROCESS:
        print("ERROR: 'FILES_TO_PROCESS' list in config is empty. Nothing to compile.")
        return None

    print(f"\nFound {len(FILES_TO_PROCESS)} file(s) listed for processing.")

    for item_index, item_details in enumerate(FILES_TO_PROCESS):
        file_path, sheet_name = None, None
        if isinstance(item_details, tuple) and len(item_details) == 2:
            file_path, sheet_name = item_details
        else:
            print(
                f"Warning: Item {item_index + 1} in 'FILES_TO_PROCESS' is not a valid (file_path, sheet_name) tuple. Skipping: {item_details}"
            )
            continue

        if not file_path:
            print(
                f"Warning: 'file_path' is missing for item {item_index + 1}. Skipping."
            )
            continue

        df_for_period = read_and_prepare_ntd_file(file_path, sheet_name)

        if df_for_period is not None and not df_for_period.empty:
            all_dfs.append(df_for_period)
        elif df_for_period is not None and df_for_period.empty:
            print(
                f"  Note: File processed but resulted in an empty DataFrame: {os.path.basename(file_path)}"
            )

    if not all_dfs:
        print(
            "\nNo DataFrames were successfully read and prepared with data. Cannot compile."
        )
        return None

    print(f"\nConcatenating {len(all_dfs)} DataFrame(s)...")
    try:
        compiled_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        print(
            f"Concatenation successful. Total rows in compiled DataFrame: {len(compiled_df)}"
        )

        # Access global OUTPUT_FILE_PATH
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        current_output_path = OUTPUT_FILE_PATH  # Use a mutable variable for path

        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as ose:
                print(f"ERROR: Could not create output directory {output_dir}: {ose}")
                current_output_path = os.path.basename(
                    OUTPUT_FILE_PATH
                )  # Fallback path
                print(
                    f"Attempting to save to current directory as: {os.path.abspath(current_output_path)}"
                )

        if current_output_path.endswith(".csv"):
            compiled_df.to_csv(current_output_path, index=False)
        elif current_output_path.endswith(".xlsx"):
            compiled_df.to_excel(current_output_path, index=False)
        else:
            default_output_path = os.path.splitext(current_output_path)[0] + ".csv"
            print(
                f"Warning: Unknown output file extension for '{current_output_path}'. Saving as CSV to '{default_output_path}'."
            )
            compiled_df.to_csv(default_output_path, index=False)
            current_output_path = default_output_path

        print(f"Compiled NTD data saved to: {os.path.abspath(current_output_path)}")
        return compiled_df

    except Exception as e:
        print(f"ERROR: An error occurred during DataFrame concatenation or saving: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("--- Starting NTD Data Compilation Script ---")

    valid_config = True
    # Check global configuration variables
    # Note: Using 'globals().get("VAR_NAME")' is a way to check if a global var exists
    # without causing a NameError if it was accidentally deleted, but direct access is fine too.
    if not FILES_TO_PROCESS:  # Checks if the list is empty
        print(
            "CRITICAL ERROR: 'FILES_TO_PROCESS' list is empty. Please configure input files."
        )
        valid_config = False
    if not OUTPUT_FILE_PATH:  # Checks if the string is empty
        print("CRITICAL ERROR: 'OUTPUT_FILE_PATH' is not specified.")
        valid_config = False
    if not EXISTING_PERIOD_COLUMN_NAME:  # Checks if the string is empty
        print(
            "CRITICAL WARNING: 'EXISTING_PERIOD_COLUMN_NAME' is not specified. "
            "The script will not specifically look for or validate an existing period column."
        )
        # This is a warning, not an error that prevents running, but user should be aware.

    if valid_config:
        compiled_dataframe = compile_ntd_data()  # No argument passed
        if compiled_dataframe is not None:
            print(f"\nCompilation summary: DataFrame shape: {compiled_dataframe.shape}")
        else:
            print("\nCompilation process failed or resulted in no data being compiled.")
    else:
        print("\nScript cannot proceed due to critical configuration errors.")

    print("\n--- Script Finished ---")
