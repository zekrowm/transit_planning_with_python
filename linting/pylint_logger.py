"""
pylint_logger.py

This script will:
1) Gather all .py files from FILES_OR_FOLDERS.
2) Skip any paths found in SKIP_PATHS (folder or file).
3) Run pylint on each remaining file.
4) Log detailed pylint output (both stdout and stderr) to a timestamped .log file.
5) Generate an Excel summary with:
   - Script name
   - Immediate parent folder
   - Pylint score
   - Number of issues
   - Full file path
   - Stderr (if any)
   ...and saves to a timestamped .xlsx in OUTPUT_FOLDER.

CONFIGURATION:
    - FILES_OR_FOLDERS:  A list of .py files and/or folders
    - SKIP_PATHS:        A list of folders or files you want to exclude from linting
    - OUTPUT_FOLDER:     Where to write the .xlsx and .log files
    - LOG_LEVEL:         Logging detail level (INFO, DEBUG, etc.)
    - DETAILED_LOG_FILENAME_PREFIX: Prefix for the .log file name
"""

import os
import sys
import subprocess
import re
from datetime import datetime
import logging

from openpyxl import Workbook

# =============================================================================
# CONFIGURATION
# =============================================================================

FILES_OR_FOLDERS = [
    # Example: mix .py files and/or folders
    r"C:\Path\to\some_folder",
    # r"C:\Path\to\single_file.py",
]

SKIP_PATHS = [
    # Example: you can skip entire folders or single files by listing them here
    # r"C:\Path\to\folder_to_skip",
    r"C:\Path\to\file_to_skip.py",
]

# Default output folder for both the Excel file AND the detailed log file
OUTPUT_FOLDER = r"C:\Path\to\Your\Logs\Folder"

# Logging config
LOG_LEVEL = logging.INFO  # Adjust to logging.DEBUG for more verbose internal logs
DETAILED_LOG_FILENAME_PREFIX = "pylint_detailed_log"

# -----------------------------------------------------------------------------
# LOGGING SETUP (for minimal console logging)
# -----------------------------------------------------------------------------
# This logger is for basic progress info to console. 
# We'll create a separate "detailed logger" for the .log file below.
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# DETAILED LOGGER SETUP
# -----------------------------------------------------------------------------
def setup_detailed_logger(output_folder, filename_prefix, log_level):
    """
    Sets up a dedicated logger to write detailed pylint output to a timestamped file.
    Returns the logger instance and the full path to the log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{filename_prefix}_{timestamp}.log"

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
            console_logger.info(f"Created output directory: {output_folder}")
        except OSError as e:
            console_logger.error(f"ERROR: Could not create output directory {output_folder}: {e}")
            sys.exit(1)  # Exit if we can't create the log directory

    log_filepath = os.path.join(output_folder, log_filename)

    # Create a specific logger for detailed pylint logs
    detail_logger = logging.getLogger('PylintDetailLogger')
    detail_logger.setLevel(log_level)
    # Prevent messages from going to the root logger
    detail_logger.propagate = False

    # Remove existing handlers (helpful if re-run in a notebook)
    for handler in detail_logger.handlers[:]:
        detail_logger.removeHandler(handler)
        handler.close()

    # Create a file handler
    try:
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    except OSError as e:
        console_logger.error(f"ERROR: Could not open log file {log_filepath} for writing: {e}")
        sys.exit(1)

    # Minimal format: we want the raw output of pylint
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    detail_logger.addHandler(file_handler)

    return detail_logger, log_filepath


# -----------------------------------------------------------------------------
# FILE GATHERING AND SKIP LOGIC
# -----------------------------------------------------------------------------
def gather_python_files_in_folder(folder_path):
    """Recursively finds all .py files within folder_path. Returns a list of full paths."""
    py_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.py'):
                py_files.append(os.path.join(root, filename))
    return py_files


def is_skipped(path_str, skip_list):
    """
    Returns True if `path_str` is inside or exactly matches 
    any of the items in `skip_list`.
    - If skip_list has a folder, all .py under it are skipped.
    - If skip_list has a file, that single file is skipped.
    """
    norm_path = os.path.normpath(os.path.abspath(path_str)).lower()

    for s in skip_list:
        norm_skip = os.path.normpath(os.path.abspath(s)).lower()
        # If exactly the same
        if norm_path == norm_skip:
            return True
        # Or if path is under that skip folder
        if norm_path.startswith(norm_skip + os.sep):
            return True
    return False


# -----------------------------------------------------------------------------
# PYLINT EXECUTION & PARSING
# -----------------------------------------------------------------------------
def run_pylint_on_file(py_file):
    """
    Runs pylint on a single Python file using 'python -m pylint'
    and returns the (stdout, stderr) as strings.
    """
    try:
        process = subprocess.run(
            [sys.executable, "-m", "pylint", py_file],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return process.stdout, process.stderr
    except FileNotFoundError:
        console_logger.error(
            f"ERROR: Could not find python executable at '{sys.executable}' "
            f"or pylint is not installed correctly."
        )
        return None, "Execution Error: Python or pylint not found."
    except Exception as e:
        console_logger.error(f"ERROR: Unexpected error running pylint on {py_file}: {e}")
        return None, f"Execution Error: {e}"


def parse_pylint_output(stdout_text):
    """
    Parse:
      - The numeric score from "Your code has been rated at X.XX/10"
      - The number of issues: lines that match severity prefixes (C, R, W, E, F).
    Returns (score_float, issues_count).
    """
    if stdout_text is None:
        return 0.0, 0

    score = 0.0
    issues_count = 0

    # We'll do a regex approach that catches lines with a code like C0103, W1202, etc.
    # e.g. "myscript.py:10:0: C0103: ..."
    issue_line_regex = re.compile(r':\s*([CRWEF]\d{4}):')

    for line in stdout_text.splitlines():
        if issue_line_regex.search(line):
            issues_count += 1

        if "Your code has been rated at" in line:
            try:
                part = line.split("rated at")[-1].split("(")[0].strip()  # e.g. "8.15/10"
                number_str = part.split("/")[0].strip()
                score = float(number_str)
            except (IndexError, ValueError):
                console_logger.warning(f"Could not parse pylint score from line: '{line}'")

    return score, issues_count


# -----------------------------------------------------------------------------
# MAIN PROCESSOR
# -----------------------------------------------------------------------------
def lint_and_create_outputs(files_or_folders, skip_list, output_folder):
    """
    1) Sets up the detailed logger.
    2) Collects .py files from each entry in 'files_or_folders', skipping any in 'skip_list'.
    3) Runs pylint on each .py file, logs the full output to the .log file, 
       and prepares summary data for Excel.
    4) Writes an Excel summary with additional info.
    """

    # 1) Detailed logger setup
    detail_logger, log_filepath = setup_detailed_logger(output_folder, DETAILED_LOG_FILENAME_PREFIX, LOG_LEVEL)
    console_logger.info(f"Detailed pylint log file: {log_filepath}")

    # 2) Collect all .py files
    all_py_files = []
    for entry in files_or_folders:
        abs_entry = os.path.abspath(entry)
        if not os.path.exists(abs_entry):
            console_logger.warning(f"Path not found, skipping: {abs_entry}")
            continue

        if os.path.isdir(abs_entry):
            console_logger.info(f"Scanning folder: {abs_entry}...")
            found = gather_python_files_in_folder(abs_entry)
            console_logger.info(f"  Found {len(found)} .py files in that folder.")
            all_py_files.extend(found)
        elif os.path.isfile(abs_entry) and abs_entry.lower().endswith('.py'):
            console_logger.info(f"Adding Python file: {abs_entry}")
            all_py_files.append(abs_entry)
        else:
            console_logger.warning(f"Skipping non-Python path: {abs_entry}")

    # Remove duplicates while preserving order
    unique_py_files = list(dict.fromkeys(all_py_files))

    # Apply skip paths
    final_py_files = []
    for pyf in unique_py_files:
        if is_skipped(pyf, skip_list):
            console_logger.info(f"Skipping (in skip list): {pyf}")
        else:
            final_py_files.append(pyf)

    if not final_py_files:
        console_logger.info("No .py files to lint after skipping. Exiting.")
        # Close the detail logger
        for handler in detail_logger.handlers[:]:
            handler.close()
            detail_logger.removeHandler(handler)
        return

    console_logger.info(f"\nWill lint {len(final_py_files)} files.\n")

    # 3) Run pylint and log everything
    results = []
    for idx, py_file in enumerate(final_py_files, start=1):
        console_logger.info(f"Linting {idx}/{len(final_py_files)}: {py_file}")
        detail_logger.info(f"{'='*80}")
        detail_logger.info(f"PYLINT ANALYSIS FOR: {py_file}")
        detail_logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        detail_logger.info(f"{'='*80}\n")

        stdout, stderr = run_pylint_on_file(py_file)

        # Log to file
        if stdout:
            detail_logger.info("--- Pylint Standard Output ---")
            detail_logger.info(stdout)
            detail_logger.info("--- End Pylint Standard Output ---\n")
        else:
            detail_logger.warning(f"No stdout from pylint for {py_file}")

        if stderr:
            detail_logger.info("--- Pylint Standard Error ---")
            detail_logger.info(stderr)
            detail_logger.info("--- End Pylint Standard Error ---\n")

        score, issues = parse_pylint_output(stdout)

        results.append({
            "script_name": os.path.basename(py_file),
            "immediate_folder": os.path.basename(os.path.dirname(py_file)),
            "full_path": py_file,
            "score": score,
            "issues": issues,
            "pylint_stderr": stderr if stderr else ""
        })

    # 4) Write Excel summary
    console_logger.info("Creating Excel summary...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Pylint Results Summary"

    # Header row
    ws.append(["Script Name", "Immediate Folder", "Pylint Score", "Number of Issues", "Full Path", "Pylint Stderr"])

    # Data rows
    for item in results:
        ws.append([
            item["script_name"],
            item["immediate_folder"],
            item["score"],
            item["issues"],
            item["full_path"],
            item["pylint_stderr"]
        ])

    # Optional: auto-size columns
    for col in ws.columns:
        max_length = 0
        column_letter = col[0].column_letter
        for cell in col:
            if cell.value is not None:
                value_length = len(str(cell.value))
                if value_length > max_length:
                    max_length = value_length
        ws.column_dimensions[column_letter].width = max_length + 2

    # Build the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"pylint_results_{timestamp}.xlsx"
    excel_filepath = os.path.join(output_folder, excel_filename)

    try:
        wb.save(excel_filepath)
        console_logger.info(f"Pylint summary written to: {excel_filepath}")
    except Exception as e:
        console_logger.error(f"ERROR: Failed to save Excel file: {e}")

    # Clean up detail logger
    for handler in detail_logger.handlers[:]:
        handler.close()
        detail_logger.removeHandler(handler)

    console_logger.info("Done.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point using the config at the top of the script."""
    lint_and_create_outputs(FILES_OR_FOLDERS, SKIP_PATHS, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
