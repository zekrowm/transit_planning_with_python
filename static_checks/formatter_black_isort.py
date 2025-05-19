"""
formatter_black_isort.py

A unified formatter that checks—and optionally fixes—Python code with both
isort and black.

Operation
---------
1. Recursively walk TARGET_DIRECTORY, skipping anything whose base name is
   listed in SKIP_NAMES.
2. For each *.py file that is not skipped:
   a. Run `isort --check --diff` to detect unsorted imports.
   b. Run `black --check --diff` (line length set by BLACK_LINE_LENGTH) to
      detect style issues.
   c. If READ_ONLY is False and either tool reports changes, run the
      corresponding fix command (`isort` or `black`) and log the outcome.
3. Collect all stdout/stderr, diffs, and status lines in LOG_FILENAME,
   followed by separate isort and black summaries.

Configuration
-------------
TARGET_DIRECTORY   root folder to scan
LOG_DIRECTORY      where the log file is written
LOG_FILENAME       full log path (derived from LOG_DIRECTORY)
READ_ONLY          True → check only; False → check and apply
SKIP_NAMES         folder or file names to ignore during traversal
BLACK_LINE_LENGTH  max line length passed to black

Usage
-----
$ python format_checker.py

Requires isort ≥ 5 and black ≥ 24 to be discoverable on PATH.
"""

import os
import subprocess
import sys

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

TARGET_DIRECTORY = r"C:\Your\Input\Folder\Path"  # <<< UPDATE THIS PATH
LOG_DIRECTORY = r"C:\Your\Output\Folder\Path" # <<< UPDATE THIS PATH
LOG_FILENAME = os.path.join(LOG_DIRECTORY, "format_check.log") # Generic log file name

# Set to True to only check files and log diffs (no modifications).
# Set to False to check files, log diffs, and apply formatting/sorting changes.
READ_ONLY = False

# Use simple directory/file names for os.walk skipping
SKIP_NAMES = [
    "__pycache__",
    ".venv",
    ".git",
    "node_modules",
    "build",
    "dist",
    # Add other directory or file names to skip here
]

# Black specific arguments (isort uses defaults or pyproject.toml)
BLACK_LINE_LENGTH = "100"


# Create log directory if it doesn't exist.
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    print(f"Log directory '{LOG_DIRECTORY}' ensured.")
except OSError as e:
    print(f"Error creating log directory '{LOG_DIRECTORY}': {e}", file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------------------------------------------------------

def process_file_with_tool(tool_name, check_cmd_args, fix_cmd_args, py_file, log_file, read_only_mode):
    """
    Processes a file with a given tool (check and optionally fix).

    Args:
        tool_name (str): Name of the tool (e.g., "isort", "black").
        check_cmd_args (list): List of arguments for the check command (file path will be appended).
        fix_cmd_args (list): List of arguments for the fix command (file path will be appended).
        py_file (str): Path to the python file.
        log_file (file_object): Opened log file for writing.
        read_only_mode (bool): If True, only check, don't fix.

    Returns:
        tuple: (needs_change, change_applied, error_occurred, critical_tool_not_found)
               needs_change (bool): True if the tool indicates changes are needed.
               change_applied (bool): True if changes were successfully applied.
               error_occurred (bool): True if any error occurred during check or fix.
               critical_tool_not_found (bool): True if the tool itself was not found.
    """
    needs_change = False
    change_applied = False
    error_occurred = False
    critical_tool_not_found = False

    log_file.write(f"--- {tool_name.capitalize()} processing for: {py_file} ---\n")
    print(f"  Running {tool_name} check on: {py_file}")

    # 1) Check + Diff
    try:
        check_process = subprocess.run(
            check_cmd_args + [py_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError:
        error_msg = f"Error: '{tool_name}' command not found. Please ensure it is installed and available in PATH.\n"
        print(error_msg, file=sys.stderr)
        log_file.write(error_msg)
        return False, False, True, True # error_occurred, critical_tool_not_found
    except Exception as e:
        error_msg = f"Unexpected error running {tool_name} check on {py_file}: {e}\n"
        print(error_msg, file=sys.stderr)
        log_file.write(error_msg)
        return False, False, True, False # error_occurred

    # --- Analyze check_process results ---
    if check_process.returncode == 0: # Exit code 0: OK, no changes needed
        log_file.write(f"Status ({tool_name}): OK (Already compliant)\n")
    elif check_process.returncode == 1: # Exit code 1: Needs changes
        needs_change = True
        log_file.write(f"Status ({tool_name}): Needs changes\n")
        log_file.write("Diff proposal:\n")
        log_file.write(check_process.stdout) # Diff is printed to stdout by both tools
        if check_process.stderr: # Log stderr from check if any (e.g., black might mention files it would change)
            log_file.write(f"{tool_name} check stderr:\n")
            log_file.write(check_process.stderr)
        log_file.write("\n")

        if not read_only_mode:
            print(f"    Applying {tool_name} to: {py_file}")
            log_file.write(f"Attempting {tool_name} modifications (Read & Modify Mode):\n")
            try:
                fix_process = subprocess.run(
                    fix_cmd_args + [py_file],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )

                # Log output from the fix command
                if fix_process.stdout: # Black "reformatted file", isort is usually silent on stdout for fix
                    log_file.write(f"{tool_name} fix stdout:\n{fix_process.stdout}\n")
                if fix_process.stderr: # Black "X files reformatted", isort "Fixing file.py"
                    log_file.write(f"{tool_name} fix stderr:\n{fix_process.stderr}\n")

                if fix_process.returncode != 0:
                    log_file.write(f"Reformatting with {tool_name} returned non-zero exit code: {fix_process.returncode}\n")
                    error_occurred = True
                else:
                    log_file.write(f"Status ({tool_name}): Modifications applied successfully.\n")
                    change_applied = True
            except FileNotFoundError: # Should not happen if check worked, but as a safeguard
                error_msg = f"Error: '{tool_name}' command not found during reformatting attempt.\n"
                print(error_msg, file=sys.stderr)
                log_file.write(error_msg)
                error_occurred = True
                critical_tool_not_found = True # If tool vanishes between check and fix
            except Exception as e:
                error_msg = f"Unexpected error running {tool_name} reformat on {py_file}: {e}\n"
                print(error_msg, file=sys.stderr)
                log_file.write(error_msg)
                error_occurred = True
        else: # READ_ONLY is True
            log_file.write(f"Status ({tool_name}): Read-Only Mode - No changes applied.\n")
    else: # Other non-zero return codes from check (e.g., 123 for black internal error)
        error_occurred = True
        log_file.write(f"Status ({tool_name}): Error during check (exit code {check_process.returncode})\n")
        log_file.write(f"{tool_name} check stdout:\n{check_process.stdout}\n")
        log_file.write(f"{tool_name} check stderr:\n{check_process.stderr}\n")

    log_file.write("\n") # Extra newline after this tool's section for the file
    return needs_change, change_applied, error_occurred, critical_tool_not_found

# --------------------------------------------------------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------------------------------------------------------

def main():
    py_files = []
    print(f"Searching for .py files in: {TARGET_DIRECTORY}")
    print(f"Skipping names: {SKIP_NAMES}")
    print(f"Run Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}")

    for root, dirs, files in os.walk(TARGET_DIRECTORY, topdown=True):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for filename in files:
            if filename.endswith(".py") and filename not in SKIP_NAMES:
                full_path = os.path.join(root, filename)
                py_files.append(full_path)

    if not py_files:
        print("No Python files found (or all were skipped). Exiting.")
        try:
            with open(LOG_FILENAME, "w", encoding="utf-8") as log_file:
                log_file.write(f"Formatter Log - Target: {TARGET_DIRECTORY}\n")
                log_file.write(f"Mode: {'Read-Only' if READ_ONLY else 'Read & Modify'}\n")
                log_file.write("=" * 80 + "\n\n")
                log_file.write("No Python files found or all were skipped.\n")
            print(f"Log file created/emptied: {LOG_FILENAME}")
        except IOError as e:
            print(f"Error writing initial message to log file '{LOG_FILENAME}': {e}", file=sys.stderr)
        return

    print(f"Found {len(py_files)} Python files to process.")

    # Counters for isort
    isort_files_needing_sorting = 0
    isort_files_sorted = 0
    isort_files_with_errors = 0
    isort_tool_missing = False

    # Counters for black
    black_files_needing_reformat = 0
    black_files_reformatted = 0
    black_files_with_errors = 0
    black_tool_missing = False

    try:
        with open(LOG_FILENAME, "w", encoding="utf-8") as log_file:
            log_file.write(f"Formatter Log - Target: {TARGET_DIRECTORY}\n")
            log_file.write(f"Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}\n")
            log_file.write("=" * 80 + "\n\n")

            for py_file in py_files:
                log_file.write(f"Processing file: {py_file}\n")
                print(f"Processing: {py_file}")

                # --- Step 1: Run isort ---
                if not isort_tool_missing:
                    isort_check_cmd = ["isort", "--check", "--diff"]
                    isort_fix_cmd = ["isort"]
                    needs_sorting, sorted_applied, isort_err, isort_missing = process_file_with_tool(
                        "isort", isort_check_cmd, isort_fix_cmd, py_file, log_file, READ_ONLY
                    )
                    if isort_missing:
                        isort_tool_missing = True # Stop trying to run isort if not found
                    if isort_err:
                        isort_files_with_errors += 1
                    if needs_sorting:
                        isort_files_needing_sorting += 1
                    if sorted_applied:
                        isort_files_sorted += 1
                else:
                    log_file.write(f"--- isort processing for: {py_file} ---\n")
                    log_file.write("Skipping isort: command was not found earlier.\n\n")


                # --- Step 2: Run black ---
                if not black_tool_missing:
                    black_check_cmd = ["black", "--check", "--diff", "--line-length", BLACK_LINE_LENGTH]
                    black_fix_cmd = ["black", "--line-length", BLACK_LINE_LENGTH]
                    needs_reformat, reformat_applied, black_err, black_missing = process_file_with_tool(
                        "black", black_check_cmd, black_fix_cmd, py_file, log_file, READ_ONLY
                    )
                    if black_missing:
                        black_tool_missing = True # Stop trying to run black if not found
                    if black_err:
                        black_files_with_errors += 1
                    if needs_reformat:
                        black_files_needing_reformat += 1
                    if reformat_applied:
                        black_files_reformatted += 1
                else:
                    log_file.write(f"--- black processing for: {py_file} ---\n")
                    log_file.write("Skipping black: command was not found earlier.\n\n")

                log_file.write("-" * 60 + "\n\n") # Separator after all tools for a file

            # --- Write Summary ---
            log_file.write("=" * 80 + "\n")
            log_file.write(f"Summary: Processed {len(py_files)} files.\n")
            log_file.write("=" * 80 + "\n")

            # isort Summary
            log_file.write("isort Summary:\n")
            if isort_tool_missing:
                log_file.write(" - isort command not found. All isort operations skipped.\n")
            else:
                log_file.write(f" - {isort_files_needing_sorting} file(s) needed import sorting.\n")
                if READ_ONLY:
                    log_file.write("   - Running in Read-Only mode: No files were sorted by this script.\n")
                else:
                    log_file.write(f"   - {isort_files_sorted} file(s) had imports successfully sorted.\n")
                    if isort_files_needing_sorting > isort_files_sorted and not READ_ONLY:
                        log_file.write(f"   - Note: {isort_files_needing_sorting - isort_files_sorted} file(s) needed sorting but encountered errors or were not fixed.\n")
                log_file.write(f" - {isort_files_with_errors} file(s) encountered errors during isort processing.\n")
            log_file.write("\n")

            # black Summary
            log_file.write("black Summary:\n")
            if black_tool_missing:
                log_file.write(" - black command not found. All black operations skipped.\n")
            else:
                log_file.write(f" - {black_files_needing_reformat} file(s) needed reformatting by black.\n")
                if READ_ONLY:
                    log_file.write("   - Running in Read-Only mode: No files were reformatted by this script.\n")
                else:
                    log_file.write(f"   - {black_files_reformatted} file(s) were successfully reformatted by black.\n")
                    if black_files_needing_reformat > black_files_reformatted and not READ_ONLY:
                         log_file.write(f"   - Note: {black_files_needing_reformat - black_files_reformatted} file(s) needed reformatting but encountered errors or were not fixed.\n")
                log_file.write(f" - {black_files_with_errors} file(s) encountered errors during black processing.\n")

    except IOError as e:
        print(f"Error writing to log file '{LOG_FILENAME}': {e}", file=sys.stderr)
        return

    # --- Final Console Output ---
    print("-" * 40)
    print(f"Formatting check completed. Processed {len(py_files)} files.")
    print(f"Run Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}")
    print("-" * 40)

    # isort console summary
    print("isort Results:")
    if isort_tool_missing:
        print("  isort command not found. Skipped.")
    else:
        if isort_files_needing_sorting > 0:
            if READ_ONLY:
                print(f"  {isort_files_needing_sorting} file(s) would need import sorting (no changes applied).")
            else:
                print(f"  {isort_files_needing_sorting} file(s) needed import sorting.")
                print(f"  {isort_files_sorted} file(s) had imports successfully sorted.")
                if isort_files_needing_sorting > isort_files_sorted:
                    print(f"  {isort_files_needing_sorting - isort_files_sorted} file(s) could not be sorted due to errors.")
        else:
            print("  All checked files already had correctly sorted imports.")
        if isort_files_with_errors > 0:
            print(f"  {isort_files_with_errors} file(s) encountered errors during isort processing.")
    print("-" * 40)

    # black console summary
    print("black Results:")
    if black_tool_missing:
        print("  black command not found. Skipped.")
    else:
        if black_files_needing_reformat > 0:
            if READ_ONLY:
                print(f"  {black_files_needing_reformat} file(s) would need reformatting by black (no changes applied).")
            else:
                print(f"  {black_files_needing_reformat} file(s) needed reformatting by black.")
                print(f"  {black_files_reformatted} file(s) were successfully reformatted by black.")
                if black_files_needing_reformat > black_files_reformatted:
                     print(f"  {black_files_needing_reformat - black_files_reformatted} file(s) could not be reformatted by black due to errors.")
        else:
            print("  All checked files were already correctly black-formatted.")
        if black_files_with_errors > 0:
            print(f"  {black_files_with_errors} file(s) encountered errors during black processing.")

    print("-" * 40)
    print(f"See '{LOG_FILENAME}' for detailed logs and diffs.")
    print("-" * 40)


if __name__ == "__main__":
    main()
