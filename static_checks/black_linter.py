"""
black_linter.py

This script will:
1) Gather all .py files within TARGET_DIRECTORY.
2) Skip any paths/files found in SKIP_NAMES.
3) For each file not skipped:
   - Run `black --check --diff` to see if reformatting is needed.
   - Log the result (OK, Needs reformatting, or Error).
   - If changes are needed AND READ_ONLY is False:
     - Log the diff.
     - Run `black` to apply the reformatting.
     - Log the reformatting outcome.
   - If changes are needed AND READ_ONLY is True:
     - Log the diff.
     - Log that changes were NOT applied due to read-only mode.
4) Write all diffs and logs to the specified LOG_FILENAME.
"""

import os
import subprocess
import sys

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

TARGET_DIRECTORY = r"C:\Your\Input\Folder\Path"
LOG_DIRECTORY = r"C:\Your\Output\Folder\Path"
LOG_FILENAME = os.path.join(LOG_DIRECTORY, "black_check.log")

# Set to True to only check files and log diffs (no modifications).
# Set to False to check files, log diffs, and apply formatting changes.
READ_ONLY = True

# Use simple directory/file names for os.walk skipping
SKIP_NAMES = [
    "__pycache__",
    ".venv",  # Example: common directory to skip
    ".git",  # Example: common directory to skip
    # Add other directory or file names to skip here
]

# Create log directory if it doesn't exist.
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    print(f"Log directory '{LOG_DIRECTORY}' ensured.")
except OSError as e:
    print(f"Error creating log directory '{LOG_DIRECTORY}': {e}", file=sys.stderr)
    sys.exit(1)  # Exit if we can't create the log directory

# --------------------------------------------------------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------------------------------------------------------

def main():
    py_files = []
    print(f"Searching for .py files in: {TARGET_DIRECTORY}")
    print(f"Skipping names: {SKIP_NAMES}")
    print(
        f"Run Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}"
    )  # Notify user of mode

    for root, dirs, files in os.walk(TARGET_DIRECTORY, topdown=True):
        # --- Modify dirs in-place to prevent descending into skipped directories ---
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]

        for filename in files:
            if filename.endswith(".py") and filename not in SKIP_NAMES:
                full_path = os.path.join(root, filename)
                # print(f"  Found: {full_path}") # Optional debug print
                py_files.append(full_path)

    if not py_files:
        print("No Python files found (or all were skipped). Exiting.")
        # Create an empty log file anyway, or add a note to it
        try:
            with open(LOG_FILENAME, "w", encoding="utf-8") as log_file:
                log_file.write(f"Black Linter Log - Target: {TARGET_DIRECTORY}\n")
                log_file.write(
                    f"Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}\n"
                )
                log_file.write("=" * 80 + "\n\n")
                log_file.write("No Python files found or all were skipped.\n")
            print(f"Log file created/emptied: {LOG_FILENAME}")
        except IOError as e:
            print(
                f"Error writing initial message to log file '{LOG_FILENAME}': {e}", file=sys.stderr
            )
        return

    print(f"Found {len(py_files)} Python files to check.")
    files_needing_reformat = 0
    files_reformatted = 0  # Track files actually reformatted
    files_with_errors = 0

    try:
        with open(LOG_FILENAME, "w", encoding="utf-8") as log_file:
            log_file.write(f"Black Linter Log - Target: {TARGET_DIRECTORY}\n")
            log_file.write(
                f"Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}\n"
            )  # Log the mode
            log_file.write("=" * 80 + "\n\n")

            for py_file in py_files:
                print(f"Checking: {py_file}")
                log_file.write(f"--- Checking: {py_file} ---\n")

                # 1) Check + Diff to see if any reformatting is needed
                check_cmd = ["black", "--check", "--diff", "--line-length", "100", py_file]
                try:
                    check_process = subprocess.run(
                        check_cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        check=False,  # Don't raise exception on non-zero exit code
                    )
                except FileNotFoundError:
                    error_msg = "Error: 'black' command not found. Please ensure Black is installed and available in PATH.\n"
                    print(error_msg, file=sys.stderr)
                    log_file.write(error_msg)
                    return  # Exit the script if black is not found
                except Exception as e:
                    error_msg = f"Unexpected error running black check on {py_file}: {e}\n"
                    print(error_msg, file=sys.stderr)
                    log_file.write(error_msg)
                    files_with_errors += 1
                    continue  # Skip to the next file

                # --- Analyze check_process results ---
                if check_process.returncode == 0:
                    log_file.write("Status: OK (Already formatted correctly)\n\n")
                elif check_process.returncode == 1:
                    files_needing_reformat += 1
                    log_file.write("Status: Needs reformatting\n")
                    log_file.write("Diff:\n")
                    log_file.write(check_process.stdout)  # Diff is printed to stdout
                    if check_process.stderr:
                        log_file.write("Check stderr:\n")
                        log_file.write(check_process.stderr)
                    log_file.write("\n")

                    # --- Conditional Reformatting based on READ_ONLY flag ---
                    if not READ_ONLY:
                        # 2) Actually reformat (fix) the file
                        print(f"  Reformatting: {py_file}")
                        fix_cmd = ["black", "--line-length", "100", py_file]
                        try:
                            fix_process = subprocess.run(
                                fix_cmd,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                check=False,
                            )

                            log_file.write("Reformatting attempt (Read & Modify Mode):\n")
                            if fix_process.stderr:
                                log_file.write(f"Reformat output (stderr):\n{fix_process.stderr}\n")
                            if fix_process.stdout:  # Usually empty on success, but log if present
                                log_file.write(f"Reformat output (stdout):\n{fix_process.stdout}\n")

                            if fix_process.returncode != 0:
                                log_file.write(
                                    f"Reformatting returned non-zero exit code: {fix_process.returncode}\n"
                                )
                                files_with_errors += 1
                            else:
                                log_file.write("Status: Reformatting applied successfully.\n")
                                files_reformatted += (
                                    1  # Increment count of actually reformatted files
                                )

                        except FileNotFoundError:
                            error_msg = "Error: 'black' command not found during reformatting.\n"
                            print(error_msg, file=sys.stderr)
                            log_file.write(error_msg)
                            return
                        except Exception as e:
                            error_msg = (
                                f"Unexpected error running black reformat on {py_file}: {e}\n"
                            )
                            print(error_msg, file=sys.stderr)
                            log_file.write(error_msg)
                            files_with_errors += 1

                        log_file.write("\n")  # Extra newline after reformatting block
                    else:
                        # READ_ONLY is True, just log that we didn't modify
                        log_file.write("Status: Read-Only Mode - No changes applied.\n\n")

                else:  # Handle other non-zero return codes (e.g., 123 for internal errors)
                    files_with_errors += 1
                    log_file.write(
                        f"Status: Error during check (exit code {check_process.returncode})\n"
                    )
                    log_file.write("Check stdout:\n")
                    log_file.write(check_process.stdout)
                    log_file.write("Check stderr:\n")
                    log_file.write(check_process.stderr)
                    log_file.write("\n\n")

            # --- Write Summary ---
            log_file.write("=" * 80 + "\n")
            log_file.write(f"Summary: Checked {len(py_files)} files.\n")
            log_file.write(f" - {files_needing_reformat} file(s) needed reformatting.\n")
            if READ_ONLY:
                log_file.write(
                    " - Running in Read-Only mode: No files were modified by this script.\n"
                )
            else:
                log_file.write(
                    f" - Running in Read & Modify mode: {files_reformatted} file(s) were successfully reformatted.\n"
                )
                if files_needing_reformat > files_reformatted:
                    log_file.write(
                        f" - Note: {files_needing_reformat - files_reformatted} file(s) needed reformatting but encountered errors during the fix attempt.\n"
                    )
            log_file.write(
                f" - {files_with_errors} file(s) encountered errors during processing (check or reformat).\n"
            )

    except IOError as e:
        print(f"Error writing to log file '{LOG_FILENAME}': {e}", file=sys.stderr)
        return  # Exit if logging fails

    # --- Final Console Output ---
    print("-" * 40)
    print(f"Linting completed. Processed {len(py_files)} files.")
    print(
        f"Run Mode: {'Read-Only (check only)' if READ_ONLY else 'Read & Modify (check and apply)'}"
    )
    if files_needing_reformat > 0:
        if READ_ONLY:
            print(
                f"  {files_needing_reformat} file(s) would need reformatting (no changes applied)."
            )
        else:
            print(f"  {files_needing_reformat} file(s) needed reformatting.")
            print(f"  {files_reformatted} file(s) were successfully reformatted.")
            if files_needing_reformat > files_reformatted:
                print(
                    f"  {files_needing_reformat - files_reformatted} file(s) could not be reformatted due to errors."
                )
    else:
        print("  All checked files were already correctly formatted.")

    if files_with_errors > 0:
        print(f"  {files_with_errors} file(s) encountered errors during processing.")
    print(f"See '{LOG_FILENAME}' for detailed logs and diffs.")
    print("-" * 40)


if __name__ == "__main__":
    main()
