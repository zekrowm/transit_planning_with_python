"""
black_linter.py

This script will:
1) Gather all .py files within TARGET_DIRECTORY.
2) Skip any paths found in SKIP_PATHS.
3) For each file not skipped:
   - Run `black --check --diff` to see if reformatting is needed. 
   - If changes are needed, log the diff to LOG_FILENAME and then run `black` to fix it.
4) Writes all diffs and reformat logs to the specified LOG_FILENAME.
"""

import os
import subprocess
import sys

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

TARGET_DIRECTORY = r'C:\Your\Input\Folder\Path'
LOG_DIRECTORY = r'C:\Your\Output\Folder\Path'
LOG_FILENAME = os.path.join(LOG_DIRECTORY, "black_check.log")
# --- Updated SKIP_PATHS ---
# Use simple directory/file names for os.walk skipping
SKIP_NAMES = [
    "__pycache__",
    ".venv",        # Example: common directory to skip
    ".git",         # Example: common directory to skip
    # Add other directory or file names to skip here
]

# Create log directory if it doesn't exist.
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    print(f"Log directory '{LOG_DIRECTORY}' ensured.")
except OSError as e:
    print(f"Error creating log directory '{LOG_DIRECTORY}': {e}", file=sys.stderr)
    sys.exit(1) # Exit if we can't create the log directory

# --------------------------------------------------------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------------------------------------------------------

def main():
    py_files = []
    print(f"Searching for .py files in: {TARGET_DIRECTORY}")
    print(f"Skipping names: {SKIP_NAMES}")

    for root, dirs, files in os.walk(TARGET_DIRECTORY, topdown=True):
        # --- Modify dirs in-place to prevent descending into skipped directories ---
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]

        for filename in files:
            if filename.endswith('.py') and filename not in SKIP_NAMES:
                full_path = os.path.join(root, filename)
                print(f"  Found: {full_path}") # Debug print
                py_files.append(full_path)

    if not py_files:
        print("No Python files found (or all were skipped). Exiting.")
        # Create an empty log file anyway, or add a note to it
        try:
            with open(LOG_FILENAME, 'w', encoding='utf-8') as log_file:
                log_file.write("No Python files found or all were skipped.\n")
            print(f"Log file created/emptied: {LOG_FILENAME}")
        except IOError as e:
            print(f"Error writing initial message to log file '{LOG_FILENAME}': {e}", file=sys.stderr)
        return

    print(f"Found {len(py_files)} Python files to check.")
    files_needing_reformat = 0
    files_with_errors = 0

    try:
        with open(LOG_FILENAME, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Black Linter Log - Target: {TARGET_DIRECTORY}\n")
            log_file.write("=" * 80 + "\n\n")

            for py_file in py_files:
                print(f"Checking: {py_file}") # Debug print
                log_file.write(f"--- Checking: {py_file} ---\n")

                # 1) Check + Diff to see if any reformatting is needed
                # Ensure line length is treated as a string for the command list
                check_cmd = ['black', '--check', '--diff', '--line-length', '100', py_file]
                try:
                    check_process = subprocess.run(
                        check_cmd,
                        capture_output=True, # Captures both stdout and stderr
                        text=True,
                        encoding='utf-8',
                        check=False # Don't raise exception on non-zero exit code
                    )
                except FileNotFoundError:
                    error_msg = "Error: 'black' command not found. Please ensure Black is installed and available in PATH.\n"
                    print(error_msg, file=sys.stderr)
                    log_file.write(error_msg)
                    return # Exit the script if black is not found
                except Exception as e:
                    error_msg = f"Unexpected error running black check on {py_file}: {e}\n"
                    print(error_msg, file=sys.stderr)
                    log_file.write(error_msg)
                    files_with_errors += 1
                    continue # Skip to the next file

                # --- Analyze check_process results ---
                # Black exit codes:
                # 0: Nothing would change.
                # 1: Something would change.
                # 123: Internal error.

                if check_process.returncode == 0:
                    log_file.write("Status: OK (Already formatted correctly)\n\n")
                elif check_process.returncode == 1:
                    files_needing_reformat += 1
                    log_file.write("Status: Needs reformatting\n")
                    log_file.write("Diff:\n")
                    log_file.write(check_process.stdout) # Diff is printed to stdout
                    if check_process.stderr: # Sometimes black --check prints warnings/info to stderr too
                        log_file.write("Check stderr:\n")
                        log_file.write(check_process.stderr)
                    log_file.write("\n")

                    # 2) Actually reformat (fix) the file
                    print(f"  Reformatting: {py_file}") # Debug print
                    fix_cmd = ['black', '--line-length', '100', py_file]
                    try:
                        fix_process = subprocess.run(
                            fix_cmd,
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            check=False
                        )

                        log_file.write("Reformatting attempt:\n")
                        # Log stderr for reformatting status, stdout might be empty
                        if fix_process.stderr:
                             log_file.write(f"Reformat output (stderr):\n{fix_process.stderr}\n")
                        if fix_process.stdout:
                             log_file.write(f"Reformat output (stdout):\n{fix_process.stdout}\n")

                        if fix_process.returncode != 0:
                            log_file.write(f"Reformatting returned non-zero exit code: {fix_process.returncode}\n")
                            files_with_errors += 1
                        else:
                             log_file.write("Status: Reformatting applied successfully.\n")

                    except FileNotFoundError:
                         # Should not happen if check worked, but handle defensively
                        error_msg = "Error: 'black' command not found during reformatting.\n"
                        print(error_msg, file=sys.stderr)
                        log_file.write(error_msg)
                        return
                    except Exception as e:
                        error_msg = f"Unexpected error running black reformat on {py_file}: {e}\n"
                        print(error_msg, file=sys.stderr)
                        log_file.write(error_msg)
                        files_with_errors += 1

                    log_file.write("\n") # Extra newline after reformatting block

                else: # Handle other non-zero return codes (e.g., 123 for internal errors)
                    files_with_errors += 1
                    log_file.write(f"Status: Error during check (exit code {check_process.returncode})\n")
                    log_file.write("Check stdout:\n")
                    log_file.write(check_process.stdout)
                    log_file.write("Check stderr:\n")
                    log_file.write(check_process.stderr)
                    log_file.write("\n\n")

            log_file.write("=" * 80 + "\n")
            log_file.write(f"Summary: Checked {len(py_files)} files. ")
            log_file.write(f"{files_needing_reformat} needed reformatting. ")
            log_file.write(f"{files_with_errors} encountered errors during processing.\n")

    except IOError as e:
        print(f"Error writing to log file '{LOG_FILENAME}': {e}", file=sys.stderr)
        return # Exit if logging fails

    print(f"Linting completed. Processed {len(py_files)} files.")
    if files_needing_reformat > 0:
        print(f"  {files_needing_reformat} file(s) were reformatted.")
    if files_with_errors > 0:
        print(f"  {files_with_errors} file(s) encountered errors during processing.")
    print(f"See {LOG_FILENAME} for details.")

if __name__ == "__main__":
    main()
