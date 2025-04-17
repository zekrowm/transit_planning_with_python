"""
black_lintfix.py

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

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_DIRECTORY = r'C:\Your\Input\Folder\Path'
LOG_DIRECTORY = r'C:\Your\Output\Folder\Path'
LOG_FILENAME = os.path.join(LOG_DIRECTORY, "black_check.log")
SKIP_PATHS = [
    "__pycache__",
]

# Create log directory if it doesn't exist.
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# -----------------------------------------------------------------------------
# SKIP LOGIC
# -----------------------------------------------------------------------------
def is_skipped(path_str, skip_list):
    path_str_abs = os.path.abspath(path_str)
    for skip_entry in skip_list:
        skip_entry_abs = os.path.abspath(skip_entry)
        if path_str_abs == skip_entry_abs:
            return True
        if path_str_abs.startswith(skip_entry_abs + os.sep):
            return True
    return False

# -----------------------------------------------------------------------------
# MAIN PROCESS
# -----------------------------------------------------------------------------
def main():
    # Gather all .py files in TARGET_DIRECTORY (recursively).
    py_files = []
    for root, _, files in os.walk(TARGET_DIRECTORY):
        for filename in files:
            if filename.endswith('.py'):
                full_path = os.path.join(root, filename)
                if is_skipped(full_path, SKIP_PATHS):
                    continue
                py_files.append(full_path)

    if not py_files:
        print("No Python files found (or all were skipped). Exiting.")
        return

    with open(LOG_FILENAME, 'w', encoding='utf-8') as log_file:
        for py_file in py_files:
            # 1) Check + Diff to see if any reformatting is needed
            check_cmd = ['black', '--check', '--diff', py_file]
            try:
                check_process = subprocess.run(
                    check_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',  # Explicitly decode using UTF-8
                )
            except FileNotFoundError:
                log_file.write("Error: 'black' command not found. Please ensure Black is installed and available in PATH.\n")
                return

            if check_process.returncode == 1:
                log_file.write(f"File needs reformatting: {py_file}\n")
                log_file.write(check_process.stdout)
                log_file.write("\n\n")

                # 2) Actually reformat (fix) the file
                fix_cmd = ['black', py_file]
                try:
                    fix_process = subprocess.run(
                        fix_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',  # Explicitly decode using UTF-8
                    )
                except FileNotFoundError:
                    log_file.write("Error: 'black' command not found during reformatting. Please ensure Black is installed and available in PATH.\n")
                    return

                log_file.write(f"Reformatted: {py_file}\n")
                log_file.write(fix_process.stdout)
                log_file.write("\n\n")

            elif check_process.returncode not in (0, 1):
                log_file.write(f"Error checking {py_file}:\n")
                log_file.write(check_process.stderr)
                log_file.write("\n\n")

    print(f"Linting completed successfully. See {LOG_FILENAME} for details.")

if __name__ == "__main__":
    main()
