"""
Script Name:
    pylint_logger.py

Purpose:
    This script performs static analysis on Python files using Pylint and isort.
    It gathers all .py files from specified locations, skips excluded paths,
    and then for each file:
    1. Runs `pylint` to assess code quality and identify issues.
    2. Runs `isort` in check-only mode with a diff to detect unsorted imports.
    3. Validates the module-level docstring against a defined standard, checking for
       the presence and order of specific headers (Script Name, Purpose, Inputs, Outputs, Dependencies).
    The script logs detailed stdout/stderr from both tools to a timestamped .log file
    and generates an Excel summary (.xlsx) of the findings.

Inputs:
    1. List of Python files and/or folder paths to scan (specified by `FILES_OR_FOLDERS` list in the script).
    2. List of file or folder paths to skip during scanning (specified by `SKIP_PATHS` list in the script).
    3. Configuration constants defined within the script:
        - `OUTPUT_FOLDER`: Directory path for saving log and Excel summary files.
        - `LOG_LEVEL`: Logging level for console output (e.g., `logging.INFO`).
        - `DETAILED_LOG_FILENAME_PREFIX`: Prefix for the detailed log file name.
        - `REQUIRED_DOC_HEADERS`: Tuple of strings defining mandatory headers for module docstrings.

Outputs:
    1. Detailed Log File: A timestamped .log file (e.g., `lint_detailed_log_YYYYMMDD_HHMMSS.log`)
       in `OUTPUT_FOLDER`, containing the complete stdout and stderr from `pylint` and `isort`
       for each processed file.
    2. Excel Summary File: A timestamped .xlsx file (e.g., `lint_results_YYYYMMDD_HHMMSS.xlsx`)
       in `OUTPUT_FOLDER`, with a sheet "Lint Summary" detailing for each file:
        - Script Name
        - Immediate Parent Folder
        - Pylint Score
        - Number of Pylint Issues
        - Needs Isort Fix (Yes/No)
        - Docstring OK (Yes/No)
        - Full File Path
        - Combined Stderr from tools (if any)
    3. Console output: Status messages, progress updates, and a summary of the linting process.

Dependencies:
    1. Python standard libraries:
        - `datetime` (for timestamping)
        - `logging` (for console and file logging)
        - `os` (for file and path operations)
        - `pathlib` (for path manipulation)
        - `re` (for parsing Pylint output)
        - `subprocess` (for running external tools)
        - `sys` (for system-specific parameters and functions)
        - `ast` (for parsing Python code to check docstrings)
        - `typing` (for type hints)
    2. `openpyxl` (for writing Excel files).
    3. External command-line tools:
        - `pylint`: Must be installed and accessible (e.g., via `python -m pylint`).
        - `isort`: Must be installed and accessible (e.g., via `python -m isort`).
"""

from __future__ import annotations

import ast  # ← NEW: for parsing module‑level docstrings
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from openpyxl import Workbook

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

FILES_OR_FOLDERS: List[str] = [
    # Example: mix .py files and/or folders
    r"C:\Path\to\some_folder",
    # r"C:\Path\to\single_file.py",
]

SKIP_PATHS: List[str] = [
    # r"C:\Path\to\folder_to_skip",
    r"C:\Path\to\file_to_skip.py",
]

OUTPUT_FOLDER: str = r"C:\Path\to\Your\Logs\Folder"

LOG_LEVEL: int = logging.INFO  # set to logging.DEBUG for more detail
DETAILED_LOG_FILENAME_PREFIX: str = "lint_detailed_log"

# Required headers (case‑insensitive, must appear in this order) for a valid module docstring
REQUIRED_DOC_HEADERS: tuple[str, ...] = (
    "Script Name:",
    "Purpose:",
    "Inputs:",
    "Outputs:",
    "Dependencies:",
)

# ==================================================================================================
# LOGGING SETUP
# ==================================================================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_logger = logging.getLogger(__name__)

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


def setup_detailed_logger(
    output_folder: str, filename_prefix: str, log_level: int
) -> Tuple[logging.Logger, str]:
    """Create a dedicated file logger and return (logger, log_filepath)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{filename_prefix}_{timestamp}.log"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    log_filepath = os.path.join(output_folder, log_filename)

    detail_logger = logging.getLogger("LintDetailLogger")
    detail_logger.setLevel(log_level)
    detail_logger.propagate = False
    detail_logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    detail_logger.addHandler(file_handler)

    return detail_logger, log_filepath


def gather_python_files_in_folder(folder_path: str) -> List[str]:
    """Recursively list *.py inside folder_path."""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.endswith(".py")
    ]


def is_skipped(path_str: str, skip_list: List[str]) -> bool:
    """Return True if path_str matches or is inside any path in skip_list."""
    norm_path = os.path.abspath(path_str).lower()
    for s in skip_list:
        norm_skip = os.path.abspath(s).lower()
        if norm_path == norm_skip or norm_path.startswith(norm_skip + os.sep):
            return True
    return False


# ---------------------------  Docstring helpers ---------------------------------------------------


def docstring_is_valid(py_file: str) -> bool:
    """
    Return True when the module‑level docstring exists and contains every header
    in REQUIRED_DOC_HEADERS, in order (case‑insensitive).
    """
    try:
        with open(py_file, "r", encoding="utf-8") as fh:
            module = ast.parse(fh.read())
    except (SyntaxError, OSError):
        # unreadable or syntactically invalid file
        return False

    doc = ast.get_docstring(module)
    if not doc:
        return False

    lines = [ln.strip().lower() for ln in doc.splitlines() if ln.strip()]
    headers = [h.lower() for h in REQUIRED_DOC_HEADERS]

    idx = 0
    for line in lines:
        if line.startswith(headers[idx]):
            idx += 1
            if idx == len(headers):
                return True  # saw every header in correct order
    return False


# ---------------------------  Pylint helpers ------------------------------------------------------


def run_pylint_on_file(py_file: str) -> Tuple[str | None, str | None]:
    """Run pylint and return (stdout, stderr)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pylint", py_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.stdout, proc.stderr
    except FileNotFoundError:
        console_logger.error("python -m pylint not found. Is pylint installed?")
        return None, "Execution Error: pylint not found."
    except Exception as exc:  # pylint: disable=broad-except
        console_logger.error("Unexpected error running pylint on %s: %s", py_file, exc)
        return None, f"Execution Error: {exc}"


def parse_pylint_output(stdout_text: str | None) -> Tuple[float, int]:
    """Extract pylint score and issue count from stdout."""
    if not stdout_text:
        return 0.0, 0

    score = 0.0
    issues = 0
    issue_re = re.compile(r":\s*([CRWEF]\d{4}):")

    for line in stdout_text.splitlines():
        if issue_re.search(line):
            issues += 1
        if "Your code has been rated at" in line:
            try:
                score_part = line.split("rated at")[1].split("(")[0].strip()
                score = float(score_part.split("/")[0])
            except (IndexError, ValueError):
                console_logger.debug("Could not parse pylint score from: %s", line)
    return score, issues


# ----------------------------  isort helpers ------------------------------------------------------


def run_isort_check_on_file(py_file: str) -> Tuple[str | None, str | None, bool]:
    """
    Run isort in check‑only+diff mode.
    Returns (stdout, stderr, needs_reformat).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "isort", "--check-only", "--diff", py_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        needs_fix = proc.returncode != 0
        return proc.stdout, proc.stderr, needs_fix
    except FileNotFoundError:
        console_logger.error("python -m isort not found. Is isort installed?")
        return None, "Execution Error: isort not found.", False
    except Exception as exc:  # pylint: disable=broad-except
        console_logger.error("Unexpected error running isort on %s: %s", py_file, exc)
        return None, f"Execution Error: {exc}", False


# ==================================================================================================
# MAIN LINTING WORKFLOW
# ==================================================================================================


def lint_and_create_outputs(
    files_or_folders: List[str], skip_list: List[str], output_folder: str
) -> None:
    """Run pylint + isort, log details, build Excel summary."""
    detail_logger, log_filepath = setup_detailed_logger(
        output_folder, DETAILED_LOG_FILENAME_PREFIX, LOG_LEVEL
    )
    console_logger.info("Detailed lint log: %s", log_filepath)

    # -------------------------------------------------------------------------
    # Collect .py files
    # -------------------------------------------------------------------------
    collected: List[str] = []
    for entry in files_or_folders:
        abs_entry = os.path.abspath(entry)
        if not os.path.exists(abs_entry):
            console_logger.warning("Path not found, skipping: %s", abs_entry)
            continue
        if os.path.isdir(abs_entry):
            console_logger.info("Scanning folder: %s", abs_entry)
            collected.extend(gather_python_files_in_folder(abs_entry))
        elif abs_entry.lower().endswith(".py"):
            collected.append(abs_entry)
        else:
            console_logger.debug("Non‑Python path skipped: %s", abs_entry)

    # Remove duplicates (preserve order)
    unique_files = list(dict.fromkeys(collected))
    final_files = [f for f in unique_files if not is_skipped(f, skip_list)]

    if not final_files:
        console_logger.info("No Python files to lint after applying skips.")
        return

    console_logger.info("Linting %d files…", len(final_files))

    # -------------------------------------------------------------------------
    # Lint each file
    # -------------------------------------------------------------------------
    results: List[dict] = []
    for idx, py_file in enumerate(final_files, start=1):
        console_logger.info("[%d/%d] %s", idx, len(final_files), py_file)
        detail_logger.info("=" * 80)
        detail_logger.info("FILE: %s", py_file)
        detail_logger.info("Time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        detail_logger.info("=" * 80)

        # ---- Pylint ---------------------------------------------------------
        stdout_pylint, stderr_pylint = run_pylint_on_file(py_file)
        detail_logger.info("\n--- pylint stdout ---\n%s", stdout_pylint or "")
        if stderr_pylint:
            detail_logger.info("\n--- pylint stderr ---\n%s", stderr_pylint)

        score, issues = parse_pylint_output(stdout_pylint)

        # ---- Docstring check -----------------------------------------------
        doc_ok = "Yes" if docstring_is_valid(py_file) else "No"

        # ---- isort ----------------------------------------------------------
        stdout_isort, stderr_isort, needs_isort = run_isort_check_on_file(py_file)
        detail_logger.info("\n--- isort diff (check‑only) ---\n%s", stdout_isort or "")
        if stderr_isort:
            detail_logger.info("\n--- isort stderr ---\n%s", stderr_isort)

        results.append(
            {
                "script_name": os.path.basename(py_file),
                "immediate_folder": os.path.basename(os.path.dirname(py_file)),
                "full_path": py_file,
                "score": score,
                "issues": issues,
                "needs_isort": "Yes" if needs_isort else "No",
                "docstring_ok": doc_ok,
                "stderr": (stderr_pylint or "") + (stderr_isort or ""),
            }
        )

    # -------------------------------------------------------------------------
    # Excel summary
    # -------------------------------------------------------------------------
    console_logger.info("Building Excel summary…")
    wb = Workbook()
    ws = wb.active
    ws.title = "Lint Summary"

    ws.append(
        [
            "Script Name",
            "Immediate Folder",
            "Pylint Score",
            "Pylint Issues",
            "Needs Isort Fix",
            "Docstring OK",
            "Full Path",
            "Stderr (if any)",
        ]
    )

    for row in results:
        ws.append(
            [
                row["script_name"],
                row["immediate_folder"],
                row["score"],
                row["issues"],
                row["needs_isort"],
                row["docstring_ok"],
                row["full_path"],
                row["stderr"],
            ]
        )

    # Simple column autosize
    for col in ws.columns:
        length = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = length + 2

    excel_name = f"lint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = os.path.join(output_folder, excel_name)
    wb.save(excel_path)
    console_logger.info("Excel summary written: %s", excel_path)

    # Close file handlers
    for h in detail_logger.handlers:
        h.close()

    console_logger.info("Done.")


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:  # pylint: disable=missing-function-docstring
    lint_and_create_outputs(FILES_OR_FOLDERS, SKIP_PATHS, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
