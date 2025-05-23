"""
Script Name:
    mypy_checker.py

Purpose:
    Checks Python files with mypy, logs full output, and creates an Excel
    summary.  If any file has mypy errors, the script ends with exit code 1.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from openpyxl import Workbook

# =============================================================================
# CONFIGURATION
# =============================================================================

FILES_OR_FOLDERS: List[str] = [
    # r"C:\Path\to\some_folder",
    # r"C:\Path\to\single_file.py",
]

SKIP_PATHS: List[str] = [
    # r"C:\Path\to\skip_this_folder",
    # r"C:\Path\to\skip_this_file.py",
]

OUTPUT_FOLDER: str = r"C:\Path\to\Your\Logs\Folder"

LOG_LEVEL: int = logging.INFO  # use logging.DEBUG for more detail
DETAILED_LOG_FILENAME_PREFIX = "mypy_detailed_log"

# Extra mypy flags (e.g. ["--ignore-missing-imports"])
MYPY_ADDITIONAL_ARGS: List[str] = []

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================


def setup_detailed_logger(
    out_folder: str, prefix: str, level: int
) -> Tuple[logging.Logger, str]:
    """Return (dedicated file logger, log_filepath)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(out_folder).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    log_filepath = str(log_path / f"{prefix}_{timestamp}.log")

    detail_logger = logging.getLogger("MypyDetailLogger")
    detail_logger.setLevel(level)
    detail_logger.propagate = False
    detail_logger.handlers.clear()

    fh = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    detail_logger.addHandler(fh)
    return detail_logger, log_filepath


def gather_python_files(folder_path: str) -> List[str]:
    """Recursively list *.py inside folder_path."""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.endswith(".py")
    ]


def is_skipped(path_str: str, skip_list: List[str]) -> bool:
    """True if path_str equals/is inside any skip entry."""
    norm_path = os.path.abspath(path_str).lower()
    for skip in skip_list:
        norm_skip = os.path.abspath(skip).lower()
        if norm_path == norm_skip or norm_path.startswith(norm_skip + os.sep):
            return True
    return False


# ------------------------------ mypy helpers ---------------------------------

_MYPY_RE = re.compile(r"Found\s+(\d+)\s+error")


def run_mypy(py_file: str) -> Tuple[str | None, str | None]:
    """Run mypy; return (stdout, stderr)."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--show-error-codes",
                "--no-color-output",
                *MYPY_ADDITIONAL_ARGS,
                py_file,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.stdout, proc.stderr
    except FileNotFoundError:
        console_logger.error("python -m mypy not found. Is mypy installed?")
        return None, "Execution Error: mypy not found."
    except Exception as exc:  # pylint: disable=broad-except
        console_logger.error("Unexpected error running mypy on %s: %s", py_file, exc)
        return None, f"Execution Error: {exc}"


def parse_mypy(stdout: str | None) -> Tuple[int, bool]:
    """Return (#errors, success_flag)."""
    if not stdout:
        return 1, False
    if "Success: no issues found" in stdout:
        return 0, True
    if m := _MYPY_RE.search(stdout):
        return int(m.group(1)), False
    # fallback – count explicit ': error:' lines
    errs = [ln for ln in stdout.splitlines() if ": error:" in ln]
    return len(errs), len(errs) == 0


# =============================================================================
# CORE WORKFLOW
# =============================================================================


def run_checks(
    files_or_folders: List[str], skip_list: List[str], out_folder: str
) -> int:
    """
    Run mypy on each target file, create Excel + log.
    Returns the number of files with mypy errors.
    """
    detail_logger, log_fp = setup_detailed_logger(
        out_folder, DETAILED_LOG_FILENAME_PREFIX, LOG_LEVEL
    )
    console_logger.info("Detailed mypy log: %s", log_fp)

    # -------- collect .py files
    collected: List[str] = []
    for entry in files_or_folders:
        entry = os.path.abspath(entry)
        if not os.path.exists(entry):
            console_logger.warning("Path not found, skipping: %s", entry)
            continue
        if os.path.isdir(entry):
            console_logger.info("Scanning folder: %s", entry)
            collected.extend(gather_python_files(entry))
        elif entry.lower().endswith(".py"):
            collected.append(entry)

    unique_files = list(dict.fromkeys(collected))
    targets = [f for f in unique_files if not is_skipped(f, skip_list)]
    if not targets:
        console_logger.info("No Python files to check after applying skips.")
        return 0

    console_logger.info("Type-checking %d files…", len(targets))

    # -------- run mypy
    results: List[dict] = []
    for i, py in enumerate(targets, 1):
        console_logger.info("[%d/%d] %s", i, len(targets), py)
        detail_logger.info("=" * 80)
        detail_logger.info("FILE: %s", py)
        detail_logger.info("Time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        detail_logger.info("=" * 80)

        out, err = run_mypy(py)
        detail_logger.info("\n--- mypy stdout ---\n%s", out or "")
        if err:
            detail_logger.info("\n--- mypy stderr ---\n%s", err)

        n_errors, ok = parse_mypy(out)

        results.append(
            {
                "script_name": os.path.basename(py),
                "folder": os.path.basename(os.path.dirname(py)),
                "full_path": py,
                "errors": n_errors,
                "success": "Yes" if ok else "No",
                "stderr": err or "",
            }
        )

    # -------- Excel summary
    wb = Workbook()
    ws = wb.active
    ws.title = "mypy Summary"
    ws.append(
        [
            "Script Name",
            "Immediate Folder",
            "# mypy Errors",
            "Mypy Success",
            "Full Path",
            "Stderr (if any)",
        ]
    )
    for row in results:
        ws.append(
            [
                row["script_name"],
                row["folder"],
                row["errors"],
                row["success"],
                row["full_path"],
                row["stderr"],
            ]
        )
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = (
            max(len(str(c.value or "")) for c in col) + 2
        )

    excel_path = Path(out_folder) / f"mypy_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    wb.save(excel_path)
    console_logger.info("Excel summary written: %s", excel_path)

    # -------- overall summary / exit code
    files_with_errors = sum(1 for r in results if r["errors"] > 0)
    if files_with_errors == 0:
        console_logger.info(
            "✅ Success: no mypy issues found in %d files.", len(results)
        )
    else:
        console_logger.warning(
            "❌ %d of %d file(s) have mypy errors.", files_with_errors, len(results)
        )

    for h in detail_logger.handlers:
        h.close()

    return files_with_errors


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:  # pylint: disable=missing-function-docstring
    exit_code = 1 if run_checks(FILES_OR_FOLDERS, SKIP_PATHS, OUTPUT_FOLDER) else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
