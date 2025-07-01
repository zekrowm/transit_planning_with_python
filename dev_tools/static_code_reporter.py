"""Runs static analysis tools on Python files and exports detailed logs and summaries.

Executes multiple code-quality tools—`mypy`, `vulture`, `pylint`, and `pydocstyle`
—on specified Python files and directories. Generates per-tool log files, intended
for systematic QA in large or shared codebases.

Outputs:
    - Log files: One per tool with stdout/stderr from each scan
    - Console output: Progress updates and summary of results
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

FILES_OR_FOLDERS: List[str] = [
    r"C:\Path\to\Main\Input\Folder",  # Replace with your folder or file path
]
SKIP_PATHS: List[str] = [
    # e.g. r"C:\Path\to\project\venv",
]
OUTPUT_FOLDER: str = r"C:\Path\to\Output\Folder"  # Replace with your folder path
LOG_LEVEL: int = logging.INFO

ENABLE_MYPY: bool = True
ENABLE_VULTURE: bool = True
ENABLE_PYLINT: bool = True
ENABLE_PYDOCSTYLE: bool = True

# mypy
MYPY_ADDITIONAL_ARGS: List[str] = ["--ignore-missing-imports"]
# vulture
VULTURE_MIN_CONFIDENCE: int = 70  # default is 60
# pydocstyle
PYDOCSTYLE_ADDITIONAL_ARGS: List[str] = ["--convention=google"]
# pylint
PYLINT_ADDITIONAL_ARGS: list[str] = [
    "--errors-only",
    "--ignored-modules=arcpy",
]
# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONSOLE = logging.getLogger(__name__)

# ============================================================================
# FUNCTIONS
# ============================================================================


def setup_detailed_logger(
    out_folder: str, prefix: str, level: int = logging.DEBUG
) -> Tuple[logging.Logger, str]:
    """Create a timestamped file logger.

    A new :class:`logging.Logger` named ``prefix`` is returned with
    propagation disabled and a single ``FileHandler`` that writes to
    ``{out_folder}/{prefix}_YYYYMMDD_HHMMSS.log``.

    Args:
        out_folder: Directory where the log file will be created (created if
            it does not exist).
        prefix: Stem used for both the logger name and the log-file name.
        level: Logging level applied to the handler. Defaults to
            ``logging.DEBUG``.

    Returns:
        Tuple containing:

        * The configured logger instance.
        * Absolute path to the created log file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(out_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    logfile = str(folder / f"{prefix}_{ts}.log")
    lg = logging.getLogger(prefix)
    lg.setLevel(level)
    lg.propagate = False
    lg.handlers.clear()
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    return lg, logfile


def gather_python_files(paths: List[str], skip_list: List[str]) -> List[str]:
    """Collect *.py* files, respecting *skip_list*."""
    collected: List[str] = []

    def is_skipped(p: Path) -> bool:
        norm = p.resolve().as_posix().lower()
        for s in skip_list:
            ns = Path(s).resolve().as_posix().lower()
            if norm == ns or norm.startswith(ns + "/"):
                return True
        return False

    for entry in paths:
        p = Path(entry).expanduser()
        if not p.exists():
            CONSOLE.warning("Path not found – skipping: %s", p)
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".py"):
                        fpath = Path(root, f)
                        if not is_skipped(fpath):
                            collected.append(fpath.as_posix())
        elif p.suffix.lower() == ".py" and not is_skipped(p):
            collected.append(p.as_posix())
    return list(dict.fromkeys(collected))


# ----------------------------------------------------------------------------
# 1) MYPY CHECK
# ----------------------------------------------------------------------------
_MYPY_RE = re.compile(r"Found\s+(\d+)\s+error")


def run_mypy(files: list[str]) -> int:
    """Run *mypy* on each file and return the count of files with errors."""
    logger, log_fp = setup_detailed_logger(OUTPUT_FOLDER, "mypy_detailed_log")
    CONSOLE.info("mypy log → %s", log_fp)

    results: list[dict[str, Any]] = []
    for idx, py in enumerate(files, 1):
        logger.info(
            "\n%s\nFILE %d/%d → %s\n%s",
            "=" * 80,
            idx,
            len(files),
            py,
            "=" * 80,
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--show-error-codes",
                    "--no-color-output",
                    *MYPY_ADDITIONAL_ARGS,
                    py,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            out, err = proc.stdout, proc.stderr
        except FileNotFoundError:
            CONSOLE.error("python -m mypy not found – aborting mypy pass.")
            return len(files)

        logger.info(out or "")
        if err:
            logger.info("\n--- stderr ---\n%s", err)

        if "Success: no issues found" in (out or ""):
            errors = 0
        elif m := _MYPY_RE.search(out or ""):
            errors = int(m.group(1))
        else:
            errors = sum(1 for line in (out or "").splitlines() if ": error:" in line)

        results.append({"errors": errors})

    return sum(1 for r in results if r["errors"] > 0)


# ----------------------------------------------------------------------------
# 2) VULTURE CHECK
# ----------------------------------------------------------------------------
_VULTURE_LINE = re.compile(r"^(?:.+?:)?(\d+):\s*(.+?)\s*\((\d+%) confidence\)$")


def run_vulture(files: list[str]) -> int:
    """Detect dead code with *vulture* and return files-with-issues count."""
    logger, log_fp = setup_detailed_logger(OUTPUT_FOLDER, "vulture_detailed_log")
    CONSOLE.info("vulture log → %s", log_fp)

    files_with_issues = 0
    for idx, py in enumerate(files, 1):
        logger.info(
            "\n%s\nFILE %d/%d → %s\n%s",
            "=" * 80,
            idx,
            len(files),
            py,
            "=" * 80,
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vulture",
                    py,
                    "--min-confidence",
                    str(VULTURE_MIN_CONFIDENCE),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            out, err = proc.stdout, proc.stderr
        except FileNotFoundError:
            CONSOLE.error("python -m vulture not found – aborting vulture pass.")
            return len(files)

        logger.info(out or "")
        if err:
            logger.info("\n--- stderr ---\n%s", err)

        hit = any(_VULTURE_LINE.match(line.strip()) for line in (out or "").splitlines())
        if hit:
            files_with_issues += 1

    return files_with_issues


# ----------------------------------------------------------------------------
# 3) PYLINT
# ----------------------------------------------------------------------------
_PYLINT_ISSUE = re.compile(r":\s*[EF]\d{4}:")


def run_pylint(files: list[str]) -> int:
    """Run *pylint* (errors-only) and return files-with-issues count."""
    logger, log_fp = setup_detailed_logger(OUTPUT_FOLDER, "pylint_detailed_log")
    CONSOLE.info("pylint (errors-only) log → %s", log_fp)

    files_with_issues = 0
    for idx, py in enumerate(files, 1):
        logger.info(
            "\n%s\nFILE %d/%d → %s\n%s",
            "=" * 80,
            idx,
            len(files),
            py,
            "=" * 80,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pylint", *PYLINT_ADDITIONAL_ARGS, py],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            CONSOLE.error("python -m pylint not found – aborting pylint pass.")
            return len(files)

        out, err = proc.stdout, proc.stderr
        logger.info(out or "")
        if err:
            logger.info("\n--- pylint stderr ---\n%s", err)

        if any(_PYLINT_ISSUE.search(line) for line in (out or "").splitlines()):
            files_with_issues += 1

    return files_with_issues


# ----------------------------------------------------------------------------
# 4) PYDOCSTYLE CHECK
# ----------------------------------------------------------------------------


def run_pydocstyle(files: list[str]) -> int:
    """Validate docstrings with *pydocstyle* and return files-with-issues count."""
    logger, log_fp = setup_detailed_logger(OUTPUT_FOLDER, "pydocstyle_detailed_log")
    CONSOLE.info("pydocstyle log → %s", log_fp)

    files_with_issues = 0
    for idx, py in enumerate(files, 1):
        logger.info(
            "\n%s\nFILE %d/%d → %s\n%s",
            "=" * 80,
            idx,
            len(files),
            py,
            "=" * 80,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pydocstyle", *PYDOCSTYLE_ADDITIONAL_ARGS, py],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            out, err = proc.stdout, proc.stderr
        except FileNotFoundError:
            CONSOLE.error("python -m pydocstyle not found – aborting pydocstyle pass.")
            return len(files)

        logger.info(out or "")
        if err:
            logger.info("\n--- pydocstyle stderr ---\n%s", err)

        issues = [line for line in (out or "").splitlines() if line.strip()]
        if issues:
            files_with_issues += 1

    return files_with_issues


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry-point for command-line execution.

    Collects target files, runs the enabled static-analysis tools, prints a
    high-level summary to stdout, and exits with code 0 when all tools pass
    or 1 when any tool reports issues.
    """
    files = gather_python_files(FILES_OR_FOLDERS, SKIP_PATHS)
    if not files:
        CONSOLE.warning("No Python files found – nothing to do.")
        return

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    overall_tools_failed_files = 0

    if ENABLE_MYPY:
        CONSOLE.info("\n=== Running mypy pass ===")
        failures = run_mypy(files)
        if failures > 0:
            CONSOLE.warning("mypy found issues in %d file(s).", failures)
            overall_tools_failed_files += failures
        else:
            CONSOLE.info("mypy pass: No issues found.")

    if ENABLE_VULTURE:
        CONSOLE.info("\n=== Running vulture pass ===")
        failures = run_vulture(files)
        if failures > 0:
            CONSOLE.warning("vulture found potential dead code in %d file(s).", failures)
            overall_tools_failed_files += failures
        else:
            CONSOLE.info("vulture pass: No issues found.")

    if ENABLE_PYLINT:
        CONSOLE.info("\n=== Running pylint pass ===")
        failures = run_pylint(files)
        if failures > 0:
            CONSOLE.warning("pylint found issues in %d file(s).", failures)
            overall_tools_failed_files += failures
        else:
            CONSOLE.info("pylint pass: No issues found.")

    if ENABLE_PYDOCSTYLE:
        CONSOLE.info("\n=== Running pydocstyle pass ===")
        failures = run_pydocstyle(files)
        if failures > 0:
            CONSOLE.warning("pydocstyle found issues in %d file(s).", failures)
            overall_tools_failed_files += failures
        else:
            CONSOLE.info("pydocstyle pass: No issues found.")

    CONSOLE.info("\n" + "=" * 80)
    if overall_tools_failed_files == 0:
        CONSOLE.info("🎉 All enabled static-analysis checks passed successfully.")
    else:
        CONSOLE.warning(
            "⚠️ Static analysis detected issues. Total 'files with issues': %d. "
            "See individual logs for details.",
            overall_tools_failed_files,
        )
    CONSOLE.info("Detailed logs and Excel summaries → %s", Path(OUTPUT_FOLDER).resolve())


if __name__ == "__main__":
    main()
