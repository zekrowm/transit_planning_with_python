"""Runs ruff on Python source files, exporting detailed logs.

The script recursively scans one or more target paths, optionally skipping
specified sub-paths, then:

1. Executes ``ruff check`` on each file (or ``ruff check --fix`` when
   ``READ_ONLY`` is ``False``).
2. Writes per-file stdout/stderr to a timestamped log file.
3. Exits with code 0 if all files pass, or 1 if any file reports issues.

Outputs
-------
Detailed log
    <OUTPUT_FOLDER>/ruff_detailed_<timestamp>.log
Exit status
    0 when every file is clean; 1 otherwise.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

FILES_OR_FOLDERS: List[str] = [
    r"C:\Path\to\Your\Project",  # <<< CHANGE ME
]

SKIP_PATHS: List[str] = [
    r"C:\Path\to\Your\Project\Folder1_to_Skip",
    r"C:\Path\to\Your\Project\Folder2_to_Skip",
]

OUTPUT_FOLDER: str = r"C:\Path\to\Your\Logs_Output_Folder"  # <<< CHANGE ME
READ_ONLY: bool = True  # False → apply fixes

RUFF_ADDITIONAL_ARGS: List[str] = []  # e.g. ["--extend-exclude", "migrations"]

LOG_LEVEL: int = logging.INFO

RUFF_CLI_ARGS: list[str] = [
    # top-level settings
    "--line-length",
    "100",
    "--target-version",
    "py310",
    # rule enablement
    "--select",
    "I,F,D,ANN,TC",
    "--fixable",
    "F401,D,I,TC003",
    "--ignore",
    "ANN401",
    # pydocstyle: Google convention
    "--config",
    'lint.pydocstyle.convention="google"',
]

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONSOLE = logging.getLogger(__name__)


# =============================================================================
# SCRIPTS
# =============================================================================


def _is_skipped(p: Path, skip_list: List[str]) -> bool:
    """Return *True* when *p* matches or is inside any ``skip_list`` path."""
    norm = p.resolve().as_posix().lower()
    for s in skip_list:
        ns = Path(s).expanduser().resolve().as_posix().lower()
        if norm == ns or norm.startswith(ns + "/"):
            return True
    return False


def gather_python_files(targets: List[str], skip_list: List[str]) -> List[str]:
    """Collect ``.py`` files from *targets*, excluding *skip_list*."""
    collected: List[str] = []
    for entry in targets:
        p = Path(entry).expanduser()
        if not p.exists():
            CONSOLE.warning("Path not found – skipping: %s", p)
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".py"):
                        fpath = Path(root, f)
                        if not _is_skipped(fpath, skip_list):
                            collected.append(fpath.as_posix())
        elif p.suffix.lower() == ".py" and not _is_skipped(p, skip_list):
            collected.append(p.as_posix())

    # de-duplicate but keep order
    return list(dict.fromkeys(collected))


# -----------------------------------------------------------------------------
# RUFF PASS
# -----------------------------------------------------------------------------


def _setup_detailed_logger(out_folder: str) -> Tuple[logging.Logger, str]:
    """Create and return a file logger plus its log-file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(out_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    logfile = str(folder / f"ruff_detailed_{ts}.log")

    lg = logging.getLogger("ruff_detailed")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    lg.handlers.clear()

    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    return lg, logfile


def run_ruff(files: List[str], read_only: bool) -> int:
    """Run Ruff on *files*, write a detailed log, and return the number of files with issues.

    Args:
    ----------
    files : List[str]
        Absolute or relative paths to Python source files.
    read_only : bool
        • ``True``  → run ``ruff check`` (report only)
        • ``False`` → run ``ruff check --fix`` (apply autofixes)

    Returns:
    -------
    int
        Number of files that Ruff reported as having at least one issue.
    """
    logger, log_fp = _setup_detailed_logger(OUTPUT_FOLDER)
    CONSOLE.info("ruff log → %s", log_fp)

    files_with_issues = 0
    total_issues = 0

    ruff_base_cmd: list[str] = ["ruff", "check", *RUFF_CLI_ARGS, *RUFF_ADDITIONAL_ARGS]
    if not read_only:
        ruff_base_cmd.append("--fix")

    for idx, py in enumerate(files, start=1):
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
                [*ruff_base_cmd, py],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:  # Ruff not installed / not on PATH
            CONSOLE.error("'ruff' command not found. Install it via 'pip install ruff'.")
            sys.exit(1)

        out, err, rc = proc.stdout, proc.stderr, proc.returncode
        logger.info(out or "")
        if err:
            logger.info("\n--- ruff stderr ---\n%s", err)

        # Ruff exit codes: 0 = clean, 1 = issues, ≥2 = internal error
        if rc == 1:
            issue_count = sum(1 for line in (out or "").splitlines() if ":" in line)
            total_issues += issue_count
            files_with_issues += 1
        elif rc not in (0, 1):
            # Treat internal errors as a single issue
            files_with_issues += 1

    logger.info(
        "ruff flagged %d issue(s) across %d file(s).",
        total_issues,
        files_with_issues,
    )
    return files_with_issues


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entrypoint – collect files, run ruff, print high-level summary."""
    files = gather_python_files(FILES_OR_FOLDERS, SKIP_PATHS)
    if not files:
        CONSOLE.warning("No Python files found – nothing to do.")
        return

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    CONSOLE.info(
        "=== Running ruff (%s mode) on %d file(s) ===",
        "check-only" if READ_ONLY else "check + fix",
        len(files),
    )
    failed = run_ruff(files, READ_ONLY)

    CONSOLE.info("\n" + "=" * 80)
    if failed == 0:
        CONSOLE.info("🎉  ruff reports no outstanding issues.")
        sys.exit(0)
    else:
        CONSOLE.warning(
            "⚠️  ruff detected problems in %d file(s). See the detailed log for details.",
            failed,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
