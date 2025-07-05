"""Runs ruff on Python source files, exporting detailed logs, plus an informational ty pass.

The script recursively scans one or more target paths, optionally skipping
specified sub-paths, then:

1. Executes ``ruff check`` on each file (or ``ruff check --fix`` when
   ``READ_ONLY`` is ``False``).
2. Writes per-file stdout/stderr to a timestamped Ruff log file.
3. Optionally runs ``ty check --exit-zero`` on the same files, exporting
   an informational type-check log.
4. Exits with code 0 if all files pass Ruff, or 1 if any file reports issues.

Outputs
-------
Detailed Ruff log
    <OUTPUT_FOLDER>/ruff_detailed_<timestamp>.log
Detailed ty log (if enabled)
    <OUTPUT_FOLDER>/ty_detailed_<timestamp>.log
Exit status
    0 when Ruff reports no outstanding issues; 1 otherwise.
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

FILES_OR_FOLDERS: List[str] = [r"C:\Path\to\Project"]  # <<< CHANGE ME

SKIP_PATHS: List[str] = [
 #   r"C:\Path\to\Project\.venv",
 #   r"C:\Path\to\Project\tests",
]

OUTPUT_FOLDER: str = r"C:\Path\to\Project\Logs"  # <<< CHANGE ME

READ_ONLY: bool = True  # False → apply fixes
RUFF_ADDITIONAL_ARGS: List[str] = []  # e.g. ["--extend-exclude", "migrations"]
RUFF_TOML_PATH: str | None = None    # e.g. r"C:\Path\to\Project\ruff.toml"

ENABLE_TY: bool = True           # False → skip ty altogether
TY_ADDITIONAL_ARGS: List[str] = []       # e.g. ["--project", "src"]
TY_TOML_PATH: str | None = None          # e.g. r"C:\Path\to\Project\ty.toml"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

LOG_LEVEL: int = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONSOLE = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================

def _is_skipped(p: Path, skip_list: List[str]) -> bool:
    """Return True when p matches or is inside any path in skip_list."""
    norm = p.resolve().as_posix().lower()
    for s in skip_list:
        ns = Path(s).expanduser().resolve().as_posix().lower()
        if norm == ns or norm.startswith(ns + "/"):
            return True
    return False


def gather_python_files(targets: List[str], skip_list: List[str]) -> List[str]:
    """Collect .py files from targets, excluding any in skip_list."""
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


def _setup_detailed_logger(out_folder: str) -> Tuple[logging.Logger, str]:
    """Create and return a file logger plus its log-file path for Ruff."""
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
    """Run Ruff on files, write a detailed log, and return count of files with issues."""
    logger, log_fp = _setup_detailed_logger(OUTPUT_FOLDER)
    CONSOLE.info("ruff log → %s", log_fp)

    files_with_issues = 0
    total_issues = 0

    ruff_base_cmd: List[str] = ["ruff", "check", *RUFF_ADDITIONAL_ARGS]
    if not read_only:
        ruff_base_cmd.append("--fix")
    if RUFF_TOML_PATH:
        ruff_base_cmd.extend(["--config", RUFF_TOML_PATH])

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
        except FileNotFoundError:
            CONSOLE.error("'ruff' command not found. Install via 'pip install ruff'.")
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
            files_with_issues += 1

    logger.info(
        "ruff flagged %d issue(s) across %d file(s).",
        total_issues,
        files_with_issues,
    )
    return files_with_issues


def _setup_ty_logger(out_folder: str) -> Tuple[logging.Logger, str]:
    """Create and return a file logger plus its log-file path for ty."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(out_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    logfile = str(folder / f"ty_detailed_{ts}.log")

    lg = logging.getLogger("ty_detailed")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    lg.handlers.clear()

    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    return lg, logfile


def run_ty(files: List[str]) -> int:
    """Run ty in info-only mode (--exit-zero) and export a detailed log."""
    logger, log_fp = _setup_ty_logger(OUTPUT_FOLDER)
    CONSOLE.info("ty log     → %s", log_fp)

    ty_cmd: List[str] = ["ty", "check", "--exit-zero", *TY_ADDITIONAL_ARGS]
    if TY_TOML_PATH:
        ty_cmd.extend(["--config-file", TY_TOML_PATH])
    ty_cmd.extend(files)

    try:
        proc = subprocess.run(
            ty_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        CONSOLE.warning(
            "'ty' not found – skipping type-check. "
            "Install via `pip install ty` or your preferred tool installer."
        )
        return 0

    logger.info(proc.stdout or "")
    if proc.stderr:
        logger.info("\n--- ty stderr ---\n%s", proc.stderr)

    diag_count = sum(1 for ln in (proc.stdout or "").splitlines() if ":" in ln)
    logger.info("ty reported %d diagnostic(s).", diag_count)
    return diag_count


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Entrypoint – gather files, run Ruff, optionally run ty, emit summary."""
    files = gather_python_files(FILES_OR_FOLDERS, SKIP_PATHS)
    if not files:
        CONSOLE.warning("No Python files found – nothing to do.")
        return

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    CONSOLE.info(
        "=== Running Ruff (%s mode) on %d file(s) ===",
        "check-only" if READ_ONLY else "check + fix",
        len(files),
    )
    failed = run_ruff(files, READ_ONLY)

    if ENABLE_TY:
        CONSOLE.info("=== Running ty (information-only) on %d file(s) ===", len(files))
        run_ty(files)

    CONSOLE.info("\n" + "=" * 80)
    if failed == 0:
        CONSOLE.info("🎉 Ruff reports no outstanding issues.")
        sys.exit(0)
    else:
        CONSOLE.warning(
            "⚠️ Ruff detected problems in %d file(s). "
            "See the detailed log for details.",
            failed,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
