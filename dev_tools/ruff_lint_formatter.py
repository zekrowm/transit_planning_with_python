"""Run Ruff on a collection of Python source files with a single invocation.

This script is designed for efficient and consistent static analysis using
Ruff, supporting use cases like pre-commit hooks, CI pipelines, or local
batch validation with optional autofix.

Features:
- Accepts explicit file or folder paths (via command-line arguments or hardcoded fallback).
- Excludes predefined paths (e.g. tests, venv).
- Applies a single Ruff run for performance.
- Disables ANSI color codes for log cleanliness.
- Automatically logs detailed output to a timestamped log file.
- Supports custom per-file suppressions for known missing stubs (e.g., `arcpy`, `geopandas`).
- Outputs a concise rule-level summary (e.g., `D212:3, ANN001:2`).

Configuration constants can be edited in-place to adapt to local environments.

Dependencies:
    - Ruff must be installed and accessible via CLI (`ruff` or `python -m ruff`).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to scan (empty → auto-discover repo root or use CLI args)
FILES_OR_FOLDERS: list[str] = []

# Folders or files to exclude
SKIP_PATHS: List[str] = ["tests"]  # e.g. ["venv", "build", "tests"]

# Where to save Ruff logs and any artifacts
OUTPUT_FOLDER: str = ".artifacts" # relative path is fine, replace with local folder path if desired

# If True, only check; if False, allow autofix
READ_ONLY: bool = False # False → ruff --fix

# Extra flags you want to pass straight through to Ruff
RUFF_ADDITIONAL_ARGS: list[str] = [] # e.g. ["--extend-exclude", "migrations"]

# Allow explicit file list from the command line (e.g. CI “changed files” step)
if len(sys.argv) > 1:
    FILES_OR_FOLDERS[:] = sys.argv[1:]

# -----------------------------------------------------------------------------  
# Backup Ruff flags (used **only** when no project config is found)  
# -----------------------------------------------------------------------------  

BACKUP_RUFF_CLI_ARGS: list[str] = [
    "--line-length", "100",
    "--target-version", "py310",
    "--select", "I,F,D,ANN,TCH",
    "--fixable", "F401,D,I,TCH003",
    "--ignore", "ANN401",
    # "--pydocstyle-convention", "google",
]

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONSOLE = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ruff configuration discovery
# -----------------------------------------------------------------------------

def _find_ruff_config(start: Path | None = None) -> Path | None:
    """Return the first Ruff config file found when walking upward."""
    for parent in (start or Path(__file__).resolve().parent).resolve().parents:
        for name in ("pyproject.toml", "ruff.toml", ".ruff.toml"):
            cfg = parent / name
            if cfg.is_file():
                return cfg
    return None


_HAS_USER_RUFF_CONFIG = _find_ruff_config() is not None
_EFFECTIVE_RUFF_CLI_ARGS: list[str] = (
    [] if _HAS_USER_RUFF_CONFIG else BACKUP_RUFF_CLI_ARGS
)

CONSOLE.info(
    "Using %s Ruff configuration.",
    "project" if _HAS_USER_RUFF_CONFIG else "embedded backup",
)

# ── fine-grained suppressions for libs without stubs ──────────────────────────
_NO_STUB_LIBS = {"geopandas", "arcpy"}
_SILENCED_CODES = ("ANN", "TCH")

# =============================================================================
# FUNCTIONS
# =============================================================================


def _default_targets() -> list[str]:
    """Return a default list of paths to scan.

    If command-line arguments are provided, they are used as the targets.
    Otherwise, the parent directory of this script is returned as a fallback.

    Returns:
        list[str]: List of file or directory paths to scan.
    """
    if FILES_OR_FOLDERS:
        return FILES_OR_FOLDERS
    # fallback: repo root (folder above this script)
    return [Path(__file__).resolve().parents[1].as_posix()]


def _is_skipped(p: Path, skip: list[str]) -> bool:
    """Determine whether a given path should be excluded from scanning.

    Args:
        p (Path): Path to evaluate.
        skip (list[str]): List of paths to skip, normalized to lowercase.

    Returns:
        bool: True if the path should be skipped, False otherwise.
    """
    norm = p.resolve().as_posix().lower()
    for s in skip:
        tgt = Path(s).expanduser().resolve().as_posix().lower()
        if norm == tgt or norm.startswith(tgt + "/"):
            return True
    return False


def gather_python_files(targets: list[str], skip: list[str]) -> list[str]:
    """Recursively collect all Python files from the given targets, excluding skip paths.

    Args:
        targets (list[str]): Files or directories to scan.
        skip (list[str]): Directories to exclude (e.g., 'tests', 'venv').

    Returns:
        list[str]: List of discovered `.py` file paths.
    """
    out: list[str] = []
    for entry in targets:
        p = Path(entry).expanduser()
        if not p.exists():
            CONSOLE.warning("Path not found – skipping: %s", p)
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                root_p = Path(root)
                parts_lower = {part.lower() for part in root_p.parts}
                if _is_skipped(root_p, skip) or "tests" in parts_lower:
                    continue
                for f in files:
                    if f.endswith(".py"):
                        fp = root_p / f
                        if not _is_skipped(fp, skip):
                            out.append(fp.as_posix())
        elif p.suffix.lower() == ".py" and not _is_skipped(p, skip):
            out.append(p.as_posix())
    # remove duplicates, preserving order
    return list(dict.fromkeys(out))


def _setup_detailed_logger(out_dir: str) -> Tuple[logging.Logger, Path]:
    """Create and configure a logger that writes detailed output to a timestamped file.

    Args:
        out_dir (str): Directory where the log file should be saved.

    Returns:
        Tuple[logging.Logger, Path]: The logger object and the path to the log file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(out_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"ruff_detailed_{ts}.log"

    lg = logging.getLogger("ruff_detailed")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    lg.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    return lg, log_path


def _ruff_cmd() -> list[str]:
    """Return the base command to invoke Ruff.

    Returns:
        list[str]: Command for subprocess call (either ['ruff'] or ['python', '-m', 'ruff']).
    """
    return ["ruff"] if shutil.which("ruff") else [sys.executable, "-m", "ruff"]


def run_ruff(files: list[str], read_only: bool) -> int:
    """Run Ruff once, echoing its output and returning the number of problem files.

    Args:
        files (list[str]): Paths to the Python files to lint.
        read_only (bool): If True, Ruff runs in check-only mode; otherwise, it
            applies autofixes where possible.

    Returns:
        int: The number of unique files that still contain violations after the
        Ruff run (zero when clean). The script exits with a code mirroring this.
    """
    # Determine project root for cleaner relative paths in per-file ignores
    root_dir = Path(_default_targets()[0]).expanduser().resolve()
    per_file_ignores_flags: list[str] = []

    for fp in files:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Add suppressions for libraries lacking type stubs
        if any(
            re.search(rf"^\s*(?:from|import)\s+{lib}\b", text, re.MULTILINE)
            for lib in _NO_STUB_LIBS
        ):
            try:
                rel = Path(fp).resolve().relative_to(root_dir).as_posix()
            except ValueError:
                rel = Path(fp).name

            for code in _SILENCED_CODES:
                per_file_ignores_flags += ["--per-file-ignores", f"{rel}:{code}"]

    # Assemble the Ruff CLI invocation
    cmd = [
        *_ruff_cmd(),
        "check",
        *_EFFECTIVE_RUFF_CLI_ARGS,     # ← single-source of truth
        *per_file_ignores_flags,
        *RUFF_ADDITIONAL_ARGS,
    ]
    if not read_only:
        cmd.append("--fix")

    # Set up detailed logging and execute Ruff
    detailed_logger, _ = _setup_detailed_logger(OUTPUT_FOLDER)
    proc = subprocess.run(
        [*cmd, *files],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    # Echo Ruff’s stdout/stderr to console and detailed log
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        detailed_logger.debug(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        detailed_logger.debug(proc.stderr)

    # Count distinct files that still have issues
    issue_lines = re.findall(r"^(.+?):\d+:\d+:", proc.stdout or "", flags=re.MULTILINE)
    return len(set(issue_lines)) if proc.returncode == 1 else 0

# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entry point for the script.

    Gathers Python files, filters out skipped paths, and runs Ruff on the resulting list.
    Prints summary output to the console and exits with an appropriate status code.

    Exit Codes:
        0: No issues found by Ruff.
        1: Ruff found violations in one or more files.
    """
    files = gather_python_files(_default_targets(), SKIP_PATHS)
    if not files:
        CONSOLE.warning("No Python files found – nothing to do.")
        return

    CONSOLE.info(
        "Running Ruff (%s) on %d files …",
        "check-only" if READ_ONLY else "check+fix",
        len(files),
    )
    failed = run_ruff(files, READ_ONLY)

    if failed == 0:
        CONSOLE.info("🎉  Ruff reports no outstanding issues.")
        sys.exit(0)
    CONSOLE.warning("⚠️  Ruff detected problems in %d file(s).", failed)
    sys.exit(1)


if __name__ == "__main__":
    main()
