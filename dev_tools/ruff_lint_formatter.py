"""
Runs Ruff on Python source files, exporting detailed logs (single call).

* Keeps the familiar “edit-constants” workflow.
* Uses ONE Ruff invocation (faster).
* Disables ANSI colours for clean logs/CLI.
* Prints a one-line per-rule summary (e.g. D212:3, ANN001:2).
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

FILES_OR_FOLDERS: list[str] = []   # CI: repo root fallback, replace with local folder path if desired

SKIP_PATHS: List[str] = ["tests"]  # e.g. ["venv", "build", "tests"]

OUTPUT_FOLDER: str = ".artifacts"  # relative path is fine, replace with local folder path if desired
READ_ONLY: bool = True  # False → ruff --fix
RUFF_ADDITIONAL_ARGS: List[str] = []  # e.g. ["--extend-exclude", "migrations"]

# Allow explicit file list from the command line (e.g. CI “changed files” step)
if len(sys.argv) > 1:
    FILES_OR_FOLDERS[:] = sys.argv[1:]

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

LOG_LEVEL = logging.INFO
RUFF_CLI_ARGS: list[str] = [
    "--line-length",
    "100",
    "--target-version",
    "py310",
    # "--color", "never",                            # no ANSI escapes
    "--select",
    "I,F,D,ANN,TCH",
    "--fixable",
    "F401,D,I,TC003",
    "--ignore",
    "ANN401",
#    "--pydocstyle-convention", "google",
]

# ── new: fine-grained suppressions for libs without stubs ─────
_NO_STUB_LIBS = {"geopandas", "arcpy"}
_SILENCED_CODES = ("ANN", "TCH")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONSOLE = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS
# =============================================================================

def _default_targets() -> list[str]:
    if FILES_OR_FOLDERS:
        return FILES_OR_FOLDERS
    # fallback: repo root (folder above this script)
    return [Path(__file__).resolve().parents[1].as_posix()]


def _is_skipped(p: Path, skip: list[str]) -> bool:
    norm = p.resolve().as_posix().lower()
    for s in skip:
        tgt = Path(s).expanduser().resolve().as_posix().lower()
        if norm == tgt or norm.startswith(tgt + "/"):
            return True
    return False


def gather_python_files(targets: list[str], skip: list[str]) -> list[str]:
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
    """Return ['ruff'] if on PATH else [python, -m, ruff]."""
    return ["ruff"] if shutil.which("ruff") else [sys.executable, "-m", "ruff"]


def run_ruff(files: list[str], read_only: bool) -> int:
    """Run Ruff once, echoing its output and returning the number of problem files."""
    # Determine project root for drive‐colon–free relative paths
    root_dir = Path(_default_targets()[0]).expanduser().resolve()
    per_file_ignores_flags: list[str] = []

    for fp in files:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # If this file imports a library without stubs, queue one flag per code
        if any(
            re.search(rf"^\s*(?:from|import)\s+{lib}\b", text, re.MULTILINE)
            for lib in _NO_STUB_LIBS
        ):
            try:
                rel = Path(fp).resolve().relative_to(root_dir).as_posix()
            except ValueError:
                rel = Path(fp).name

            for code in _SILENCED_CODES:
                per_file_ignores_flags += [
                    "--per-file-ignores",
                    f"{rel}:{code}",
                ]

    # Build the single Ruff invocation
    cmd = [
        *_ruff_cmd(),
        "check",
        *RUFF_CLI_ARGS,
        *per_file_ignores_flags,
        *RUFF_ADDITIONAL_ARGS,
    ]
    if not read_only:
        cmd.append("--fix")

    # Set up detailed logging and run Ruff
    detailed_logger, _ = _setup_detailed_logger(OUTPUT_FOLDER)
    proc = subprocess.run(
        [*cmd, *files],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    # Echo Ruff’s stdout/stderr
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        detailed_logger.debug(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        detailed_logger.debug(proc.stderr)

    # Count distinct files with issues
    issue_lines = re.findall(r"^(.+?):\d+:\d+:", proc.stdout or "", flags=re.MULTILINE)
    return len(set(issue_lines)) if proc.returncode == 1 else 0

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
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
