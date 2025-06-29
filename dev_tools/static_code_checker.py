#!/usr/bin/env python
"""Batch-run mypy, vulture, pylint, and pydocstyle with CI-friendly logging.

The script prioritises *determinism in automation* while remaining ergonomic
for ad-hoc local use or interactive notebooks.

Typical usage
-------------
# Lint everything under the repo root (auto-detected)
python run_static_checks.py

# From a notebook cell
from run_static_checks import main
main(read_only=True)      # returns 0 on success, ≥1 on failure
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Final, Iterable, List, Sequence

# =============================================================================
# CONFIGURATION
# =============================================================================

#: Default folders/files to scan when none are supplied via the CLI.
DEFAULT_TARGETS: Final[list[str]] = ["."]
#: Directories that should never be scanned (rel-paths ok).
DEFAULT_EXCLUDES: Final[list[str]] = ["tests", "venv", ".venv", ".mypy_cache"]
#: Default output directory for detailed logs.
DEFAULT_OUT_DIR: Final[str] = os.getenv("STATIC_LINT_OUTDIR", ".artifacts")
#: Roughly “info” in human mode, “warning” in CI.
DEFAULT_LOG_LEVEL: Final[int] = (
    logging.WARNING if os.getenv("CI") else logging.INFO
)

# Tips
# ---------------------------------
# Fully-qualified *absolute* paths work fine here; a few examples:
#
#   DEFAULT_TARGETS  = [r"C:\Users\alice\dev\my_repo",  # Windows
#                       "./scripts"]                    # relative OK
#
#   DEFAULT_EXCLUDES = [r"C:\Users\alice\dev\my_repo\tests",
#                       "build"]                        # relative OK
#
#   DEFAULT_OUT_DIR  = r"C:\Users\alice\dev\my_repo\logs"       # writes *.log files here
#

# -----------------------------------------------------------------------------
# FALL-BACK CLI OPTIONS
# -----------------------------------------------------------------------------

# These are **only** injected when no project-level toml config is discovered.
MYPY_ARGS_FALLBACK: Final[list[str]] = [
    "--python-version", "3.10",
    "--pretty",
    "--show-error-codes",
    "--explicit-package-bases",
    "--namespace-packages",
    "--no-strict-optional",      # strict_optional = false
    "--ignore-missing-imports",  # overridable stanzas aren’t possible via CLI
    "--allow-untyped-calls",
]

VULTURE_MIN_CONFIDENCE_FALLBACK: Final[int] = 70  # matches [tool.vulture]

PYLINT_ARGS_FALLBACK: Final[list[str]] = [
    "--errors-only",
    "--output-format=text",
    "--ignored-modules=arcpy",
    "--disable=duplicate-code",
    "-j", "0",                   # jobs = 0  (all CPUs)
]

PYDOCSTYLE_ARGS_FALLBACK: Final[list[str]] = [
    "--convention=google",
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def _in_notebook() -> bool:
    """Return ``True`` if executed inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython  # pylint: disable=import-error
        return "IPKernelApp" in get_ipython().config
    except Exception:  # pragma: no cover
        return False


def _find_file_upwards(names: Sequence[str]) -> Path | None:
    """Return the first file named *names* found by walking parents upward."""
    here = Path.cwd()
    for parent in [here, *here.parents]:
        for name in names:
            fp = parent / name
            if fp.is_file():
                return fp
    return None


def _detailed_logger(prefix: str, out_dir: str = DEFAULT_OUT_DIR) -> logging.Logger:
    """Return a logger that writes 100 % of output to *{out_dir}/{prefix}_ts.log*."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(out_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"{prefix}_{ts}.log"

    lg = logging.getLogger(prefix)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    lg.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    return lg


def _gather_py_files(
    targets: Iterable[str], excludes: Iterable[str]
) -> list[str]:
    """Return a *deduped* list of Python files under *targets* excluding *excludes*."""
    out: list[str] = []
    excl_norm = {Path(p).resolve().as_posix().lower() for p in excludes}

    def is_skipped(pth: Path) -> bool:
        norm = pth.resolve().as_posix().lower()
        return any(norm == x or norm.startswith(f"{x}/") for x in excl_norm)

    for entry in targets:
        p = Path(entry).expanduser()
        if not p.exists():
            logging.warning("Path not found – skipping: %s", p)
            continue
        if p.is_dir():
            for root, _, files in os.walk(p):
                root_p = Path(root)
                if is_skipped(root_p):
                    continue
                for f in files:
                    if f.endswith(".py"):
                        fp = root_p / f
                        if not is_skipped(fp):
                            out.append(fp.as_posix())
        elif p.suffix.lower() == ".py" and not is_skipped(p):
            out.append(p.as_posix())
    return list(dict.fromkeys(out))


def _tool(
    name: str,
    module: str,
    cli_args: list[str],
    files: list[str],
    parser: re.Pattern[str] | None = None,
) -> int:
    """Run *module* once per *files*, log everything, return “files with issues”.

    The logic mirrors all four tools so we keep it DRY.
    """
    log = _detailed_logger(f"{name}_log")
    not_found_msg = f"python -m {module} not found – aborting {name} pass."
    issue_count = 0

    for idx, py in enumerate(files, 1):
        log.info("\n%s\nFILE %d/%d → %s\n%s", "=" * 80, idx, len(files), py, "=" * 80)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module, *cli_args, py],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            logging.error(not_found_msg)
            return len(files)

        out, err = proc.stdout, proc.stderr
        log.info(out or "")
        if err:
            log.info("\n--- stderr ---\n%s", err)

        if parser and any(parser.search(line) for line in (out or "").splitlines()):
            issue_count += 1
        elif not parser and (out or "").strip():  # pydocstyle simple heuristic
            issue_count += 1
    return issue_count

# -----------------------------------------------------------------------------
# Per-tool wrappers (auto-detect external config → else fall-back args)
# -----------------------------------------------------------------------------

_MYPY_OUT_RE = re.compile(r": error:")  # minimal but works
_PYLINT_RE = re.compile(r":\s*[EF]\d{4}:")
_VULTURE_RE = re.compile(r"^\S+:\d+:.+\(\d+% confidence\)$", re.M)

def run_mypy(files: list[str]) -> int:
    cfg_exists = bool(_find_file_upwards(["mypy.ini", "pyproject.toml"]))
    args = [] if cfg_exists else MYPY_ARGS_FALLBACK
    return _tool("mypy", "mypy", ["--show-error-codes", "--no-color-output", *args], files, _MYPY_OUT_RE)

def run_vulture(files: list[str]) -> int:
    cfg_exists = bool(_find_file_upwards([".vulture_ignore", "pyproject.toml"]))
    min_conf = [] if cfg_exists else ["--min-confidence", str(VULTURE_MIN_CONFIDENCE_FALLBACK)]
    return _tool("vulture", "vulture", [*min_conf], files, _VULTURE_RE)

def run_pylint(files: list[str]) -> int:
    cfg_exists = bool(_find_file_upwards([".pylintrc", "pyproject.toml"]))
    args = [] if cfg_exists else PYLINT_ARGS_FALLBACK
    return _tool("pylint", "pylint", args, files, _PYLINT_RE)

def run_pydocstyle(files: list[str]) -> int:
    cfg_exists = bool(_find_file_upwards([".pydocstyle", "pyproject.toml"]))
    args = [] if cfg_exists else PYDOCSTYLE_ARGS_FALLBACK
    # parser = None → treat any non-empty stdout as issues
    return _tool("pydocstyle", "pydocstyle", args, files)


# -----------------------------------------------------------------------------
# CLI / orchestration
# -----------------------------------------------------------------------------

def _parse_cli(argv: Sequence[str]) -> argparse.Namespace:
    """Return parsed CLI arguments."""
    ap = argparse.ArgumentParser(
        prog="run_static_checks",
        description="Batch-run mypy, vulture, pylint, and pydocstyle.",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Files or folders to scan (default: repo root).",
    )
    ap.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        help="Extra paths to exclude (repeatable).",
    )
    ap.add_argument(
        "--read-only",
        action="store_true",
        help="Reserved for interface parity with Ruff script (no effect here).",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for detailed logs (default: {DEFAULT_OUT_DIR}).",
    )
    return ap.parse_args(argv)

# =============================================================================
# MAIN
# =============================================================================

def main(argv: Sequence[str] | None = None, *, read_only: bool | None = None) -> int:
    """Run the full tool-chain.  Returns *files-with-issues* aggregate count."""
    args = _parse_cli(argv or sys.argv[1:])
    # Optional positional *read_only* allows notebook callers to override flag.
    args.read_only = bool(read_only) if read_only is not None else args.read_only

    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("Collecting Python files …")
    files = _gather_py_files(args.paths, [*DEFAULT_EXCLUDES, *args.exclude])
    if not files:
        logging.warning("No Python files found – nothing to do.")
        return 0

    overall = 0
    for runner, label in (
        (run_mypy, "mypy"),
        (run_vulture, "vulture"),
        (run_pylint, "pylint"),
        (run_pydocstyle, "pydocstyle"),
    ):
        logging.info("=== Running %s ===", label)
        failed = runner(files)
        overall += failed
        msg = "pass ✅" if failed == 0 else f"❌  {failed} file(s) with issues"
        logging.info("%s → %s", label, msg)

    summary = (
        "All checks passed 🎉" if overall == 0 else f"Static analysis found issues in {overall} file(s)"
    )
    logging.info("=" * 79 + "\n%s", summary)
    return overall


if __name__ == "__main__":
    rc = main()
    # Exit unless we are in a notebook; kernels hate sys.exit
    if not _in_notebook():
        sys.exit(0 if rc == 0 else 1)
