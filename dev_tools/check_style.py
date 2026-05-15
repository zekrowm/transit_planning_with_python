#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check in-house style rules from CONTRIBUTING.md across repository scripts.

Walks a target directory of Python files and flags violations of the house
style rules defined in CONTRIBUTING.md.  Results are printed to the console
and written to a timestamped log file in LOG_DIR.

Checks performed (per-file):
    1.  no_utils_import   – No runtime imports from utils/ at module level.
    2.  config_section    – A CONFIGURATION section is present.
    3.  run_log_present   – Run-log sidecar machinery present for output-writing scripts.
    4.  raw_string_paths  – Raw-string literals (r"…") used for path/dir config variables.
    5.  dc_crs            – Washington DC CRS referenced when a CRS config variable is defined.
    6.  imperial_units    – Imperial units (feet/miles) referenced when metric distances appear.
    7.  notebook_guard    – `if __name__ == "__main__": main()` guard present.
    8.  main_function     – Top-level `def main()` function present.
    9.  logging_present   – `import logging` and at least one `logging.` call present.
    10. success_message   – A success/completion message via `logging` present.

Exit status:
    0 – all enabled checks passed for all files
    1 – one or more violations detected
    2 – unrecoverable error (e.g. SCRIPTS_ROOT directory not found)

Configuration constants are grouped in the CONFIGURATION section below.
"""

from __future__ import annotations

import ast
import datetime as _dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

# =============================================================================
# CONFIGURATION
# =============================================================================
# === BEGIN CONFIG ===

# Directory to audit — relative to cwd or absolute.
SCRIPTS_ROOT: str = r"scripts"

# Where timestamped audit logs are written.
LOG_DIR: str = r"logs"

PY_EXTENSIONS: tuple[str, ...] = (".py",)
SKIP_DIRS: tuple[str, ...] = (
    "__pycache__",
    ".git",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dev_tools",
    "tests",
)

# Set any check to False to disable it entirely.
CHECKS_ENABLED: dict[str, bool] = {
    "no_utils_import": True,
    "config_section": True,
    "run_log_present": True,
    "raw_string_paths": True,
    "dc_crs": True,
    "imperial_units": True,
    "notebook_guard": True,
    "main_function": True,
    "logging_present": True,
    "success_message": True,
}

# === END CONFIG ===

# =============================================================================
# TYPES
# =============================================================================


class Violation(NamedTuple):
    """A single style-check finding."""

    check: str
    line: int | None
    message: str


class FileResult(NamedTuple):
    """Outcome of all checks on one file."""

    path: Path
    violations: list[Violation]

    @property
    def passed(self) -> bool:
        """Return True if no violations were found."""
        return not self.violations


# =============================================================================
# HELPERS
# =============================================================================

# EPSG codes appropriate for the Washington DC area.
_DC_EPSG: frozenset[int] = frozenset({
    32618,  # WGS 84 / UTM zone 18N (meters)
    26918,  # NAD83 / UTM zone 18N (meters)
    2248,   # NAD83 / Maryland State Plane (feet)
    2804,   # NAD83 / Maryland State Plane (old)
    2893,   # NAD83(HARN) / Maryland
    4326,   # WGS84 geographic (universal baseline)
})

# Patterns that suggest a script writes file output to disk.
_OUTPUT_WRITE_PAT: re.Pattern[str] = re.compile(
    r"\b(?:to_excel|to_csv|to_file|ExcelWriter|write_text|to_parquet|to_shapefile)\b"
    r'|open\s*\([^)]+,\s*["\']w',
    re.IGNORECASE,
)


def _read_source(path: Path) -> str | None:
    """Return file source as UTF-8 text, or None on read error."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logging.warning("Cannot read %s: %s", path, exc)
        return None


def _extract_config_block(src: str) -> str | None:
    """Return text between # === BEGIN CONFIG === / # === END CONFIG === markers, or None."""
    m = re.search(
        r"# === BEGIN CONFIG ===\n(.*?)# === END CONFIG ===",
        src,
        re.DOTALL,
    )
    return m.group(1) if m else None


# =============================================================================
# CHECK FUNCTIONS
# Each returns a list of Violation — empty means the check passed.
# =============================================================================


def check_no_utils_import(src: str, path: Path) -> list[Violation]:
    """Flag any `import utils` or `from utils …` statement at module level."""
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "utils" or alias.name.startswith("utils."):
                    found.append(Violation(
                        check="no_utils_import",
                        line=node.lineno,
                        message=(
                            f"`import {alias.name}` detected — copy helpers from "
                            "utils/ into the script instead of importing at runtime"
                        ),
                    ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "utils" or module.startswith("utils."):
                names = ", ".join(a.name for a in node.names)
                found.append(Violation(
                    check="no_utils_import",
                    line=node.lineno,
                    message=(
                        f"`from {module} import {names}` detected — copy helpers "
                        "from utils/ into the script instead of importing at runtime"
                    ),
                ))
    return found


def check_config_section(src: str) -> list[Violation]:
    """Flag a missing or incomplete CONFIGURATION section."""
    has_header = bool(re.search(r"#[^\n]*\bCONFIGURATION\b", src, re.IGNORECASE))
    has_begin = "# === BEGIN CONFIG ===" in src
    has_end = "# === END CONFIG ===" in src

    if not has_header:
        return [Violation(
            check="config_section",
            line=None,
            message=(
                "No CONFIGURATION section found — add a config block near the top "
                "of the script with `# === BEGIN CONFIG ===` / `# === END CONFIG ===` markers"
            ),
        )]

    violations: list[Violation] = []
    if has_begin and not has_end:
        violations.append(Violation(
            check="config_section",
            line=None,
            message="Found `# === BEGIN CONFIG ===` but no matching `# === END CONFIG ===`",
        ))
    elif has_end and not has_begin:
        violations.append(Violation(
            check="config_section",
            line=None,
            message="Found `# === END CONFIG ===` but no matching `# === BEGIN CONFIG ===`",
        ))
    elif not has_begin:
        violations.append(Violation(
            check="config_section",
            line=None,
            message=(
                "CONFIGURATION section present but lacks BEGIN/END markers — "
                "add `# === BEGIN CONFIG ===` / `# === END CONFIG ===` so "
                "write_run_log can capture settings verbatim"
            ),
        ))
    return violations


def check_run_log_present(src: str) -> list[Violation]:
    """Flag output-writing scripts that lack run-log sidecar machinery."""
    if not _OUTPUT_WRITE_PAT.search(src):
        return []  # script doesn't appear to write files — check not applicable

    indicators = ["_runlog.txt", "write_run_log", "REQUIRE_RUN_LOG", "extract_config_block"]
    if not any(ind in src for ind in indicators):
        return [Violation(
            check="run_log_present",
            line=None,
            message=(
                "Script writes output file(s) but has no run-log machinery "
                "(_runlog.txt / write_run_log / REQUIRE_RUN_LOG / extract_config_block) — "
                "add a _runlog.txt sidecar per CONTRIBUTING.md"
            ),
        )]
    return []


def check_raw_string_paths(src: str) -> list[Violation]:
    """Flag path/dir/file config variables assigned to plain (non-raw) strings."""
    # Prefer to scan only the config block; fall back to the full source.
    search_in = _extract_config_block(src) or src

    # Matches uppercase variable names with PATH|DIR|FOLDER|FILE|ROOT, then an
    # assignment whose right-hand side starts with a plain (non-raw) quote.
    assignment_pat = re.compile(
        r'^[ \t]*([A-Z_]*(?:PATH|DIR|FOLDER|FILE|ROOT)[A-Z_]*)[ \t]*(?::[^\n=]+)?=[ \t]*'
        r'(?:Path\s*\(\s*)?(["\'])',
        re.MULTILINE,
    )

    found: list[Violation] = []
    for m in assignment_pat.finditer(search_in):
        var_name = m.group(1)
        # Walk back to start of line and check whether a raw prefix exists.
        line_start = search_in.rfind("\n", 0, m.start()) + 1
        line_end = search_in.find("\n", m.start())
        line = search_in[line_start:line_end if line_end != -1 else None]
        if not re.search(r'=[ \t]*(?:Path\s*\(\s*)?[rR]["\']', line):
            found.append(Violation(
                check="raw_string_paths",
                line=None,
                message=(
                    f'Config variable `{var_name}` uses a plain string for a path — '
                    r'use r"…" (raw string) to avoid backslash-escape issues on Windows'
                ),
            ))
    return found


def check_dc_crs(src: str) -> list[Violation]:
    """Flag CRS config variables whose EPSG codes don't match a DC projection."""
    # Match uppercase config names containing CRS, EPSG, PROJECTION, or SPATIAL_REF.
    crs_var_pat = re.compile(
        r'^[ \t]*([A-Z_]*(?:CRS|EPSG|PROJECTION|SPATIAL_REF)[A-Z_]*)[ \t]*(?::[^\n=]+)?=[ \t]*([^\n#]+)',
        re.MULTILINE,
    )

    found: list[Violation] = []
    for m in crs_var_pat.finditer(src):
        var_name = m.group(1)
        raw_value = m.group(2).strip().rstrip(",")

        # Skip type annotations, None, or empty assignments.
        if not raw_value or raw_value in {"None", "Optional[int]", "Optional[str]"}:
            continue

        epsg_numbers = [int(n) for n in re.findall(r'\b(\d{4,6})\b', raw_value)]
        for epsg in epsg_numbers:
            if epsg not in _DC_EPSG:
                found.append(Violation(
                    check="dc_crs",
                    line=None,
                    message=(
                        f"Config variable `{var_name}` uses EPSG:{epsg} — "
                        "default to a Washington DC CRS "
                        "(EPSG:32618, 26918, or 2248) per CONTRIBUTING.md; "
                        "add an inline comment if a different CRS is intentional"
                    ),
                ))
    return found


def check_imperial_units(src: str) -> list[Violation]:
    """Flag metric-only distance variables that lack an imperial counterpart."""
    metric_vars = re.findall(
        r'\b([A-Z_]*(?:METERS?|METRES?|_KM|_KILOMETERS?|_KILOMETRES?)[A-Z_]*)\b',
        src,
    )
    has_imperial = bool(re.search(
        r'\b(?:feet|foot|FEET|FOOT|_FT\b|MILES?|_MI\b)',
        src,
    ))

    if metric_vars and not has_imperial:
        preview = ", ".join(sorted({v for v in metric_vars})[:4])
        return [Violation(
            check="imperial_units",
            line=None,
            message=(
                f"Metric distance variable(s) found ({preview}) "
                "but no feet/miles reference detected — "
                "imperial units should be the default per CONTRIBUTING.md, "
                "with metric available as a secondary option"
            ),
        )]
    return []


def check_notebook_guard(src: str) -> list[Violation]:
    """Flag the absence of an `if __name__ == '__main__': main()` guard."""
    if not re.search(
        r'if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n[ \t]+main\s*\(',
        src,
    ):
        return [Violation(
            check="notebook_guard",
            line=None,
            message=(
                'Missing `if __name__ == "__main__": main()` guard — '
                "required so the script runs from both a Jupyter notebook and the CLI"
            ),
        )]
    return []


def check_main_function(src: str, path: Path) -> list[Violation]:
    """Flag the absence of a top-level `def main()` function."""
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return []
    return [Violation(
        check="main_function",
        line=None,
        message=(
            "No top-level `def main()` function — "
            "scripts must be modular with a clearly defined main() entry point"
        ),
    )]


def check_logging_present(src: str) -> list[Violation]:
    """Flag scripts that don't import and use the logging module."""
    violations: list[Violation] = []
    if not re.search(r'^import logging\b', src, re.MULTILINE):
        violations.append(Violation(
            check="logging_present",
            line=None,
            message=(
                "Missing `import logging` — "
                "prefer the logging module over print() for all diagnostics and status messages"
            ),
        ))
    elif not re.search(r'\blogging\.', src):
        violations.append(Violation(
            check="logging_present",
            line=None,
            message=(
                "`import logging` present but no `logging.` calls found — "
                "replace print() calls with logging.info() / logging.warning() etc."
            ),
        ))
    return violations


def check_success_message(src: str) -> list[Violation]:
    """Flag scripts that don't log a success or completion message."""
    if not re.search(
        r'logging\.\w+\s*\([^)]*(?:success|complet|finish|done)',
        src,
        re.IGNORECASE,
    ):
        return [Violation(
            check="success_message",
            line=None,
            message=(
                "No success/completion log message detected — "
                'add e.g. logging.info("Script completed successfully.") at the end of main()'
            ),
        )]
    return []


# =============================================================================
# ORCHESTRATION
# =============================================================================


def audit_file(path: Path) -> FileResult:
    """Run all enabled checks against a single Python file."""
    src = _read_source(path)
    if src is None:
        return FileResult(path=path, violations=[
            Violation(check="io", line=None, message="File could not be read")
        ])

    enabled = CHECKS_ENABLED
    violations: list[Violation] = []

    if enabled.get("no_utils_import", True):
        violations.extend(check_no_utils_import(src, path))
    if enabled.get("config_section", True):
        violations.extend(check_config_section(src))
    if enabled.get("run_log_present", True):
        violations.extend(check_run_log_present(src))
    if enabled.get("raw_string_paths", True):
        violations.extend(check_raw_string_paths(src))
    if enabled.get("dc_crs", True):
        violations.extend(check_dc_crs(src))
    if enabled.get("imperial_units", True):
        violations.extend(check_imperial_units(src))
    if enabled.get("notebook_guard", True):
        violations.extend(check_notebook_guard(src))
    if enabled.get("main_function", True):
        violations.extend(check_main_function(src, path))
    if enabled.get("logging_present", True):
        violations.extend(check_logging_present(src))
    if enabled.get("success_message", True):
        violations.extend(check_success_message(src))

    return FileResult(path=path, violations=violations)


def collect_python_files(root: Path) -> list[Path]:
    """Walk *root* and return all Python files, skipping SKIP_DIRS."""
    found: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        for fname in sorted(files):
            if any(fname.endswith(ext) for ext in PY_EXTENSIONS):
                found.append(Path(dirpath) / fname)
    return found


# =============================================================================
# LOGGING SETUP
# =============================================================================


def _setup_logging() -> Path:
    """Configure console + file logging and return the log file path."""
    log_root = Path(LOG_DIR)
    log_root.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_root / f"style_check_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Audit Python scripts for in-house style compliance."""
    log_file = _setup_logging()
    logging.info("Style check log → %s", log_file)

    root = Path(SCRIPTS_ROOT).resolve()
    if not root.is_dir():
        logging.error("SCRIPTS_ROOT does not exist or is not a directory: %s", root)
        sys.exit(2)

    enabled_names = [k for k, v in CHECKS_ENABLED.items() if v]
    logging.info("Auditing: %s", root)
    logging.info("Enabled checks (%d): %s", len(enabled_names), ", ".join(enabled_names))

    py_files = collect_python_files(root)
    if not py_files:
        logging.warning("No Python files found under %s", root)
        sys.exit(0)

    logging.info("Files to audit: %d", len(py_files))

    results: list[FileResult] = [audit_file(p) for p in py_files]

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    logging.info("─" * 72)
    logging.info(
        "RESULTS: %d passed, %d with violations (of %d total)",
        len(passed),
        len(failed),
        len(results),
    )
    logging.info("─" * 72)

    for result in failed:
        try:
            rel = result.path.relative_to(root)
        except ValueError:
            rel = result.path
        logging.warning("✗ %s  (%d violation(s))", rel, len(result.violations))
        for v in result.violations:
            loc = f"  line {v.line}" if v.line else ""
            logging.warning("    [%s]%s  %s", v.check, loc, v.message)

    if passed:
        logging.info("─" * 72)
        logging.info("✓ Files with no violations:")
        for result in passed:
            try:
                rel = result.path.relative_to(root)
            except ValueError:
                rel = result.path
            logging.info("    %s", rel)

    logging.info("─" * 72)
    if failed:
        logging.error("Style audit complete — %d file(s) have violations.", len(failed))
        sys.exit(1)
    else:
        logging.info("Style audit complete — all %d files passed.", len(results))


if __name__ == "__main__":
    main()
