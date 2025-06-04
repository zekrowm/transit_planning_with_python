"""
Script Name:
    script_layout_checker.py

Purpose:
    Audit-only tool that enforces the repository’s file-layout convention.

Inputs:
    None. Configuration is done via constants at the top of the script
    (e.g., ROOTS, SKIP, LOG_FOLDER).

Outputs:
    - A timestamped log file listing any layout violations per script.
    - Exit code 0 if all files are compliant, 1 if any issues are found.

Dependencies:
    - os, sys, re, ast, datetime, pathlib
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ======================================================================
# CONFIGURATION
# ======================================================================

ROOTS: List[str] = [r"C:\Path\To\Your\Repo"]  # folders or files to scan
SKIP: List[str] = [r".venv", r"__pycache__"]  # substrings in path to ignore
LOG_FOLDER = r"C:\tmp\structure_logs"  # where to write the log file

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

FENCE = "# " + "=" * 98  # canonical 100-char fence
FENCE_RX = re.compile(r"^#\s*={8,}\s*$")  # any 8+ "=" (detect misuse)
MAJOR_NAMES = ("CONFIGURATION", "FUNCTIONS", "MAIN")
TOLERANCE = {"CONFIGURATION": 3, "FUNCTIONS": 6, "MAIN": 3}
REQ_DOC_HDRS = [
    "script name:",
    "purpose:",
    "inputs:",
    "outputs:",
    "dependencies:",
]

# ======================================================================
# FUNCTIONS
# ======================================================================


def gather_py_files(roots: List[str], skip: List[str]) -> List[Path]:
    """
    Recursively collect .py files under *roots*, skipping any path containing
    a token in *skip*.  Returns a sorted list of Path objects.
    """
    out: List[Path] = []
    skip_low = [s.lower() for s in skip]
    for p in roots:
        path = Path(p).expanduser()
        if not path.exists():
            print(f"⚠  Path not found: {path}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix.lower() == ".py":
            out.append(path)
        else:
            for f in path.rglob("*.py"):
                if any(tok in str(f).lower() for tok in skip_low):
                    continue
                out.append(f)
    return sorted(out, key=lambda x: x.as_posix())


def top_level_indices(lines: List[str], prefixes: Tuple[str, ...]) -> List[int]:
    """
    Return line indices where a line (without leading whitespace) starts
    with any of the given *prefixes*.  Indented lines are ignored.
    """
    return [i for i, ln in enumerate(lines) if ln.startswith(prefixes)]


def find_header_blocks(lines: List[str], name: str) -> List[Tuple[int, int]]:
    """
    Return a list of (start, end) slices for every properly formed three-line
    header named *name*.  Each slice covers lines[start:end] (3 lines).
    """
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(lines) - 2:
        if (
            lines[i].rstrip("\n") == FENCE
            and lines[i + 1].strip().lower() == f"# {name.lower()}"
            and lines[i + 2].rstrip("\n") == FENCE
        ):
            blocks.append((i, i + 3))
            i += 3
        else:
            i += 1
    return blocks


def first_module_doc(lines: List[str]) -> str | None:
    """
    Return the first triple-quoted module docstring if it is the very first
    statement in the file.  Otherwise return None.
    """
    try:
        tree = ast.parse("".join(lines))
    except SyntaxError:
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    # Confirm the docstring is top-level (i.e., first node in the module)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        return doc
    return None


def check_configuration_enclosed(lines: List[str], header_start: int) -> bool:
    """
    Return True if, from the start of the file up to header_start (exclusive),
    the count of '{' exceeds the count of '}', indicating the header is
    inside a '{'… region.  This flags CONFIGURATION being encapsulated.
    """
    balance = 0
    for ln in lines[:header_start]:
        balance += ln.count("{")
        balance -= ln.count("}")
    return balance > 0


# ----------------------------------------------------------------------
# CORE ANALYSIS
# ----------------------------------------------------------------------


def analyse_file(py: Path) -> List[str]:
    """
    Analyse a single Python file for:
      1. Major section headers (presence, style, location, no '{' encapsulation)
      2. Minor misuse of '=' fences
      3. Presence of def main()
      4. Module-level docstring structure
    Return a list of issue strings found (empty if compliant).
    """
    try:
        lines = py.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        return [f"ERROR: cannot read file ({exc})"]

    issues: List[str] = []

    # Determine anchor positions
    import_idxs = top_level_indices(lines, ("import ", "from "))
    import_idx = max(import_idxs) if import_idxs else None
    def_idxs = top_level_indices(lines, ("def ",))
    first_def = min(def_idxs) if def_idxs else None
    main_def = next((i for i in def_idxs if lines[i].startswith("def main")), None)

    # 1. Major headers
    for name in MAJOR_NAMES:
        blocks = find_header_blocks(lines, name)
        if not blocks:
            issues.append(f"{name}: header missing")
            continue

        # Determine canonical target line for this header
        target: int | None = None
        if name == "CONFIGURATION" and import_idx is not None:
            target = import_idx + 1
        elif name == "FUNCTIONS" and first_def is not None:
            target = first_def - 3
        elif name == "MAIN" and main_def is not None:
            target = main_def - 3

        # Pick the block closest to target (if anchor exists), else just take first
        best = blocks[0]
        if target is not None:
            best = min(blocks, key=lambda b: abs(b[0] - target))

        # Check location tolerance if we have a target
        if target is not None:
            if abs(best[0] - target) > TOLERANCE[name]:
                issues.append(f"{name}: header mispositioned (line {best[0]+1})")

        # Check correct fence length/style
        start = best[0]
        if lines[start].rstrip("\n") != FENCE or lines[start + 2].rstrip("\n") != FENCE:
            issues.append(f"{name}: fence length != 100 chars or style")

        # Additional: CONFIGURATION block must not be inside an open '{'
        if name == "CONFIGURATION":
            if check_configuration_enclosed(lines, start):
                issues.append("CONFIGURATION: header is enclosed inside '{'")

    # 2. Minor headers using "=" misuse
    for i, ln in enumerate(lines):
        if FENCE_RX.match(ln):
            # If this fence is not part of a major-block, flag it
            if not any(
                start <= i < end
                for name in MAJOR_NAMES
                for start, end in find_header_blocks(lines, name)
            ):
                issues.append(f"Minor header uses '=' fence at line {i+1}")

    # 3. def main() presence
    if main_def is None:
        issues.append("def main() not found")

    # 4. Module docstring
    doc = first_module_doc(lines)
    if doc is None:
        issues.append("Module docstring missing or not at top")
    else:
        # Verify required headings appear in order
        want = REQ_DOC_HDRS.copy()
        idx = 0
        for ln in (l.strip().lower() for l in doc.splitlines() if l.strip()):
            if idx < len(want) and ln.startswith(want[idx]):
                idx += 1
        if idx < len(want):
            issues.append("Module docstring missing required headings or wrong order")

    return issues


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:
    files = gather_py_files(ROOTS, SKIP)
    if not files:
        print("No Python files found; nothing to audit.")
        sys.exit(0)

    log_path = Path(LOG_FOLDER).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    logfile = log_path / f"struct_audit_{datetime.now():%Y%m%d_%H%M%S}.log"

    total_files_with_issues = 0
    total_issues = 0

    with logfile.open("w", encoding="utf-8") as log:
        for py in files:
            issues = analyse_file(py)
            if issues:
                total_files_with_issues += 1
                total_issues += len(issues)
                log.write(f"{py}:\n")
                for item in issues:
                    log.write(f"  • {item}\n")
                log.write("\n")

    if total_issues == 0:
        print("✅  Structure-police: all files comply with house style.")
        sys.exit(0)
    else:
        print(
            f"❌  Structure-police: {total_issues} issue(s) in "
            f"{total_files_with_issues} file(s). See → {logfile}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
