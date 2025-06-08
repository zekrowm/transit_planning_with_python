"""
Audits function definitions across a codebase for consistency with canonical sources.

Compares top-level functions in a specified canonical file or folder against matching
functions found elsewhere in the codebase. Uses AST comparison to detect semantic
differences. Intended to help maintain reusable helper function consistency across
the project.

Inputs:
    - CANONICAL_PATH: Path to a canonical Python file or folder of canonical files.
    - SEARCH_ROOT: Root directory of the codebase to audit.
    - LOG_DIR: Directory where audit logs will be written.
    - PY_EXTENSIONS: File suffixes to treat as Python scripts.
    - IGNORE_PRIVATE: If True, skips functions whose names begin with '_'.

Outputs:
    - A timestamped log file listing all matches and mismatches.
    - Exit code 0 if all matches are consistent, 1 if mismatches found, 2 on error.
"""

import ast
import datetime as _dt
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

# Either a .py file *or* a directory that holds the canonical file set
CANONICAL_PATH = "helpers"  # ex: "helpers.py" or "lib/helpers"

SEARCH_ROOT = "."  # repo root to audit
LOG_DIR = "./logs"  # where audit logs are written
PY_EXTENSIONS = (".py",)  # recognised source suffixes
IGNORE_PRIVATE = True  # skip functions whose names start with "_"

# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# INTERNAL UTILITIES
# -----------------------------------------------------------------------------


def _normalise(node: ast.AST) -> ast.AST:
    """Strip line/column metadata for stable comparison."""
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return node


def _extract_functions(path: Path) -> Dict[str, ast.AST]:
    """
    Return a mapping of {func_name: normalised_ast} for *top-level* functions
    in *path*.  Raises RuntimeError on read/parse failure.
    """
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise RuntimeError(f"Cannot read {path}: {err}") from err

    try:
        mod = ast.parse(src, filename=str(path))
    except SyntaxError as err:
        raise RuntimeError(f"Syntax error in {path}: {err}") from err

    funcs = {}
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            if IGNORE_PRIVATE and node.name.startswith("_"):
                continue
            funcs[node.name] = _normalise(node)
    return funcs


def _compare_ast(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a, include_attributes=False) == ast.dump(
        b, include_attributes=False
    )


class AuditResult(NamedTuple):
    identical: List[Path]
    mismatched: List[Path]


# -----------------------------------------------------------------------------
# CANONICAL FUNCTION HARVEST
# -----------------------------------------------------------------------------


def _collect_canonical_funcs(
    source: Path,
) -> tuple[Dict[str, ast.AST], Set[Path]]:
    """
    Harvest canonical functions from *source* (file or directory).

    Returns
    -------
    funcs : dict
        {function_name: canonical_ast}
    files : set(Path)
        Paths of all canonical files (used to exclude them from the audit)
    """
    funcs: Dict[str, ast.AST] = {}
    canonical_files: Set[Path] = set()

    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = [
            p for p in source.iterdir() if p.is_file() and p.suffix in PY_EXTENSIONS
        ]
    else:
        raise RuntimeError(f"{source} is neither a file nor a directory.")

    for path in paths:
        canonical_files.add(path.resolve())
        try:
            fns = _extract_functions(path)
        except RuntimeError as err:
            logging.warning("%s", err)
            continue

        for name, ast_node in fns.items():
            if name in funcs:
                logging.warning(
                    "Duplicate canonical definition of %s: keeping first (%s), "
                    "ignoring %s",
                    name,
                    funcs[name].__dict__.get("__file__", "<unknown>"),  # type: ignore[attr-defined]
                    path,
                )
                continue
            ast_node.__dict__["__file__"] = str(path)  # annotate for later logs
            funcs[name] = ast_node

    return funcs, canonical_files


# -----------------------------------------------------------------------------
# AUDIT CORE
# -----------------------------------------------------------------------------


def _audit_single_function(
    name: str,
    canon_ast: ast.AST,
    skip_paths: Set[Path],
) -> AuditResult:
    """Compare *name* in every module under SEARCH_ROOT (recursive)."""
    identical, mismatched = [], []

    for root, _, files in os.walk(Path(SEARCH_ROOT)):
        for fname in files:
            if not fname.endswith(PY_EXTENSIONS):
                continue
            path = Path(root, fname).resolve()
            if path in skip_paths:
                continue

            try:
                funcs = _extract_functions(path)
            except RuntimeError as err:
                logging.warning("%s", err)
                continue

            if name not in funcs:
                continue

            if _compare_ast(canon_ast, funcs[name]):
                identical.append(path)
            else:
                mismatched.append(path)

    return AuditResult(identical=identical, mismatched=mismatched)


# -----------------------------------------------------------------------------
# LOGGING & MAINFLOW
# -----------------------------------------------------------------------------


def _setup_logging() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOG_DIR, f"helpers_audit_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Audit log → %s", log_file)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    _setup_logging()
    source = Path(CANONICAL_PATH).resolve()

    try:
        canonical_funcs, canonical_files = _collect_canonical_funcs(source)
    except RuntimeError as err:
        logging.error("%s", err)
        sys.exit(2)

    if not canonical_funcs:
        logging.error("No canonical functions discovered in %s", source)
        sys.exit(2)

    logging.info("Canonical source: %s", source)
    logging.info("Total canonical functions: %d", len(canonical_funcs))

    exit_code = 0
    for func_name, canon_ast in canonical_funcs.items():
        logging.info("── auditing function: %s ──", func_name)
        result = _audit_single_function(func_name, canon_ast, canonical_files)

        if result.identical:
            logging.info("  ✓ %d identical copy(ies)", len(result.identical))
            for p in result.identical:
                logging.info("      %s", p)

        if result.mismatched:
            logging.warning("  ✗ %d mismatching copy(ies)", len(result.mismatched))
            for p in result.mismatched:
                logging.warning("      %s", p)
            exit_code = 1

        if not result.identical and not result.mismatched:
            logging.info("  ∅ No duplicates found outside canonical set.")

    if exit_code:
        logging.error("Audit finished – discrepancies detected.")
    else:
        logging.info("Audit finished – all duplicates match the canonical code.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
