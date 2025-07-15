#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and (optionally) repair helper‑function consistency across a codebase.

The script compares top‑level functions in a canonical source file/folder
against occurrences elsewhere in the repository.  If `AUTO_FIX` is enabled,
mismatching definitions are automatically overwritten with the canonical
implementation (a `.bak` backup is made the first time each file is edited).

Exit status:
    0 – all copies match the canonical code
    1 – mismatches detected (and, if `AUTO_FIX` is False, left unchanged)
    2 – unrecoverable error (e.g. unreadable canonical source)

Configuration constants are grouped in the *CONFIGURATION* section below.
"""

from __future__ import annotations
import ast
import copy
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
CANONICAL_PATH: str = "helpers"  # e.g. "helpers.py" or "lib/helpers"

SEARCH_ROOT: str = "."  # repo root to audit
LOG_DIR: str = "./logs"  # where audit logs are written
PY_EXTENSIONS: tuple[str, ...] = (".py",)  # recognised source suffixes
IGNORE_PRIVATE: bool = True  # skip functions whose names start with "_"

# If True, mismatching definitions are *overwritten* with the canonical version.
# A per‑file backup with suffix ".bak" is created on the first edit.
AUTO_FIX: bool = False

# -----------------------------------------------------------------------------
# INTERNAL TYPES
# -----------------------------------------------------------------------------

class CanonicalFunc(NamedTuple):
    """Container holding canonical function data."""

    norm_ast: ast.AST  # normalised AST (line/col metadata stripped)
    source: str  # exact source text of the definition


class AuditResult(NamedTuple):
    """Outcome of a single canonical function audit."""

    identical: List[Path]
    mismatched: List[Path]

# =============================================================================
# FUNCTIONS
# =============================================================================

def _normalise(node: ast.AST) -> ast.AST:
    """Return *node* with all positional attributes zeroed for stable comparison."""
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, attr):
                setattr(n, attr, None)
    return node


def _compare_ast(a: ast.AST, b: ast.AST) -> bool:
    """True if *a* and *b* are semantically identical ASTs (ignoring positions)."""
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def _extract_functions(path: Path) -> Dict[str, CanonicalFunc]:
    """Return mapping {function_name: CanonicalFunc} for top‑level defs in *path*."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise RuntimeError(f"Cannot read {path}: {err}") from err

    try:
        mod = ast.parse(src, filename=str(path))
    except SyntaxError as err:
        raise RuntimeError(f"Syntax error in {path}: {err}") from err

    funcs: Dict[str, CanonicalFunc] = {}
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            if IGNORE_PRIVATE and node.name.startswith("_"):
                continue
            funcs[node.name] = CanonicalFunc(
                norm_ast=_normalise(copy.deepcopy(node)),
                source=ast.get_source_segment(src, node) or "",
            )
    return funcs


def _collect_canonical_funcs(
    source: Path,
) -> tuple[Dict[str, CanonicalFunc], Set[Path]]:
    """Harvest canonical functions from *source* (file or directory)."""
    funcs: Dict[str, CanonicalFunc] = {}
    canonical_files: Set[Path] = set()

    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = [p for p in source.iterdir() if p.is_file() and p.suffix in PY_EXTENSIONS]
    else:
        raise RuntimeError(f"{source} is neither a file nor a directory.")

    for path in paths:
        canonical_files.add(path.resolve())
        try:
            for name, cn in _extract_functions(path).items():
                if name in funcs:
                    logging.warning(
                        "Duplicate canonical definition of %s: keeping first (%s), ignoring %s",
                        name,
                        funcs[name].norm_ast.__dict__.get("__file__", "<unknown>"),  # type: ignore[attr-defined]
                        path,
                    )
                    continue
                cn.norm_ast.__dict__["__file__"] = str(path)  # type: ignore[attr-defined]
                funcs[name] = cn
        except RuntimeError as err:
            logging.warning("%s", err)

    return funcs, canonical_files


def _replace_function_in_file(path: Path, func_name: str, new_src: str) -> None:
    """Replace *func_name* in *path* with *new_src*; create a `.bak` backup once."""
    text = path.read_text(encoding="utf-8")
    mod = ast.parse(text, filename=str(path))
    target = next(
        (n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    if target is None:
        raise RuntimeError(f"{func_name} not found in {path}")

    lines = text.splitlines(keepends=True)
    start, end = target.lineno - 1, target.end_lineno  # slice [start:end)
    if not new_src.endswith("\n"):
        new_src += "\n"
    new_lines = [ln if ln.endswith("\n") else ln + "\n" for ln in new_src.splitlines()]

    # One‑time backup per file
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")

    lines[start:end] = new_lines
    path.write_text("".join(lines), encoding="utf-8")


def _audit_single_function(
    name: str,
    canon_ast: ast.AST,
    canon_src: str,
    skip_paths: Set[Path],
) -> AuditResult:
    """Compare canonical *name* against all other occurrences in the repository."""
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

            fn_info = funcs[name]
            if _compare_ast(canon_ast, fn_info.norm_ast):
                identical.append(path)
            else:
                mismatched.append(path)

                if AUTO_FIX:
                    try:
                        _replace_function_in_file(path, name, canon_src)
                        logging.info("      ↻ fixed %s", path)
                    except RuntimeError as err:
                        logging.error("      ⚠ could not fix %s – %s", path, err)

    return AuditResult(identical=identical, mismatched=mismatched)


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure console + file logging."""
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
    """Entry point."""
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
    if AUTO_FIX:
        logging.info("Auto‑repair mode: ON")

    exit_code = 0
    for func_name, canon in canonical_funcs.items():
        logging.info("── auditing function: %s ──", func_name)
        result = _audit_single_function(
            func_name,
            canon.norm_ast,
            canon.source,
            canonical_files,
        )

        if result.identical:
            logging.info("  ✓ %d identical copy(ies)", len(result.identical))
            for p in result.identical:
                logging.info("      %s", p)

        if result.mismatched:
            label = "repaired" if AUTO_FIX else "mismatching"
            logging.warning("  ✗ %d %s copy(ies)", len(result.mismatched), label)
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
