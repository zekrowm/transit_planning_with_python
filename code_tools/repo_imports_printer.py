"""Scans a Python project to classify and export imports by type.

Recursively traverses a target directory, extracts all imported top-level modules,
and categorizes them as standard-library, third-party, or local. Results are written
to a plain-text file grouped in Black/isort style. This is useful for developing or
updating a requirements.txt file.

Inputs:
    - TARGET_DIR: Root directory of the Python project to scan.
    - OUTPUT_PATH: Path to the output `.txt` file.

Outputs:
    - Text file listing imports in three sections: standard library, third-party
      (with pinned versions), and local modules.
"""

from __future__ import annotations

import ast
import importlib.metadata as importlib_metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_DIR: Path = Path(r"/path/to/transit_planning_with_python").resolve()
OUTPUT_PATH: Path = Path(r"/path/to/output_directory/imports_by_type.txt").resolve()

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

_STD_LIB_NAMES: Set[str] = set(sys.stdlib_module_names)  # Python 3.10+

# =============================================================================
# FUNCTIONS
# =============================================================================


def _gather_python_files(root: Path) -> Iterable[Path]:
    """Yield every *.py file under *root* (depth-first)."""
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(".py"):
                yield Path(dirpath, fname)


def _extract_modules_from_file(py_file: Path) -> Set[str]:
    """Return top-level module/package names imported in *py_file*."""
    source = py_file.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source, filename=str(py_file))
    modules: Set[str] = set()

    for node in ast.walk(tree):
        # `import foo, bar as baz`
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_name = alias.name.split(".")[0]
                modules.add(top_name)

        # `from foo.bar import baz`
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import => definitely local
                modules.add(node.module.split(".")[0] if node.module else "")
            else:
                if node.module:
                    top_name = node.module.split(".")[0]
                    modules.add(top_name)

    modules.discard("")  # remove any empty placeholders
    return modules


def _module_origin(name: str) -> Path | None:
    """Resolve *name* to its file location; None if resolution fails."""
    spec = importlib.util.find_spec(name)
    if spec and spec.origin and spec.origin != "built-in":
        return Path(spec.origin).resolve()
    return None


def _classify_modules(modules: Set[str], project_root: Path) -> Mapping[str, Set[str]]:
    """Split *modules* into {'stdlib', 'third_party', 'local'} according to:

      • stdlib       – present in sys.stdlib_module_names
      • local        – resolves inside *project_root*  (or relative import)
      • third_party  – everything else
    """
    stdlib, third_party, local = set(), set(), set()

    for mod in modules:
        if mod in _STD_LIB_NAMES:
            stdlib.add(mod)
            continue

        origin = _module_origin(mod)
        if origin and project_root in origin.parents:
            local.add(mod)
        else:
            third_party.add(mod)

    return {"stdlib": stdlib, "third_party": third_party, "local": local}


def _get_pinned_version(package: str) -> str:
    """Return 'pkg==x.y.z' if the version can be found, else just pkg."""
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return package  # not installed in current interpreter
    return f"{package}=={version}"


def _write_output(groups: Mapping[str, Set[str]], outfile: Path) -> None:
    """Write grouped import list to *outfile*."""
    try:
        outfile.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.exit(f"Cannot create output directory {outfile.parent}: {exc}")

    lines: list[str] = []

    # Order: stdlib, third-party, local (Black / isort default)
    header_map = {
        "stdlib": "# --- Standard library -------------------------------------------------",
        "third_party": "# --- Third-party ------------------------------------------------------",
        "local": "# --- Local -------------------------------------------------------------",
    }

    for key in ("stdlib", "third_party", "local"):
        lines.append(header_map[key])
        entries = (
            sorted(groups[key])
            if key != "third_party"
            else sorted(_get_pinned_version(p) for p in groups["third_party"])
        )
        lines.extend(entries)
        lines.append("")  # blank line between groups

    try:
        outfile.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote import list to: {outfile}")
    except OSError as exc:
        sys.exit(f"Cannot write to {outfile}: {exc}")


# =============================================================================
# MAIN
# =============================================================================


def main(root: Path, out_path: Path) -> None:
    """Driver: extract, classify, and export imports."""
    all_modules: Set[str] = set()
    for py_file in _gather_python_files(root):
        all_modules |= _extract_modules_from_file(py_file)

    groups = _classify_modules(all_modules, root)
    _write_output(groups, out_path)


if __name__ == "__main__":
    main(TARGET_DIR, OUTPUT_PATH)
