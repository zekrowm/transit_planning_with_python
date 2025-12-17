"""Generate a requirements.txt by scanning imports in a Python repository.

This script statically parses Python files to extract import statements,
maps importable package names to installed distributions, and writes a
requirements.txt into a user-specified output folder.

Limitations:
- Dynamic imports and conditional dependencies may be missed.
- Optional extras are not detected.
- Mapping depends on the *current* Python environment (what's installed).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable

# =============================================================================
# Configuration
# =============================================================================
# Set these manually.
REPO_ROOT = Path(r"C:\path\to\your\repo")
OUTPUT_DIR = Path(r"C:\path\to\output\folder")

# Optional settings.
OUTPUT_FILENAME = "requirements.txt"
INCLUDE_VERSIONS = True
INCLUDE_EDITABLE_LOCAL = False

EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "site-packages",
        "node_modules",
    }
)
INCLUDE_GLOBS = ("*.py",)


# =============================================================================
# Implementation
# =============================================================================
_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", set()))


@dataclass(frozen=True, slots=True)
class Config:
    """Derived configuration values.

    Attributes:
        repo_root: Root directory of the repository to scan.
        output_dir: Folder where requirements.txt will be written.
        output_filename: Name of the requirements file.
        include_versions: If True, pin versions as 'name==version'.
        include_editable_local: If True, include local top-level packages if discovered.
        exclude_dirs: Directory names to skip anywhere in the tree.
        include_globs: File globs to include (relative to each directory).
    """

    repo_root: Path
    output_dir: Path
    output_filename: str
    include_versions: bool
    include_editable_local: bool
    exclude_dirs: frozenset[str]
    include_globs: tuple[str, ...]


def _is_stdlib_top_level(name: str) -> bool:
    """Return True if `name` looks like a standard library top-level module."""
    return (not name) or (name in _STDLIB_MODULES)


def iter_python_files(repo_root: Path, cfg: Config) -> Iterable[Path]:
    """Yield Python files under repo_root, respecting exclusions."""
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        parts = set(path.parts)
        if any(excl in parts for excl in cfg.exclude_dirs):
            continue

        if any(path.match(glob) for glob in cfg.include_globs):
            yield path


def extract_import_toplevel_modules(py_file: Path) -> set[str]:
    """Extract top-level module names imported in a Python file.

    Args:
        py_file: Path to a Python source file.

    Returns:
        A set of top-level module names (e.g., "numpy" from "numpy.linalg").
    """
    try:
        source = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = py_file.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        return set()

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = (alias.name or "").split(".", maxsplit=1)[0]
                if top:
                    imports.add(top)

        elif isinstance(node, ast.ImportFrom):
            # `from . import x` -> node.module is None; ignore as local/relative.
            if node.level and (node.module is None):
                continue
            if node.module:
                top = node.module.split(".", maxsplit=1)[0]
                if top:
                    imports.add(top)

    return imports


def discover_imports(repo_root: Path, cfg: Config) -> set[str]:
    """Scan the repo and return all non-stdlib top-level imported module names."""
    found: set[str] = set()

    for py_file in iter_python_files(repo_root, cfg):
        for mod in extract_import_toplevel_modules(py_file):
            if _is_stdlib_top_level(mod):
                continue
            found.add(mod)

    return found


def build_import_to_distributions_map() -> dict[str, list[str]]:
    """Map importable package names -> distribution names."""
    # Example: {"yaml": ["PyYAML"], "PIL": ["Pillow"], "sklearn": ["scikit-learn"], ...}
    return metadata.packages_distributions()


def resolve_distributions(
    imported_modules: set[str],
    import_to_dists: dict[str, list[str]],
) -> set[str]:
    """Resolve imported top-level modules to installed distribution names."""
    resolved: set[str] = set()

    for mod in sorted(imported_modules):
        for dist in import_to_dists.get(mod, []):
            if dist:
                resolved.add(dist)

    return resolved


def get_distribution_versions(distributions: set[str]) -> dict[str, str]:
    """Return a dict of distribution -> version for installed distributions."""
    versions: dict[str, str] = {}
    for dist in sorted(distributions):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


def discover_local_top_level_packages(repo_root: Path, cfg: Config) -> set[str]:
    """Detect importable top-level packages that live in-repo (root or src/)."""
    local: set[str] = set()

    candidates = [repo_root]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        candidates.append(src_dir)

    for base in candidates:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if child.name in cfg.exclude_dirs:
                continue
            if (child / "__init__.py").is_file():
                local.add(child.name)

    return local


def format_requirements(distributions: set[str], include_versions: bool) -> list[str]:
    """Format requirement lines."""
    if include_versions:
        versions = get_distribution_versions(distributions)
        return [f"{name}=={ver}" for name, ver in sorted(versions.items())]

    return sorted(distributions)


def write_requirements(lines: list[str], output_path: Path) -> None:
    """Write requirements lines to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    """Run requirements export."""
    cfg = Config(
        repo_root=REPO_ROOT,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        include_versions=INCLUDE_VERSIONS,
        include_editable_local=INCLUDE_EDITABLE_LOCAL,
        exclude_dirs=EXCLUDE_DIRS,
        include_globs=INCLUDE_GLOBS,
    )

    repo_root = cfg.repo_root.resolve()
    output_dir = cfg.output_dir.resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"REPO_ROOT does not exist or is not a directory: {repo_root}")

    imported = discover_imports(repo_root, cfg)

    local_pkgs = discover_local_top_level_packages(repo_root, cfg)
    if not cfg.include_editable_local:
        imported -= local_pkgs

    import_to_dists = build_import_to_distributions_map()
    distributions = resolve_distributions(imported, import_to_dists)

    out_path = output_dir / cfg.output_filename
    lines = format_requirements(distributions, cfg.include_versions)

    write_requirements(lines, out_path)

    print(f"Wrote {len(lines)} requirement(s) to: {out_path}")
    if local_pkgs and not cfg.include_editable_local:
        print(f"Excluded local package(s): {', '.join(sorted(local_pkgs))}")


if __name__ == "__main__":
    main()
