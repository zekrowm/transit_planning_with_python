#!/usr/bin/env python3
"""Quick environment sanity check for local dev and CI.

Treats certain typing-stub wheels as *optional*:

    * They emit ⚠️  instead of ❌ when missing/outdated
    * They do not cause the script to exit with a non-zero status

Resolution order for the requirements file:

    1. REQUIREMENTS_FILE (if not None)
    2. First CLI argument
    3. requirements.txt adjacent to this script
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Union

from importlib.metadata import PackageNotFoundError, version as get_version
from packaging.version import InvalidVersion, Version

# ======================================================================
# CONFIGURATION
# ======================================================================

REQUIREMENTS_FILE: Union[str, Path, None] = (r"C:\Path\To\Your\requirements.txt")

MIN_PY_VERSION: Tuple[int, int] = (3, 9)

# Typing stubs or other niceties that should NOT fail CI
_OPTIONAL_PACKAGES: set[str] = {
    "types-networkx",
    "types-openpyxl",
}

# ======================================================================
# FUNCTIONS
# ======================================================================

def _resolve_req_file() -> Path:
    if REQUIREMENTS_FILE is not None:
        return Path(REQUIREMENTS_FILE).expanduser().resolve()
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return Path(__file__).with_name("requirements.txt")


def _strip_inline_comment(text: str) -> str:
    """Remove inline comments starting with '#' or ';'."""
    for sep in ("#", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def _parse_req_file(path: Path) -> Dict[str, str]:
    """Return mapping {package: required_version}."""
    out: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        operator = ">=" if ">=" in line else "=="
        if operator in line:
            pkg, ver = line.split(operator, 1)
        else:
            pkg, ver = line, "0"

        out[pkg.strip()] = _strip_inline_comment(ver)
    return out


def _version_satisfies(inst: str, req: str) -> bool:
    """True if installed version satisfies required (wildcards allowed)."""
    if "*" in req:
        return inst.startswith(req.split("*", 1)[0])
    try:
        return Version(inst) >= Version(req)
    except InvalidVersion:
        return False


def _check_package(pkg: str, req: str) -> tuple[bool, str]:
    try:
        inst = get_version(pkg)
    except PackageNotFoundError:
        return False, "NOT INSTALLED"
    return _version_satisfies(inst, req), inst


def _check_python() -> bool:
    print("🔧 Python:", sys.version.split()[0], end=" ")
    if sys.version_info[:2] >= MIN_PY_VERSION:
        print("✅")
        return True
    print(f"⚠️  need ≥ {MIN_PY_VERSION[0]}.{MIN_PY_VERSION[1]}")
    return False


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    req_file = _resolve_req_file()
    if not req_file.exists():
        print(f"Requirements file '{req_file}' not found.", file=sys.stderr)
        sys.exit(2)

    py_ok = _check_python()

    requirements = _parse_req_file(req_file)
    print("\n🔍 Package versions:\n")

    failures = 0
    for pkg, min_ver in sorted(requirements.items()):
        ok, inst = _check_package(pkg, min_ver)
        if ok:
            status = "✅"
            label = ""
        elif pkg in _OPTIONAL_PACKAGES:
            status = "⚠️"
            label = "optional"
        else:
            status = "❌"
            label = ""
            failures += 1

        print(
            f"{pkg:<20} installed: {inst:<15} "
            f"required: ≥ {min_ver:<10} {status} {label}"
        )

    if py_ok and failures == 0:
        print("\n🎉 Environment satisfies all minimum *required* versions.")
        sys.exit(0)

    print(f"\n🚨 {failures} required package(s) and/or Python version need attention.")
    sys.exit(1)


if __name__ == "__main__":
    main()
