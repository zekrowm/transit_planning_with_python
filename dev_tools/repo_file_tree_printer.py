"""Generates a visual tree of a project directory and writes it to disk.

Walks a target directory and builds a UTF-8/ASCII-formatted tree showing
only folders that contain at least one file. Outputs the tree to both stdout
and a specified text file. Useful for developing or updating a
directory_structure.txt file.

Inputs:
    - TARGET_DIR: Root directory to scan.
    - OUTPUT_DIR: Folder where the tree file will be saved.
    - OUTPUT_FILE: Name of the output tree file.

Outputs:
    - Console: Pretty-printed directory tree.
    - Disk: Text file containing the same structure.
"""

import os
import sys
from typing import Any, Dict, List

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

TARGET_DIR = r"/path/to/transit_planning_with_python"
OUTPUT_DIR = r"/path/to/output_directory"
OUTPUT_FILE = r"directory_structure.txt"

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


def build_tree(root: str) -> Dict[str, Any]:
    """Build nested dict of dirs containing files.

    Each dict maps:
      subdir_name -> nested dict
      '__files__' -> sorted list of filenames in this dir
    """
    tree = {}
    for dirpath, _, files in os.walk(root):
        if not files:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).extend(sorted(files))
    return tree


def build_lines(tree: Dict[str, Any], root_name: str) -> List[str]:
    """Convert nested dict to list of tree lines with connectors."""
    lines = [f"{root_name}/"]

    def recurse(node: Dict[str, Any], prefix: str) -> None:
        files = node.get("__files__", [])
        dirs = sorted(k for k in node.keys() if k != "__files__")
        entries = [(f, "file") for f in files] + [(d, "dir") for d in dirs]

        for idx, (name, typ) in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if typ == "dir" else ""
            lines.append(f"{prefix}{connector}{name}{suffix}")
            if typ == "dir":
                extension = "    " if is_last else "│   "
                recurse(node[name], prefix + extension)

    recurse(tree, "")
    return lines


# ==================================================================================================
# MAIN
# ==================================================================================================


def main(directory: str, output_dir: str, output_filename: str) -> None:
    """Generate and output the directory tree for all files."""
    tree = build_tree(directory)
    root_name = os.path.basename(os.path.abspath(directory)) or directory
    lines = build_lines(tree, root_name)

    # Print to console
    for line in lines:
        print(line)

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # Write to file
    output_path = os.path.join(output_dir, output_filename)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nWrote structure to {output_path}")
    except OSError as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(TARGET_DIR, OUTPUT_DIR, OUTPUT_FILE)
  
