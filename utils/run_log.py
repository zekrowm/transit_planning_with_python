"""Utility helpers for run-log generation in transit analysis scripts.

This module contains the canonical implementations of run-log helpers that are
intentionally reproduced (verbatim) inside each analysis script so that scripts
remain self-contained and runnable without the project on ``sys.path``.  When
updating any function here, mirror the change in every script that carries a
copy — the docstring comment in those copies names this file as the source.
"""

from __future__ import annotations

from pathlib import Path

_BEGIN = "# === BEGIN CONFIG ==="
_END = "# === END CONFIG ==="


def _try_ipython_cell() -> tuple[str | None, str | None]:
    """Return (cell_text, label) for the most recent Jupyter cell with both markers.

    Returns (None, None) if no IPython kernel is running or no matching cell is
    found.  Requires exactly one BEGIN and one END marker in the candidate cell
    so that false positives from cells that merely quote the markers are avoided.
    """
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return None, None
    if ip is None:
        return None, None
    history: list[str] = ip.user_ns.get("In", [])
    for i in range(len(history) - 1, -1, -1):
        cell = history[i]
        if cell.count(_BEGIN) == 1 and cell.count(_END) == 1:
            return cell, f"<Jupyter cell In[{i}]>"
    return None, None


def _resolve_script_source() -> tuple[str, str]:
    """Return (source_text, source_label) for the current execution context.

    Resolution order:

    1. Most recent Jupyter cell that contains exactly one BEGIN and one END
       marker (handles re-runs and edits correctly via reverse iteration).
    2. The file at ``__file__`` in the calling module's globals — works for
       both ``python script.py`` and ``%run script.py`` in a notebook.

    Raises:
        RuntimeError: if neither source can be located.
    """
    cell_text, cell_label = _try_ipython_cell()
    if cell_text is not None:
        return cell_text, cell_label

    file_path: str | None = globals().get("__file__")
    if file_path is not None:
        p = Path(file_path).resolve()
        return p.read_text(encoding="utf-8"), str(p)

    raise RuntimeError(
        "Cannot locate source for run log: no __file__ is defined and no "
        "Jupyter cell contains both CONFIG markers. Ensure the config cell "
        "has been executed before running this script."
    )


def extract_config_block_from_text(source_text: str, source_label: str) -> str:
    """Return the text between the CONFIG markers in *source_text*.

    Args:
        source_text: Full source text to scan (cell contents or file text).
        source_label: Human-readable description used in error messages.

    Returns:
        The verbatim text of the configuration block, joined with newlines.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
    """
    lines = source_text.splitlines()
    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin_idx is None and stripped == _BEGIN:
            begin_idx = i
        elif begin_idx is not None and stripped == _END:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        raise ValueError(
            f"Config markers not found in '{source_label}'. "
            f"Expected '{_BEGIN}' and '{_END}'."
        )
    return "\n".join(lines[begin_idx + 1 : end_idx])


def extract_config_block(source_file: Path) -> str:
    r"""Return the text between the CONFIG markers in *source_file*.

    Thin wrapper around :func:`extract_config_block_from_text` for callers
    that already have a file path.

    Args:
        source_file: Path to the Python source file to scan (typically
            ``Path(__file__)`` from the calling script).

    Returns:
        The verbatim text of the configuration block, joined with ``\n``.

    Raises:
        ValueError: If either marker is missing or they appear out of order.
        OSError: If ``source_file`` cannot be read.
    """
    return extract_config_block_from_text(
        source_file.read_text(encoding="utf-8"), str(source_file)
    )
