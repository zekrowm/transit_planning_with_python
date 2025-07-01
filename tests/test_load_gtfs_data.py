# tests/test_load_gtfs_data.py
"""Unit tests for load_gtfs_data().

Run locally with:
    pytest -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# --------------------------------------------------------------------------- #
# Make the top-level repo importable before the package is installed.
# Directory layout (simplified):
#
#   transit_planning_with_python-main/
#       helpers/
#           __init__.py
#           gtfs_loader.py      ← contains load_gtfs_data()
#           …
#       tests/
#           test_load_gtfs_data.py   ← this file
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

# Now regular imports work whether or not the package is pip-installed.
from helpers.gtfs_helpers import load_gtfs_data  # adjust module name only

# -----------------------------------------------------------------------------
# Helper: write the minimal valid GTFS files pytest needs for a happy-path run
# -----------------------------------------------------------------------------

REQUIRED = [
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "fare_attributes.txt",
    "fare_rules.txt",
    "feed_info.txt",
    "frequencies.txt",
    "shapes.txt",
    "transfers.txt",
]


def create_minimal_gtfs(folder: Path) -> None:
    """Write one-row CSVs for every standard GTFS file.

    The function creates a syntactically valid GTFS feed with minimal content
    for testing purposes.

    Args:
        folder (Path): The path to the directory where the GTFS files will be
            created.
    """
    minimal_content = {
        "agency.txt": "agency_id,agency_name\n1,Sample Agency\n",
        "stops.txt": "stop_id,stop_name\nS1,Main St\n",
        "routes.txt": "route_id,route_type\nR1,3\n",
        "trips.txt": "route_id,service_id,trip_id\nR1,WK,T1\n",
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nT1,08:00:00,08:00:00,S1,1\n"
        ),
    }
    for fname in REQUIRED:
        path = folder / fname
        if fname in minimal_content:
            path.write_text(minimal_content[fname], encoding="utf-8")
        else:
            path.write_text("dummy_col\nvalue\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# 1. Happy-path test
# -----------------------------------------------------------------------------


def test_load_gtfs_success(tmp_path: Path) -> None:
    """Test successful loading of a minimal GTFS feed.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.
    """
    create_minimal_gtfs(tmp_path)
    data = load_gtfs_data(tmp_path)
    assert isinstance(data, dict)
    assert set(data) == {f.replace(".txt", "") for f in REQUIRED}
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    assert data["agency"].shape == (1, 2)


# -----------------------------------------------------------------------------
# 2. Directory does not exist
# -----------------------------------------------------------------------------


def test_load_gtfs_missing_directory() -> None:
    """Test that OSError is raised when the GTFS directory does not exist."""
    with pytest.raises(OSError, match="does not exist"):
        load_gtfs_data("path/that/does/not/exist")


# -----------------------------------------------------------------------------
# 3. A required file is missing
# -----------------------------------------------------------------------------


def test_load_gtfs_missing_file(tmp_path: Path) -> None:
    """Test that OSError is raised when a required GTFS file is missing.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.
    """
    create_minimal_gtfs(tmp_path)
    os.remove(tmp_path / "agency.txt")  # make one file vanish
    with pytest.raises(OSError, match="agency.txt"):
        load_gtfs_data(tmp_path)


# -----------------------------------------------------------------------------
# 4. A file exists but is empty
# -----------------------------------------------------------------------------


def test_load_gtfs_empty_file(tmp_path: Path) -> None:
    """Test that ValueError is raised when a GTFS file is empty.

    Args:
        tmp_path (Path): pytest fixture for a temporary directory.
    """
    create_minimal_gtfs(tmp_path)
    (tmp_path / "stops.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        load_gtfs_data(tmp_path)
