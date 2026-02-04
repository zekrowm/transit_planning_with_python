from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest
from utils.gtfs_helpers import load_gtfs_data


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _mk_gtfs_dir(tmp_path, files: Iterable[tuple[str, str]]) -> str:
    """Create a minimal GTFS folder with (name, contents) pairs."""
    base = tmp_path / "gtfs"
    base.mkdir()
    for name, contents in files:
        _write(base / name, contents)
    return str(base)


def test_load_gtfs_data_happy_path_minimal(tmp_path) -> None:
    """Loads specified files; keys are stems; values preserve content."""
    folder = _mk_gtfs_dir(
        tmp_path,
        files=[
            (
                "stops.txt",
                "stop_id,stop_name,stop_lat,stop_lon\n001,Main,38.9,-77.0\n",
            ),
            (
                "trips.txt",
                "route_id,service_id,trip_id\n10,WKD,10A\n",
            ),
        ],
    )

    result = load_gtfs_data(folder, files=("stops.txt", "trips.txt"))

    assert set(result.keys()) == {"stops", "trips"}
    assert isinstance(result["stops"], pd.DataFrame)
    assert len(result["stops"]) == 1
    # Default dtype=str should preserve leading zeros (object/string-like in pandas)
    assert result["stops"].loc[0, "stop_id"] == "001"


def test_load_gtfs_data_missing_folder_raises(tmp_path) -> None:
    """Missing directory → OSError with clear path in message."""
    missing = tmp_path / "no_such_folder"
    with pytest.raises(OSError) as excinfo:
        load_gtfs_data(str(missing), files=("stops.txt",))
    assert str(missing) in str(excinfo.value)


def test_load_gtfs_data_missing_file_detection(tmp_path) -> None:
    """If any requested file is absent, raise OSError listing it."""
    folder = _mk_gtfs_dir(
        tmp_path,
        files=[
            ("stops.txt", "stop_id,stop_name\n001,Main\n"),
        ],
    )
    with pytest.raises(OSError) as excinfo:
        load_gtfs_data(folder, files=("stops.txt", "trips.txt"))
    # Should enumerate the missing file(s)
    msg = str(excinfo.value)
    assert "Missing GTFS files" in msg and "trips.txt" in msg


def test_load_gtfs_data_empty_file_raises(tmp_path) -> None:
    """Empty CSV → ValueError with filename in message (from EmptyDataError)."""
    folder = _mk_gtfs_dir(
        tmp_path,
        files=[
            ("stops.txt", ""),  # empty file
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        load_gtfs_data(folder, files=("stops.txt",))
    msg = str(excinfo.value)
    assert "stops.txt" in msg and "empty" in msg.lower()


def test_load_gtfs_data_parser_error_raises(tmp_path: Path) -> None:
    """Malformed CSV → ValueError wrapping pandas ParserError."""
    # Trigger a ParserError via an unclosed quote, which the default C engine rejects.
    # Note: too-few fields are tolerated (padded with NaN), so they won't fail reliably.
    malformed = 'route_id,service_id,trip_id\n10,"WKD,10A\n'
    folder = _mk_gtfs_dir(tmp_path, files=[("trips.txt", malformed)])
    with pytest.raises(ValueError) as excinfo:
        load_gtfs_data(folder, files=("trips.txt",))
    msg = str(excinfo.value)
    assert "Parser error" in msg and "trips.txt" in msg


def test_load_gtfs_data_dtype_mapping_string(tmp_path) -> None:
    """Respects dtype mapping (string dtype keeps leading zeros as strings)."""
    folder = _mk_gtfs_dir(
        tmp_path,
        files=[
            (
                "stops.txt",
                "stop_id,stop_name\n001,Main\n",
            ),
        ],
    )
    result = load_gtfs_data(
        folder,
        files=("stops.txt",),
        dtype={"stop_id": "string", "stop_name": "string"},
    )
    df = result["stops"]
    # Values are strings (pandas StringDtype), not inferred numeric
    assert pd.api.types.is_string_dtype(df["stop_id"])
    assert df.loc[0, "stop_id"] == "001"
