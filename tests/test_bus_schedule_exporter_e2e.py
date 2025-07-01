# tests/test_bus_schedule_exporter_e2e.py
r"""End-to-end smoke test for ``bus_schedule_exporter``.

What the test does
------------------
1. Builds a complete but miniature GTFS bundle (2 stops, 1 weekday trip).
   • Core logic files (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`,
     `calendar.txt`) contain real data.
   • The seven other spec files are written with **only a header row**.
     That satisfies the exporter's loader without adding noise.
2. Monkey-patches the exporter’s global paths so it reads the bundle from a
   temporary directory and writes output next to it.
3. Runs ``bus_schedule_exporter.main()`` and asserts the expected Excel
   file exists.

Running locally
---------------
    # Set the project root in the command line
    # Update with your folder path, type in and press enter
    cd "Your\Project\Root"

    # Optional, if you do not have them
    python -m pip install --upgrade pip
    python -m pip install pytest pandas openpyxl   # plus any exporter deps

    # Type and press enter
    pytest -q    # run all tests in the folders
    # or: pytest tests/test_exporter_e2e.py -q

The test is offline, takes < 2 s, and the same ``pytest -q`` command works
unchanged in GitHub Actions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# -- make sure the project root is importable ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------------------
# 1. Build the dummy GTFS bundle
# ------------------------------------------------------------------------------

CORE_FILES: dict[str, pd.DataFrame] = {
    "stops.txt": pd.DataFrame(
        {
            "stop_id": ["S1", "S2"],
            "stop_name": ["Stop 1", "Stop 2"],
            "stop_lat": [0, 0],
            "stop_lon": [0, 0],
        }
    ),
    "routes.txt": pd.DataFrame(
        {"route_id": ["R1"], "route_short_name": ["1"], "route_type": [3]}
    ),
    "trips.txt": pd.DataFrame(
        {
            "service_id": ["WEEK"],
            "route_id": ["R1"],
            "trip_id": ["t1"],
            "trip_headsign": ["Outbound"],
            "direction_id": [0],
        }
    ),
    "stop_times.txt": pd.DataFrame(
        {
            "trip_id": ["t1", "t1"],
            "arrival_time": ["08:00:00", "08:10:00"],
            "departure_time": ["08:00:00", "08:10:00"],
            "stop_id": ["S1", "S2"],
            "stop_sequence": [1, 2],
            "timepoint": [1, 1],
        }
    ),
    "calendar.txt": pd.DataFrame(
        {
            "service_id": ["WEEK"],
            "monday": [1],
            "tuesday": [1],
            "wednesday": [1],
            "thursday": [1],
            "friday": [1],
            "saturday": [0],
            "sunday": [0],
            "start_date": [20250101],
            "end_date": [20251231],
        }
    ),
}

# Minimal headers for the “stub” files
STUB_HEADERS: dict[str, str] = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone",
    "calendar_dates.txt": "service_id,date,exception_type",
    "fare_attributes.txt": "fare_id,price,currency_type,payment_method,transfers",
    "fare_rules.txt": "fare_id,route_id,origin_id,destination_id,contains_id",
    "feed_info.txt": "feed_publisher_name,feed_publisher_url,feed_lang",
    "frequencies.txt": "trip_id,start_time,end_time,headway_secs",
    "shapes.txt": "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence",
    "transfers.txt": "from_stop_id,to_stop_id,transfer_type",
}


def _write_dummy_gtfs(gtfs_dir: Path) -> None:
    """Create the minimal GTFS bundle (core data + header-only stubs)."""
    # Core data
    for name, df in CORE_FILES.items():
        df.to_csv(gtfs_dir / name, index=False)

    # Stub files with just headers
    for name, header in STUB_HEADERS.items():
        (gtfs_dir / name).write_text(header + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------
# 2. End-to-end test
# ------------------------------------------------------------------------------

def test_exporter_creates_excel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
   """Test that the exporter successfully creates an Excel schedule from a dummy GTFS bundle."""
    # Arrange
    gtfs_dir = tmp_path / "gtfs"
    out_dir = tmp_path / "out"
    gtfs_dir.mkdir()
    out_dir.mkdir()
    _write_dummy_gtfs(gtfs_dir)

    # Import exporter only after GTFS files exist
    import scripts.gtfs_exports.bus_schedule_exporter as exp  # noqa: E402

    # Point exporter to our temp dirs
    monkeypatch.setattr(exp, "GTFS_FOLDER_PATH", str(gtfs_dir))
    monkeypatch.setattr(exp, "BASE_OUTPUT_PATH", str(out_dir))
    monkeypatch.setattr(exp, "FILTER_SERVICE_IDS", [])
    monkeypatch.setattr(exp, "FILTER_IN_ROUTES", [])
    monkeypatch.setattr(exp, "FILTER_OUT_ROUTES", [])

    # Act – should run without error
    exp.main()

    # Assert
    subfolder = out_dir / "calendar_WEEK_mon_tue_wed_thu_fri"
    expected_xlsx = subfolder / "route_1_schedule_Weekday.xlsx"

    assert subfolder.is_dir(), "Exporter did not create the service-ID folder."
    assert expected_xlsx.is_file(), "Exporter did not write the Excel schedule."
