import sys
from pathlib import Path

import pandas as pd

# Add the script directory to path to import the module
# We need to make sure we point to the directory containing the script
script_dir = Path("scripts/operations_tools").resolve()
sys.path.append(str(script_dir))

from unittest.mock import MagicMock  # noqa: E402

# Mock visualization libraries to allow tests to run in environments without them
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()

import trip_event_runtime_diagnostics as target  # noqa: E402

FIXTURE_PATH = Path("tests/fixtures/trips_performed.csv")


def test_load_trip_files_tides_support() -> None:
    """Verify load_trip_files handles TIDES data correctly."""
    # 1. Load the data using the target function
    df = target.load_trip_files([FIXTURE_PATH])

    # 2. Assert basic columns are renamed
    assert "Route" in df.columns, "Route column missing (renamed from route_id)"
    assert "Direction" in df.columns, "Direction column missing (renamed from direction_id)"
    assert "TripID" in df.columns, "TripID column missing (renamed from trip_id_performed)"
    assert "Scheduled Start Time" in df.columns, "Scheduled Start Time column missing"
    assert "Actual Start Time" in df.columns, "Actual Start Time column missing"
    assert "trip_start_time" in df.columns, "trip_start_time column should be derived"

    # 3. Assert filtering
    # In fixture:
    # Row 1: In service, Scheduled -> Keep
    # Row 2: In service, Scheduled -> Keep
    # Row 3: Deadhead -> Drop
    # Row 4: In service, Scheduled -> Keep
    # Row 5: In service, Scheduled -> Keep
    # Row 6: In service, Duplicated -> Keep
    # Row 7: In service, Added -> Keep
    # Row 8: Pullout -> Drop
    # Row 9: Pullin -> Drop
    # Row 10: In service, Canceled -> Drop

    # Total rows in fixture: 10.
    # Expected kept: 6 rows.
    # Dropped: 3 (Deadhead), 8 (Pullout), 9 (Pullin), 10 (Canceled).

    assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

    # Check Canceled is gone
    # Row 10 trip_id_performed is TP20250320_010
    assert "TP20250320_010" not in df["TripID"].values, "Canceled trip should be filtered out"

    # Check Deadhead is gone
    # Row 3 TP20250318_003
    assert "TP20250318_003" not in df["TripID"].values, "Deadhead trip should be filtered out"

    # 4. Assert Time Extraction
    # Row 1: 2025-03-18T06:00:00 -> 06:00
    row1 = df[df["TripID"] == "TP20250318_001"].iloc[0]
    assert row1["trip_start_time"] == "06:00", f"Expected 06:00, got {row1['trip_start_time']}"

    # 5. Assert Direction is string "0"/"1"
    # Row 1 direction_id is 0
    assert str(row1["Direction"]) == "0", f"Expected direction '0', got {row1['Direction']}"

    # 6. Assert Is Tides flag
    assert "_is_tides" in df.columns
    assert df["_is_tides"].all()

    # 7. Check DateTime conversion
    assert pd.api.types.is_datetime64_any_dtype(df["Scheduled Start Time"]), (
        "Scheduled Start Time not datetime"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["Actual Start Time"]), (
        "Actual Start Time not datetime"
    )


def test_extract_trip_start_time_skip() -> None:
    """Verify extract_trip_start_time returns early if column exists."""
    df = pd.DataFrame({"trip_start_time": ["10:00"], "Trip": ["TRIP_1000"]})
    res = target.extract_trip_start_time(df)
    pd.testing.assert_frame_equal(df, res)
