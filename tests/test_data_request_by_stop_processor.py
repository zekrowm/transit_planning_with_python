import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the script directory to sys.path to allow importing the module
script_dir = Path("scripts/ridership_tools").resolve()
sys.path.append(str(script_dir))

# Mock openpyxl if it's not installed, as it is used at module level
# We do this before importing the target module
try:
    import openpyxl  # noqa: F401
except ImportError:
    # Create a dummy module
    mock_openpyxl = MagicMock()
    # We need to mock specific submodules/attributes accessed at top-level
    mock_openpyxl.styles = MagicMock()
    mock_openpyxl.utils = MagicMock()

    # Inject into sys.modules
    sys.modules["openpyxl"] = mock_openpyxl
    sys.modules["openpyxl.styles"] = mock_openpyxl.styles
    sys.modules["openpyxl.utils"] = mock_openpyxl.utils

import data_request_by_stop_processor as target  # noqa: E402

# Fixture path
FIXTURE_PATH = Path("tests/fixtures/ridership_by_route_and_stop.csv")


def test_full_processing_integration() -> None:
    """Verify the full processing pipeline using the CSV fixture."""
    # 1. Load the fixture data
    if not FIXTURE_PATH.exists():
        raise AssertionError(f"Fixture file not found at {FIXTURE_PATH}")

    fixture_df = pd.read_csv(FIXTURE_PATH)

    # 2. Patch dependencies and configuration
    # We patch the module-level variables and functions
    with (
        patch("data_request_by_stop_processor.read_excel_file") as mock_read,
        patch("data_request_by_stop_processor.write_to_excel") as mock_write,
        patch("data_request_by_stop_processor.INPUT_FILE_PATH", Path("dummy_input.xlsx")),
        patch("data_request_by_stop_processor.OUTPUT_DIR", Path("dummy_output")),
        patch("data_request_by_stop_processor.ROUTES", []),
        patch("data_request_by_stop_processor.ROUTES_EXCLUDE", []),
        patch("data_request_by_stop_processor.STOP_IDS", []),
        patch("data_request_by_stop_processor.TIME_PERIODS", ["AM PEAK", "PM PEAK"]),
        patch("data_request_by_stop_processor.AGGREGATE_ROUTES_TOGETHER", True),
        patch("data_request_by_stop_processor.APPLY_ROUNDING", True),
        patch("data_request_by_stop_processor.AGGREGATE_BIN_RANGES", False),
    ):
        # Setup mock return value for read_excel_file
        mock_read.return_value = fixture_df.copy()

        # 3. Run the main function
        target.main()

        # 4. Assertions
        mock_read.assert_called_once()
        mock_write.assert_called_once()

        # Extract arguments passed to write_to_excel
        args, _ = mock_write.call_args
        # signature: (output_file, filtered_data, aggregated_peaks, all_time_aggregated)
        # args[0] is output_file
        filtered_data = args[1]
        aggregated_peaks = args[2]
        all_time_aggregated = args[3]

        # Check filtered_data (Original sheet)
        # Should contain all rows since we didn't filter by route/stop
        # And it should contain MIDDAY even though it's not in TIME_PERIODS config
        assert "MIDDAY" in filtered_data["TIME_PERIOD"].values, (
            "MIDDAY rows should be preserved in Original sheet"
        )

        # Check aggregated_peaks (AM PEAK, PM PEAK)
        assert "AM PEAK" in aggregated_peaks
        assert "PM PEAK" in aggregated_peaks
        assert "MIDDAY" not in aggregated_peaks, (
            "MIDDAY should not be in aggregated_peaks as it was not in TIME_PERIODS config"
        )

        # Check specific aggregation logic
        # Focus on Stop 1001 (Main St & 1st Ave)
        # Fixture Data for Stop 1001:
        # AM PEAK, 10A: Board 12.4, Alight 3.2
        # AM PEAK, 10B: Board 18.1, Alight 6.7
        # PM PEAK, 10A: Board 22,   Alight 11.9
        # MIDDAY,  10A: Board 30,   Alight 28

        # --- Test AM PEAK Aggregation ---
        # Expected AM PEAK Board: 12.4 + 18.1 = 30.5
        # Expected AM PEAK Alight: 3.2 + 6.7 = 9.9
        am_peak_df = aggregated_peaks["AM PEAK"]
        stop_1001_am = am_peak_df[am_peak_df["STOP_ID"] == 1001]

        assert not stop_1001_am.empty, "Stop 1001 should be present in AM PEAK aggregation"
        # Access using iloc[0]
        assert stop_1001_am.iloc[0]["BOARD_ALL_TOTAL"] == 30.5, (
            f"Expected 30.5, got {stop_1001_am.iloc[0]['BOARD_ALL_TOTAL']}"
        )
        assert stop_1001_am.iloc[0]["ALIGHT_ALL_TOTAL"] == 9.9, (
            f"Expected 9.9, got {stop_1001_am.iloc[0]['ALIGHT_ALL_TOTAL']}"
        )

        # --- Test All Time Aggregation ---
        # Expected All Time Board: 12.4 (AM) + 18.1 (AM) + 30 (MIDDAY) + 22 (PM) = 82.5
        # Expected All Time Alight: 3.2 + 6.7 + 28 + 11.9 = 49.8
        stop_1001_all = all_time_aggregated[all_time_aggregated["STOP_ID"] == 1001]

        assert not stop_1001_all.empty, "Stop 1001 should be present in All Time aggregation"
        assert stop_1001_all.iloc[0]["BOARD_ALL_TOTAL"] == 82.5, (
            f"Expected 82.5, got {stop_1001_all.iloc[0]['BOARD_ALL_TOTAL']}"
        )
        assert stop_1001_all.iloc[0]["ALIGHT_ALL_TOTAL"] == 49.8, (
            f"Expected 49.8, got {stop_1001_all.iloc[0]['ALIGHT_ALL_TOTAL']}"
        )

        # Verify Routes column aggregation
        # For Stop 1001, routes are 10A and 10B in AM PEAK.
        # The script sorts and joins unique routes.
        assert stop_1001_am.iloc[0]["ROUTES"] == "10A, 10B", (
            f"Expected '10A, 10B', got {stop_1001_am.iloc[0]['ROUTES']}"
        )
