import sys
from unittest.mock import MagicMock

# -- MOCK MATPLOTLIB BEFORE IMPORTING SCRIPT --
# This prevents ModuleNotFoundError if matplotlib is not installed in the environment.
try:
    import matplotlib

    # Use Agg backend if real matplotlib is present
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    # If not present, install a mock into sys.modules
    mock_mpl = MagicMock()
    mock_plt = MagicMock()
    # Mock specific submodules used by ntd_route_trends.py
    sys.modules["matplotlib"] = mock_mpl
    sys.modules["matplotlib.pyplot"] = mock_plt
    sys.modules["matplotlib.dates"] = MagicMock()
    HAS_MATPLOTLIB = False

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

# Import the script to be tested
from scripts.ridership_tools import ntd_route_trends


# Fixture to load the CSV data
@pytest.fixture
def input_df() -> pd.DataFrame:
    csv_path = Path("tests/fixtures/ntd_monthly_multi_month.csv")
    df = pd.read_csv(csv_path)
    return df


def test_ntd_route_trends_integration(input_df, tmp_path):
    """
    Integration test for ntd_route_trends script using a multi-month CSV fixture.
    Mocks read_month_workbook to serve data from the fixture instead of Excel files.
    """

    # --- Setup Configuration Patches ---

    # 1. Define the months available in the fixture (Dec 2025 - Feb 2026)
    test_periods = {
        "Dec-2025": ntd_route_trends.PeriodSpec("dummy.xlsx", "Sheet1"),
        "Jan-2026": ntd_route_trends.PeriodSpec("dummy.xlsx", "Sheet1"),
        "Feb-2026": ntd_route_trends.PeriodSpec("dummy.xlsx", "Sheet1"),
    }

    # 2. Define the date range to process
    start_month = "Dec-2025"
    end_month = "Feb-2026"

    # 3. Define routes to process (Subset of what's in fixture to test filtering)
    test_routes = ["101", "202"]

    # --- Define Mock for read_month_workbook ---

    def mock_read_month_workbook(period: str, spec: ntd_route_trends.PeriodSpec) -> pd.DataFrame:
        """
        Intercepts read_month_workbook calls.
        Filters the loaded fixture CSV for the requested period.
        Normalizes columns and data types to match what read_month_workbook usually returns.
        """
        # Convert script period format ("Dec-2025") to fixture format ("December 2025")
        dt = datetime.strptime(period, "%b-%Y")
        fixture_month_str = dt.strftime("%B %Y")

        # Filter fixture data
        period_df = input_df[input_df["MTH_YR"] == fixture_month_str].copy()

        if period_df.empty:
            return pd.DataFrame()

        # Select required columns (matching script's REQUIRED_COLS + normalization)
        # Script expects: ROUTE_NAME, SERVICE_PERIOD, MTH_BOARD, DAYS
        # Fixture has these names already.

        # Normalize ROUTE_NAME to string (fixture has int)
        # The script's read_month_workbook usually calls normalise_route, so we simulate that output.
        period_df["ROUTE_NAME"] = period_df["ROUTE_NAME"].apply(lambda x: str(x))

        # Normalize SERVICE_PERIOD
        # Fixture has "Weekday", "Saturday", "Sunday" which are already correct,
        # but for safety we can apply the script's normalizer if needed.
        # The script's normalise_service_period handles "Weekday" -> "Weekday".
        period_df["SERVICE_PERIOD"] = period_df["SERVICE_PERIOD"].apply(
            ntd_route_trends.normalise_service_period
        )

        # Ensure numeric types for MTH_BOARD and DAYS
        period_df["MTH_BOARD"] = pd.to_numeric(period_df["MTH_BOARD"], errors="coerce")
        period_df["DAYS"] = pd.to_numeric(period_df["DAYS"], errors="coerce")

        # Add metadata columns that read_month_workbook appends
        period_df["period"] = period
        period_df["period_dt"] = dt

        return period_df

    # --- Apply Patches ---

    with (
        patch.object(ntd_route_trends, "PERIODS", test_periods),
        patch.object(ntd_route_trends, "START_MONTH", start_month),
        patch.object(ntd_route_trends, "END_MONTH", end_month),
        patch.object(ntd_route_trends, "ROUTES", test_routes),
        patch.object(ntd_route_trends, "OUTPUT_ROOT", tmp_path),
        patch.object(ntd_route_trends, "read_month_workbook", side_effect=mock_read_month_workbook),
    ):
        # Run the main function
        ntd_route_trends.main()

    # --- Verify Results ---

    # 1. Verify output directories for selected routes
    route_101_dir = tmp_path / "route_101"
    route_202_dir = tmp_path / "route_202"

    assert route_101_dir.exists(), "Output directory for route 101 should exist"
    assert route_202_dir.exists(), "Output directory for route 202 should exist"

    # 2. Verify filtering: Route 303 (present in fixture but not in ROUTES) should NOT have a directory
    route_303_dir = tmp_path / "route_303"
    assert not route_303_dir.exists(), (
        "Output directory for route 303 should NOT exist (filtering check)"
    )

    # 3. Verify files inside route directory
    expected_files = [
        "monthly_long.csv",
        "monthly_wide.csv",
        "outage_flags.csv",
        # Plots might be mocked or real depending on HAS_MATPLOTLIB, but ntd_route_trends.export_route calls plot functions regardless.
        # If mocked, files won't be created unless the mock is configured to do so (it isn't).
        # However, export_route calls plt.savefig. If plt is mocked, savefig is a Mock call.
        # If plt is real (Agg), files are created.
    ]

    # We only check for CSVs which are pandas-driven and independent of matplotlib
    csv_files = ["monthly_long.csv", "monthly_wide.csv", "outage_flags.csv"]

    for filename in csv_files:
        file_path = route_101_dir / filename
        assert file_path.exists(), f"Expected output file {filename} missing for route 101"

    # Check for plots only if we had real matplotlib
    if HAS_MATPLOTLIB:
        for filename in ["plots/monthly_totals.png", "plots/daily_averages.png"]:
            file_path = route_101_dir / filename
            assert file_path.exists(), f"Expected plot file {filename} missing for route 101"

    # 4. Verify content of monthly_long.csv for Route 101
    df_long = pd.read_csv(route_101_dir / "monthly_long.csv")
    assert len(df_long) >= 3, "Should have data rows for the 3 months"
    assert "Dec-2025" in df_long["period"].unique()
    assert "Jan-2026" in df_long["period"].unique()
    assert "Feb-2026" in df_long["period"].unique()

    # 5. Verify Combined Output
    combined_dir = tmp_path / "_combined"
    assert combined_dir.exists()
    assert (combined_dir / "all_routes_monthly_long.csv").exists()
