import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Import the script module
from scripts.operations_tools import otp_monthly_by_timepoint_order


@pytest.fixture
def tides_input_csv(tmp_path: Path) -> Path:
    """Creates a temporary TIDES-style CSV input file."""
    # Load fixture
    fixture_path = Path("tests/fixtures/stop_visits.csv")
    df = pd.read_csv(fixture_path)

    # Add simulated joined columns
    # Pattern is PAT_30_WB or PAT_30_EB.
    df["pattern_id"] = df["pattern_id"].map(
        {"shp-101-01": "PAT_30_WB", "shp-101-51": "PAT_30_EB"}
    )
    df["route_id"] = "30"
    df["direction_id"] = df["pattern_id"].apply(lambda x: "0" if "WB" in x else "1")

    # Ensure TIDES columns are present
    # They should be there: schedule_relationship, service_date, actual_departure_time, ...

    input_path = tmp_path / "tides_input.csv"
    df.to_csv(input_path, index=False)
    return input_path


def test_tides_data_processing(tides_input_csv: Path, tmp_path: Path) -> None:
    """Test that the script correctly processes TIDES-style data."""
    output_dir = tmp_path / "output"

    # Arguments for the script
    argv = [
        "otp_monthly_by_timepoint_order.py",
        "--input",
        str(tides_input_csv),
        "--outdir",
        str(output_dir),
        "--start-month",
        "2025-03",
        "--end-month",
        "2025-03",
    ]

    with patch.object(sys, "argv", argv):
        # This should not raise SystemExit now
        otp_monthly_by_timepoint_order.main()

    # Verification
    assert output_dir.exists()

    # Check for variation index
    variation_index = output_dir / "variation_index.csv"
    assert variation_index.exists()

    df_var = pd.read_csv(variation_index)
    # Expect rows for PAT_30_WB and PAT_30_EB
    assert "PAT_30_WB" in df_var["Variation"].values
    assert "PAT_30_EB" in df_var["Variation"].values

    # Check for specific output file
    # Format: {route}_{direction}_{variation_slug}_n{count}_pct.csv
    # Direction is "0" or "1" (not normalized to text)

    # Get N for one variation
    wb_row = df_var[df_var["Variation"] == "PAT_30_WB"].iloc[0]
    n_wb = int(wb_row["N"])

    expected_pct_file = output_dir / f"30_0_PAT_30_WB_n{n_wb}_pct.csv"
    assert expected_pct_file.exists()

    # Read output and verify some content
    df_pct = pd.read_csv(expected_pct_file)
    assert "Year-Month" in df_pct.columns
    assert "2025-03" in df_pct["Year-Month"].values
