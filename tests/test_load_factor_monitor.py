import pandas as pd
import pytest

from scripts.ridership_tools import load_factor_monitor


# Define a fixture for the CSV data
@pytest.fixture
def input_df() -> pd.DataFrame:
    # Load the CSV fixture
    # We assume the test is run from the root of the repo
    csv_path = "tests/fixtures/statistics_by_route_and_trip.csv"
    df = pd.read_csv(csv_path)

    # The script expects 'TRIP_START_TIME' to be time-like objects (Timestamp or time)
    # The CSV has strings "HH:MM". We need to convert them.
    # We use errors='coerce' to handle the empty string in row 9, which becomes NaT.
    # We keep them as Timestamps because assign_service_period handles Timestamp.hour.
    df["TRIP_START_TIME"] = pd.to_datetime(df["TRIP_START_TIME"], format="%H:%M", errors="coerce")

    return df


def test_process_data_structure(input_df) -> None:
    """Test that process_data adds the required columns and handles basic logic."""
    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    expected_cols = [
        "SERVICE_PERIOD",
        "LOAD_FACTOR",
        "LOAD_FACTOR_VIOLATION",
        "ROUTE_LIMIT_TYPE",
    ]
    for col in expected_cols:
        assert col in processed.columns


def test_process_data_calculations(input_df) -> None:
    """Test specific calculations for load factor and violations."""
    # Row 5 in CSV: 20B, MAX_LOAD=50. Capacity=39.
    # Load Factor = 50/39 = 1.2821
    # Route 20B is not in LOWER_LIMIT_ROUTES (default), so limit is 1.25.
    # Should be a violation.

    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    # Find the row for 20B, Inbound, 17:45 (Serial 5)
    row_5 = processed[processed["SERIAL_NUMBER"] == 5].iloc[0]

    assert row_5["LOAD_FACTOR_VIOLATION"] == "TRUE"
    assert row_5["ROUTE_LIMIT_TYPE"] == "HIGH"
    assert row_5["SERVICE_PERIOD"] == "PM Peak"  # 17:45 -> 17 is in [15, 18)
    assert round(row_5["LOAD_FACTOR"], 2) == 1.28

    # Row 8: 40D, Inbound, 03:30, MAX_LOAD=10.
    # 03:30 -> hour 3. 4 <= 3 < 6 is False. -> "Other"
    row_8 = processed[processed["SERIAL_NUMBER"] == 8].iloc[0]
    assert row_8["SERVICE_PERIOD"] == "Other"

    # Row 9: Missing time. NaT. NaT.hour is nan. "Other".
    row_9 = processed[processed["SERIAL_NUMBER"] == 9].iloc[0]
    assert row_9["SERVICE_PERIOD"] == "Other"


def test_process_data_filtering(input_df) -> None:
    """Test route filtering logic."""
    # Filter IN only 10A
    processed_in = load_factor_monitor.process_data(
        input_df.copy(),
        bus_capacity=39,
        filter_in_routes=["10A"],
        filter_out_routes=[],
        decimals=4,
    )
    assert all(processed_in["ROUTE_NAME"] == "10A")

    # Filter OUT 10A
    processed_out = load_factor_monitor.process_data(
        input_df.copy(),
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=["10A"],
        decimals=4,
    )
    assert "10A" not in processed_out["ROUTE_NAME"].values


def test_integration_exports(input_df, tmp_path, monkeypatch) -> None:
    """Integration test for the export functions using temporary directory."""
    # Monkeypatch OUTPUT_FILE so exports go to tmp_path
    # The script uses os.path.dirname(OUTPUT_FILE) for create_route_workbooks
    # And uses OUTPUT_FILE itself for export_to_excel

    fake_output_xlsx = tmp_path / "processed_stats.xlsx"
    monkeypatch.setattr(load_factor_monitor, "OUTPUT_FILE", str(fake_output_xlsx))

    # Mock export functions to avoid 'openpyxl' import errors in CI
    # We only verify that the logic flow reaches these functions
    monkeypatch.setattr(load_factor_monitor, "export_to_excel", lambda df, path: None)
    monkeypatch.setattr(load_factor_monitor, "create_route_workbooks", lambda df: None)

    processed = load_factor_monitor.process_data(
        input_df,
        bus_capacity=39,
        filter_in_routes=[],
        filter_out_routes=[],
        decimals=4,
    )

    # 1. Export CSV (Real I/O allowed for CSV)
    csv_out = tmp_path / "out.csv"
    load_factor_monitor.export_to_csv(processed, str(csv_out))
    assert csv_out.exists()

    # 2. Export Excel (Combined) - Mocked
    load_factor_monitor.export_to_excel(processed, str(fake_output_xlsx))
    # We cannot check file existence because it's mocked

    # 3. Create route workbooks - Mocked
    load_factor_monitor.create_route_workbooks(processed)
    # We cannot check file existence because it's mocked

    # 4. Violation Log (Real I/O allowed for TXT)
    log_out = tmp_path / "violations.txt"
    load_factor_monitor.write_violation_log(processed, str(log_out))
    assert log_out.exists()

    # Verify content of violation log
    content = log_out.read_text(encoding="utf-8")
    assert "20B" in content  # Should be listed as violation
