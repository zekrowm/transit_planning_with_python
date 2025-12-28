from __future__ import annotations

import pandas as pd
import pytest

from helpers.gtfs_helpers import load_gtfs_data

SAMPLE_DATA_PATH = "sample_data/gtfs"

def test_load_sample_data_exists() -> None:
    """Verify that the sample data directory and critical files exist."""
    import os

    assert os.path.exists(SAMPLE_DATA_PATH), "Sample data directory missing"
    assert os.path.exists(os.path.join(SAMPLE_DATA_PATH, "agency.txt")), "agency.txt missing"
    assert os.path.exists(os.path.join(SAMPLE_DATA_PATH, "stops.txt")), "stops.txt missing"

def test_load_gtfs_data_with_sample() -> None:
    """Verify loading the synthetic sample data works correctly."""

    # We must explicitly list the files we created, because the default list
    # in load_gtfs_data includes optional files we didn't create (e.g. transfers.txt).
    files_to_load = (
        "agency.txt",
        "stops.txt",
        "routes.txt",
        "trips.txt",
        "stop_times.txt",
        "calendar.txt",
        "shapes.txt",
    )

    data = load_gtfs_data(SAMPLE_DATA_PATH, files=files_to_load)

    # Check agency
    assert "agency" in data
    agency = data["agency"]
    assert len(agency) == 1
    assert agency.iloc[0]["agency_name"] == "Toy Transit"

    # Check stops
    assert "stops" in data
    stops = data["stops"]
    assert len(stops) == 3
    assert "S1" in stops["stop_id"].values

    # Check routes
    assert "routes" in data
    routes = data["routes"]
    assert len(routes) == 1
    assert routes.iloc[0]["route_long_name"] == "Downtown Express"

    # Check trips
    assert "trips" in data
    trips = data["trips"]
    assert len(trips) == 1
    assert trips.iloc[0]["trip_id"] == "T1"
