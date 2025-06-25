"""Generate a CSV of selected stops and the routes serving each.

Users can configure to filter by stop_id or stop_code, and optional inclusion
of direction IDs in the route listing.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from pandas._typing import DtypeArg

# =============================================================================
# CONFIGURATION
# =============================================================================

GTFS_FOLDER_PATH = (
    r"C:\Path\To\GTFS"  # folder containing stops.txt, stop_times.txt, trips.txt, routes.txt
)
OUTPUT_CSV_PATH = r"C:\Path\To\Output\selected_stops_routes.csv"

# If True, filter by stop_id (default); if False, filter by stop_code
FILTER_BY_STOP_ID: bool = True

# Provide lists for both—only the one matching FILTER_BY_STOP_ID will be applied:
SELECTED_STOP_IDS: list[str] = ["1001", "1002", "1003"]
SELECTED_STOP_CODES: list[str] = ["A100", "B200", "C300"]

# If True (default), append direction_id in parentheses after each route_short_name
INCLUDE_DIRECTION: bool = True

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_gtfs_files(folder: str, dtype: DtypeArg = str) -> dict[str, pd.DataFrame]:
    """Load core GTFS text files into DataFrames.

    Args:
        folder: Path to GTFS folder.
        dtype: Passed to pandas.read_csv to enforce column types.

    Returns:
        Mapping of filename stem → DataFrame.

    Raises:
        RuntimeError: If any required file is missing or unreadable.
    """
    required = ["stops.txt", "stop_times.txt", "trips.txt", "routes.txt"]
    data: dict[str, pd.DataFrame] = {}
    for fname in required:
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing required GTFS file: {fname}")
        try:
            df = pd.read_csv(path, dtype=dtype, low_memory=False)
            key = os.path.splitext(fname)[0]
            data[key] = df
            logging.info(f"Loaded {fname} ({len(df)} rows)")
        except Exception as e:
            raise RuntimeError(f"Error loading {fname}: {e}") from e
    return data


def map_stops_to_routes(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    filter_by_id: bool,
    selected_ids: list[str],
    selected_codes: list[str],
    include_direction: bool,
) -> pd.DataFrame:
    """Build a DataFrame of each selected stop and its serving routes.

    The output will include the filter key (stop_id or stop_code),
    stop_name, and a comma-separated list of route_short_name (with optional
    direction_id in parentheses).

    Args:
        stops: GTFS stops.txt DataFrame.
        stop_times: GTFS stop_times.txt DataFrame.
        trips: GTFS trips.txt DataFrame.
        routes: GTFS routes.txt DataFrame.
        filter_by_id: If True, filter by stop_id; else by stop_code.
        selected_ids: List of stop_id values to include.
        selected_codes: List of stop_code values to include.
        include_direction: Whether to append direction_id to each route name.

    Returns:
        DataFrame with columns [filter_key, 'stop_name', 'routes'].
    """
    # Determine which column and values to use for filtering
    if filter_by_id:
        filter_col = "stop_id"
        values = selected_ids
    else:
        filter_col = "stop_code"
        values = selected_codes

    # 1) Select stops by the chosen key
    sel = stops[stops[filter_col].isin(values)].copy()
    if sel.empty:
        logging.error("No stops matched your selection criteria.")
        sys.exit(1)

    # 2) Gather stop_times for those stops (always by stop_id)
    st = stop_times[stop_times["stop_id"].isin(sel["stop_id"])]

    # 3) Join through trips → routes (and direction_id)
    merged = (
        st[["stop_id", "trip_id"]]
        .merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id", how="left")
        .merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
    )

    # 4) Build a column with or without direction
    if include_direction:
        # ensure direction_id is integer or string
        merged["direction_id"] = merged["direction_id"].fillna(-1).astype(int)
        merged["route_label"] = merged.apply(
            lambda r: f"{r['route_short_name']}({r['direction_id']})",
            axis=1,
        )
    else:
        merged["route_label"] = merged["route_short_name"]

    # 5) Aggregate unique labels per stop_id
    agg = merged.groupby("stop_id")["route_label"].agg(lambda seq: sorted(set(seq))).reset_index()
    agg["routes"] = agg["route_label"].apply(lambda lst: ", ".join(lst))
    agg.drop(columns=["route_label"], inplace=True)

    # 6) Merge back to get stop_name and the filter key
    sel = sel.merge(agg, on="stop_id", how="left")
    sel["routes"].fillna("", inplace=True)

    # 7) Return only the filter key, stop_name, and routes
    return sel[[filter_col, "stop_name", "routes"]]


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the stop→routes CSV exporter."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1) Load GTFS files
    try:
        data = load_gtfs_files(GTFS_FOLDER_PATH)
    except RuntimeError as err:
        logging.error(err)
        sys.exit(1)

    # 2) Map selected stops to their serving routes
    df_out = map_stops_to_routes(
        stops=data["stops"],
        stop_times=data["stop_times"],
        trips=data["trips"],
        routes=data["routes"],
        filter_by_id=FILTER_BY_STOP_ID,
        selected_ids=SELECTED_STOP_IDS,
        selected_codes=SELECTED_STOP_CODES,
        include_direction=INCLUDE_DIRECTION,
    )

    # 3) Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 4) Write CSV
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info(f"Wrote {len(df_out)} records to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
