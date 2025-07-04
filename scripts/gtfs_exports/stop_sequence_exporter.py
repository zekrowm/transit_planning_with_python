"""Extract the dominant stop pattern in each direction for every GTFS route.

For every route in the input feed, the script identifies the most
frequently occurring ordered list of stops (“pattern”) for `direction_id`
0 and 1. It then writes one CSV per route that lists every stop in either
pattern together with its sequence number in each direction.

Typical use cases include schedule validation, pattern visualisation, and
network QA in **ArcGIS Pro**, **QGIS**, or Jupyter Notebooks.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR: str = r"Path\To\Your\GTFS\Folder"
OUTPUT_DIR: str = r"Folder\To\Hold\PerRouteCSVs"  # <- folder, not file

# Optional route filters
FILTER_IN_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["306", "50A"]
FILTER_OUT_ROUTE_SHORT_NAMES: list[str] = []  # e.g. ["999"]

# =============================================================================
# FUNCTIONS
# =============================================================================


def compute_most_common_pattern(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    route_id: str,
    direction_id: str,
) -> list[str]:
    """Return the most frequent stop pattern for a route-direction pair.

    Args:
        trips_df: GTFS **trips** table.
        stop_times_df: GTFS **stop_times** table.
        route_id: Route identifier to analyse.
        direction_id: `"0"` or `"1"` (treated as string).

    Returns:
        List[str]: Ordered `stop_id` sequence of the most common pattern, or an
        empty list when no pattern is found.
    """
    route_trips = trips_df[
        (trips_df["route_id"] == route_id) & (trips_df["direction_id"] == direction_id)
    ]
    if route_trips.empty:
        return []

    stop_times = stop_times_df[stop_times_df["trip_id"].isin(route_trips["trip_id"])].copy()
    if stop_times.empty:
        return []

    if not pd.api.types.is_numeric_dtype(stop_times["stop_sequence"]):
        stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="raise")

    counter: Counter[tuple[str, ...]] = Counter()
    for _, grp in stop_times.groupby("trip_id"):
        seq = tuple(grp.sort_values("stop_sequence")["stop_id"])
        if seq:
            counter[seq] += 1

    if not counter:
        return []

    most_common_seq, _ = counter.most_common(1)[0]
    return list(most_common_seq)


def load_core_gtfs_tables(input_dir: str) -> tuple[pd.DataFrame, ...]:
    """Convenience wrapper to load the four tables required for extraction.

    Args:
        input_dir: Directory containing the GTFS feed.

    Returns:
        Tuple in the order ``(routes, trips, stop_times, stops)``.
    """
    gtfs = load_gtfs_data(input_dir, dtype=str)
    return gtfs["routes"], gtfs["trips"], gtfs["stop_times"], gtfs["stops"]


def filter_routes(
    routes_df: pd.DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Filter *routes_df* on its ``route_short_name`` column.

    Args:
        routes_df: GTFS *routes* table.
        include: List of short names to **keep**.  When *None* the set is
            unrestricted.
        exclude: List of short names to **drop** after *include* is applied.

    Returns:
        A new, filtered data frame; never modifies *routes_df* in place.
    """
    routes_df = routes_df.copy()
    routes_df["route_short_name"] = routes_df["route_short_name"].astype(str)

    if include:
        routes_df = routes_df[routes_df["route_short_name"].isin(include)]
    if exclude:
        routes_df = routes_df[~routes_df["route_short_name"].isin(exclude)]

    return routes_df


def build_stop_lookup(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Create a ``stop_id``-indexed lookup of codes and names.

    Args:
        stops_df: GTFS *stops* table.

    Returns:
        Data frame with index ``stop_id`` and columns
        ``["stop_code", "stop_name"]``.
    """
    stops_df = stops_df.astype({"stop_id": str, "stop_code": str, "stop_name": str})
    return stops_df.set_index("stop_id")[["stop_code", "stop_name"]]


def build_pattern_rows(
    routes_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stop_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Generate one row per stop appearing in either dominant pattern.

    For every *route_short_name* the function:

    1. Identifies the most common pattern for each direction.
    2. Computes sequence positions (`seq_dir_0`, `seq_dir_1`).
    3. Builds a tidy table suitable for per-route CSV export.

    Args:
        routes_df: Filtered subset of the GTFS *routes* table.
        trips_df: Full GTFS *trips* table.
        stop_times_df: Full GTFS *stop_times* table.
        stop_lookup: Output of :func:`build_stop_lookup`.

    Returns:
        Normalised data frame with the columns::

            route_short_name  stop_id  stop_code  stop_name
            seq_dir_0         seq_dir_1
    """
    rows: list[dict[str, object]] = []

    for _, route in routes_df.iterrows():
        route_id = route["route_id"]
        short = route["route_short_name"]

        pattern_0 = compute_most_common_pattern(trips_df, stop_times_df, route_id, "0")
        pattern_1 = compute_most_common_pattern(trips_df, stop_times_df, route_id, "1")

        all_stops = set(pattern_0).union(pattern_1)
        if not all_stops:
            continue

        seq0 = {sid: i + 1 for i, sid in enumerate(pattern_0)}
        seq1 = {sid: i + 1 for i, sid in enumerate(pattern_1)}
        order_key = lambda s: (seq0.get(s, float("inf")), seq1.get(s, float("inf")))

        for sid in sorted(all_stops, key=order_key):
            code, name = stop_lookup.loc[sid] if sid in stop_lookup.index else ("", "")
            rows.append(
                {
                    "route_short_name": short,
                    "stop_id": sid,
                    "stop_code": code,
                    "stop_name": name,
                    "seq_dir_0": seq0.get(sid, pd.NA),
                    "seq_dir_1": seq1.get(sid, pd.NA),
                }
            )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "route_short_name",
            "stop_id",
            "stop_code",
            "stop_name",
            "seq_dir_0",
            "seq_dir_1",
        ],
    )


def export_patterns_by_route(result_df: pd.DataFrame, out_dir: str) -> None:
    """Write one CSV per *route_short_name*.

    Args:
        result_df: Table created by :func:`build_pattern_rows`.
        out_dir: Output directory.  Created recursively when absent.

    Raises:
        OSError: When *out_dir* cannot be created.
        RuntimeError: Propagates unexpected I/O errors from Pandas.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Safe filename pattern: keep alnum, dash, underscore
    fil_safe = lambda s: re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")

    for short_name, group in result_df.groupby("route_short_name", sort=True):
        fname = f"{fil_safe(short_name)}_patterns.csv"
        path = os.path.join(out_dir, fname)
        group.to_csv(path, index=False)
        logging.info("Wrote %s (%d rows)", fname, len(group))


# -----------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# -----------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: str,
    files: Optional[Sequence[str]] = None,
    dtype: str | type[str] | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load one or more GTFS text files into memory.

    Args:
        gtfs_folder_path: Absolute or relative path to the folder
            containing the GTFS feed.
        files: Explicit sequence of file names to load. If ``None``,
            the standard 13 GTFS text files are attempted.
        dtype: Value forwarded to :pyfunc:`pandas.read_csv(dtype=…)` to
            control column dtypes. Supply a mapping for per-column dtypes.

    Returns:
        Mapping of file stem → :class:`pandas.DataFrame`; for example,
        ``data["trips"]`` holds the parsed *trips.txt* table.

    Raises:
        OSError: Folder missing or one of *files* not present.
        ValueError: Empty file or CSV parser failure.
        RuntimeError: Generic OS error while reading a file.

    Notes:
        All columns default to ``str`` to avoid pandas’ type-inference
        pitfalls (e.g. leading zeros in IDs).
    """
    if not os.path.exists(gtfs_folder_path):
        raise OSError(f"The directory '{gtfs_folder_path}' does not exist.")

    if files is None:
        files = (
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
        )

    missing = [
        file_name
        for file_name in files
        if not os.path.exists(os.path.join(gtfs_folder_path, file_name))
    ]
    if missing:
        raise OSError(f"Missing GTFS files in '{gtfs_folder_path}': {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        file_path = os.path.join(gtfs_folder_path, file_name)
        try:
            df = pd.read_csv(file_path, dtype=dtype, low_memory=False)
            data[key] = df
            logging.info("Loaded %s (%d records).", file_name, len(df))

        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"File '{file_name}' in '{gtfs_folder_path}' is empty.") from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Parser error in '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"OS error reading file '{file_name}' in '{gtfs_folder_path}': {exc}"
            ) from exc

    return data


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Entrypoint for ad hoc execution.

    Loads data, applies filters, derives patterns, and exports the CSVs.
    All operational messages—including failures—are written via
    :pydata:`logging`.
    """
    try:
        routes, trips, stop_times, stops = load_core_gtfs_tables(INPUT_DIR)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error(exc)
        return

    routes = filter_routes(routes, FILTER_IN_ROUTE_SHORT_NAMES, FILTER_OUT_ROUTE_SHORT_NAMES)
    if routes.empty:
        logging.info("No routes to process after applying filters.")
        return

    stop_lookup = build_stop_lookup(stops)
    result_df = build_pattern_rows(routes, trips, stop_times, stop_lookup)

    if result_df.empty:
        logging.info("No patterns found for any route/direction.")
        return

    try:
        export_patterns_by_route(result_df, OUTPUT_DIR)
    except Exception as exc:  # noqa: BLE001
        logging.error("Export failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
