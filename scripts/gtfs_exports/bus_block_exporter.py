"""Generate minute‑by‑minute vehicle‑block schedules from a GTFS feed.

This utility ingests a local GTFS zip (or extracted directory) and produces
Excel workbooks that trace each vehicle block—optionally aggregated to
*Route × Direction* level—at a fixed time resolution (default = 1 minute).
Each spreadsheet shows, for every minute in the evaluation horizon:

- the active Trip Start Time for context,
- stop‑level metadata when the vehicle is at a stop, and
- operational states such as DWELL, LAYOVER, DEADHEAD, or
  TRAVELING BETWEEN STOPS when it is not.

Typical use‑cases include block‑level runtime diagnostics, interlining
analysis, operator schedule reviews, and automated QA pipelines.

The behaviour is controlled entirely by the *CONFIGURATION* constants below;
no command‑line flags are required.  Adjust paths, time‑window parameters, or
filters (e.g. selected `service_id`s, maximum trips per block, or specific
`route_short_name`s) to match the target feed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

GTFS_FOLDER_PATH: str = r"Path\To\Your\GTFS_Folder"
OUTPUT_FOLDER: str = r"Path\To\Your\Output_Folder"

ROUTE_SHORTNAME_FILTER: list[str] = []  # e.g. ["350", "353"]; [] = all
AGGREGATE_BY_ROUTE_DIR: bool = False  # False → block XLSX; True → route/dir XLSX

CALENDAR_SERVICE_IDS: list[str] = ["3"]
DEFAULT_HOURS: int = 26
TIME_INTERVAL_MIN: int = 1

DWELL_THRESHOLD: int = 3  # minutes
LAYOVER_THRESHOLD: int = 20  # minutes
MAX_TRIPS_PER_BLOCK: int = 150

LOG_LEVEL: str = "INFO"  # DEBUG / INFO / WARNING / ERROR

# ==============================================================================
# FUNCTIONS
# ==============================================================================


def time_to_minutes(time_str: str) -> int:
    """Convert HH:MM[:SS] → integer minutes (supports 24 + hours)."""
    parts = time_str.split(":")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 60 + minutes + seconds // 60


def minutes_to_hhmm(total: int) -> str:
    """Convert integer minutes → HH:MM (zero‑padded, >24 h allowed)."""
    return f"{total // 60:02d}:{total % 60:02d}"


def validate_folders(input_path: Path, output_path: Path) -> None:
    """Ensure input is a directory and create output directory if needed."""
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_path} is not a directory.")
    output_path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# REUSABLE FUNCTIONS
# ------------------------------------------------------------------------------


def load_gtfs_data(
    gtfs_folder_path: Path,
    files: Optional[list[str]] = None,
    dtype: str | type | Mapping[str, Any] = str,
) -> dict[str, pd.DataFrame]:
    """Load GTFS text files into pandas DataFrames."""
    if files is None:
        files = [
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
            "calendar.txt",
            "calendar_dates.txt",
        ]
    missing = [f for f in files if not (gtfs_folder_path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing GTFS files: {', '.join(missing)}")

    data: dict[str, pd.DataFrame] = {}
    for file_name in files:
        key = file_name.replace(".txt", "")
        df = pd.read_csv(gtfs_folder_path / file_name, dtype=dtype, low_memory=False)
        data[key] = df
        logging.debug("Loaded %-20s %7d rows", file_name, len(df))
    return data


# ------------------------------------------------------------------------------
# STOP‑CLUSTER TOOLING
# ------------------------------------------------------------------------------


def find_cluster(stop_id: str, clusters: list[dict[str, Any]]) -> Optional[str]:
    """Return cluster name containing the given stop ID, if any."""
    for cluster_item in clusters:
        if stop_id in cluster_item["stops"]:
            return cluster_item["name"]
    return None


# ------------------------------------------------------------------------------
# STOP TIME AUGMENTERS
# ------------------------------------------------------------------------------


def mark_first_and_last_stops(df_in: pd.DataFrame) -> pd.DataFrame:
    """Mark first and last stops per trip using boolean flags."""
    df_out = df_in.sort_values(["trip_id", "stop_sequence"]).copy()
    df_out["is_first_stop"] = False
    df_out["is_last_stop"] = False
    for _trip_id, group in df_out.groupby("trip_id"):
        df_out.loc[group.index.min(), "is_first_stop"] = True
        df_out.loc[group.index.max(), "is_last_stop"] = True
    return df_out


# ------------------------------------------------------------------------------
# STATUS LOGIC (unchanged from your original script)
# ------------------------------------------------------------------------------


def _status_for_same_trip(minute: int, stop_info: tuple[Any, ...]) -> Optional[tuple[Any, ...]]:
    (arr, dep, s_id, s_name, t_id, is_first, is_last, s_seq, t_val) = stop_info
    if minute == arr and is_last:
        return (
            "ARRIVE",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if minute == dep and is_first:
        return (
            "DEPART",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if arr == dep and minute == arr:
        return (
            "ARRIVE/DEPART",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    if arr < minute < dep:
        return (
            "DWELL",
            s_id,
            s_name,
            minutes_to_hhmm(arr),
            minutes_to_hhmm(dep),
            t_id,
            s_seq,
            t_val,
        )
    return None


def _status_for_different_trip(
    dep: int,
    next_arr: int,
    current_stop_id: str,
    current_stop_name: str,
    next_stop_id: str,
) -> tuple[str, str, str]:
    """Classify the layover, dwell, or dead‑heading state between two trips.

    Parameters
    ----------
    dep
        Departure minute of the current trip’s last processed stop.
    next_arr
        Arrival minute of the next trip’s first stop.
    current_stop_id
        Stop ID of the current stop.
    current_stop_name
        Human‑readable name of the current stop.
    next_stop_id
        Stop ID of the next trip’s first stop.

    Returns:
    -------
    tuple[str, str, str]
        Status string plus the stop‑ID and stop‑name that the fill‑row
        should carry for layover/dwell/long‑break/deadhead situations.
    """
    gap = next_arr - dep
    same_stop = current_stop_id == next_stop_id

    if same_stop:
        if gap <= DWELL_THRESHOLD:
            return ("DWELL", current_stop_id, current_stop_name)
        if gap > LAYOVER_THRESHOLD:
            return ("LONG BREAK", current_stop_id, current_stop_name)
        return ("LAYOVER", current_stop_id, current_stop_name)

    return ("DEADHEAD", current_stop_id, current_stop_name)


def get_status_for_minute(
    minute: int,
    sequence: list[tuple[Any, ...]],
) -> tuple[
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    int,
]:
    """Return the stop‑level status tuple for a single minute.

    The tuple layout is identical to that produced by `_status_for_same_trip`.
    """
    if not sequence:
        return ("EMPTY", None, None, None, None, None, None, 0)

    for i, item in enumerate(sequence):
        same_trip = _status_for_same_trip(minute, item)
        if same_trip:
            return same_trip

        arr, dep = item[0], item[1]
        if i < len(sequence) - 1:
            next_item = sequence[i + 1]
            next_arr = next_item[0]

            if dep < minute < next_arr:
                trip_id = item[4]
                next_trip_id = next_item[4]
                stop_id = item[2]
                stop_name = item[3]
                stop_seq = item[7]
                t_val = item[8]

                if trip_id == next_trip_id:
                    return (
                        "TRAVELING BETWEEN STOPS",
                        None,
                        None,
                        None,
                        None,
                        trip_id,
                        None,
                        0,
                    )

                status, fill_id, fill_name = _status_for_different_trip(
                    dep, next_arr, stop_id, stop_name, next_item[2]
                )
                return (
                    status,
                    fill_id,
                    fill_name,
                    minutes_to_hhmm(arr),
                    minutes_to_hhmm(dep),
                    next_trip_id,
                    stop_seq,
                    t_val,
                )

    return ("EMPTY", None, None, None, None, None, None, 0)


# ------------------------------------------------------------------------------
# BLOCK‑BUILDING UTILITIES
# ------------------------------------------------------------------------------


def _create_trips_summary(block_subset: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize every trip in one block for quick look‑ups.

    Each item stores the trip’s time span plus a pre‑built sequence of
    stop‑time tuples so we can answer status queries quickly.
    """
    trips_summary: list[dict[str, Any]] = []

    for trip_id, trip_df in block_subset.groupby("trip_id"):
        t_sorted = trip_df.sort_values("stop_sequence")

        # numeric minutes for fast comparisons
        start_time = t_sorted["arrival_min"].min()
        end_time = t_sorted["departure_min"].max()

        # human‑readable version computed **once**
        start_str: str = minutes_to_hhmm(start_time)

        sequence: list[tuple[Any, ...]] = []
        for _, row in t_sorted.iterrows():
            sequence.append(
                (
                    row["arrival_min"],
                    row["departure_min"],
                    row["stop_id"],
                    row["stop_name"],
                    row["trip_id"],
                    row["is_first_stop"],
                    row["is_last_stop"],
                    row["stop_sequence"],
                    row["timepoint"],
                )
            )

        trips_summary.append(
            {
                "trip_id": trip_id,
                "start": start_time,
                "start_str": start_str,  # ← added field
                "end": end_time,
                "stop_times_sequence": sequence,
                "route_id": t_sorted.iloc[0]["route_short_name"],
                "direction_id": t_sorted.iloc[0]["direction_id"],
            }
        )

    trips_summary.sort(key=lambda x: x["start"])
    return trips_summary


def _status_for_active_trips(
    minute: int,
    active_trips: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], tuple[Any, ...]]:
    """Choose the most appropriate status among concurrently active trips."""
    candidates: list[tuple[dict[str, Any], tuple[Any, ...]]] = [
        (trip, get_status_for_minute(minute, trip["stop_times_sequence"])) for trip in active_trips
    ]

    valid = [c for c in candidates if c[1][0] != "EMPTY"]
    if not valid:
        return None, ("EMPTY", None, None, None, None, None, None, 0)
    if len(valid) == 1:
        return valid[0]

    def sort_key(item: tuple[dict[str, Any], tuple[Any, ...]]) -> tuple[bool, int]:
        stat = item[1]
        stop_seq = stat[6] if stat[6] is not None else 999_999
        tp = stat[7]
        return (tp == 0, stop_seq)

    valid.sort(key=sort_key)
    return valid[0]


def _row_for_inactive(minute: int, block_id: str, trips: list[dict[str, Any]]) -> dict[str, Any]:
    prev_trip = next_trip = None
    for trip in trips:
        if trip["end"] < minute:
            if prev_trip is None or trip["end"] > prev_trip["end"]:
                prev_trip = trip
        elif trip["start"] > minute:
            if next_trip is None or trip["start"] < next_trip["start"]:
                next_trip = trip

    if prev_trip and next_trip:
        gap = next_trip["start"] - prev_trip["end"]
        if gap <= DWELL_THRESHOLD:
            status = "DWELL"
        elif gap <= LAYOVER_THRESHOLD:
            status = "LAYOVER"
        else:
            status = "LONG BREAK"
    else:
        status = "INACTIVE"

    if next_trip and next_trip["start"] == minute + 1 and status in {"DWELL", "LAYOVER"}:
        status = "LOADING"

    return {
        "Timestamp": minutes_to_hhmm(minute),
        "Trip Start Time": "",  # NEW – keep schema stable
        "Block": block_id,
        "Route": "",
        "Direction": "",
        "Trip ID": "",
        "Stop ID": "",
        "Stop Name": "",
        "Stop Sequence": "",
        "Arrival Time": "",
        "Departure Time": "",
        "Status": status,
        "Timepoint": 0,
    }


def fill_stop_ids_for_dwell_layover_loading(df_in: pd.DataFrame) -> pd.DataFrame:
    """Populate empty stop fields for DWELL / LAYOVER / LOADING statuses.

    The refactored schedule rows purposely omit stop‑level metadata for rows where the
    vehicle is stationary *between* trips.  For operational QA and easier Excel review
    we replicate the original behavior:

    *Remember the most recent non‑blank stop information* and copy it forward to any
    subsequent row whose **Status** is one of::

        {"DWELL", "LAYOVER", "LOADING"}

    and whose **Stop ID** is empty.

    Args:
        df_in: A per‑block schedule DataFrame produced by
            :func:`process_block` *before* aggregation.

    Returns:
        A new DataFrame with missing values filled.  All columns that exist in
        the input are preserved unchanged; only the affected cells are updated.
    """
    df_out: pd.DataFrame = df_in.copy()

    last: dict[str, Any] = {
        "Stop ID": None,
        "Stop Name": None,
        "Stop Sequence": None,
        "Arrival Time": None,
        "Departure Time": None,
        "Trip ID": None,
    }

    target_statuses: set[str] = {"DWELL", "LAYOVER", "LOADING"}

    for idx in df_out.index:
        stop_id = df_out.at[idx, "Stop ID"]

        # ── update the “memory” whenever we see a fully populated stop row ───────────
        if pd.notna(stop_id) and str(stop_id).strip():
            last["Stop ID"] = stop_id
            last["Stop Name"] = df_out.at[idx, "Stop Name"]
            last["Stop Sequence"] = df_out.at[idx, "Stop Sequence"]
            last["Arrival Time"] = df_out.at[idx, "Arrival Time"]
            last["Departure Time"] = df_out.at[idx, "Departure Time"]
            last["Trip ID"] = df_out.at[idx, "Trip ID"]
            continue

        # ── backfill rows that qualify ─────────────────────────────────────────────
        status = df_out.at[idx, "Status"]
        if status in target_statuses and last["Stop ID"] is not None:
            for col, val in last.items():
                if not df_out.at[idx, col]:
                    df_out.at[idx, col] = val

    return df_out


def _build_schedule_rows(  # type: ignore[override]
    trips_summary: list[dict[str, Any]],
    timeline: range,
    block_id: str,
) -> list[dict[str, Any]]:
    """Generate a minute‑by‑minute schedule matrix for one block.

    Adds a human‑readable **Trip Start Time** so every row can be traced back
    to the origin of the active trip.
    """
    rows: list[dict[str, Any]] = []

    for minute in timeline:
        active = [t for t in trips_summary if t["start"] <= minute <= t["end"]]

        if active:
            chosen_trip, chosen_stat = _status_for_active_trips(minute, active)
            # `chosen_trip` is never None when `active` is truthy, but be defensive
            if chosen_trip is not None:
                (
                    status,
                    stop_id,
                    stop_name,
                    arr_str,
                    dep_str,
                    stat_trip_id,
                    stop_seq,
                    tp_val,
                ) = chosen_stat

                if status == "EMPTY":
                    status = "TRAVELING BETWEEN STOPS"

                # Promote DWELL/LAYOVER → LOADING when a departure is imminent
                if status in {"DWELL", "LAYOVER"}:
                    nxt_min = minute + TIME_INTERVAL_MIN
                    if nxt_min <= chosen_trip["end"]:
                        if (
                            get_status_for_minute(nxt_min, chosen_trip["stop_times_sequence"])[0]
                            == "DEPART"
                        ):
                            status = "LOADING"

                row: dict[str, Any] = {
                    "Timestamp": minutes_to_hhmm(minute),
                    "Trip Start Time": chosen_trip["start_str"],  # ← ★ keeps column
                    "Block": block_id,
                    "Route": chosen_trip["route_id"],
                    "Direction": chosen_trip["direction_id"],
                    "Trip ID": stat_trip_id or "",
                    "Stop ID": stop_id or "",
                    "Stop Name": stop_name or "",
                    "Stop Sequence": stop_seq or "",
                    "Arrival Time": arr_str or "",
                    "Departure Time": dep_str or "",
                    "Status": status,
                    "Timepoint": tp_val,
                }
            else:  # unreachable but keeps schema intact
                row = {
                    "Timestamp": minutes_to_hhmm(minute),
                    "Trip Start Time": "",
                    "Block": block_id,
                    "Route": "",
                    "Direction": "",
                    "Trip ID": "",
                    "Stop ID": "",
                    "Stop Name": "",
                    "Stop Sequence": "",
                    "Arrival Time": "",
                    "Departure Time": "",
                    "Status": "TRAVELING BETWEEN STOPS",
                    "Timepoint": 0,
                }
        else:
            row = _row_for_inactive(minute, block_id, trips_summary)

        rows.append(row)

    return rows


def process_block(
    block_subset: pd.DataFrame,
    block_id: str,
    timeline: range,
) -> pd.DataFrame:
    """Return a minute‑by‑minute schedule DataFrame for one vehicle block.

    The function

    1. summarises all trips in the block (``_create_trips_summary``);
    2. generates one row per minute (``_build_schedule_rows``);
    3. **back‑fills** DWELL / LAYOVER / LOADING rows so they inherit the most
       recent stop metadata (``fill_stop_ids_for_dwell_layover_loading``).

    Args:
        block_subset: All GTFS stop‑times for the selected block, already
            filtered and augmented with helper columns.
        block_id:     The block identifier.
        timeline:     ``range`` object covering every minute to evaluate.

    Returns:
    -------
        A fully populated :class:`pandas.DataFrame` ready for Excel export or
        aggregation.
    """
    # ── 1. trip‑level summary ────────────────────────────────────────────────
    trips_summary = _create_trips_summary(block_subset)

    # ── 2. build raw minute‑by‑minute rows ───────────────────────────────────
    rows = _build_schedule_rows(trips_summary, timeline, block_id)
    df = pd.DataFrame(rows)

    # ── 3. restore stop context for idle statuses ────────────────────────────
    return fill_stop_ids_for_dwell_layover_loading(df)


# ------------------------------------------------------------------------------
# MERGE + FILTER PRIMARY DATASET
# ------------------------------------------------------------------------------


def _merge_and_filter_data(
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    stops_df: pd.DataFrame,
) -> pd.DataFrame:
    if CALENDAR_SERVICE_IDS:
        trips_df = trips_df[trips_df["service_id"].isin(CALENDAR_SERVICE_IDS)]

    stop_times_df["arrival_min"] = stop_times_df["arrival_time"].apply(time_to_minutes)
    stop_times_df["departure_min"] = stop_times_df["departure_time"].apply(time_to_minutes)

    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips_df["trip_id"])]

    if "stop_code" not in stops_df.columns:
        stops_df["stop_code"] = None

    merged = stop_times_df.merge(trips_df, on="trip_id", how="left")
    merged = merged.merge(stops_df[["stop_id", "stop_name", "stop_code"]], on="stop_id", how="left")

    merged = mark_first_and_last_stops(merged)

    if "timepoint" not in merged.columns:
        merged["timepoint"] = 0
    else:
        merged["timepoint"] = (
            pd.to_numeric(merged["timepoint"], errors="coerce").fillna(0).astype(int)
        )

    merged.loc[(merged["is_first_stop"]) & (merged["timepoint"] == 0), "timepoint"] = 2
    merged.loc[(merged["is_last_stop"]) & (merged["timepoint"] == 0), "timepoint"] = 2

    # route_short_name filtering
    if ROUTE_SHORTNAME_FILTER:
        route_match = merged["route_short_name"].isin(ROUTE_SHORTNAME_FILTER)
        blocks_to_keep = merged.loc[route_match, "block_id"].unique()
        merged = merged[merged["block_id"].isin(blocks_to_keep)]
        logging.info("After route_short_name filter → %d rows", len(merged))

    return merged


def _status_for_active_trips(
    minute: int,
    active_trips: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], tuple[Any, ...]]:
    """Choose the most appropriate status among concurrently active trips."""
    candidates: list[tuple[dict[str, Any], tuple[Any, ...]]] = [
        (trip, get_status_for_minute(minute, trip["stop_times_sequence"])) for trip in active_trips
    ]

    valid = [c for c in candidates if c[1][0] != "EMPTY"]
    if not valid:
        return None, ("EMPTY", None, None, None, None, None, None, 0)
    if len(valid) == 1:
        return valid[0]

    def sort_key(item: tuple[dict[str, Any], tuple[Any, ...]]) -> tuple[bool, int]:
        stat = item[1]
        stop_seq = stat[6] if stat[6] is not None else 999_999
        tp = stat[7]
        return (tp == 0, stop_seq)

    valid.sort(key=sort_key)
    return valid[0]


# ------------------------------------------------------------------------------
# AGGREGATION TO ROUTE/DIRECTION
# ------------------------------------------------------------------------------


def _aggregate_by_route_dir(
    block_frames: list[pd.DataFrame],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Combine per‑block spreadsheets into Route × Direction workbooks."""
    combined = pd.concat(block_frames, ignore_index=True)

    keep = [
        "Timestamp",
        "Trip Start Time",  # ← new column retained
        "Block",
        "Route",
        "Direction",
        "Trip ID",
        "Stop ID",
        "Stop Name",
        "Status",
        "Interlined Route",
    ]
    combined = combined[keep]

    grouped: dict[tuple[str, str], pd.DataFrame] = {}
    for (rte, direc), grp in combined.groupby(["Route", "Direction"], sort=False):
        grp_sorted = grp.sort_values(["Timestamp", "Block"])
        grouped[(rte, str(direc))] = grp_sorted.reset_index(drop=True)
    return grouped


# ==============================================================================
# MAIN
# ==============================================================================


def run() -> None:
    """End‑to‑end pipeline: load GTFS, build per‑block schedules, write XLSX."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    gtfs_path = Path(GTFS_FOLDER_PATH)
    out_path = Path(OUTPUT_FOLDER)
    validate_folders(gtfs_path, out_path)

    logging.info("Loading GTFS …")
    gtfs = load_gtfs_data(gtfs_path)

    # ── Merge, mark, filter ────────────────────────────────────────────────────
    trips = gtfs["trips"]
    if "route_short_name" not in trips.columns:
        trips = trips.merge(
            gtfs["routes"][["route_id", "route_short_name"]],
            on="route_id",
            how="left",
        )

    merged_df = _merge_and_filter_data(trips, gtfs["stop_times"], gtfs["stops"])

    # ── Build blocks ───────────────────────────────────────────────────────────
    blocks = merged_df["block_id"].dropna().unique()
    timeline = range(0, DEFAULT_HOURS * 60, TIME_INTERVAL_MIN)

    block_frames: list[pd.DataFrame] = []
    for blk_id in blocks:
        blk_data = merged_df[merged_df["block_id"] == blk_id].copy()

        if blk_data["trip_id"].nunique() > MAX_TRIPS_PER_BLOCK:
            logging.warning(
                "Block %s has >%d trips – skipped.",
                blk_id,
                MAX_TRIPS_PER_BLOCK,
            )
            continue

        # no clusters needed anymore
        block_df = process_block(blk_data, blk_id, timeline)

        # determine interlined routes for this block
        route_set = set(block_df["Route"].dropna())
        interlined_map = {
            r: ",".join(sorted(route_set - {r})) if len(route_set) > 1 else "" for r in route_set
        }
        block_df["Interlined Route"] = block_df["Route"].map(interlined_map).fillna("")

        block_frames.append(block_df)

        if not AGGREGATE_BY_ROUTE_DIR:
            fname = f"block_{blk_id}.xlsx"
            block_df.to_excel(out_path / fname, index=False)
            logging.info("Wrote %s", fname)

    # ── Aggregate or finish ────────────────────────────────────────────────────
    if AGGREGATE_BY_ROUTE_DIR:
        grouped = _aggregate_by_route_dir(block_frames)
        for (rte, direc), df in grouped.items():
            fname = f"route_{rte}_dir_{direc}.xlsx"
            df.to_excel(out_path / fname, index=False)
            logging.info("Wrote %s", fname)


if __name__ == "__main__":
    run()
