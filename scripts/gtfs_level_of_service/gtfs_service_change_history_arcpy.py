"""Analyze and compare multiple GTFS datasets to detect service changes.

The script loads successive GTFS snapshots, calculates route-level service
metrics (span, trip count, median headway), detects stop additions/removals/
relocations, and identifies interlining changes.  Results are written to three
Excel workbooks:

* ``stop_change_report.xlsx`` – stop additions, removals, relocations, and
  route–stop service changes.
* ``route_metrics_by_signup.xlsx`` – per-signup route metrics plus delta-only
  sheets.
* ``service_level_changes.xlsx`` – route-level span/trip/headway deltas,
  created routes, and deleted routes.
"""

from __future__ import annotations

import math
import os
from typing import Any, cast  # ← NEW import

import numpy as np
import pandas as pd
from pandas._libs.tslibs.nattype import NaTType  # ← NEW import

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# Chronological GTFS snapshots to compare
MULTIPLE_GTFS_CONFIGS = [
    {
        "name": "Jan_2025",
        "path": r"Path\To\Your\GTFS_Folder",
    },
    {
        "name": "Jun_2025",
        "path": r"Path\To\Your\GTFS_Folder",
    },
    # add more …
]

# Output locations & filenames
OUTPUT_DIRECTORY = r"Path\To\Your\Output_Folder"
OUTPUT_EXCEL_NAME_STOPS = "stop_change_report.xlsx"
OUTPUT_EXCEL_NAME_METRIC = "route_metrics_by_signup.xlsx"
OUTPUT_EXCEL_NAME_DELTA = "service_level_changes.xlsx"
COMPARISON_EXCEL = "detailed_service_change_comparison.xlsx"

# Tolerance for GPS drift when detecting “moved” stops: ~5 m
COORD_TOLERANCE_DEG = 0.00005

# Routes to exclude from *all* outputs
ROUTE_FILTER_OUT = ["9999A", "9999B", "9999C"]
FILTER_SET: set[str] = set(ROUTE_FILTER_OUT)

# Time blocks for headway & schedule analysis
TIME_BLOCKS = {
    "AM": ("04:00", "09:00"),
    "MIDDAY": ("09:00", "15:00"),
    "PM": ("15:00", "21:00"),
    "NIGHT": ("21:00", "28:00"),
}

# Schedule types for per-signup metrics
SCHEDULE_TYPES = {
    "Weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "Saturday": ["saturday"],
    "Sunday": ["sunday"],
}

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================


def _keep_changed(df: pd.DataFrame) -> pd.DataFrame:
    """Return only those routes where at least one metric actually changed."""
    mask = (
        df["span_delta"].fillna(0).astype(float).ne(0)
        | df["trips_delta"].fillna(0).astype(float).ne(0)
        | df["hdwy_delta"].fillna(0).astype(float).ne(0)
    )
    return df.loc[mask]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (optional helper)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _parse_gtfs_time(t: str | float | int) -> pd.Timedelta | NaTType:  # ← widened
    """Return *t* converted from GTFS HH:MM:SS to ``pd.Timedelta``."""
    try:
        hh, mm, ss = map(int, str(t).split(":"))
        return pd.Timedelta(seconds=hh * 3600 + mm * 60 + ss)
    except Exception:
        return pd.NaT  # type: ignore[return-value]


def _format_td(td: pd.Timedelta | NaTType | None) -> str | None:  # ← narrowed
    """Format Timedelta as HH:MM or None."""
    if td is None or isinstance(td, NaTType) or pd.isna(td):
        return None
    mins = int(td.total_seconds() // 60)
    hh, mm = divmod(mins, 60)
    return f"{hh:02d}:{mm:02d}"


def _assign_block(td: pd.Timedelta | None) -> str | None:
    """Assign a TIME_BLOCKS label to a pandas.Timedelta."""
    if pd.isna(td):
        return None
    for blk, (s, e) in TIME_BLOCKS.items():
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        start, end = (
            pd.Timedelta(hours=sh, minutes=sm),
            pd.Timedelta(hours=eh, minutes=em),
        )
        if start <= td < end:
            return blk
    return None


# --------------------------------------------------------------------------------------------------
# GTFS DATA LOADING & METRICS
# --------------------------------------------------------------------------------------------------

FILES_NEEDED_STOP = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
FILES_NEEDED_METR = FILES_NEEDED_STOP + ["calendar.txt"]


def _check_files(base: str, files: list[str]) -> None:
    for f in files:
        if not os.path.exists(os.path.join(base, f)):
            raise FileNotFoundError(f"Required GTFS file missing: {f} in {base}")


def load_gtfs_basic(path: str):
    """Load the minimal stop-level tables needed for stop comparison.

    Args:
        path: Absolute or relative filesystem path to a single GTFS bundle
            containing at least ``stops.txt``, ``routes.txt``, ``trips.txt``,
            and ``stop_times.txt``.

    Returns:
        A 2-tuple ``(stops, stop_to_routes)`` where:

        * **stops** – ``pd.DataFrame`` with columns
          ``stop_id``, ``stop_code``, ``stop_name``, ``stop_lat``,
          ``stop_lon`` (one row per stop).
        * **stop_to_routes** – mapping ``stop_id -> {route_short_name, …}``
          representing all routes that call at each stop after applying
          ``ROUTE_FILTER_OUT``.

    Raises:
        FileNotFoundError: If any required GTFS text file is absent.
    """
    _check_files(path, FILES_NEEDED_STOP)
    stops = pd.read_csv(
        os.path.join(path, "stops.txt"),
        dtype={
            "stop_id": str,
            "stop_code": str,
            "stop_name": str,
            "stop_lat": float,
            "stop_lon": float,
        },
        usecols=["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"],
    )
    routes = pd.read_csv(
        os.path.join(path, "routes.txt"),
        usecols=["route_id", "route_short_name"],
        dtype=str,
    )
    trips = pd.read_csv(
        os.path.join(path, "trips.txt"),
        usecols=["trip_id", "route_id"],
        dtype=str,
    )
    stimes = pd.read_csv(
        os.path.join(path, "stop_times.txt"),
        usecols=["trip_id", "stop_id"],
        dtype=str,
    )

    merged = (
        stimes.merge(trips, on="trip_id", how="left")
        .merge(routes, on="route_id", how="left")
        .dropna(subset=["route_short_name"])
    )
    if FILTER_SET:
        merged = merged.loc[~merged["route_short_name"].isin(FILTER_SET)]

    stop_to_routes = merged.groupby("stop_id")["route_short_name"].apply(set).to_dict()
    return stops, stop_to_routes


def load_route_metrics(path: str, schedule_type: str = "Weekday") -> pd.DataFrame:
    """Compute weekday / Saturday / Sunday service metrics for each route.

    Metrics per ``route_short_name`` include first trip time, last trip time,
    span (minutes), total trips, and median headway across user-defined
    ``TIME_BLOCKS``.
    """
    _check_files(path, FILES_NEEDED_METR)

    # 1 ─ read files
    routes = pd.read_csv(
        os.path.join(path, "routes.txt"),
        usecols=["route_id", "route_short_name"],
        dtype=str,
    )
    trips = pd.read_csv(
        os.path.join(path, "trips.txt"),
        usecols=["trip_id", "route_id", "service_id"],
        dtype=str,
    )
    stimes = pd.read_csv(
        os.path.join(path, "stop_times.txt"),
        usecols=["trip_id", "stop_sequence", "departure_time"],
        dtype=str,
    )
    cal = pd.read_csv(os.path.join(path, "calendar.txt"), dtype=str)

    # 2 ─ filter by schedule_type days
    days = SCHEDULE_TYPES[schedule_type]
    mask = pd.Series(True, index=cal.index)
    for d in days:
        mask &= cal[d] == "1"
    valid_sids = set(cal.loc[mask, "service_id"])
    trips = trips.loc[trips["service_id"].isin(valid_sids)]

    # 3 ─ merge trips→routes, apply route filter
    trip_rt = trips.merge(routes, on="route_id", how="left")
    if FILTER_SET:
        trip_rt = trip_rt.loc[~trip_rt["route_short_name"].isin(FILTER_SET)]

    # 4 ─ find first stop per trip
    stimes["stop_sequence"] = pd.to_numeric(stimes["stop_sequence"], errors="coerce")
    idx_first = stimes.groupby("trip_id")["stop_sequence"].idxmin()
    st = stimes.loc[idx_first].merge(trip_rt, on="trip_id", how="inner")
    st["td"] = st["departure_time"].apply(_parse_gtfs_time)
    st = st.dropna(subset=["td"])
    st["block"] = st["td"].apply(_assign_block)

    # 5 ─ compute metrics per route
    metrics: list[dict[str, float | str | None]] = []
    for rt, grp in st.groupby("route_short_name"):
        times = grp["td"].sort_values()
        first_td, last_td = times.iloc[0], times.iloc[-1]
        span_min = int((last_td - first_td).total_seconds() // 60)
        trips_ct = len(times)
        # median headway
        hw_list: list[pd.Series] = []
        for _, bdf in grp.groupby("block"):
            if len(bdf) >= 2:
                diffs = bdf["td"].sort_values().diff().dropna()
                hw_list.append(diffs.dt.total_seconds() / 60)
        med_hw = float(pd.concat(hw_list).median()) if hw_list else None

        metrics.append(
            {
                "route_short_name": rt,
                "first_trip_time": _format_td(first_td),
                "last_trip_time": _format_td(last_td),
                "span_minutes": span_min,
                "trips_count": trips_ct,
                "median_headway_min": med_hw,
            }
        )

    df = pd.DataFrame(metrics).sort_values("route_short_name").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------------------
# STOP-LEVEL COMPARISON
# --------------------------------------------------------------------------------------------------


def compare_signups(
    name_old: str,
    name_new: str,
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    routes_old: dict[str, set[str]],
    routes_new: dict[str, set[str]],
    tol: float = COORD_TOLERANCE_DEG,
) -> dict[str, pd.DataFrame]:
    """Builds a DataFrame comparing each route between two signups with flags."""
    idx_old, idx_new = df_old.set_index("stop_id"), df_new.set_index("stop_id")
    old_ids, new_ids = set(idx_old.index), set(idx_new.index)

    # added / removed
    added = new_ids - old_ids
    removed = old_ids - new_ids
    added_df = idx_new.loc[list(added)].reset_index()
    added_df["change_type"] = "added"
    removed_df = idx_old.loc[list(removed)].reset_index()
    removed_df["change_type"] = "removed"

    # moved (using simple absolute-difference test)
    common = old_ids & new_ids
    moved_rows: list[dict[str, float | str | None]] = []
    for sid in common:
        lon_old = cast(float, idx_old.at[sid, "stop_lon"])  # ← cast + .at
        lat_old = cast(float, idx_old.at[sid, "stop_lat"])
        lon_new = cast(float, idx_new.at[sid, "stop_lon"])
        lat_new = cast(float, idx_new.at[sid, "stop_lat"])
        if abs(lon_old - lon_new) > tol or abs(lat_old - lat_new) > tol:
            moved_rows.append(
                {
                    "stop_id": sid,
                    "stop_code": idx_new.at[sid, "stop_code"],
                    "stop_name": idx_new.at[sid, "stop_name"],
                    "lat_old": lat_old,
                    "lon_old": lon_old,
                    "lat_new": lat_new,
                    "lon_new": lon_new,
                    "change_type": "moved",
                }
            )
    moved_df = pd.DataFrame(moved_rows)

    # route-service changes
    svc_rows: list[dict[str, str | float | None]] = []
    for sid in old_ids | new_ids:
        r_old = routes_old.get(sid, set())
        r_new = routes_new.get(sid, set())
        started, stopped = r_new - r_old, r_old - r_new
        if started or stopped:
            base = idx_new if sid in idx_new.index else idx_old
            svc_rows.append(
                {
                    "stop_id": sid,
                    "stop_code": base.at[sid, "stop_code"],
                    "stop_name": base.at[sid, "stop_name"],
                    "lat_old": cast(float | None, idx_old.at[sid, "stop_lat"])
                    if sid in idx_old.index
                    else None,
                    "lon_old": cast(float | None, idx_old.at[sid, "stop_lon"])
                    if sid in idx_old.index
                    else None,
                    "lat_new": cast(float | None, idx_new.at[sid, "stop_lat"])
                    if sid in idx_new.index
                    else None,
                    "lon_new": cast(float | None, idx_new.at[sid, "stop_lon"])
                    if sid in idx_new.index
                    else None,
                    "routes_started": ", ".join(sorted(started)),
                    "routes_stopped": ", ".join(sorted(stopped)),
                    "change_type": "route_service_change",
                }
            )
    svc_df = pd.DataFrame(svc_rows)

    # harmonize columns
    base_cols = [
        "stop_id",
        "stop_code",
        "stop_name",
        "lat_old",
        "lon_old",
        "lat_new",
        "lon_new",
        "change_type",
    ]
    added_df = added_df.reindex(columns=base_cols, fill_value=None)
    removed_df = removed_df.reindex(columns=base_cols, fill_value=None)
    moved_df = moved_df.reindex(columns=base_cols, fill_value=None)
    svc_df = svc_df.reindex(
        columns=base_cols + ["routes_started", "routes_stopped"], fill_value=None
    )

    key = f"{name_old}→{name_new}"
    return {
        f"Added_{key}": added_df,
        f"Removed_{key}": removed_df,
        f"Moved_{key}": moved_df,
        f"RouteSvcChange_{key}": svc_df,
    }


# --------------------------------------------------------------------------------------------------
# SERVICE-LEVEL METRIC COMPARISONS
# --------------------------------------------------------------------------------------------------


def build_service_level_changes(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    prev_label: str,
    curr_label: str,
) -> dict[str, pd.DataFrame]:
    """Produce route-level delta tables between two signups."""
    prev_i = prev_df.set_index("route_short_name")
    curr_i = curr_df.set_index("route_short_name")

    added = sorted(set(curr_i.index) - set(prev_i.index))
    deleted = sorted(set(prev_i.index) - set(curr_i.index))
    common = sorted(set(prev_i.index) & set(curr_i.index))

    deltas: list[dict[str, Any]] = []  # ← relaxed value typing
    for rt in common:
        p, c = prev_i.loc[rt], curr_i.loc[rt]

        def dcol(col: str):
            a, b = p[col], c[col]
            return (b - a) if pd.notna(a) and pd.notna(b) else np.nan

        deltas.append(
            {
                "route_short_name": rt,
                "span_old_min": p["span_minutes"],
                "span_new_min": c["span_minutes"],
                "span_delta": dcol("span_minutes"),
                "trips_old": p["trips_count"],
                "trips_new": c["trips_count"],
                "trips_delta": dcol("trips_count"),
                "hdwy_old_min": p["median_headway_min"],
                "hdwy_new_min": c["median_headway_min"],
                "hdwy_delta": dcol("median_headway_min"),
            }
        )

    df_delta = pd.DataFrame(deltas).sort_values("route_short_name")
    df_add = pd.DataFrame({"route_short_name": added})
    df_del = pd.DataFrame({"route_short_name": deleted})

    key = f"{prev_label}→{curr_label}"
    return {
        f"ServiceChange_{key}": df_delta,
        f"Routes_Added_{key}": df_add,
        f"Routes_Deleted_{key}": df_del,
    }


# --------------------------------------------------------------------------------------------------
# INTERLINING & DETAILED CLASSIFICATION (NO GEOMETRY)
# --------------------------------------------------------------------------------------------------


def _build_interlining_map(trips_df: pd.DataFrame, routes_df: pd.DataFrame) -> dict[str, str]:
    """Returns dict: route_short_name -> comma-joined list of other routes sharing the same block_id."""
    merged = trips_df.merge(routes_df[["route_id", "route_short_name"]], on="route_id", how="left")
    blk = merged.groupby("block_id")["route_short_name"].apply(lambda s: set(s.dropna()))
    inter: dict[str, set[str]] = {}
    for route_set in blk:
        for r in route_set:
            others = route_set - {r}
            if others:
                inter.setdefault(r, set()).update(others)
    return {r: ", ".join(sorted(v)) for r, v in inter.items()}


def compare_signups_detailed(
    prev_label: str,
    curr_label: str,
    metrics_prev: pd.DataFrame,
    metrics_curr: pd.DataFrame,
    inter_prev: dict[str, str],
    inter_curr: dict[str, str],
) -> pd.DataFrame:
    """Build a DataFrame comparing each route between two signups with flags."""
    all_routes = sorted(
        set(metrics_prev["route_short_name"]).union(metrics_curr["route_short_name"])
    )
    rows: list[dict[str, str]] = []

    # prepare delta table once for speed
    delta_key = f"ServiceChange_{prev_label}→{curr_label}"
    delta_df = build_service_level_changes(metrics_prev, metrics_curr, prev_label, curr_label)[
        delta_key
    ]
    delta_changed = set(_keep_changed(delta_df)["route_short_name"])

    for rt in all_routes:
        changes: list[str] = []
        in_prev = rt in metrics_prev["route_short_name"].values
        in_curr = rt in metrics_curr["route_short_name"].values

        if in_prev and not in_curr:
            changes.append("Route eliminated")
        elif in_curr and not in_prev:
            changes.append("Route created")
        else:
            # interlining
            if inter_prev.get(rt, "") != inter_curr.get(rt, ""):
                changes.append("Interlining change")
            # metric delta
            if rt in delta_changed:
                changes.append("Span/Trips/Headway change")

        if not changes:
            changes = ["No change"]
        rows.append({"route_short_name": rt, "change_flags": ", ".join(changes)})

    return pd.DataFrame(rows)


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:
    """Entry-point: run the full multi-signup GTFS comparison pipeline."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # ─── LOAD GTFS & BUILD STOP-LEVEL DATA ────────────────────────────────────
    signups_stops: dict[str, pd.DataFrame] = {}
    signups_routes_map: dict[str, dict[str, set[str]]] = {}
    signups_metrics: dict[str, pd.DataFrame] = {}

    for cfg in MULTIPLE_GTFS_CONFIGS:
        name, path = cfg["name"], cfg["path"]
        print(f"Loading GTFS for {name} …")
        stops, stop_routes = load_gtfs_basic(path)
        signups_stops[name] = stops
        signups_routes_map[name] = stop_routes

        print(f"Building Weekday route metrics for {name} …")
        signups_metrics[name] = load_route_metrics(path, schedule_type="Weekday")
    print("All GTFS loads complete.\n")

    # ─── STOP-LEVEL CHANGE WORKBOOK ───────────────────────────────────────────
    all_sheets_stop: dict[str, pd.DataFrame] = {}
    for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
        prev = MULTIPLE_GTFS_CONFIGS[i - 1]["name"]
        curr = MULTIPLE_GTFS_CONFIGS[i]["name"]
        sheets = compare_signups(
            prev,
            curr,
            signups_stops[prev],
            signups_stops[curr],
            signups_routes_map[prev],
            signups_routes_map[curr],
        )
        all_sheets_stop.update(sheets)

    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_STOPS), engine="openpyxl"
    ) as xls:
        for sheet_name, df in all_sheets_stop.items():
            df.to_excel(xls, sheet_name=sheet_name[:31], index=False)
    print("✓ stop-level change workbook written.")

    # ─── ROUTE METRICS + DELTA-ONLY SHEETS ───────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_METRIC)
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as xls:
        # 1) one sheet per signup
        for label, df in signups_metrics.items():
            df.to_excel(xls, sheet_name=label[:31], index=False)

        # 2) delta-only sheets
        for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
            prev = MULTIPLE_GTFS_CONFIGS[i - 1]["name"]
            curr = MULTIPLE_GTFS_CONFIGS[i]["name"]

            delta_dict = build_service_level_changes(
                signups_metrics[prev], signups_metrics[curr], prev, curr
            )
            full = delta_dict[f"ServiceChange_{prev}→{curr}"]
            changed = _keep_changed(full)
            if not changed.empty:
                sheet_nm = f"Changes_{prev}_to_{curr}"[:31]
                changed.to_excel(xls, sheet_name=sheet_nm, index=False)
    print("✓ route metrics workbook written.")

    # ─── SERVICE-LEVEL CHANGE WORKBOOK ───────────────────────────────────────
    delta_sheets: dict[str, pd.DataFrame] = {}
    for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
        prev = MULTIPLE_GTFS_CONFIGS[i - 1]["name"]
        curr = MULTIPLE_GTFS_CONFIGS[i]["name"]
        sheets = build_service_level_changes(
            signups_metrics[prev], signups_metrics[curr], prev, curr
        )
        delta_sheets.update(sheets)

    with pd.ExcelWriter(
        os.path.join(OUTPUT_DIRECTORY, OUTPUT_EXCEL_NAME_DELTA), engine="openpyxl"
    ) as xls:
        for sheet_name, df in delta_sheets.items():
            df.to_excel(xls, sheet_name=sheet_name[:31], index=False)
    print("✓ service-level change workbook written.")

    # ─── INTERLINING & DETAILED COMPARISON (NO GEOMETRY) ─────────────────────
    interlining_map: dict[str, dict[str, str]] = {}
    for cfg in MULTIPLE_GTFS_CONFIGS:
        name, path = cfg["name"], cfg["path"]
        trips = pd.read_csv(
            os.path.join(path, "trips.txt"), dtype=str, usecols=["route_id", "block_id"]
        )
        routes = pd.read_csv(
            os.path.join(path, "routes.txt"),
            dtype=str,
            usecols=["route_id", "route_short_name"],
        )
        if FILTER_SET:
            routes = routes.loc[~routes["route_short_name"].isin(FILTER_SET)]
        interlining_map[name] = _build_interlining_map(trips, routes)

    # detailed comparison workbook
    comp_path = os.path.join(OUTPUT_DIRECTORY, COMPARISON_EXCEL)
    with pd.ExcelWriter(comp_path, engine="openpyxl") as xls:
        for i in range(1, len(MULTIPLE_GTFS_CONFIGS)):
            prev = MULTIPLE_GTFS_CONFIGS[i - 1]["name"]
            curr = MULTIPLE_GTFS_CONFIGS[i]["name"]
            df_comp = compare_signups_detailed(
                prev,
                curr,
                signups_metrics[prev],
                signups_metrics[curr],
                interlining_map[prev],
                interlining_map[curr],
            )
            sheet = f"{prev}→{curr}"[:31]
            df_comp.to_excel(xls, sheet_name=sheet, index=False)
    print("✓ detailed comparison workbook written.")


if __name__ == "__main__":
    main()
